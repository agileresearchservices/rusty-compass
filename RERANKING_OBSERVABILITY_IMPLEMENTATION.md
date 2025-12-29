# Reranking Observability Enhancement - Implementation Summary

## Overview
Enhanced the reranking pipeline to emit detailed observability events during the tools node execution, providing real-time visibility into reranker performance, score changes, and document ranking adjustments.

## Changes Made

### 1. **File: `/Users/kevin/github/personal/rusty-compass/langchain_agent/main.py`**

#### Modified: `tools_node()` method (lines 979-1009)

**What was changed:**
- Added storage of **original rank** (position before reranking) in document metadata
- Added storage of **reranker score** in document metadata
- Stores this data while executing the reranker pipeline

**Code added:**
```python
# Capture original order for comparison and store original ranks
original_sources = [doc.metadata.get('source', 'unknown') for doc in results]
for i, doc in enumerate(results, 1):
    doc.metadata['original_rank'] = i

# ... reranking happens ...

# Store reranker scores in metadata and update metadata for observability
for i, (doc, score) in enumerate(results_with_scores, 1):
    doc.metadata['reranker_score'] = score
```

**Why:**
These metadata fields are later extracted by the observable_agent service to compute detailed RerankedDocument objects with full ranking and scoring information.

---

### 2. **File: `/Users/kevin/github/personal/rusty-compass/langchain_agent/api/services/observable_agent.py`**

#### Modified: `_emit_node_events()` method - "tools" node section (lines 437-452)

**What was changed:**
- Removed the simple `RerankerStartEvent` emission
- Added conditional logic that:
  1. Emits `RerankerStartEvent` with model and candidate count
  2. Computes detailed `RerankedDocument` objects using new helper method
  3. Checks if reranking changed the document order
  4. Emits `RerankerResultEvent` with full reranking results

**Code added:**
```python
# Emit reranking events if enabled
if ENABLE_RERANKING:
    # Emit reranker start event with candidate count
    await emit(RerankerStartEvent(
        model=RERANKER_MODEL,
        candidate_count=len(documents),
    ))

    # Emit reranker result event with detailed document information
    reranked_docs = self._compute_reranked_documents(documents)
    reranking_changed_order = self._check_if_order_changed(documents, reranked_docs)

    await emit(RerankerResultEvent(
        results=reranked_docs,
        reranking_changed_order=reranking_changed_order,
    ))
```

#### Added: `_compute_reranked_documents()` helper method (lines 586-622)

**Purpose:** Transform LangChain Document objects into structured RerankedDocument objects

**Returns:** List of `RerankedDocument` objects with:
- `source`: Document source identifier
- `score`: Reranker cross-encoder score (0.0-1.0)
- `rank`: Current position after reranking (1-indexed)
- `original_rank`: Position before reranking
- `snippet`: First 200 characters of document content
- `rank_change`: Integer representing position change (original_rank - rank)
  - Positive values = moved up in ranking
  - Negative values = moved down
  - Zero = no change

**Implementation details:**
```python
def _compute_reranked_documents(self, documents: List) -> List:
    """Compute RerankedDocument objects from retrieved documents."""
    reranked_docs = []

    for rank, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "unknown")
        # Extract reranker score if available, otherwise use 0.0
        score = doc.metadata.get("reranker_score", 0.0)
        # Extract original rank if available, otherwise estimate based on position
        original_rank = doc.metadata.get("original_rank", rank)
        # Calculate rank change (positive = moved up)
        rank_change = original_rank - rank

        snippet = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content

        reranked_docs.append(RerankedDocument(
            source=source,
            score=score,
            rank=rank,
            original_rank=original_rank,
            snippet=snippet,
            rank_change=rank_change,
        ))

    return reranked_docs
```

#### Added: `_check_if_order_changed()` helper method (lines 624-639)

**Purpose:** Determine whether reranking altered the document order

**Returns:** Boolean indicating if any document's rank changed

**Implementation:**
```python
def _check_if_order_changed(self, documents: List, reranked_docs: List) -> bool:
    """Determine if reranking changed the document order."""
    return any(doc.rank_change != 0 for doc in reranked_docs)
```

---

## Data Flow

### Request Flow During Tools Node Execution:

1. **Agent calls `knowledge_base` tool**
   - Retriever invokes hybrid search
   - Returns initial results

2. **If ENABLE_RERANKING is True:**
   - `tools_node()` in main.py:
     - Stores original ranks in `doc.metadata['original_rank']`
     - Calls reranker on documents
     - Stores reranker scores in `doc.metadata['reranker_score']`

3. **Observable agent processes output:**
   - `_emit_node_events()` is called with "tools" node output
   - Contains `retrieved_documents` with scores and original ranks in metadata

4. **Reranking events are emitted:**
   - `RerankerStartEvent`: Model name and candidate count
   - `RerankerResultEvent`: Array of RerankedDocument objects with full details
     - Each includes score, rank, original_rank, and rank_change
     - Boolean flag indicating if order changed

---

## Event Schema

### RerankerStartEvent
```python
{
    "type": "reranker_start",
    "node": "tools",
    "model": "Qwen/Qwen3-Reranker-8B",
    "candidate_count": 12,
    "timestamp": "2024-01-15T10:30:45.123456"
}
```

### RerankerResultEvent
```python
{
    "type": "reranker_result",
    "node": "tools",
    "results": [
        {
            "source": "doc_id_1",
            "score": 0.9234,
            "rank": 1,
            "original_rank": 3,
            "snippet": "The quick brown fox jumps over the lazy dog...",
            "rank_change": 2  # Moved up 2 positions
        },
        {
            "source": "doc_id_2",
            "score": 0.8567,
            "rank": 2,
            "original_rank": 1,
            "snippet": "Lorem ipsum dolor sit amet...",
            "rank_change": -1  # Moved down 1 position
        }
    ],
    "reranking_changed_order": true,
    "timestamp": "2024-01-15T10:30:45.234567"
}
```

---

## Configuration

The implementation uses these config values from `/Users/kevin/github/personal/rusty-compass/langchain_agent/config.py`:

```python
ENABLE_RERANKING = True              # Enable reranking pipeline
RERANKER_MODEL = "Qwen/Qwen3-Reranker-8B"  # Model to use
RERANKER_TOP_K = 4                   # Top K documents to keep after reranking
RERANKER_FETCH_K = 12                # Initial fetch before reranking
```

---

## Observability Benefits

### For UI/Frontend:
1. **Real-time visibility** into reranker execution
2. **Score visualization** - show cross-encoder relevance scores
3. **Ranking changes** - highlight documents that were reordered
4. **Performance metrics** - track reranker latency via event timestamps

### For Debugging:
1. **Identify ranking discrepancies** - see where hybrid search was wrong
2. **Score distribution** - understand relevance score patterns
3. **Order changes** - trace which documents improved/degraded in relevance
4. **Model performance** - evaluate Qwen3 reranker effectiveness

### For Analytics:
1. **Reranking impact** - measure how often order changes
2. **Score patterns** - analyze relevance score distributions
3. **Document quality** - identify documents that benefit most from reranking
4. **Model effectiveness** - A/B test different reranking models

---

## Event Emission Timing

Events are emitted in the `_emit_node_events()` method during the tools node processing:

1. **HybridSearchResultEvent** - Initial search results (before reranking)
2. **RerankerStartEvent** - Reranker begins processing (if ENABLE_RERANKING=True)
3. **RerankerResultEvent** - Reranker completes with scored results (if ENABLE_RERANKING=True)
4. All events flow through WebSocket to frontend for real-time UI updates

---

## Backward Compatibility

- Events only emit when `ENABLE_RERANKING = True`
- No changes to core retrieval or reranking logic
- Metadata additions are non-intrusive (new fields in doc.metadata)
- Existing code paths unaffected
- Works with both legacy and improved streaming modes

---

## Testing Recommendations

### Unit Tests:
1. `_compute_reranked_documents()` - verify document transformation
2. `_check_if_order_changed()` - verify order change detection
3. Metadata storage in main.py - verify scores and ranks are stored

### Integration Tests:
1. Full pipeline with reranking enabled - verify events are emitted
2. Event sequence validation - RerankerStart → RerankerResult order
3. Score and rank accuracy - verify values match reranker output

### E2E Tests:
1. WebSocket event delivery - verify frontend receives events
2. UI rendering - verify reranking events display correctly
3. Performance - measure added latency from observability

---

## Implementation Checklist

- [x] Enhanced main.py tools_node to store original ranks
- [x] Enhanced main.py tools_node to store reranker scores
- [x] Enhanced observable_agent.py to emit RerankerStartEvent
- [x] Created _compute_reranked_documents() method
- [x] Created _check_if_order_changed() method
- [x] Enhanced observable_agent.py to emit RerankerResultEvent
- [x] Event classes already defined in events.py
- [x] ENABLE_RERANKING check implemented
- [x] Syntax validation passed
- [x] Backward compatibility verified

---

## Files Modified

1. `/Users/kevin/github/personal/rusty-compass/langchain_agent/main.py`
   - Modified `tools_node()` method (3 new lines in reranking section)

2. `/Users/kevin/github/personal/rusty-compass/langchain_agent/api/services/observable_agent.py`
   - Modified `_emit_node_events()` method (16 new lines in tools section)
   - Added `_compute_reranked_documents()` method (37 lines)
   - Added `_check_if_order_changed()` method (16 lines)

3. No changes needed to:
   - `/Users/kevin/github/personal/rusty-compass/langchain_agent/api/schemas/events.py` (already has needed classes)
   - Configuration files (already have required settings)

---

## Notes

- The implementation gracefully handles missing metadata (defaults to 0.0 for scores)
- Rank change calculation is accurate even for partial reranking (fewer results after reranking)
- Event emission is conditional on ENABLE_RERANKING flag
- Compatible with both async streaming modes (legacy and improved)

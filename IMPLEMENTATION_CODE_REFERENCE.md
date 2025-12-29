# Reranking Observability - Code Reference

## Quick Reference: Exact Changes Made

### 1. File: langchain_agent/main.py

**Location:** `tools_node()` method, reranking section (lines 979-1009)

**Before:**
```python
                # Apply reranking if enabled
                if ENABLE_RERANKING and self.reranker and results:
                    # Capture original order for comparison
                    original_sources = [doc.metadata.get('source', 'unknown') for doc in results]

                    print(f"\n[Reranker] Processing {len(results)} candidates...")
                    reranked_results = self.reranker.rerank(query, results, RERANKER_TOP_K)

                    # Extract documents with scores for logging
                    results_with_scores = [(doc, score) for doc, score in reranked_results]
                    results = [doc for doc, score in results_with_scores]

                    # Log reranking results with scores
                    print(f"[Reranker] Reranking complete → top {len(results)} selected:")
                    for i, (doc, score) in enumerate(results_with_scores, 1):
                        source = doc.metadata.get('source', 'unknown')
                        relevance_bar = "█" * int(score * 20)
                        print(f"  {i}. score={score:.4f} {relevance_bar} [{source}]")

                    # Log order changes if applicable
                    reranked_sources = [doc.metadata.get('source', 'unknown') for doc in results]
                    if original_sources[:len(reranked_sources)] != reranked_sources:
                        print(f"[Reranker] Order changed: {original_sources[:len(reranked_sources)]} → {reranked_sources}")
                    else:
                        print(f"[Reranker] Order unchanged (already optimally ranked)")
```

**After:**
```python
                # Apply reranking if enabled
                if ENABLE_RERANKING and self.reranker and results:
                    # Capture original order for comparison and store original ranks
                    original_sources = [doc.metadata.get('source', 'unknown') for doc in results]
                    for i, doc in enumerate(results, 1):
                        doc.metadata['original_rank'] = i

                    print(f"\n[Reranker] Processing {len(results)} candidates...")
                    reranked_results = self.reranker.rerank(query, results, RERANKER_TOP_K)

                    # Extract documents with scores and store in metadata
                    results_with_scores = [(doc, score) for doc, score in reranked_results]
                    results = [doc for doc, score in results_with_scores]

                    # Store reranker scores in metadata and update metadata for observability
                    for i, (doc, score) in enumerate(results_with_scores, 1):
                        doc.metadata['reranker_score'] = score

                    # Log reranking results with scores
                    print(f"[Reranker] Reranking complete → top {len(results)} selected:")
                    for i, (doc, score) in enumerate(results_with_scores, 1):
                        source = doc.metadata.get('source', 'unknown')
                        relevance_bar = "█" * int(score * 20)
                        print(f"  {i}. score={score:.4f} {relevance_bar} [{source}]")

                    # Log order changes if applicable
                    reranked_sources = [doc.metadata.get('source', 'unknown') for doc in results]
                    if original_sources[:len(reranked_sources)] != reranked_sources:
                        print(f"[Reranker] Order changed: {original_sources[:len(reranked_sources)]} → {reranked_sources}")
                    else:
                        print(f"[Reranker] Order unchanged (already optimally ranked)")
```

**Key additions:**
1. Line 984: Store original ranks in metadata before reranking
2. Lines 994-995: Store reranker scores in metadata after reranking

---

### 2. File: langchain_agent/api/services/observable_agent.py

#### Change #1: Modify `_emit_node_events()` method

**Location:** Lines 422-452 (the "tools" node section)

**Before:**
```python
        elif node_name == "tools":
            # Emit search events
            documents = output.get("retrieved_documents", [])
            if documents:
                await emit(HybridSearchResultEvent(
                    candidate_count=len(documents),
                    candidates=[
                        SearchCandidate(
                            source=doc.metadata.get("source", "unknown"),
                            snippet=doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        )
                        for doc in documents[:10]  # Limit to 10 for UI
                    ],
                ))

                # Emit reranker events if enabled
                if ENABLE_RERANKING:
                    await emit(RerankerStartEvent(
                        model=RERANKER_MODEL,
                        candidate_count=len(documents),
                    ))
```

**After:**
```python
        elif node_name == "tools":
            # Emit search events
            documents = output.get("retrieved_documents", [])
            if documents:
                await emit(HybridSearchResultEvent(
                    candidate_count=len(documents),
                    candidates=[
                        SearchCandidate(
                            source=doc.metadata.get("source", "unknown"),
                            snippet=doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        )
                        for doc in documents[:10]  # Limit to 10 for UI
                    ],
                ))

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

**Key additions:**
1. Lines 445-447: Compute RerankedDocument objects and check if order changed
2. Lines 449-452: Emit RerankerResultEvent with full reranking details

#### Change #2: Add `_compute_reranked_documents()` method

**Location:** After `_parse_ollama_response()` method (lines 586-622)

**Full implementation:**
```python
    def _compute_reranked_documents(self, documents: List) -> List:
        """
        Compute RerankedDocument objects from retrieved documents.

        Since the documents in output["retrieved_documents"] are already reranked
        by the tools_node before reaching this method, we construct RerankedDocument
        objects using their current positions and extract scores from metadata.

        Args:
            documents: List of LangChain Document objects (already reranked)

        Returns:
            List of RerankedDocument objects with ranking information
        """
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

#### Change #3: Add `_check_if_order_changed()` method

**Location:** After `_compute_reranked_documents()` method (lines 624-639)

**Full implementation:**
```python
    def _check_if_order_changed(self, documents: List, reranked_docs: List) -> bool:
        """
        Determine if reranking changed the document order.

        This is a heuristic check based on whether any document moved from its
        original position. Since we don't have the pre-reranking order directly,
        we check if any document has a non-zero rank_change value.

        Args:
            documents: List of LangChain Document objects (reranked)
            reranked_docs: List of RerankedDocument objects with rank info

        Returns:
            Boolean indicating if any document's rank changed
        """
        return any(doc.rank_change != 0 for doc in reranked_docs)
```

---

## Event Classes (Already Exist in events.py)

### RerankerStartEvent
```python
class RerankerStartEvent(BaseEvent):
    """Emitted when reranking begins."""

    type: Literal["reranker_start"] = "reranker_start"
    node: Literal["tools"] = "tools"
    model: str
    candidate_count: int
```

### RerankedDocument
```python
class RerankedDocument(BaseModel):
    """A document after reranking with its new score and rank."""

    source: str
    score: float  # Cross-encoder score (0.0-1.0)
    rank: int  # New rank after reranking
    original_rank: int  # Rank before reranking
    snippet: str
    rank_change: int = 0  # How much the rank changed
```

### RerankerResultEvent
```python
class RerankerResultEvent(BaseEvent):
    """Emitted when reranking completes with scored documents."""

    type: Literal["reranker_result"] = "reranker_result"
    node: Literal["tools"] = "tools"
    results: List[RerankedDocument]
    reranking_changed_order: bool = False
```

---

## Configuration Values Used

From `langchain_agent/config.py`:

```python
ENABLE_RERANKING = True
RERANKER_MODEL = "Qwen/Qwen3-Reranker-8B"
RERANKER_TOP_K = 4
RERANKER_FETCH_K = 12
```

---

## Data Flow Diagram

```
User Query
    ↓
Agent asks knowledge_base tool
    ↓
Hybrid Retriever (main.py)
    ├─ Vector search (semantic)
    └─ Text search (lexical)
    ↓
Results (12 documents) → Store original_rank in metadata
    ↓
[IF ENABLE_RERANKING]
    ↓
Qwen3 Reranker (main.py)
    ├─ Score each document
    └─ Sort by relevance
    ↓
Results (4 best documents) → Store reranker_score in metadata
    ↓
Return from tools_node
    ↓
Observable Agent (observable_agent.py)
    ├─ HybridSearchResultEvent (initial 12)
    ├─ RerankerStartEvent (12 candidates)
    └─ RerankerResultEvent (computed from metadata)
         ├─ source
         ├─ score (from metadata)
         ├─ rank (position in final list)
         ├─ original_rank (from metadata)
         ├─ snippet
         └─ rank_change (calculated)
    ↓
WebSocket → Frontend UI
```

---

## Testing the Implementation

### Quick Smoke Test
```python
# Verify the methods exist
from langchain_agent.api.services.observable_agent import ObservableAgentService

service = ObservableAgentService()
assert hasattr(service, '_compute_reranked_documents')
assert hasattr(service, '_check_if_order_changed')
print("✓ Methods exist")
```

### Unit Test Example
```python
import pytest
from langchain_agent.api.schemas.events import RerankedDocument
from langchain_core.documents import Document

def test_compute_reranked_documents():
    service = ObservableAgentService()

    # Create mock documents with metadata
    docs = [
        Document(
            page_content="Test content 1",
            metadata={"source": "doc1", "reranker_score": 0.9, "original_rank": 2}
        ),
        Document(
            page_content="Test content 2",
            metadata={"source": "doc2", "reranker_score": 0.85, "original_rank": 1}
        ),
    ]

    result = service._compute_reranked_documents(docs)

    assert len(result) == 2
    assert result[0].rank == 1
    assert result[0].original_rank == 2
    assert result[0].rank_change == 1  # Moved up
    assert result[1].rank == 2
    assert result[1].original_rank == 1
    assert result[1].rank_change == -1  # Moved down
```

### Integration Test Example
```python
import asyncio
from langchain_agent.api.services.observable_agent import ObservableAgentService

async def test_reranking_events():
    service = ObservableAgentService()
    await service.ensure_initialized()

    events_captured = []

    async def emit_callback(event):
        events_captured.append(event)

    # Process message
    await service.process_message(
        "What is the capital of France?",
        "test-thread-id",
        emit_callback
    )

    # Verify reranking events were emitted
    reranker_start_events = [e for e in events_captured
                            if hasattr(e, 'type') and e.type == 'reranker_start']
    reranker_result_events = [e for e in events_captured
                             if hasattr(e, 'type') and e.type == 'reranker_result']

    assert len(reranker_start_events) > 0, "RerankerStartEvent not emitted"
    assert len(reranker_result_events) > 0, "RerankerResultEvent not emitted"

    # Verify result event has proper data
    result_event = reranker_result_events[0]
    assert result_event.reranking_changed_order in [True, False]
    assert len(result_event.results) > 0
    assert all(hasattr(r, 'rank') for r in result_event.results)
    assert all(hasattr(r, 'score') for r in result_event.results)
```

---

## Common Issues & Solutions

### Issue: Events not emitting
**Solution:** Verify `ENABLE_RERANKING = True` in config.py

### Issue: Score is 0.0 for all documents
**Solution:** Verify main.py correctly stores `reranker_score` in metadata (line 995)

### Issue: rank_change calculation is wrong
**Solution:** Verify `original_rank` is stored before reranking (line 984 in main.py)

### Issue: RerankerResultEvent not received in frontend
**Solution:** Check WebSocket connection and verify events are being emitted (lines 449-452 in observable_agent.py)

---

## Summary of Lines Changed

| File | Lines | Change |
|------|-------|--------|
| main.py | 983-984 | Store original_rank in metadata |
| main.py | 994-995 | Store reranker_score in metadata |
| observable_agent.py | 445-452 | Emit RerankerStartEvent and RerankerResultEvent |
| observable_agent.py | 586-622 | Add _compute_reranked_documents method |
| observable_agent.py | 624-639 | Add _check_if_order_changed method |

**Total additions: ~70 lines of code**
**Total modifications: ~5 lines of code**

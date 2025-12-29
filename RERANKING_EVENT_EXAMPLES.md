# Reranking Observability - Event Examples

## Real-World Event Sequence Example

### Scenario: User asks "What is the capital of France?"

The following events would be emitted during the tools node execution:

---

## 1. HybridSearchResultEvent (Before Reranking)

Initial hybrid search returns 12 candidates:

```json
{
    "type": "hybrid_search_result",
    "node": "tools",
    "candidate_count": 12,
    "candidates": [
        {
            "source": "wikipedia_france_page_1",
            "snippet": "France is a country in Western Europe, located on the Atlantic coast..."
        },
        {
            "source": "geography_cities_2",
            "snippet": "Paris is the largest city in France and serves as the capital..."
        },
        {
            "source": "history_europe_3",
            "snippet": "During the medieval period, several cities competed for prominence..."
        },
        {
            "source": "tourism_guide_4",
            "snippet": "Paris offers iconic landmarks like the Eiffel Tower and museums..."
        },
        {
            "source": "economy_industrial_5",
            "snippet": "France's economy is diversified with manufacturing and services..."
        },
        {
            "source": "climate_regions_6",
            "snippet": "Northern France has a temperate climate with moderate rainfall..."
        },
        {
            "source": "culture_arts_7",
            "snippet": "French art and culture have influenced world aesthetics for centuries..."
        },
        {
            "source": "agriculture_production_8",
            "snippet": "France produces extensive vineyards and agricultural products..."
        },
        {
            "source": "sports_events_9",
            "snippet": "Paris hosted the 1924 and 2024 Olympic Games..."
        },
        {
            "source": "politics_government_10",
            "snippet": "The French government operates as a semi-presidential system..."
        },
        {
            "source": "transport_rail_11",
            "snippet": "Paris has an extensive metro system and rail networks..."
        },
        {
            "source": "entertainment_theater_12",
            "snippet": "The Palais Garnier opera house is located in central Paris..."
        }
    ],
    "timestamp": "2024-01-15T10:30:45.123456"
}
```

**Note:** Documents are in their original hybrid search ranking order. Notice:
- Index 0: general France article (good but not specific)
- Index 1: direct answer about Paris/capital (relevant)
- Index 2-11: various tangentially related topics

---

## 2. RerankerStartEvent

Reranker begins processing these 12 candidates:

```json
{
    "type": "reranker_start",
    "node": "tools",
    "model": "Qwen/Qwen3-Reranker-8B",
    "candidate_count": 12,
    "timestamp": "2024-01-15T10:30:45.234567"
}
```

**Note:** The reranker will use the query "What is the capital of France?" to score each document's relevance.

---

## 3. RerankerResultEvent

After reranking, top 4 documents are selected and reordered:

```json
{
    "type": "reranker_result",
    "node": "tools",
    "results": [
        {
            "source": "geography_cities_2",
            "score": 0.9876,
            "rank": 1,
            "original_rank": 2,
            "snippet": "Paris is the largest city in France and serves as the capital...",
            "rank_change": 1
        },
        {
            "source": "wikipedia_france_page_1",
            "score": 0.8234,
            "rank": 2,
            "original_rank": 1,
            "snippet": "France is a country in Western Europe, located on the Atlantic coast...",
            "rank_change": -1
        },
        {
            "source": "politics_government_10",
            "score": 0.7123,
            "rank": 3,
            "original_rank": 10,
            "snippet": "The French government operates as a semi-presidential system...",
            "rank_change": 7
        },
        {
            "source": "transport_rail_11",
            "score": 0.6789,
            "rank": 4,
            "original_rank": 11,
            "snippet": "Paris has an extensive metro system and rail networks...",
            "rank_change": 7
        }
    ],
    "reranking_changed_order": true,
    "timestamp": "2024-01-15T10:30:45.345678"
}
```

**Analysis of Results:**

| Document | Original Position | New Position | Score | Movement | Reason |
|----------|-------------------|--------------|-------|----------|--------|
| geography_cities_2 | 2 | 1 | 0.9876 | +1 up | **Highly relevant** - Direct answer to "capital" |
| wikipedia_france_page_1 | 1 | 2 | 0.8234 | -1 down | Relevant but less specific than direct answer |
| politics_government_10 | 10 | 3 | 0.7123 | +7 up | Mentions government/capital structure |
| transport_rail_11 | 11 | 4 | 0.6789 | +7 up | Mentions Paris infrastructure |

**Key Observations:**
- Reranker moved the most relevant document (geography_cities_2) to position 1
- Pushed down the generic France article that was initially ranked higher
- Promoted documents about government/infrastructure that mention Paris
- Successfully corrected the initial hybrid search ranking

---

## Frontend/UI Visualization Example

### Timeline View:
```
[10:30:45.123] HybridSearchResultEvent: 12 candidates retrieved
                └─ First result: "France is a country..." (generic)
                └─ Second result: "Paris is the largest..." (specific)

[10:30:45.234] RerankerStartEvent: Starting Qwen3 reranker
                └─ Processing 12 candidates...

[10:30:45.345] RerankerResultEvent: Reranking complete
                ✓ Order changed: YES
                ✓ Top result: "Paris is the largest..." (score: 0.9876)
                ✓ Best match moved from position 2 → 1 (+1)
```

### Ranking Changes Visualization:
```
Initial Ranking (from hybrid search):
  1. ▓▓▓▓▓▓░░░░ France article (0.65)
  2. ▓▓▓▓▓▓▓░░░ Paris capital (0.72)
  3. ▓▓▓▓░░░░░░ Medieval history (0.45)
  ...

After Reranking (Qwen3-Reranker-8B):
  1. ▓▓▓▓▓▓▓▓▓░ Paris capital (0.9876) ⬆ moved from #2
  2. ▓▓▓▓▓▓▓░░░ France article (0.8234) ⬇ moved from #1
  3. ▓▓▓▓▓▓░░░░ Government system (0.7123) ⬆ moved from #10
  4. ▓▓▓▓▓░░░░░ Paris transport (0.6789) ⬆ moved from #11
```

### Score Distribution:
```
Score Range Analysis:
  0.9-1.0: 1 document (9% of reranked results)
  0.8-0.9: 1 document (9%)
  0.7-0.8: 1 document (9%)
  0.6-0.7: 1 document (9%)

Reranking Impact:
  ✓ Most relevant document: Correctly ranked #1 (score 0.9876)
  ✓ Order changed: 4 out of 4 top results had rank shifts
  ✓ Average score improvement: +0.15 (before reranking median)
```

---

## Code Integration Example

### How the Frontend Would Use These Events:

```javascript
// Example WebSocket event handler
websocket.onmessage = (event) => {
    const message = JSON.parse(event.data);

    if (message.type === 'reranker_start') {
        // Show spinner: "Reranking 12 candidates..."
        showRerankerProgress(message.candidate_count);
    }

    if (message.type === 'reranker_result') {
        // Hide spinner
        hideRerankerProgress();

        // Update document list with new ranking
        updateDocumentRanking(message.results);

        // Show ranking changes
        if (message.reranking_changed_order) {
            showRankingChanges(message.results);
        }

        // Highlight the best result
        const topResult = message.results[0];
        showHighlight(topResult, {
            score: topResult.score,
            rankChange: topResult.rank_change,
            icon: 'rank-up' // If moved up
        });
    }
};

// Render ranking changes
function showRankingChanges(results) {
    results.forEach(doc => {
        if (doc.rank_change > 0) {
            // Green arrow up
            console.log(`⬆ ${doc.source}: moved up ${doc.rank_change} positions`);
        } else if (doc.rank_change < 0) {
            // Red arrow down
            console.log(`⬇ ${doc.source}: moved down ${-doc.rank_change} positions`);
        } else {
            // No change
            console.log(`= ${doc.source}: unchanged`);
        }
    });
}
```

---

## Performance Metrics Example

### Combined with MetricsEvent:

```json
{
    "type": "metrics",
    "retrieval_ms": 125.43,
    "reranking_ms": 45.67,
    "document_grading_ms": 89.23,
    "llm_generation_ms": 234.56,
    "total_ms": 495.89,
    "timestamp": "2024-01-15T10:30:46.000000"
}
```

**Reranking Overhead:**
- Retrieval (hybrid search): 125ms
- Reranking (Qwen3-Reranker-8B on 12 docs): 45ms
- Total retrieval + reranking: 170ms (~34% of total pipeline)

---

## Edge Cases

### Case 1: No Reranking Changes
```json
{
    "type": "reranker_result",
    "results": [
        {
            "source": "doc_1",
            "score": 0.95,
            "rank": 1,
            "original_rank": 1,
            "rank_change": 0
        },
        {
            "source": "doc_2",
            "score": 0.87,
            "rank": 2,
            "original_rank": 2,
            "rank_change": 0
        }
    ],
    "reranking_changed_order": false
}
```
**Interpretation:** Hybrid search already had optimal ranking. Reranker confirmed quality.

### Case 2: Significant Reordering
```json
{
    "results": [
        {
            "source": "doc_8",
            "rank": 1,
            "original_rank": 8,
            "rank_change": 7
        }
    ],
    "reranking_changed_order": true
}
```
**Interpretation:** Document buried at position 8 was actually most relevant. Reranker significantly improved ranking.

### Case 3: Partial Results
```json
{
    "candidate_count": 12,
    "results": [
        // 4 results (top_k=4 in config)
    ]
}
```
**Interpretation:** Hybrid search returned 12 candidates, but reranker selected only top 4 to keep.

---

## Summary

The reranking observability events provide:

1. **Transparency** - See exactly how reranker scored documents
2. **Validation** - Confirm reranker improved upon hybrid search
3. **Debugging** - Identify when reranking helps vs hurts
4. **Analytics** - Track reranker effectiveness over time
5. **User Feedback** - Show why results are ranked the way they are

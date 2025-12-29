# Reranking Observability - Testing Guide

## Overview
This guide provides comprehensive testing strategies for the enhanced reranking observability implementation.

## Test Categories

### 1. Unit Tests

#### Test: _compute_reranked_documents()

**File:** Create `test_observable_agent.py`

```python
import pytest
from langchain_core.documents import Document
from api.services.observable_agent import ObservableAgentService
from api.schemas.events import RerankedDocument

class TestComputeRerankedDocuments:
    """Tests for _compute_reranked_documents method"""

    def setup_method(self):
        self.service = ObservableAgentService()

    def test_basic_transformation(self):
        """Verify documents are correctly transformed to RerankedDocument"""
        docs = [
            Document(
                page_content="Test content 1",
                metadata={
                    "source": "doc_1",
                    "reranker_score": 0.95,
                    "original_rank": 2
                }
            ),
        ]

        result = self.service._compute_reranked_documents(docs)

        assert len(result) == 1
        assert isinstance(result[0], RerankedDocument)
        assert result[0].source == "doc_1"
        assert result[0].score == 0.95
        assert result[0].rank == 1
        assert result[0].original_rank == 2

    def test_rank_change_calculation_moved_up(self):
        """Verify rank_change is positive when document moved up"""
        docs = [
            Document(
                page_content="Content",
                metadata={
                    "source": "doc_1",
                    "reranker_score": 0.9,
                    "original_rank": 5
                }
            ),
        ]

        result = self.service._compute_reranked_documents(docs)

        assert result[0].rank_change == 4  # 5 - 1 = 4 (moved up 4 positions)

    def test_rank_change_calculation_moved_down(self):
        """Verify rank_change is negative when document moved down"""
        docs = [
            Document(
                page_content="Content",
                metadata={
                    "source": "doc_1",
                    "reranker_score": 0.5,
                    "original_rank": 1
                }
            ),
        ]

        result = self.service._compute_reranked_documents(docs)

        assert result[0].rank_change == -0  # Document is at rank 1, was at 1

    def test_rank_change_no_change(self):
        """Verify rank_change is zero when document didn't move"""
        docs = [
            Document(
                page_content="Content",
                metadata={
                    "source": "doc_1",
                    "reranker_score": 0.9,
                    "original_rank": 1
                }
            ),
        ]

        result = self.service._compute_reranked_documents(docs)

        assert result[0].rank_change == 0

    def test_score_extraction_with_default(self):
        """Verify score defaults to 0.0 if not in metadata"""
        docs = [
            Document(
                page_content="Content",
                metadata={"source": "doc_1"}
                # No reranker_score or original_rank
            ),
        ]

        result = self.service._compute_reranked_documents(docs)

        assert result[0].score == 0.0
        assert result[0].original_rank == 1  # Defaults to current rank

    def test_snippet_truncation(self):
        """Verify long content is truncated to 200 chars"""
        long_content = "x" * 500
        docs = [
            Document(
                page_content=long_content,
                metadata={"source": "doc_1", "original_rank": 1}
            ),
        ]

        result = self.service._compute_reranked_documents(docs)

        assert len(result[0].snippet) == 203  # 200 + "..."
        assert result[0].snippet.endswith("...")

    def test_snippet_no_truncation(self):
        """Verify short content is not truncated"""
        short_content = "Short text"
        docs = [
            Document(
                page_content=short_content,
                metadata={"source": "doc_1", "original_rank": 1}
            ),
        ]

        result = self.service._compute_reranked_documents(docs)

        assert result[0].snippet == short_content

    def test_multiple_documents(self):
        """Verify multiple documents are correctly ranked"""
        docs = [
            Document(
                page_content="First",
                metadata={"source": "doc_1", "reranker_score": 0.9, "original_rank": 3}
            ),
            Document(
                page_content="Second",
                metadata={"source": "doc_2", "reranker_score": 0.8, "original_rank": 1}
            ),
            Document(
                page_content="Third",
                metadata={"source": "doc_3", "reranker_score": 0.7, "original_rank": 2}
            ),
        ]

        result = self.service._compute_reranked_documents(docs)

        assert len(result) == 3
        assert result[0].rank == 1
        assert result[1].rank == 2
        assert result[2].rank == 3
        assert result[0].rank_change == 2  # 3 -> 1
        assert result[1].rank_change == -1  # 1 -> 2
        assert result[2].rank_change == -1  # 2 -> 3
```

#### Test: _check_if_order_changed()

```python
class TestCheckIfOrderChanged:
    """Tests for _check_if_order_changed method"""

    def setup_method(self):
        self.service = ObservableAgentService()

    def test_order_changed_document_moved_up(self):
        """Verify returns True when document moved up"""
        docs = [Document(page_content="Content", metadata={})]
        reranked_docs = [
            RerankedDocument(
                source="doc_1",
                score=0.9,
                rank=1,
                original_rank=3,
                snippet="content",
                rank_change=2  # Moved up
            )
        ]

        result = self.service._check_if_order_changed(docs, reranked_docs)

        assert result is True

    def test_order_changed_document_moved_down(self):
        """Verify returns True when document moved down"""
        docs = [Document(page_content="Content", metadata={})]
        reranked_docs = [
            RerankedDocument(
                source="doc_1",
                score=0.9,
                rank=3,
                original_rank=1,
                snippet="content",
                rank_change=-2  # Moved down
            )
        ]

        result = self.service._check_if_order_changed(docs, reranked_docs)

        assert result is True

    def test_no_order_change(self):
        """Verify returns False when all documents unchanged"""
        docs = [
            Document(page_content="Content 1", metadata={}),
            Document(page_content="Content 2", metadata={}),
        ]
        reranked_docs = [
            RerankedDocument(
                source="doc_1",
                score=0.9,
                rank=1,
                original_rank=1,
                snippet="content",
                rank_change=0
            ),
            RerankedDocument(
                source="doc_2",
                score=0.8,
                rank=2,
                original_rank=2,
                snippet="content",
                rank_change=0
            )
        ]

        result = self.service._check_if_order_changed(docs, reranked_docs)

        assert result is False

    def test_partial_order_change(self):
        """Verify returns True if any document changed"""
        docs = [
            Document(page_content="Content 1", metadata={}),
            Document(page_content="Content 2", metadata={}),
        ]
        reranked_docs = [
            RerankedDocument(
                source="doc_1",
                score=0.9,
                rank=1,
                original_rank=1,
                snippet="content",
                rank_change=0
            ),
            RerankedDocument(
                source="doc_2",
                score=0.8,
                rank=2,
                original_rank=3,
                snippet="content",
                rank_change=1  # This one changed
            )
        ]

        result = self.service._check_if_order_changed(docs, reranked_docs)

        assert result is True
```

### 2. Integration Tests

#### Test: Full tools_node with reranking

```python
import pytest
import asyncio
from main import LangChainAgent
from config import ENABLE_RERANKING

class TestToolsNodeReranking:
    """Integration tests for tools_node reranking"""

    @pytest.fixture
    def agent(self):
        agent = LangChainAgent()
        agent.initialize_components()
        return agent

    def test_original_rank_stored_in_metadata(self, agent):
        """Verify original_rank is stored before reranking"""
        if not ENABLE_RERANKING:
            pytest.skip("Reranking not enabled")

        # This requires mocking the reranker or using a real query
        # We'll verify through the tools_node output
        query = "What is machine learning?"
        results = agent.vector_store.as_retriever(
            search_type="hybrid",
            search_kwargs={"k": 10, "fetch_k": 20, "lambda_mult": 0.5}
        ).invoke(query)

        # Simulate what tools_node does
        if ENABLE_RERANKING and agent.reranker:
            for i, doc in enumerate(results, 1):
                doc.metadata['original_rank'] = i

            # Verify metadata was set
            assert all('original_rank' in doc.metadata for doc in results)
            assert results[0].metadata['original_rank'] == 1

    def test_reranker_score_stored_in_metadata(self, agent):
        """Verify reranker_score is stored after reranking"""
        if not ENABLE_RERANKING:
            pytest.skip("Reranking not enabled")

        query = "What is machine learning?"
        results = agent.vector_store.as_retriever(
            search_type="hybrid",
            search_kwargs={"k": 10, "fetch_k": 20, "lambda_mult": 0.5}
        ).invoke(query)

        if ENABLE_RERANKING and agent.reranker:
            # Store original ranks
            for i, doc in enumerate(results, 1):
                doc.metadata['original_rank'] = i

            # Rerank
            reranked_results = agent.reranker.rerank(query, results, 4)

            # Simulate what tools_node does
            results_reranked = [doc for doc, score in reranked_results]
            for i, (doc, score) in enumerate(reranked_results, 1):
                doc.metadata['reranker_score'] = score

            # Verify metadata was set
            assert all('reranker_score' in doc.metadata for doc in results_reranked)
            # Scores should be between 0 and 1
            assert all(0 <= doc.metadata['reranker_score'] <= 1
                      for doc in results_reranked)
```

### 3. Event Emission Tests

#### Test: RerankerStartEvent and RerankerResultEvent emission

```python
import asyncio
from api.services.observable_agent import ObservableAgentService
from api.schemas.events import RerankerStartEvent, RerankerResultEvent
from config import ENABLE_RERANKING

@pytest.mark.asyncio
async def test_reranking_events_emitted():
    """Verify reranking events are emitted during tools node"""
    if not ENABLE_RERANKING:
        pytest.skip("Reranking not enabled")

    service = ObservableAgentService()
    await service.ensure_initialized()

    events_captured = []

    async def emit_callback(event):
        events_captured.append(event)

    # Process a message that will trigger reranking
    await service.process_message(
        "What is the capital of France?",
        "test-thread-id",
        emit_callback
    )

    # Filter reranking events
    reranker_start = [e for e in events_captured
                      if isinstance(e, RerankerStartEvent)]
    reranker_result = [e for e in events_captured
                       if isinstance(e, RerankerResultEvent)]

    # Verify events were emitted
    assert len(reranker_start) > 0, "RerankerStartEvent not emitted"
    assert len(reranker_result) > 0, "RerankerResultEvent not emitted"

    # Verify event content
    start_event = reranker_start[0]
    assert start_event.node == "tools"
    assert start_event.candidate_count > 0

    result_event = reranker_result[0]
    assert result_event.node == "tools"
    assert len(result_event.results) > 0
    assert isinstance(result_event.reranking_changed_order, bool)

    # Verify each result has required fields
    for doc in result_event.results:
        assert hasattr(doc, 'source')
        assert hasattr(doc, 'score')
        assert hasattr(doc, 'rank')
        assert hasattr(doc, 'original_rank')
        assert hasattr(doc, 'snippet')
        assert hasattr(doc, 'rank_change')

@pytest.mark.asyncio
async def test_reranking_events_disabled():
    """Verify no reranking events when ENABLE_RERANKING is False"""
    # This requires temporarily disabling reranking in config
    # Or mocking the config value
    pytest.skip("Requires config mocking")
```

### 4. Manual Testing

#### Setup
```bash
cd /Users/kevin/github/personal/rusty-compass
python3 -m pip install -r requirements.txt
```

#### Test Scenario 1: Visual Event Sequence

1. Start the web UI (if available)
2. Ask a query: "What is machine learning?"
3. Open browser DevTools → Network → WebSocket
4. Look for event sequence:
   - `node_start` (tools)
   - `hybrid_search_result` (12 candidates)
   - `reranker_start` (12 candidates)
   - `reranker_result` (4 results with scores and rank_change)
   - `node_end` (tools)

#### Test Scenario 2: Event Data Verification

Create a test script:

```python
import asyncio
from langchain_agent.api.services.observable_agent import ObservableAgentService
from langchain_agent.api.schemas.events import (
    RerankerStartEvent, RerankerResultEvent
)

async def test_manual():
    service = ObservableAgentService()
    await service.ensure_initialized()

    events = []

    async def capture_event(event):
        events.append(event)
        if isinstance(event, RerankerStartEvent):
            print(f"[RerankerStart] Model: {event.model}, Candidates: {event.candidate_count}")
        elif isinstance(event, RerankerResultEvent):
            print(f"[RerankerResult] Order changed: {event.reranking_changed_order}")
            for doc in event.results:
                direction = "↑" if doc.rank_change > 0 else "↓" if doc.rank_change < 0 else "="
                print(f"  {direction} {doc.source}: rank {doc.original_rank} → {doc.rank}, score: {doc.score:.4f}")

    await service.process_message(
        "What is artificial intelligence?",
        "manual-test",
        capture_event
    )

if __name__ == "__main__":
    asyncio.run(test_manual())
```

Run:
```bash
cd /Users/kevin/github/personal/rusty-compass/langchain_agent
python3 test_reranking_manual.py
```

Expected output:
```
[RerankerStart] Model: Qwen/Qwen3-Reranker-8B, Candidates: 12
[RerankerResult] Order changed: True
  ↑ doc_1: rank 3 → 1, score: 0.9234
  ↓ doc_2: rank 1 → 2, score: 0.8567
  ↑ doc_3: rank 5 → 3, score: 0.7890
  ↑ doc_4: rank 8 → 4, score: 0.6543
```

### 5. Load Testing

Test with multiple concurrent requests to verify observability doesn't bottleneck:

```python
import asyncio
import time
from langchain_agent.api.services.observable_agent import ObservableAgentService

async def stress_test():
    service = ObservableAgentService()
    await service.ensure_initialized()

    event_counts = {"reranker_start": 0, "reranker_result": 0}

    async def count_events(event):
        if hasattr(event, 'type'):
            if event.type == 'reranker_start':
                event_counts['reranker_start'] += 1
            elif event.type == 'reranker_result':
                event_counts['reranker_result'] += 1

    # Run 10 concurrent requests
    start_time = time.time()
    queries = [
        "What is machine learning?",
        "What is artificial intelligence?",
        "What is deep learning?",
        "What is neural networks?",
        "What is natural language processing?"
    ] * 2  # 10 queries

    tasks = [
        service.process_message(query, f"thread-{i}", count_events)
        for i, query in enumerate(queries)
    ]

    await asyncio.gather(*tasks)
    elapsed = time.time() - start_time

    print(f"Completed {len(queries)} queries in {elapsed:.2f}s")
    print(f"RerankerStart events: {event_counts['reranker_start']}")
    print(f"RerankerResult events: {event_counts['reranker_result']}")
    print(f"Avg time per query: {elapsed/len(queries):.2f}s")
```

## Test Coverage Goals

| Component | Coverage Goal | Status |
|-----------|--------------|--------|
| _compute_reranked_documents | 100% | To implement |
| _check_if_order_changed | 100% | To implement |
| Event emission logic | 95% | To implement |
| Metadata storage | 90% | To implement |
| Full pipeline | 80% | To implement |

## Continuous Integration

Add to CI/CD pipeline:

```yaml
# .github/workflows/test.yml
- name: Test reranking observability
  run: |
    pytest langchain_agent/tests/test_observable_agent.py -v
    pytest langchain_agent/tests/test_tools_node.py -v
```

## Regression Testing

After implementation, verify:
- [ ] Existing queries still work
- [ ] Reranking still improves results
- [ ] Performance impact < 10%
- [ ] Memory usage stable
- [ ] No duplicate events

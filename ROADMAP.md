# Rusty Compass: Optimization Roadmap

## Executive Summary

Rusty Compass is a **well-architected, feature-complete LangGraph ReAct agent** with sophisticated hybrid search, cross-encoder reranking, and self-improving reflection loops. The architecture demonstrates strong software engineering fundamentals. This roadmap prioritizes improvements across **6 domains** with actionable recommendations.

---

## Priority 1: Security Hardening (Critical)

### Findings

| Severity | Issue | Location |
|----------|-------|----------|
| **HIGH** | No authentication on WebSocket/REST endpoints | `api/routes/chat.py` |
| **HIGH** | No rate limiting (DoS vulnerability) | All API endpoints |
| **HIGH** | No input validation on user messages | `ChatMessage.message` - no max length |
| **MEDIUM** | Prompt injection risk in RAG pipeline | Document content injected into prompts |
| **MEDIUM** | CORS too permissive for production | `api/main.py:22-33` |
| **LOW** | No request logging for audit trail | All endpoints |

### Recommendations

```python
# 1. Add API key authentication middleware
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API key")

# 2. Add rate limiting
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)
@app.post("/api/chat")
@limiter.limit("10/minute")  # 10 requests per minute per IP

# 3. Add input validation
class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    thread_id: Optional[str] = Field(None, regex=r"^[a-zA-Z0-9_-]{1,64}$")
```

### Prompt Injection Mitigation
- Add document content sanitization before LLM injection
- Implement system prompt hardening with instruction hierarchy
- Add output validation to detect/filter leaked system prompts

---

## Priority 2: LangGraph Optimization (High Impact)

### Current Architecture Assessment

**Strengths:**
- Clean StateGraph with `CustomAgentState` TypedDict
- Sophisticated reflection loop with document/response grading
- Dynamic λ adjustment via query evaluator
- Connection pooling for database performance

### Optimization Opportunities

#### 2.1 Parallel Node Execution
Currently nodes execute sequentially. Add parallel execution for independent operations:

```python
from langgraph.graph import StateGraph
from langgraph.types import Send

# Run document grading in parallel with query transformation prep
def route_after_retrieval(state):
    return [
        Send("grade_documents", state),
        Send("prepare_transformation", state),  # Precompute in parallel
    ]
```

#### 2.2 LangSmith Integration (Observability)
```python
# Add to config.py
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = "rusty-compass"

# Add to main.py
from langsmith import trace
import langsmith

@trace(name="knowledge_base_search")
def search_knowledge_base(query: str, lambda_mult: float):
    # Automatic tracing of retrieval, reranking, grading
    ...
```

#### 2.3 Streaming Improvements
```python
# Use astream_events for granular streaming
async for event in graph.astream_events(
    input_state,
    config={"configurable": {"thread_id": thread_id}},
    version="v2",
):
    if event["event"] == "on_chat_model_stream":
        yield event["data"]["chunk"].content
```

#### 2.4 Checkpointing Optimization
- Current: Full state checkpoint on every node
- Recommended: Add selective checkpointing for expensive operations only

```python
# Add checkpoint metadata for debugging
config = {
    "configurable": {
        "thread_id": thread_id,
        "checkpoint_ns": "reflection_loop",
    },
    "metadata": {
        "query": original_query,
        "iteration": iteration_count,
    }
}
```

---

## Priority 3: RAG Pipeline Enhancements (Medium Impact)

### 3.1 Chunking Strategy Optimization

**Current:** 1000 char chunks, 200 char overlap
**Issue:** Fixed chunking loses semantic boundaries

```python
# Implement semantic chunking
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Option 1: Markdown-aware splitting
splitter = RecursiveCharacterTextSplitter.from_language(
    language="markdown",
    chunk_size=1000,
    chunk_overlap=200,
)

# Option 2: Sentence-aware splitting with semantic boundaries
from langchain_experimental.text_splitter import SemanticChunker
splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95,
)
```

### 3.2 Hybrid Search Relevancy Tuning

**Current RRF Implementation:**
```sql
(vector_weight / (60.0 + vector_rank)) + (text_weight / (60.0 + text_rank))
```

**Recommendations:**
1. **Tune k parameter:** Default k=60 is OpenSearch standard; test k=20-40 for smaller corpora
2. **Add score normalization:** Raw RRF scores vary by query; normalize before reranking
3. **Query expansion:** Add synonyms/related terms for lexical search boost

```python
# Enhanced RRF with score normalization
def normalized_rrf_score(vector_rank, text_rank, k=40):
    raw_score = (1/(k + vector_rank)) + (1/(k + text_rank))
    # Normalize to [0, 1] range
    max_possible = 2 / k  # Both rank 0
    return raw_score / max_possible
```

### 3.3 Reranking Optimization

**Current:** BGE-Reranker-v2-m3 processes all candidates
**Improvement:** Add early termination and score thresholding

```python
def rerank_with_early_exit(self, query, documents, top_k, min_score=0.7):
    """Stop reranking when we have enough high-confidence results."""
    scored = []
    for doc in documents:
        score = self.score_single(query, doc)
        scored.append((doc, score))

        # Early exit: if we have top_k docs above threshold, stop
        high_conf = [s for s in scored if s[1] >= min_score]
        if len(high_conf) >= top_k:
            break

    return sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]
```

### 3.4 Query Understanding Improvements

Add query classification categories:
```python
QUERY_TYPES = {
    "factual": {"lambda": 0.1, "description": "Specific fact lookup"},
    "conceptual": {"lambda": 0.4, "description": "Explain a concept"},
    "procedural": {"lambda": 0.3, "description": "How-to questions"},
    "comparative": {"lambda": 0.5, "description": "Compare X vs Y"},
    "exploratory": {"lambda": 0.6, "description": "Open-ended exploration"},
}
```

---

## Priority 4: Frontend UI/UX (Medium Impact)

### 4.1 Accessibility Gaps (WCAG 2.1)

| Issue | Location | Fix |
|-------|----------|-----|
| Missing skip navigation | `Layout.tsx` | Add skip-to-main link |
| No focus management | `MessageInput` | Auto-focus after send |
| Missing ARIA landmarks | All panels | Add `role="region"` |
| Low contrast text | Gray text on dark bg | Ensure 4.5:1 ratio |
| No keyboard shortcuts | Chat interface | Add Ctrl+Enter to send |

```tsx
// Add to Layout.tsx
<a href="#main-content" className="sr-only focus:not-sr-only">
  Skip to main content
</a>

// Add to MessageInput.tsx
<textarea
  aria-label="Type your message"
  aria-describedby="input-hint"
  onKeyDown={(e) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      handleSubmit();
    }
  }}
/>
<span id="input-hint" className="sr-only">
  Press Ctrl+Enter to send
</span>
```

### 4.2 Performance Optimization

```tsx
// 1. Virtualize message list for long conversations
import { FixedSizeList } from 'react-window';

// 2. Memoize expensive components
const MemoizedMessage = React.memo(Message, (prev, next) =>
  prev.message.id === next.message.id &&
  prev.message.content === next.message.content
);

// 3. Debounce WebSocket reconnection
const connect = useMemo(() =>
  debounce((threadId) => wsConnect(threadId), 300),
  [wsConnect]
);

// 4. Lazy load observability panel
const ObservabilityPanel = lazy(() => import('./ObservabilityPanel'));
```

### 4.3 UX Improvements

1. **Message pagination:** Load 50 messages initially, load more on scroll
2. **Typing indicators:** Show "Agent is thinking..." during processing
3. **Error recovery:** Add retry button on failed messages
4. **Dark/light mode:** Add theme toggle (Tailwind `dark:` classes ready)
5. **Mobile responsive:** Current layout breaks below 768px

---

## Priority 5: Code Architecture (Maintainability)

### 5.1 Split main.py (2,697 lines → modular structure)

```
langchain_agent/
├── core/
│   ├── __init__.py
│   ├── agent.py          # LangChainAgent class
│   ├── state.py          # CustomAgentState, DocumentGrade
│   └── graph.py          # StateGraph construction
├── search/
│   ├── __init__.py
│   ├── vector_store.py   # SimplePostgresVectorStore
│   ├── retriever.py      # PostgresRetriever
│   ├── reranker.py       # BGEReranker
│   └── hybrid.py         # RRF implementation
├── reflection/
│   ├── __init__.py
│   ├── document_grader.py
│   ├── response_grader.py
│   └── query_transformer.py
└── main.py               # Entry point only (~50 lines)
```

### 5.2 Add Custom Exception Hierarchy

```python
# exceptions.py
class RustyCompassError(Exception):
    """Base exception for all agent errors."""
    pass

class RetrievalError(RustyCompassError):
    """Failed to retrieve documents from knowledge base."""
    pass

class GradingError(RustyCompassError):
    """Document or response grading failed."""
    pass

class ReflectionLimitError(RustyCompassError):
    """Max reflection iterations reached without satisfactory response."""
    pass
```

### 5.3 Add Database Migrations (Alembic)

```bash
# Initialize Alembic
alembic init migrations

# Create initial migration
alembic revision --autogenerate -m "Initial schema"

# Run migrations
alembic upgrade head
```

---

## Priority 6: Testing & Observability (Long-term)

### 6.1 Test Coverage Targets

| Component | Current | Target | Priority |
|-----------|---------|--------|----------|
| Reranker | 6 tests | 15 tests | High |
| Reflection loop | 6 tests | 20 tests | High |
| API endpoints | 0 tests | 15 tests | Critical |
| WebSocket flow | 0 tests | 10 tests | High |
| Frontend components | 0 tests | 20 tests | Medium |

### 6.2 Add Structured Logging

```python
import structlog

logger = structlog.get_logger()

# Replace print statements
logger.info(
    "document_grading_complete",
    query=query,
    relevant_count=len([d for d in grades if d["relevant"]]),
    total_docs=len(grades),
    duration_ms=duration * 1000,
)
```

### 6.3 Add Metrics Export (Prometheus)

```python
from prometheus_client import Counter, Histogram, start_http_server

QUERY_LATENCY = Histogram(
    'agent_query_latency_seconds',
    'Time spent processing queries',
    ['query_type', 'reflection_count']
)

RETRIEVAL_DOCS = Counter(
    'agent_retrieval_docs_total',
    'Documents retrieved by relevance',
    ['relevant']
)
```

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
- [ ] Add API authentication (API keys)
- [ ] Add rate limiting (slowapi)
- [ ] Add input validation (Pydantic)
- [ ] Split main.py into modules
- [ ] Add structured logging

### Phase 2: Core Improvements (Weeks 3-4)
- [ ] LangSmith integration
- [ ] Semantic chunking
- [ ] RRF k-parameter tuning
- [ ] Add pytest suite for API
- [ ] Database migrations (Alembic)

### Phase 3: UX & Accessibility (Weeks 5-6)
- [ ] WCAG 2.1 AA compliance
- [ ] Message virtualization
- [ ] Mobile responsive layout
- [ ] Dark mode toggle
- [ ] Keyboard navigation

### Phase 4: Advanced (Weeks 7-8)
- [ ] Parallel node execution
- [ ] Reranker early termination
- [ ] Query expansion
- [ ] Prometheus metrics
- [ ] Production Docker setup

---

## Quick Wins (Can Implement Today)

1. **Add input validation** to `ChatMessage` model (5 min)
2. **Add ARIA labels** to chat interface (15 min)
3. **Enable LangSmith** tracing with env var (10 min)
4. **Tune RRF k parameter** from 60→40 (5 min)
5. **Add skip-to-content link** for accessibility (5 min)
6. **Replace print() with logger.info()** (30 min)

---

This roadmap provides a structured path from the current feature-complete state to a production-hardened, accessible, and observable system. Prioritize Phase 1 (Security) before any production deployment.

# LangChain Agent Improvements Test Suite

Comprehensive test suite for the LangChain agent improvements, covering batch document grading, query evaluator caching, token budget tracking, confidence-based early stopping, and positive/negative feedback learning.

## Test File

`test_improvements.py` - Complete test suite with unit tests, integration tests, and performance benchmarks.

## Prerequisites

Install test dependencies:

```bash
pip install -r requirements-dev.txt
```

Or install pytest directly:

```bash
pip install pytest pytest-timeout pytest-cov
```

## Running Tests

### Run all tests with verbose output:
```bash
pytest test_improvements.py -v
```

### Run with coverage report:
```bash
pytest test_improvements.py --cov=main --cov-report=html
```

### Run specific test categories:

**Unit tests only:**
```bash
pytest test_improvements.py -v -k "test_batch or test_query or test_token or test_confidence"
```

**Integration tests only:**
```bash
pytest test_improvements.py -v -k "test_full"
```

**Performance tests only:**
```bash
pytest test_improvements.py -v -k "faster or latency"
```

### Run a single test:
```bash
pytest test_improvements.py::test_batch_document_grading -v
```

## Test Coverage

### Unit Tests (5 tests)

#### 1. `test_batch_document_grading`
**Purpose:** Verify batch document grading produces same results as individual grading.

**What it tests:**
- Processes multiple documents in a single batch
- Compares batch results to individual grading results
- Verifies accuracy is maintained with batching optimization

**Expected behavior:**
- All documents are graded
- Grade structure is consistent
- Batch and individual results match

**Configuration used:**
- `DOCUMENT_GRADING_BATCH_SIZE` from config.py

---

#### 2. `test_query_evaluator_caching`
**Purpose:** Verify query evaluator cache works correctly.

**What it tests:**
- Cache misses trigger LLM evaluation
- Cache hits return stored results without LLM call
- Cache updates correctly on miss

**Expected behavior:**
- First query evaluation creates cache entry
- Duplicate queries retrieve from cache
- LLM calls are reduced for cached queries

**Configuration used:**
- `ENABLE_QUERY_EVAL_CACHE` from config.py
- `QUERY_EVAL_CACHE_MAX_SIZE` from config.py

---

#### 3. `test_token_budget_tracking`
**Purpose:** Verify token budget enforcement works correctly.

**What it tests:**
- Token usage is tracked across operations
- Operations are blocked when budget is exceeded
- Warning thresholds are respected

**Expected behavior:**
- Token usage accumulates across operations
- Agent respects max token budget
- Operations complete under budget

**Configuration used:**
- `REFLECTION_MAX_TOKENS_TOTAL` from config.py
- `REFLECTION_TOKEN_WARNING_THRESHOLD` from config.py

---

#### 4. `test_confidence_early_stop`
**Purpose:** Verify confidence-based early stopping triggers correctly.

**What it tests:**
- High confidence (>threshold) with passing grade ends flow
- Low confidence forces retry even with passing grade
- Routing logic respects confidence thresholds

**Expected behavior:**
- High confidence (≥0.85) routes to END
- Low confidence routes to retry (if under max iterations)
- Max iterations always routes to END

**Configuration used:**
- `RESPONSE_GRADING_HIGH_CONFIDENCE_THRESHOLD` from config.py
- `RESPONSE_GRADING_LOW_CONFIDENCE_RETRY` from config.py
- `REFLECTION_MAX_ITERATIONS` from config.py

---

#### 5. `test_query_transformer_positive_feedback`
**Purpose:** Verify query transformer uses both positive and negative examples.

**What it tests:**
- Transformer considers both relevant and irrelevant documents
- Positive examples guide transformation
- Negative examples help avoid bad patterns
- Transformation prompt includes both types of feedback

**Expected behavior:**
- Query is transformed based on feedback
- Prompt mentions learning from successes and failures
- Iteration count is incremented

**Implementation location:**
- `query_transformer_node()` in main.py (lines 1137-1233)

---

### Integration Tests (3 tests)

#### 6. `test_full_agent_with_batch_grading`
**Purpose:** Test complete agent flow with batch document grading.

**What it tests:**
- Documents are retrieved
- Batch grading processes all documents
- Agent generates response based on graded documents
- Flow completes successfully

**Expected behavior:**
- All documents are graded in batch
- Summary grade is computed
- Routing decision is made based on grade

---

#### 7. `test_full_agent_with_caching`
**Purpose:** Test complete agent flow with query evaluator caching.

**What it tests:**
- First query evaluation creates cache entry
- Subsequent identical queries use cache
- Different queries create new cache entries
- Cache reduces LLM calls

**Expected behavior:**
- Cache populates on first query
- Cache hits avoid LLM calls
- Different queries create new entries

---

#### 8. `test_token_budget_prevents_runaway`
**Purpose:** Test that token budget prevents excessive retries when budget is low.

**What it tests:**
- Agent tracks cumulative token usage
- Retries are blocked when approaching budget limit
- Agent fails gracefully when budget exceeded
- Warning threshold triggers appropriate behavior

**Expected behavior:**
- Near-budget operations are restricted
- Over-budget operations are blocked
- Agent routes to END when budget exhausted

---

### Performance Tests (2 tests)

#### 9. `test_batch_grading_faster_than_individual`
**Purpose:** Verify batch grading is significantly faster than individual grading.

**What it tests:**
- Batch processing completes faster than individual calls
- Speedup is at least 3x (with real LLM)
- Results are equivalent between methods

**Expected behavior:**
- Batch grading shows performance improvement
- Both methods produce same number of grades
- With real LLM: ≥3x speedup

**Note:** With mocking, timing differences may be minimal. Run with real LLM to see actual speedup.

---

#### 10. `test_cache_reduces_latency`
**Purpose:** Verify query evaluator cache significantly reduces latency.

**What it tests:**
- Cached queries return near-instantly
- Cache hit is at least 10x faster than cache miss
- Results from cache match original evaluation

**Expected behavior:**
- Cache hit completes in <10ms
- Cache miss takes longer (LLM call time)
- Results are identical

**Note:** With mocking, timing differences may be minimal. Run with real LLM to see actual speedup.

---

## Test Implementation Details

### Fixtures

**`mock_llm`** - Provides predictable LLM responses without API calls
- Returns JSON for document/response grading
- Returns query analysis for evaluation
- Returns transformed queries

**`mock_embeddings`** - Returns fixed 768-dimensional vectors

**`mock_vector_store`** - Generates mock documents for testing

**`mock_agent`** - Fully configured agent with mocked dependencies

**`sample_documents`** - 4 sample documents for testing

**`sample_state`** - Pre-configured CustomAgentState for testing

### Mocking Strategy

Tests use unittest.mock to:
1. Avoid actual LLM API calls
2. Ensure deterministic test results
3. Test component logic in isolation
4. Verify component interactions

### State Management

All tests properly initialize agent state with:
- messages
- lambda_mult
- query_analysis
- iteration_count
- response_retry_count
- retrieved_documents
- document_grades
- document_grade_summary
- response_grade
- original_query
- transformed_query

### Test Isolation

Each test:
- Uses fixtures for clean state
- Clears caches between tests
- Doesn't depend on other tests
- Can run independently

## Interpreting Results

### Successful Test Output

```
test_improvements.py::test_batch_document_grading PASSED           [ 10%]
test_improvements.py::test_query_evaluator_caching PASSED          [ 20%]
test_improvements.py::test_token_budget_tracking PASSED            [ 30%]
test_improvements.py::test_confidence_early_stop PASSED            [ 40%]
test_improvements.py::test_query_transformer_positive_feedback PASSED [ 50%]
test_improvements.py::test_full_agent_with_batch_grading PASSED    [ 60%]
test_improvements.py::test_full_agent_with_caching PASSED          [ 70%]
test_improvements.py::test_token_budget_prevents_runaway PASSED    [ 80%]
test_improvements.py::test_batch_grading_faster_than_individual PASSED [ 90%]
test_improvements.py::test_cache_reduces_latency PASSED            [100%]

========== 10 passed in 2.34s ==========
```

### Performance Test Notes

Performance tests (`test_batch_grading_faster_than_individual` and `test_cache_reduces_latency`) use mocked LLM calls, so actual speedup may not be visible.

**To see real performance improvements:**

1. Create a version of tests with real LLM:
   ```python
   # Use actual agent instead of mock
   from main import LangChainAgent
   agent = LangChainAgent()
   agent.verify_prerequisites()
   agent.initialize_components()
   ```

2. Run with real Ollama backend:
   ```bash
   # Start Ollama
   ollama serve

   # Run performance tests with real LLM
   pytest test_improvements.py -v -k "faster or latency" --real-llm
   ```

## Configuration Dependencies

Tests respect these configuration values from `config.py`:

- `DOCUMENT_GRADING_BATCH_SIZE` - Batch size for document grading
- `ENABLE_QUERY_EVAL_CACHE` - Enable/disable query caching
- `QUERY_EVAL_CACHE_MAX_SIZE` - Max cache entries
- `REFLECTION_MAX_TOKENS_TOTAL` - Token budget limit
- `REFLECTION_TOKEN_WARNING_THRESHOLD` - Warning threshold
- `RESPONSE_GRADING_HIGH_CONFIDENCE_THRESHOLD` - Early stop threshold
- `RESPONSE_GRADING_LOW_CONFIDENCE_RETRY` - Retry threshold
- `REFLECTION_MAX_ITERATIONS` - Max retry attempts

To test different configurations, modify `config.py` before running tests.

## Troubleshooting

### Import Errors

If you see import errors:
```bash
# Ensure you're in the langchain_agent directory
cd /path/to/rusty-compass/langchain_agent

# Run with python module mode
python3 -m pytest test_improvements.py -v
```

### Fixture Not Found

Ensure pytest is installed:
```bash
pip install pytest
```

### Database Connection Errors

Tests use mocked components, so database should not be needed. If you see connection errors, the test may need additional mocking.

### Performance Test Timing Issues

Performance tests may not show significant timing differences with mocking. This is expected. To see real performance:

1. Use real LLM backend (Ollama)
2. Remove mocking from performance tests
3. Test with production configuration

## Adding New Tests

To add new tests:

1. **Choose test category:** Unit, Integration, or Performance
2. **Add fixture if needed:** Define in fixtures section
3. **Write test function:** Follow naming convention `test_*`
4. **Add docstring:** Explain what the test verifies
5. **Use assertions:** Clear, specific assertions
6. **Update this README:** Document the new test

### Example Test Template

```python
def test_new_feature(mock_agent, sample_state):
    """
    Test description here.

    Verifies:
    1. First thing tested
    2. Second thing tested
    3. Third thing tested
    """
    # Setup
    sample_state["some_field"] = "test_value"

    # Execute
    result = mock_agent.some_node(sample_state)

    # Verify
    assert "expected_field" in result
    assert result["expected_field"] == "expected_value"
```

## Continuous Integration

To add tests to CI pipeline:

```yaml
# .github/workflows/test.yml
name: Test Suite
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install -r requirements-dev.txt
      - run: pytest test_improvements.py -v --cov=main
```

## Further Reading

- [pytest documentation](https://docs.pytest.org/)
- [unittest.mock documentation](https://docs.python.org/3/library/unittest.mock.html)
- [LangChain testing guide](https://python.langchain.com/docs/contributing/testing)

## Contact

For questions or issues with tests, please open an issue on the project repository.

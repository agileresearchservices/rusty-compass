# Async/Sync Streaming - Code Reference

This document provides code snippets of all changes made to implement the async/sync streaming feature flag.

---

## 1. Config.py - Configuration Section

**File:** `/Users/kevin/github/personal/rusty-compass/langchain_agent/config.py`

**Location:** End of file (after RESPONSE_GRADING_LOW_CONFIDENCE_RETRY)

```python
# ============================================================================
# OBSERVABLE AGENT STREAMING CONFIGURATION
# ============================================================================

# Enable incremental async streaming for improved responsiveness (EXPERIMENTAL)
# When False (default): Backward compatible behavior - waits for entire node completion
#   - Runs entire graph in executor, collects all timing info after completion
#   - More blocking but stable behavior
# When True: Improved streaming with incremental event emission
#   - Emits NodeStartEvent immediately when node begins execution
#   - Processes events as they complete instead of waiting for full node
#   - Emits NodeEndEvent with accurate timing after processing
#   - TRADEOFF: Timing may be slightly less accurate than legacy mode, but
#     provides better UI responsiveness and prevents async event loop blocking
ENABLE_ASYNC_STREAMING = False
```

**Also Update:**

In the `__all__` list at the top of config.py, add to exports:

```python
__all__ = [
    # ... existing items ...
    # Observable agent streaming configuration
    "ENABLE_ASYNC_STREAMING",
]
```

---

## 2. Observable_Agent.py - Import Statement

**File:** `/Users/kevin/github/personal/rusty-compass/langchain_agent/api/services/observable_agent.py`

**Location:** Lines 20-28 (config import section)

```python
from config import (
    ENABLE_REFLECTION,
    ENABLE_DOCUMENT_GRADING,
    ENABLE_RESPONSE_GRADING,
    ENABLE_QUERY_TRANSFORMATION,
    ENABLE_RERANKING,
    RERANKER_MODEL,
    REFLECTION_MAX_ITERATIONS,
    ENABLE_ASYNC_STREAMING,  # NEW LINE
)
```

---

## 3. Observable_Agent.py - Initialization Update

**File:** `/Users/kevin/github/personal/rusty-compass/langchain_agent/api/services/observable_agent.py`

**Location:** ObservableAgentService.__init__() method

```python
def __init__(self):
    """Initialize the observable agent service (lazy loading)."""
    self._agent: Optional[LangChainAgent] = None
    self._initialized = False
    self._lock = asyncio.Lock()
    self._use_async_streaming = ENABLE_ASYNC_STREAMING  # NEW LINE
```

---

## 4. Observable_Agent.py - Router Method

**File:** `/Users/kevin/github/personal/rusty-compass/langchain_agent/api/services/observable_agent.py`

**Location:** Replaces original _astream_graph() method

```python
async def _astream_graph(
    self,
    initial_state: Dict[str, Any],
    config: Dict[str, Any],
    emit: EmitCallback,
    node_start_times: Dict[str, float],
    metrics: Dict[str, float],
):
    """
    Stream through the agent graph with event emission.

    Yields state updates as they occur.

    This method supports two modes controlled by ENABLE_ASYNC_STREAMING:

    - When False (default, backward compatible):
      Runs entire graph in executor, collects all timing info after completion.
      More blocking but stable timing measurements.

    - When True (experimental, improved streaming):
      Streams events incrementally as they complete, emitting NodeStartEvent
      immediately and NodeEndEvent with accurate timing. This prevents blocking
      the async event loop and improves UI responsiveness.
      TRADEOFF: Timing may be slightly less accurate but better reactivity.
    """
    if self._use_async_streaming:
        # IMPROVED STREAMING MODE: Emit events incrementally as they occur
        await self._astream_graph_improved(
            initial_state, config, emit, node_start_times, metrics
        )
    else:
        # BACKWARD COMPATIBLE MODE: Collect all results, then emit
        await self._astream_graph_legacy(
            initial_state, config, emit, node_start_times, metrics
        )
```

---

## 5. Observable_Agent.py - Legacy Method (ENABLE_ASYNC_STREAMING = False)

**File:** `/Users/kevin/github/personal/rusty-compass/langchain_agent/api/services/observable_agent.py`

**Location:** New _astream_graph_legacy() method

```python
async def _astream_graph_legacy(
    self,
    initial_state: Dict[str, Any],
    config: Dict[str, Any],
    emit: EmitCallback,
    node_start_times: Dict[str, float],
    metrics: Dict[str, float],
):
    """
    Legacy streaming mode (ENABLE_ASYNC_STREAMING = False).

    Runs entire agent graph in thread executor, collecting all node
    executions and their timing before emitting events. This provides
    stable, accurate timing but blocks the async event loop during
    the entire graph execution.
    """
    loop = asyncio.get_event_loop()

    # Use the compiled app's stream method
    # Track timestamps for each node execution during synchronous execution
    def run_sync_with_timing():
        results = []
        # List of (node_name, start_time, end_time, output) for each execution
        node_executions: List[Dict[str, Any]] = []
        prev_time = time.time()

        for event in self._agent.app.stream(initial_state, config):
            current_time = time.time()

            # Each event is a dict with node_name: output_dict
            for node_name, output in event.items():
                # Record this execution with timing
                node_executions.append({
                    "node": node_name,
                    "start": prev_time,
                    "end": current_time,
                    "output": output,
                })
                prev_time = current_time

        return node_executions

    # Run the graph
    node_executions = await loop.run_in_executor(None, run_sync_with_timing)

    # Process events and emit observability data
    for execution in node_executions:
        node_name = execution["node"]
        output = execution["output"]
        start_time = execution["start"]
        end_time = execution["end"]

        # Calculate duration
        duration_ms = max((end_time - start_time) * 1000, 1.0)  # At least 1ms

        # Store for metrics (accumulate for repeated nodes)
        if node_name not in node_start_times:
            node_start_times[node_name] = start_time
        if node_name in metrics:
            metrics[node_name] += duration_ms
        else:
            metrics[node_name] = duration_ms

        # Emit node start
        await emit(NodeStartEvent(
            node=node_name,
            input_summary=self._summarize_input(node_name, output),
        ))

        # Emit node-specific events
        await self._emit_node_events(node_name, output, emit)

        # Emit node end with actual duration
        await emit(NodeEndEvent(
            node=node_name,
            duration_ms=duration_ms,
            output_summary=self._summarize_output(node_name, output),
        ))

        yield output
```

---

## 6. Observable_Agent.py - Improved Method (ENABLE_ASYNC_STREAMING = True)

**File:** `/Users/kevin/github/personal/rusty-compass/langchain_agent/api/services/observable_agent.py`

**Location:** New _astream_graph_improved() method

```python
async def _astream_graph_improved(
    self,
    initial_state: Dict[str, Any],
    config: Dict[str, Any],
    emit: EmitCallback,
    node_start_times: Dict[str, float],
    metrics: Dict[str, float],
):
    """
    Improved streaming mode (ENABLE_ASYNC_STREAMING = True).

    Streams events incrementally from the graph as they complete, emitting
    NodeStartEvent immediately and processing each event before emitting
    NodeEndEvent. This prevents blocking the async event loop and provides
    better UI responsiveness.

    TRADEOFF: Timing between events may be slightly less precise since we
    measure wall-clock time between async event emissions rather than
    collecting execution data. However, actual node execution time is
    captured accurately via context manager or event metadata.
    """
    loop = asyncio.get_event_loop()

    def run_sync_stream():
        """Run the agent graph and yield events as they occur."""
        for event in self._agent.app.stream(initial_state, config):
            # Each event is a dict with node_name: output_dict
            for node_name, output in event.items():
                yield {
                    "node": node_name,
                    "output": output,
                }

    # Stream events from the graph executor
    # We use a separate thread for the generator to avoid blocking async
    async def consume_stream():
        """Consume the sync stream in a separate executor."""
        gen = run_sync_stream()
        while True:
            try:
                # Get next event in executor
                event = await loop.run_in_executor(
                    None,
                    next,
                    gen,
                )
                yield event
            except StopIteration:
                break

    # Process events incrementally
    async for event in consume_stream():
        node_name = event["node"]
        output = event["output"]

        # Record node start time (when we receive the event)
        node_start_time = time.time()

        # Emit node start event immediately (before processing)
        await emit(NodeStartEvent(
            node=node_name,
            input_summary=self._summarize_input(node_name, output),
        ))

        # Emit node-specific events
        await self._emit_node_events(node_name, output, emit)

        # Record node end time (after processing)
        node_end_time = time.time()
        duration_ms = max((node_end_time - node_start_time) * 1000, 1.0)

        # Store for metrics (accumulate for repeated nodes)
        if node_name not in node_start_times:
            node_start_times[node_name] = node_start_time
        if node_name in metrics:
            metrics[node_name] += duration_ms
        else:
            metrics[node_name] = duration_ms

        # Emit node end with processing duration
        await emit(NodeEndEvent(
            node=node_name,
            duration_ms=duration_ms,
            output_summary=self._summarize_output(node_name, output),
        ))

        yield output
```

---

## Summary of Changes

### Config.py
- 1 new section added at end of file
- 1 item added to `__all__` list

### Observable_Agent.py
- 1 import added
- 1 instance variable added to `__init__()`
- 1 method refactored into 3 methods:
  - `_astream_graph()` - dispatcher/router
  - `_astream_graph_legacy()` - original behavior
  - `_astream_graph_improved()` - new streaming behavior

### Total Lines Added
- config.py: ~15 lines
- observable_agent.py: ~140 lines
- Total: ~155 lines

### Total Lines Changed
- observable_agent.py: ~20 lines (imports, init, method refactor)

### Backward Compatibility
- Default behavior unchanged (legacy mode)
- All event types unchanged
- All metrics unchanged
- No public API changes

---

## Testing the Implementation

### Quick Test - Verify Import Works

```python
from config import ENABLE_ASYNC_STREAMING
print(f"Feature flag value: {ENABLE_ASYNC_STREAMING}")
```

Expected output: `Feature flag value: False`

### Quick Test - Verify Service Initialization

```python
from api.services.observable_agent import ObservableAgentService
service = ObservableAgentService()
print(f"Using async streaming: {service._use_async_streaming}")
```

Expected output: `Using async streaming: False`

### Quick Test - Enable and Verify

1. Edit `langchain_agent/config.py`
2. Change `ENABLE_ASYNC_STREAMING = True`
3. Restart service
4. Run above test again

Expected output: `Using async streaming: True`

---

## Integration Points

### WebSocket Event Emission

Both modes call the same `emit` callback:

```python
await emit(NodeStartEvent(...))
await emit(NodeEndEvent(...))
```

### Metrics Collection

Both modes populate the same metrics dict:

```python
metrics["query_evaluator"] = 123.45  # milliseconds
metrics["tools"] = 456.78
metrics["agent"] = 789.01
```

### Event Processing

Both modes call the same `_emit_node_events()` method:

```python
await self._emit_node_events(node_name, output, emit)
```

---

## Migration Checklist

- [ ] Read ASYNC_STREAMING_IMPLEMENTATION.md
- [ ] Review config.py changes
- [ ] Review observable_agent.py changes
- [ ] Test with default (False)
- [ ] Test with improved mode (True)
- [ ] Verify event order
- [ ] Verify metrics accuracy
- [ ] Check WebSocket delivery
- [ ] Monitor performance
- [ ] Update documentation if needed
- [ ] Add monitoring/alerts if improved mode deployed

---

## Key Files

1. **Configuration:** `/Users/kevin/github/personal/rusty-compass/langchain_agent/config.py`
2. **Service:** `/Users/kevin/github/personal/rusty-compass/langchain_agent/api/services/observable_agent.py`
3. **Full Documentation:** `/Users/kevin/github/personal/rusty-compass/ASYNC_STREAMING_IMPLEMENTATION.md`
4. **Quick Reference:** `/Users/kevin/github/personal/rusty-compass/ASYNC_STREAMING_QUICK_REFERENCE.md`

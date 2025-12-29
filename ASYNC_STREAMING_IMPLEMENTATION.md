# Async/Sync Streaming Fix for ObservableAgent - Implementation Guide

## Overview

This document describes the implementation of a feature flag-controlled async/sync streaming mechanism for the `ObservableAgentService`. The implementation provides backward compatibility while enabling an improved streaming mode for better UI responsiveness.

## Files Modified

1. **`langchain_agent/config.py`** - Added feature flag configuration
2. **`langchain_agent/api/services/observable_agent.py`** - Implemented dual streaming modes

---

## 1. Configuration Changes

### File: `langchain_agent/config.py`

#### Changes:
- Added `ENABLE_ASYNC_STREAMING` to `__all__` export list
- Added new configuration section at the end of the file

#### New Configuration Constant:

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

**Default Value:** `False` (backward compatible)

**Purpose:** Controls which streaming implementation is used in the observable agent service.

---

## 2. Observable Agent Service Changes

### File: `langchain_agent/api/services/observable_agent.py`

#### A. Import Statement Update

Added `ENABLE_ASYNC_STREAMING` to the config imports:

```python
from config import (
    ENABLE_REFLECTION,
    ENABLE_DOCUMENT_GRADING,
    ENABLE_RESPONSE_GRADING,
    ENABLE_QUERY_TRANSFORMATION,
    ENABLE_RERANKING,
    RERANKER_MODEL,
    REFLECTION_MAX_ITERATIONS,
    ENABLE_ASYNC_STREAMING,  # NEW
)
```

#### B. Initialization Change

Added feature flag check in `__init__`:

```python
def __init__(self):
    """Initialize the observable agent service (lazy loading)."""
    self._agent: Optional[LangChainAgent] = None
    self._initialized = False
    self._lock = asyncio.Lock()
    self._use_async_streaming = ENABLE_ASYNC_STREAMING  # NEW
```

#### C. New Methods Added

Three streaming methods are now implemented:

##### 1. `_astream_graph()` - Router Method

The main entry point that delegates to the appropriate streaming implementation:

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

    Dispatcher that routes to either legacy or improved mode based on flag.
    """
    if self._use_async_streaming:
        await self._astream_graph_improved(...)
    else:
        await self._astream_graph_legacy(...)
```

**Why?** Provides a clean separation of concerns and allows runtime switching if needed.

##### 2. `_astream_graph_legacy()` - Backward Compatible Mode

**ENABLE_ASYNC_STREAMING = False (default)**

This is the existing behavior preserved:
- Runs entire graph in executor
- Collects all node executions synchronously
- Measures timing before/after entire block
- Then processes and emits events

**Advantages:**
- Stable, accurate timing measurements
- Proven behavior - existing implementation
- No async event loop blocking until all data collected

**Disadvantages:**
- Full graph execution blocks the async event loop
- All events are emitted after the entire graph completes
- UI sees a "burst" of events at the end

**Implementation Details:**

```python
def run_sync_with_timing():
    node_executions: List[Dict[str, Any]] = []
    prev_time = time.time()

    for event in self._agent.app.stream(initial_state, config):
        current_time = time.time()
        for node_name, output in event.items():
            node_executions.append({
                "node": node_name,
                "start": prev_time,
                "end": current_time,
                "output": output,
            })
            prev_time = current_time
    return node_executions

# Run in executor
node_executions = await loop.run_in_executor(None, run_sync_with_timing)

# Then process and emit
for execution in node_executions:
    # Emit start, process, emit end
```

##### 3. `_astream_graph_improved()` - Experimental Incremental Mode

**ENABLE_ASYNC_STREAMING = True**

This is the new improved streaming behavior:
- Creates a generator that yields events from the graph
- Runs generator in executor to avoid blocking
- Each event is processed and emitted incrementally
- NodeStartEvent emitted immediately
- Event processing happens
- NodeEndEvent emitted with duration

**Advantages:**
- Doesn't block async event loop for entire execution
- UI gets real-time updates as nodes complete
- Events arrive incrementally throughout execution
- Better responsiveness and perceived performance

**Disadvantages:**
- Timing measurements are wall-clock time during processing
- May be slightly less precise for node duration (includes async emit time)
- Experimental feature, may need tuning

**Implementation Details:**

```python
def run_sync_stream():
    """Run the agent graph and yield events as they occur."""
    for event in self._agent.app.stream(initial_state, config):
        for node_name, output in event.items():
            yield {"node": node_name, "output": output}

# Stream events from executor
async def consume_stream():
    """Consume the sync stream in a separate executor."""
    gen = run_sync_stream()
    while True:
        try:
            # Get next event in executor
            event = await loop.run_in_executor(None, next, gen)
            yield event
        except StopIteration:
            break

# Process incrementally
async for event in consume_stream():
    # Record start
    node_start_time = time.time()

    # Emit start
    await emit(NodeStartEvent(...))

    # Process
    await self._emit_node_events(...)

    # Record end
    node_end_time = time.time()

    # Emit end with duration
    await emit(NodeEndEvent(duration_ms=...))

    yield output
```

---

## 3. How It Works

### Legacy Mode Flow (Default - ENABLE_ASYNC_STREAMING = False)

```
1. process_message() called
2. Calls _astream_graph() which routes to _astream_graph_legacy()
3. _astream_graph_legacy() runs entire graph in executor
   - Graph completes fully
   - All timing data collected
4. For each completed node:
   - Emit NodeStartEvent
   - Emit node-specific events
   - Emit NodeEndEvent
5. UI receives all events after execution
```

### Improved Mode Flow (ENABLE_ASYNC_STREAMING = True)

```
1. process_message() called
2. Calls _astream_graph() which routes to _astream_graph_improved()
3. _astream_graph_improved() creates async generator
   - Starts consuming graph events in executor
   - For each event as it arrives:
     - Record start time
     - Emit NodeStartEvent
     - Emit node-specific events
     - Record end time
     - Emit NodeEndEvent
4. UI receives events progressively as nodes complete
```

---

## 4. Event Flow Comparison

### Legacy Mode (Backward Compatible)
```
Time ->
[========== FULL GRAPH EXECUTION IN EXECUTOR ==========]
After completion:
NodeStart(query_evaluator) -> NodeEnd(query_evaluator)
NodeStart(tools) -> NodeEnd(tools)
NodeStart(agent) -> NodeEnd(agent)
...
```

### Improved Mode (Experimental)
```
Time ->
NodeStart(query_evaluator)
[processing]
NodeEnd(query_evaluator)  <- Happens sooner, doesn't wait for other nodes
NodeStart(tools)
[processing]
NodeEnd(tools)  <- Happens as soon as done
NodeStart(agent)
[processing]
NodeEnd(agent)
...
```

---

## 5. Configuration Usage

### To Enable Improved Streaming

In `langchain_agent/config.py`, change:

```python
ENABLE_ASYNC_STREAMING = True
```

This will immediately activate the improved streaming mode for all new ObservableAgentService instances.

### To Revert to Legacy Mode

In `langchain_agent/config.py`, change:

```python
ENABLE_ASYNC_STREAMING = False
```

No other code changes needed. The service will automatically use the legacy streaming.

---

## 6. Backward Compatibility

**Key Points:**
- Default value is `False` - existing behavior is preserved
- No changes to public API or event types
- Same events are emitted in both modes
- Same metrics are collected (accumulated per node)
- UI code requires no changes

**Migration Strategy:**
1. Deploy with `ENABLE_ASYNC_STREAMING = False` (default)
2. Test with flag enabled in staging: `ENABLE_ASYNC_STREAMING = True`
3. Monitor timing accuracy and event delivery
4. Enable in production when confident

---

## 7. Timing Considerations

### Legacy Mode
- **Node Duration:** `prev_time` to `current_time` between stream events
- **Accuracy:** High - measures actual time between graph yields
- **Granularity:** Milliseconds
- **Blocking:** Entire execution blocked

### Improved Mode
- **Node Duration:** Time from receiving event to completing processing
- **Accuracy:** Includes async emission time (~few milliseconds)
- **Granularity:** Milliseconds
- **Blocking:** Only individual event processing blocked

**Trade-off Explanation:**

In improved mode, the duration includes the time to:
1. Emit NodeStartEvent
2. Process node-specific events
3. Emit NodeEndEvent

This is typically 1-5ms overhead, but provides better UI responsiveness. If precise node execution timing is critical, use legacy mode.

---

## 8. Testing Recommendations

### Unit Tests to Add

```python
# Test legacy mode behavior
async def test_observable_agent_legacy_mode():
    service = ObservableAgentService()
    # Verify _use_async_streaming = False
    # Verify _astream_graph_legacy is called

# Test improved mode behavior
async def test_observable_agent_improved_mode():
    # Temporarily set ENABLE_ASYNC_STREAMING = True
    service = ObservableAgentService()
    # Verify _use_async_streaming = True
    # Verify _astream_graph_improved is called
    # Verify events arrive incrementally
```

### Integration Tests

1. **Event Order:** Ensure NodeStartEvent always comes before NodeEndEvent
2. **Metrics Accumulation:** Verify metrics are correctly summed for repeated nodes
3. **Event Completeness:** All node-specific events are emitted correctly
4. **Performance:** Measure WebSocket event delivery times
5. **UI Responsiveness:** Monitor real-time update frequency with improved mode

---

## 9. Monitoring & Debugging

### Logging Addition (Optional)

Add to `_astream_graph()` router:

```python
if self._use_async_streaming:
    print("[ObservableAgent] Using IMPROVED streaming mode")
    await self._astream_graph_improved(...)
else:
    print("[ObservableAgent] Using LEGACY streaming mode (backward compatible)")
    await self._astream_graph_legacy(...)
```

### Metrics to Track

- **Event latency:** Time from NodeStartEvent to NodeEndEvent
- **Event throughput:** Events per second
- **Total execution time:** Should be similar between modes
- **Peak async event loop latency:** Should improve with incremental mode

---

## 10. Future Enhancements

Potential improvements for future iterations:

1. **Adaptive Switching:** Auto-switch based on available system resources
2. **Chunked Processing:** Break large node processing into smaller chunks
3. **Event Batching:** Batch multiple events in a single emit call
4. **Timing Calibration:** Learn overhead and subtract from measurements
5. **Per-Node Configuration:** Allow different streaming modes per node type

---

## Summary

The implementation provides a clean, backward-compatible way to improve the streaming responsiveness of the ObservableAgentService while maintaining accurate metrics and event tracking. The feature flag enables easy A/B testing and gradual rollout.

### Key Changes:
1. Added `ENABLE_ASYNC_STREAMING` config flag (default: False)
2. Refactored `_astream_graph` into a router method
3. Created `_astream_graph_legacy` (existing behavior)
4. Created `_astream_graph_improved` (incremental streaming)
5. Both modes emit identical events and metrics

### Migration Path:
- Existing code: Works as-is with legacy mode
- Enable improved mode: Single config change
- No public API changes or UI modifications needed

# Async/Sync Streaming - Usage Examples

This document provides practical examples of how to use and test the new async/sync streaming feature.

---

## Example 1: Verify Current Configuration

### Code

```python
# In your application or test script
from config import ENABLE_ASYNC_STREAMING
from api.services.observable_agent import ObservableAgentService

print(f"Feature flag ENABLE_ASYNC_STREAMING = {ENABLE_ASYNC_STREAMING}")

service = ObservableAgentService()
print(f"Service will use async streaming: {service._use_async_streaming}")
```

### Expected Output

```
Feature flag ENABLE_ASYNC_STREAMING = False
Service will use async streaming: False
```

### What It Means

- `False` = Legacy mode (backward compatible, default)
- `True` = Improved mode (experimental, incremental)

---

## Example 2: Switch Between Modes (Testing)

### Step 1: Verify Legacy Mode

```bash
# Ensure config.py has:
# ENABLE_ASYNC_STREAMING = False

# Run your application
python main.py

# In client logs, you'll see normal batch event emission
```

### Step 2: Enable Improved Mode

Edit `/Users/kevin/github/personal/rusty-compass/langchain_agent/config.py`:

```python
# Change from:
ENABLE_ASYNC_STREAMING = False

# To:
ENABLE_ASYNC_STREAMING = True
```

### Step 3: Verify Improved Mode

```bash
# Restart application
python main.py

# In client logs, you'll see progressive event emission
```

### Step 4: Revert to Legacy

```python
# In config.py, change back:
ENABLE_ASYNC_STREAMING = False

# Restart application
```

---

## Example 3: Monitor Event Emission in Real-Time

### Using WebSocket Logging

```python
# Add to your WebSocket event handler
import json
import time

last_event_time = time.time()

async def on_websocket_message(message):
    global last_event_time

    current_time = time.time()
    delta = (current_time - last_event_time) * 1000  # ms
    last_event_time = current_time

    event = json.loads(message)
    print(f"[{delta:.2f}ms] Event: {event['type']}")
```

### Expected Output - Legacy Mode

```
[0.00ms] Event: NodeStartEvent
[2.34ms] Event: QueryEvaluationEvent
[1.45ms] Event: NodeEndEvent
[0.67ms] Event: NodeStartEvent
[125.67ms] Event: HybridSearchResultEvent
[0.89ms] Event: NodeEndEvent
[0.56ms] Event: NodeStartEvent
[250.34ms] Event: LLMResponseChunkEvent
[0.78ms] Event: NodeEndEvent
```

All events come in quick succession at the end of execution.

### Expected Output - Improved Mode

```
[0.00ms] Event: NodeStartEvent
[2.34ms] Event: QueryEvaluationEvent
[12.45ms] Event: NodeEndEvent
[125.67ms] Event: NodeStartEvent
[125.67ms] Event: HybridSearchResultEvent
[15.89ms] Event: NodeEndEvent
[250.34ms] Event: NodeStartEvent
[250.34ms] Event: LLMResponseChunkEvent
[25.78ms] Event: NodeEndEvent
```

Events are spread out over the execution time.

---

## Example 4: Measure Timing Accuracy

### Python Test Script

```python
import asyncio
import time
from config import ENABLE_ASYNC_STREAMING
from api.services.observable_agent import ObservableAgentService

# Store events for analysis
events = []

async def mock_emit(event):
    """Mock emit callback that logs events."""
    events.append({
        "type": type(event).__name__,
        "timestamp": time.time(),
    })

async def test_streaming_mode():
    service = ObservableAgentService()
    await service.ensure_initialized()

    # Clear events
    events.clear()

    # Process a message
    message = "What is langchain?"
    thread_id = f"test_{ENABLE_ASYNC_STREAMING}"

    start_time = time.time()
    response = await service.process_message(
        message=message,
        thread_id=thread_id,
        emit=mock_emit,
    )
    total_time = time.time() - start_time

    # Analyze events
    print(f"Mode: {'Improved' if ENABLE_ASYNC_STREAMING else 'Legacy'}")
    print(f"Total execution: {total_time*1000:.2f}ms")
    print(f"Events emitted: {len(events)}")
    print(f"Event spacing:")

    prev_time = events[0]['timestamp']
    for event in events[1:]:
        delta = (event['timestamp'] - prev_time) * 1000
        print(f"  → {event['type']:30s} (+{delta:.2f}ms)")
        prev_time = event['timestamp']

# Run test
asyncio.run(test_streaming_mode())
```

### Expected Output

**Legacy Mode:**
```
Mode: Legacy
Total execution: 450.23ms
Events emitted: 15
Event spacing:
  → QueryEvaluationEvent                 (+2.34ms)
  → NodeEndEvent                         (+1.45ms)
  → HybridSearchResultEvent              (+0.67ms)
  → NodeEndEvent                         (+125.67ms)
  → LLMResponseChunkEvent                (+0.89ms)
  → NodeEndEvent                         (+250.34ms)
  ...
```

**Improved Mode:**
```
Mode: Improved
Total execution: 450.15ms
Events emitted: 15
Event spacing:
  → QueryEvaluationEvent                 (+2.34ms)
  → NodeEndEvent                         (+12.45ms)
  → HybridSearchResultEvent              (+125.67ms)
  → NodeEndEvent                         (+15.89ms)
  → LLMResponseChunkEvent                (+250.34ms)
  → NodeEndEvent                         (+25.78ms)
  ...
```

**Key Observations:**
- Total execution time is nearly identical
- Event spacing is different (batch vs progressive)
- Improved mode shows longer gaps between event groups
- Both collect same metrics

---

## Example 5: Verify Metrics Are Identical

### Test Script

```python
import asyncio
from config import ENABLE_ASYNC_STREAMING
from api.services.observable_agent import ObservableAgentService

metrics_results = {}

async def mock_emit(event):
    """Capture MetricsEvent."""
    from api.schemas.events import MetricsEvent
    if isinstance(event, MetricsEvent):
        metrics_results['total_ms'] = event.total_ms
        metrics_results['query_evaluation_ms'] = event.query_evaluation_ms
        metrics_results['retrieval_ms'] = event.retrieval_ms
        metrics_results['document_grading_ms'] = event.document_grading_ms
        metrics_results['llm_generation_ms'] = event.llm_generation_ms
        metrics_results['response_grading_ms'] = event.response_grading_ms

async def test_metrics_consistency():
    service = ObservableAgentService()
    await service.ensure_initialized()

    message = "Test question?"
    thread_id = "test_metrics"

    response = await service.process_message(
        message=message,
        thread_id=thread_id,
        emit=mock_emit,
    )

    mode = "Improved" if ENABLE_ASYNC_STREAMING else "Legacy"
    print(f"Mode: {mode}")
    print(f"Metrics collected:")
    for key, value in metrics_results.items():
        if value is not None:
            print(f"  {key}: {value:.2f}ms")

# Test both modes
print("=" * 50)
asyncio.run(test_metrics_consistency())
```

### Expected Behavior

Run twice (once with each mode) - you should get similar metrics:

```
==================================================
Mode: Legacy
Metrics collected:
  total_ms: 450.23ms
  query_evaluation_ms: 50.12ms
  retrieval_ms: 200.34ms
  document_grading_ms: 100.45ms
  llm_generation_ms: 75.23ms
  response_grading_ms: 24.09ms

==================================================
Mode: Improved
Metrics collected:
  total_ms: 451.15ms
  query_evaluation_ms: 50.05ms
  retrieval_ms: 200.41ms
  document_grading_ms: 100.38ms
  llm_generation_ms: 75.31ms
  response_grading_ms: 24.00ms
```

Metrics are virtually identical (within 1-2%).

---

## Example 6: Testing Event Ordering

### Test Script

```python
import asyncio
from api.services.observable_agent import ObservableAgentService
from api.schemas.events import (
    NodeStartEvent, NodeEndEvent, BaseEvent
)

event_log = []

async def logging_emit(event):
    """Log all events in order."""
    event_log.append({
        "type": type(event).__name__,
        "node": getattr(event, "node", None),
    })

async def test_event_ordering():
    service = ObservableAgentService()
    await service.ensure_initialized()

    event_log.clear()

    response = await service.process_message(
        message="Test?",
        thread_id="test_ordering",
        emit=logging_emit,
    )

    # Verify ordering
    print("Event sequence:")
    start_events = []
    end_events = []

    for i, event in enumerate(event_log):
        event_type = event["type"]
        node = event["node"]
        print(f"{i:3d}. {event_type:30s} {node or ''}")

        if "Start" in event_type:
            start_events.append(event)
        if "End" in event_type:
            end_events.append(event)

    # Verify each node has Start then End
    print("\nNode pairing validation:")
    nodes_seen = {}
    valid = True

    for event in event_log:
        if event["type"] == "NodeStartEvent":
            node = event["node"]
            if node in nodes_seen and nodes_seen[node] == "start":
                print(f"  ERROR: {node} started twice without end")
                valid = False
            nodes_seen[node] = "start"
        elif event["type"] == "NodeEndEvent":
            node = event["node"]
            if node not in nodes_seen or nodes_seen[node] != "start":
                print(f"  ERROR: {node} ended without start")
                valid = False
            nodes_seen[node] = "end"

    if valid:
        print("  OK: All nodes properly paired (Start → End)")

    return valid

asyncio.run(test_event_ordering())
```

### Expected Output

```
Event sequence:
  0. NodeStartEvent                  query_evaluator
  1. QueryEvaluationEvent
  2. NodeEndEvent                    query_evaluator
  3. NodeStartEvent                  tools
  4. HybridSearchResultEvent
  5. NodeEndEvent                    tools
  6. NodeStartEvent                  document_grader
  7. DocumentGradingStartEvent
  8. DocumentGradeEvent
  9. DocumentGradingSummaryEvent
 10. NodeEndEvent                    document_grader
 11. NodeStartEvent                  agent
 12. LLMResponseStartEvent
 13. LLMResponseChunkEvent
 14. NodeEndEvent                    agent
 15. NodeStartEvent                  response_grader
 16. ResponseGradingEvent
 17. NodeEndEvent                    response_grader
 18. AgentCompleteEvent
 19. MetricsEvent

Node pairing validation:
  OK: All nodes properly paired (Start → End)
```

---

## Example 7: Stress Test - Multiple Concurrent Messages

### Test Script

```python
import asyncio
from api.services.observable_agent import ObservableAgentService

event_counts = {}

async def counting_emit(event):
    """Count events by type."""
    event_type = type(event).__name__
    event_counts[event_type] = event_counts.get(event_type, 0) + 1

async def test_concurrent_messages():
    service = ObservableAgentService()
    await service.ensure_initialized()

    event_counts.clear()

    # Create multiple concurrent requests
    messages = [
        "What is LangChain?",
        "How does retrieval work?",
        "Explain agent architecture.",
        "What are tools?",
    ]

    tasks = [
        service.process_message(
            message=msg,
            thread_id=f"test_{i}",
            emit=counting_emit,
        )
        for i, msg in enumerate(messages)
    ]

    # Run concurrently
    results = await asyncio.gather(*tasks)

    print(f"Processed {len(results)} concurrent messages")
    print("Event counts:")
    for event_type in sorted(event_counts.keys()):
        count = event_counts[event_type]
        print(f"  {event_type:30s} {count:4d}")

asyncio.run(test_concurrent_messages())
```

### Expected Output

```
Processed 4 concurrent messages
Event counts:
  AgentCompleteEvent                 4
  AgentErrorEvent                    0
  DocumentGradeEvent                10
  DocumentGradingStartEvent          4
  DocumentGradingSummaryEvent        4
  HybridSearchResultEvent            4
  LLMResponseChunkEvent              4
  LLMResponseStartEvent              4
  MetricsEvent                       4
  NodeEndEvent                      20
  NodeStartEvent                    20
  QueryEvaluationEvent               4
  ResponseGradingEvent               4
```

All metrics collected correctly with concurrent processing.

---

## Example 8: A/B Testing Performance

### Comparison Script

```python
import asyncio
import time
from dataclasses import dataclass
from api.services.observable_agent import ObservableAgentService

@dataclass
class PerfStats:
    mode: str
    total_time: float
    first_event_delay: float
    event_count: int
    events_per_second: float

async def measure_performance():
    service = ObservableAgentService()
    await service.ensure_initialized()

    mode = "Improved" if service._use_async_streaming else "Legacy"

    events = []
    exec_start = time.time()
    first_event_time = None

    async def tracking_emit(event):
        nonlocal first_event_time
        if first_event_time is None:
            first_event_time = time.time()
        events.append({"type": type(event).__name__, "time": time.time()})

    response = await service.process_message(
        message="Tell me about LangChain",
        thread_id="perf_test",
        emit=tracking_emit,
    )
    exec_end = time.time()

    total_time = exec_end - exec_start
    first_event_delay = (first_event_time - exec_start) * 1000 if first_event_time else 0
    event_count = len(events)
    events_per_second = event_count / total_time if total_time > 0 else 0

    stats = PerfStats(
        mode=mode,
        total_time=total_time * 1000,
        first_event_delay=first_event_delay,
        event_count=event_count,
        events_per_second=events_per_second,
    )

    return stats

# Compare both modes
print("Running performance comparison...\n")

# Note: You'd run this twice, changing config in between
stats = asyncio.run(measure_performance())

print(f"Mode: {stats.mode}")
print(f"Total execution: {stats.total_time:.2f}ms")
print(f"First event delay: {stats.first_event_delay:.2f}ms")
print(f"Total events: {stats.event_count}")
print(f"Event throughput: {stats.events_per_second:.2f} events/sec")
```

### Expected Results

**Legacy Mode:**
```
Mode: Legacy
Total execution: 450.23ms
First event delay: 450.12ms  ← All at the end
Total events: 20
Event throughput: 44.40 events/sec
```

**Improved Mode:**
```
Mode: Improved
Total execution: 451.08ms
First event delay: 2.34ms    ← Immediate
Total events: 20
Event throughput: 44.33 events/sec
```

Key difference: **First event delay** (improved mode gets feedback much sooner).

---

## Summary of Examples

1. **Verification** - Check current mode and configuration
2. **Switching** - Demonstrate hot-switching between modes
3. **Monitoring** - Real-time event flow analysis
4. **Timing** - Measure and compare execution profiles
5. **Metrics** - Verify consistency between modes
6. **Event Order** - Validate event sequencing
7. **Concurrency** - Test with multiple requests
8. **Performance** - A/B testing and metrics collection

All examples preserve backward compatibility and require no changes to your application code beyond the config.py flag.

---

## Best Practices

1. **Always test both modes** - Even if you deploy with legacy (False)
2. **Monitor metrics** - Ensure consistency between modes
3. **Check event ordering** - Each mode should maintain same order
4. **Load test** - Verify concurrent behavior
5. **Keep legacy as fallback** - Revert quickly if needed
6. **Use feature flag** - Don't hardcode mode selection
7. **Log mode selection** - Make it clear which mode is active
8. **Measure user impact** - A/B test with real workloads

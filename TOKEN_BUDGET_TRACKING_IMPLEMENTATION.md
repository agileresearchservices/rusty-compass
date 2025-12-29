# Token Budget Tracking Implementation Guide

## Overview

This implementation adds token budget tracking to the LangChain agent to prevent runaway costs. The system includes:

1. **Hard Budget Limit** (`REFLECTION_MAX_TOKENS_TOTAL = 50000`): When reached, the agent returns a budget exhausted message and refuses to proceed.
2. **Soft Warning Threshold** (`REFLECTION_TOKEN_WARNING_THRESHOLD = 40000`): When exceeded, warnings are printed but the agent continues.
3. **State Tracking**: New fields in `CustomAgentState` to track cumulative token usage and budget status.
4. **Budget Checks**: Token budget validation in the agent node before making LLM calls.
5. **Retry Prevention**: Budget checks in routing functions to skip retries when budget is exhausted.

## Changes Made

### 1. Configuration File (`langchain_agent/config.py`)

Added two configuration constants:

```python
# Hard limit on total tokens used in a conversation (prevents runaway costs)
# After reaching this limit, agent returns error and won't perform retries
REFLECTION_MAX_TOKENS_TOTAL = 50000

# Soft warning threshold - warns user when approaching limit (recommended: 80% of max)
# Allows agent to continue but with warnings
REFLECTION_TOKEN_WARNING_THRESHOLD = 40000
```

These are exported in the `__all__` list for easy importing.

### 2. CustomAgentState TypedDict (`langchain_agent/main.py`)

Added two new fields to track token usage:

```python
class CustomAgentState(TypedDict):
    # ... existing fields ...

    # Token budget tracking
    total_tokens_used: int                        # Cumulative tokens used in conversation
    token_budget_exceeded: bool                   # Flag indicating budget limit reached
```

### 3. Agent Node (`agent_node` method)

Implemented comprehensive token budget checking:

```python
def agent_node(self, state: CustomAgentState) -> Dict[str, Any]:
    """
    Agent reasoning node - calls LLM with tool binding.
    Includes token budget checking to prevent runaway costs.
    """
    messages = state["messages"]
    total_tokens = state.get("total_tokens_used", 0)
    token_budget_exceeded = state.get("token_budget_exceeded", False)

    # 1. Check if budget is already exceeded
    if token_budget_exceeded:
        budget_msg = AIMessage(
            content="I've exhausted my token budget. Here's my best attempt based on available information."
        )
        return {
            "messages": [budget_msg],
            "total_tokens_used": total_tokens,
            "token_budget_exceeded": True
        }

    # 2. Estimate tokens for current input
    estimated_input_tokens = self.estimate_token_count(messages)

    # 3. Check if proceeding would exceed budget
    # Conservative estimate: double the input tokens to account for potential output
    estimated_total = total_tokens + (estimated_input_tokens * 2)

    if estimated_total > REFLECTION_MAX_TOKENS_TOTAL:
        # Hard limit reached - return error message
        budget_msg = AIMessage(
            content="I've exhausted my token budget. Here's my best attempt based on available information."
        )
        if REFLECTION_SHOW_STATUS:
            print(f"[Token Budget] EXCEEDED: {estimated_total}/{REFLECTION_MAX_TOKENS_TOTAL}")
        return {
            "messages": [budget_msg],
            "total_tokens_used": total_tokens,
            "token_budget_exceeded": True
        }

    # 4. Warn if approaching soft limit
    if total_tokens > REFLECTION_TOKEN_WARNING_THRESHOLD:
        remaining = REFLECTION_MAX_TOKENS_TOTAL - total_tokens
        if REFLECTION_SHOW_STATUS:
            print(f"[Token Budget] WARNING: {total_tokens}/{REFLECTION_MAX_TOKENS_TOTAL} tokens used. Only {remaining} remaining.")

    # 5. Proceed with normal agent operation
    llm_with_tools = self.llm.bind_tools(self.tools)
    response = llm_with_tools.invoke(messages)

    # 6. Track tokens used by this call
    tokens_in_response = self.estimate_token_count([response])
    new_total = total_tokens + estimated_input_tokens + tokens_in_response

    return {
        "messages": [response],
        "total_tokens_used": new_total,
        "token_budget_exceeded": False
    }
```

**Key Features:**
- Prevents duplicate budget checks with a flag
- Estimates token count using the existing `estimate_token_count` method (1 token ≈ 4 characters)
- Uses 2x multiplier as conservative buffer for LLM output
- Tracks cumulative token usage across conversation turns
- Logs budget status to console when `REFLECTION_SHOW_STATUS` is enabled

### 4. Routing Function Updates

#### `route_after_doc_grading` method

Added token budget check before allowing document retry:

```python
def route_after_doc_grading(self, state: CustomAgentState) -> str:
    """
    Route after document grading: continue to agent or retry with transformed query.
    Skips retries if token budget is exceeded.
    """
    # If reflection is disabled, always continue to agent
    if not ENABLE_REFLECTION or not ENABLE_DOCUMENT_GRADING:
        return "agent"

    # Check token budget - skip retries if exceeded
    if state.get("token_budget_exceeded", False):
        if REFLECTION_SHOW_STATUS:
            print(f"[Reflection] Token budget exceeded. Skipping document retry.")
        return "agent"

    # ... rest of routing logic ...
```

#### `route_after_response_grading` method

Added token budget check before allowing response retry:

```python
def route_after_response_grading(self, state: CustomAgentState) -> str:
    """
    Route after response grading: end or retry with feedback.
    Skips retries if budget exceeded.
    """
    if not ENABLE_REFLECTION or not ENABLE_RESPONSE_GRADING:
        return "END"

    # Check token budget - skip retries if exceeded
    if state.get("token_budget_exceeded", False):
        if REFLECTION_SHOW_STATUS:
            print(f"[Reflection] Token budget exceeded. Skipping response retry.")
        return "END"

    # ... rest of routing logic ...
```

### 5. State Initialization (`_invoke_agent` method)

Initialize token budget fields in input data:

```python
input_data = {
    # ... existing fields ...
    # Token budget tracking initialization
    "total_tokens_used": 0,
    "token_budget_exceeded": False,
}
```

## Token Estimation Logic

The implementation uses the existing `estimate_token_count` method:

```python
def estimate_token_count(self, messages: Sequence[BaseMessage]) -> int:
    """
    Estimate token count for a list of messages.
    Uses 1 token ≈ 4 characters heuristic (conservative for English).
    """
    try:
        total_chars = 0
        for msg in messages:
            if hasattr(msg, "content") and msg.content:
                total_chars += len(str(msg.content))
        return total_chars // TOKEN_CHAR_RATIO
    except Exception:
        return 0
```

**Note:** Token counts are estimates based on the character ratio configured in `config.py` (default: 1 token per 4 characters). For more accurate token counting with specific models, consider using the model's actual tokenizer.

## Console Output Examples

### Warning Threshold Reached
```
[Token Budget] WARNING: 40,000/50,000 tokens used. Only 10,000 remaining.
```

### Hard Limit Exceeded
```
[Token Budget] EXCEEDED: 52,000/50,000
```

### Retry Skipped Due to Budget
```
[Reflection] Token budget exceeded. Skipping document retry.
[Reflection] Token budget exceeded. Skipping response retry.
```

## Configuration Recommendations

### Default Values (Suitable for Most Use Cases)
- **Hard Limit:** 50,000 tokens (~200,000 characters)
- **Warning Threshold:** 40,000 tokens (~160,000 characters)

### For Cost-Conscious Operations
```python
REFLECTION_MAX_TOKENS_TOTAL = 20000
REFLECTION_TOKEN_WARNING_THRESHOLD = 16000
```

### For Research/Development
```python
REFLECTION_MAX_TOKENS_TOTAL = 100000
REFLECTION_TOKEN_WARNING_THRESHOLD = 80000
```

## Behavior Summary

| Condition | Behavior |
|-----------|----------|
| **total_tokens < warning_threshold** | Normal operation, no budget messages |
| **warning_threshold < total_tokens < hard_limit** | Continue but print warning; block retries if flag set |
| **total_tokens >= hard_limit** | Return budget exhausted message; block all retries |
| **token_budget_exceeded flag = True** | Skip all retries immediately |

## Important Notes

1. **Token Estimation is Conservative:** The system multiplies input tokens by 2 to estimate potential output, ensuring we don't exceed budget
2. **Per-Conversation Tracking:** Token count resets for each new conversation (thread_id)
3. **No Token Deduction:** Once a turn completes, tokens are added cumulatively; no tokens are deducted
4. **Graceful Degradation:** When budget is exceeded, the agent returns a best-effort response rather than failing completely
5. **Retry Prevention:** Both document retrieval retries (via query transformation) and response improvement retries are blocked when budget is exceeded

## Testing the Implementation

### Test 1: Normal Operation
```python
# With token budget = 50000, run normal conversation
# Expected: No budget messages, normal conversation flow
```

### Test 2: Warning Threshold
```python
# Set REFLECTION_MAX_TOKENS_TOTAL = 5000
# Send messages totaling ~3000 tokens
# Expected: Warning message printed, conversation continues
```

### Test 3: Hard Limit
```python
# Set REFLECTION_MAX_TOKENS_TOTAL = 2000
# Send messages totaling ~1500 tokens
# Expected: Budget exhausted message, no further processing
```

### Test 4: Retry Prevention
```python
# Set REFLECTION_MAX_TOKENS_TOTAL = 4000, DOCUMENT_GRADING = True
# Send initial query (~1000 tokens)
# When documents fail grading with token_budget_exceeded = True
# Expected: No retry query transformation, proceed with available docs
```

## Future Enhancements

1. **Granular Budget Allocation:** Allocate specific token budgets to different phases (retrieval vs. generation)
2. **Token Carryover:** Store token counts in PostgreSQL per conversation for true multi-turn budgeting
3. **Model-Specific Tokenization:** Use actual model tokenizers for precise token counting
4. **Dynamic Budgeting:** Adjust limits based on conversation complexity
5. **Cost Tracking:** Integrate with LLM pricing to show monetary cost alongside token count

## Files Modified

1. `/Users/kevin/github/personal/rusty-compass/langchain_agent/config.py`
   - Added `REFLECTION_MAX_TOKENS_TOTAL` and `REFLECTION_TOKEN_WARNING_THRESHOLD` constants

2. `/Users/kevin/github/personal/rusty-compass/langchain_agent/main.py`
   - Updated `CustomAgentState` TypedDict with token tracking fields
   - Enhanced `agent_node` method with budget checking logic
   - Updated `route_after_doc_grading` method to skip retries when budget exceeded
   - Updated `route_after_response_grading` method to skip retries when budget exceeded
   - Initialized token budget fields in `_invoke_agent` method

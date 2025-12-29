# Token Budget Tracking - Implementation Steps

## Quick Summary

This document provides step-by-step instructions to implement token budget tracking in the LangChain agent. The implementation prevents runaway costs by enforcing hard limits and soft warnings on token usage.

## What's Already Done

✓ **config.py** - Token budget constants added:
- `REFLECTION_MAX_TOKENS_TOTAL = 50000` (hard limit)
- `REFLECTION_TOKEN_WARNING_THRESHOLD = 40000` (soft warning)
- Constants exported in `__all__` list

## What Needs to Be Done

The following changes need to be applied to `/Users/kevin/github/personal/rusty-compass/langchain_agent/main.py`:

### Step 1: Import Token Budget Constants

**Location:** Lines 51-100 (import section)

**Action:** Add these two imports after existing config imports:

```python
    # Token budget tracking
    REFLECTION_MAX_TOKENS_TOTAL,
    REFLECTION_TOKEN_WARNING_THRESHOLD,
```

**Current Location Reference:** After `DOCUMENT_GRADING_BATCH_SIZE,`

### Step 2: Update CustomAgentState TypedDict

**Location:** Lines 266-292 (CustomAgentState class definition)

**Action:** Add two new fields at the end of the TypedDict:

```python
    # Token budget tracking
    total_tokens_used: int                        # Cumulative tokens used in conversation
    token_budget_exceeded: bool                   # Flag indicating budget limit reached
```

**Docstring Update:** Change the docstring from:
```
This extends the default agent state to include query analysis,
dynamic search parameter adjustment, and reflection loop state.
```

To:
```
This extends the default agent state to include query analysis,
dynamic search parameter adjustment, reflection loop state, and token budget tracking.
```

### Step 3: Replace agent_node Method

**Location:** Lines 899-911 (agent_node method)

**Action:** Replace the entire `agent_node` method with the new implementation that includes token budget checking.

See `/Users/kevin/github/personal/rusty-compass/IMPLEMENTATION_PATCH.txt` for the complete new method code.

**Key Changes:**
- Check if budget already exceeded
- Estimate tokens for current input
- Check if proceeding would exceed budget
- Warn if approaching soft limit
- Track tokens after LLM call
- Return updated state with token counts

### Step 4: Update route_after_doc_grading Method

**Location:** Lines 1000-1023 (route_after_doc_grading method)

**Action:** Add token budget check right after the reflection disabled check:

```python
# Check token budget - skip retries if exceeded
if state.get("token_budget_exceeded", False):
    if REFLECTION_SHOW_STATUS:
        print(f"[Reflection] Token budget exceeded. Skipping document retry.")
    return "agent"
```

**Placement:** After the `if not ENABLE_REFLECTION or not ENABLE_DOCUMENT_GRADING:` block

**Docstring Update:** Change from:
```
Route after document grading: continue to agent or retry with transformed query.
```

To:
```
Route after document grading: continue to agent or retry with transformed query.
Skips retries if token budget is exceeded.
```

### Step 5: Update route_after_response_grading Method

**Location:** Lines 1025-1049 (route_after_response_grading method)

**Action:** Add token budget check right after the reflection disabled check:

```python
# Check token budget - skip retries if exceeded
if state.get("token_budget_exceeded", False):
    if REFLECTION_SHOW_STATUS:
        print(f"[Reflection] Token budget exceeded. Skipping response retry.")
    return "END"
```

**Placement:** After the `if not ENABLE_REFLECTION or not ENABLE_RESPONSE_GRADING:` block

**Docstring Update:** Change from:
```
Route after response grading: end or retry with feedback.

If the response failed grading and retries are available, routes back
to the agent with feedback to improve the response.
```

To:
```
Route after response grading: end or retry with feedback.

If the response failed grading and retries are available, routes back
to the agent with feedback to improve the response. Skips retries if budget exceeded.
```

### Step 6: Initialize Token Budget in _invoke_agent

**Location:** Lines 1993-2007 (_invoke_agent method, input_data initialization)

**Action:** Add two new fields to the `input_data` dictionary:

```python
                # Token budget tracking initialization
                "total_tokens_used": 0,
                "token_budget_exceeded": False,
```

**Placement:** After the `"transformed_query": None,` line

## Implementation Checklist

- [ ] Step 1: Import token budget constants from config
- [ ] Step 2: Update CustomAgentState TypedDict with token tracking fields
- [ ] Step 3: Replace agent_node method with token budget checking logic
- [ ] Step 4: Add token budget check to route_after_doc_grading
- [ ] Step 5: Add token budget check to route_after_response_grading
- [ ] Step 6: Initialize token budget fields in _invoke_agent state

## Verification Steps

After implementing all changes:

```bash
# 1. Check Python syntax
python3 -m py_compile langchain_agent/main.py

# 2. Check imports are valid
python3 -c "from langchain_agent.config import REFLECTION_MAX_TOKENS_TOTAL, REFLECTION_TOKEN_WARNING_THRESHOLD; print('✓ Imports OK')"

# 3. Check the implementation
grep -n "total_tokens_used" langchain_agent/main.py
grep -n "token_budget_exceeded" langchain_agent/main.py
grep -n "Token Budget" langchain_agent/main.py
```

## Testing the Implementation

### Test 1: Default Behavior (No Budget Warnings)
```
Expected: Normal conversation without budget messages
Command: Start agent and have normal conversation
```

### Test 2: Warning Threshold
```
Configuration:
  REFLECTION_MAX_TOKENS_TOTAL = 5000
  REFLECTION_TOKEN_WARNING_THRESHOLD = 3000
Expected: "[Token Budget] WARNING" message appears after ~3000 tokens used
```

### Test 3: Hard Limit
```
Configuration:
  REFLECTION_MAX_TOKENS_TOTAL = 2000
Expected: "[Token Budget] EXCEEDED" message and "exhausted my token budget" response
```

### Test 4: Retry Prevention
```
Configuration: Default token limits
Enable: ENABLE_DOCUMENT_GRADING and ENABLE_QUERY_TRANSFORMATION
Send: Query that causes document retry with token_budget_exceeded = True
Expected: "[Reflection] Token budget exceeded. Skipping document retry."
```

## Configuration Values

### Recommended Settings

**Production (Cost-Conscious):**
```python
REFLECTION_MAX_TOKENS_TOTAL = 20000
REFLECTION_TOKEN_WARNING_THRESHOLD = 16000
```

**Standard (Default):**
```python
REFLECTION_MAX_TOKENS_TOTAL = 50000
REFLECTION_TOKEN_WARNING_THRESHOLD = 40000
```

**Development/Research:**
```python
REFLECTION_MAX_TOKENS_TOTAL = 100000
REFLECTION_TOKEN_WARNING_THRESHOLD = 80000
```

To use different values, edit `langchain_agent/config.py` lines 299-303.

## Token Estimation Reference

The implementation uses this formula for token estimation:

```
Estimated Tokens = Character Count / TOKEN_CHAR_RATIO

Where TOKEN_CHAR_RATIO = 4 (1 token ≈ 4 characters)
```

**Example:**
- 1000 characters ≈ 250 tokens
- 4000 characters ≈ 1000 tokens
- 200,000 characters ≈ 50,000 tokens (hard limit)

## Expected Console Output

### Normal Operation
```
You: What is Python?
[Query Evaluator] Lambda: 0.85 | Conceptual question about Python
[Agent proceeds without budget warnings]
```

### Warning Threshold Reached
```
You: [Another question after many messages]
[Query Evaluator] Lambda: 0.50 | Balanced query
[Token Budget] WARNING: 40200/50000 tokens used. Only 9800 remaining.
[Agent proceeds but cannot retry]
```

### Hard Limit Exceeded
```
You: [Another question after token limit]
[Token Budget] EXCEEDED: 52000/50000
I've exhausted my token budget. Here's my best attempt based on available information.
```

### Retry Skipped
```
[Document Grader] ✗ 1/4 documents relevant (avg score: 0.35)
[Reflection] Token budget exceeded. Skipping document retry.
[Agent continues with available documents]
```

## Troubleshooting

### Import Error: "cannot import name 'REFLECTION_MAX_TOKENS_TOTAL'"
**Solution:** Ensure Step 1 is completed and both constants are in the import statement.

### TypeError: "unhashable type: 'int' for total_tokens_used"
**Solution:** Ensure TypedDict field type is `int` not `Int` (Python's type hints use lowercase).

### Budget checks not working
**Solution:** Verify REFLECTION_SHOW_STATUS = True in config.py to see debug messages.

### State fields not initialized
**Solution:** Check Step 6 - ensure input_data dict has both token budget fields.

## Additional Resources

- **Token Budget Documentation:** See `TOKEN_BUDGET_TRACKING_IMPLEMENTATION.md`
- **Implementation Patch:** See `IMPLEMENTATION_PATCH.txt` for complete code diffs
- **Configuration Options:** Edit `langchain_agent/config.py` lines 299-303

## Post-Implementation Next Steps

1. **Run tests** to verify token tracking works correctly
2. **Monitor production** to observe actual token usage patterns
3. **Adjust budgets** based on observed token consumption
4. **Consider implementing** token carryover to PostgreSQL for true per-conversation budgeting
5. **Integrate cost tracking** by multiplying tokens by LLM pricing

---

**Status:** Config changes complete. Main.py changes pending manual application.

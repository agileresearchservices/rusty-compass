# Before & After Code Comparison

## The Complete Enhanced Implementation

### Location
File: `/Users/kevin/github/personal/rusty-compass/langchain_agent/main.py`
Method: `query_transformer_node` (lines 1137-1233)

### Side-by-Side Comparison

#### BEFORE: Lines extracting feedback
```python
# OLD: Only negative feedback
failed_reasons = [
    g["reasoning"] for g in document_grades if not g["relevant"]
]

transform_prompt = f"""Rewrite this search query to improve document retrieval.

Original Query: {original_query}

The initial search returned documents that were not relevant because:
{chr(10).join(f"- {r}" for r in failed_reasons[:3])}

Write a new query that:
1. Captures the same user intent
2. Uses different keywords/synonyms
3. Is more specific or more general as needed
4. Focuses on core concepts

Respond with ONLY the rewritten query, nothing else."""
```

#### AFTER: Lines extracting feedback
```python
# NEW: Both positive and negative feedback

# Extract relevant and irrelevant documents
relevant_docs = [g for g in document_grades if g["relevant"]]
irrelevant_docs = [g for g in document_grades if not g["relevant"]]

# Sort by score (highest first) to get top quality examples
relevant_docs_sorted = sorted(
    relevant_docs,
    key=lambda x: x.get("score", 0.5),
    reverse=True
)
irrelevant_docs_sorted = sorted(
    irrelevant_docs,
    key=lambda x: x.get("score", 0.5),
    reverse=True
)

# Build positive feedback section
positive_feedback = ""
if relevant_docs_sorted:
    top_relevant = relevant_docs_sorted[:2]
    positive_feedback = "DOCUMENTS THAT WERE RELEVANT:\n"
    for doc in top_relevant:
        positive_feedback += f"- {doc['reasoning']}\n"

# Build negative feedback section
negative_feedback = ""
if irrelevant_docs_sorted:
    top_irrelevant = irrelevant_docs_sorted[:3]
    negative_feedback = "DOCUMENTS THAT WERE NOT RELEVANT:\n"
    for doc in top_irrelevant:
        negative_feedback += f"- {doc['reasoning']}\n"

# Build the enhanced transformation prompt
transform_prompt = f"""Rewrite this search query to improve document retrieval.

Original Query: {original_query}

Learning from what worked and what didn't:

{positive_feedback}

{negative_feedback}

Write a new query that:
1. Preserves the aspects and keywords that led to RELEVANT documents
2. Avoids terms and phrasings that led to irrelevant results
3. Emphasizes what worked in the relevant documents
4. Captures the same user intent but more effectively
5. Uses different keywords/synonyms where needed
6. Is more specific or more general as appropriate

Focus on: What made the relevant documents relevant? Amplify those signals.
Avoid: What made other documents irrelevant? Don't repeat those patterns.

Respond with ONLY the rewritten query, nothing else."""
```

#### BEFORE: Logging
```python
if REFLECTION_SHOW_STATUS:
    print(f"[Query Transformer] '{original_query}' → '{transformed}'")
```

#### AFTER: Logging
```python
if REFLECTION_SHOW_STATUS:
    feedback_info = f"({len(relevant_docs)} relevant, {len(irrelevant_docs)} irrelevant)"
    print(f"[Query Transformer] {feedback_info} '{original_query}' → '{transformed}'")
```

## Example Outputs

### Old Prompt (Negative Only)
```
Rewrite this search query to improve document retrieval.

Original Query: machine learning algorithms

The initial search returned documents that were not relevant because:
- Discusses marketing strategies for AI products
- Covers only historical perspective without technical content
- Focuses on career advice rather than algorithms

Write a new query that:
1. Captures the same user intent
2. Uses different keywords/synonyms
3. Is more specific or more general as needed
4. Focuses on core concepts

Respond with ONLY the rewritten query, nothing else.
```

### New Prompt (Positive + Negative)
```
Rewrite this search query to improve document retrieval.

Original Query: machine learning algorithms

Learning from what worked and what didn't:

DOCUMENTS THAT WERE RELEVANT:
- Discusses supervised learning algorithms and their implementations
- Covers neural networks, deep learning, and classification techniques

DOCUMENTS THAT WERE NOT RELEVANT:
- Discusses marketing strategies for AI products
- Covers only historical perspective without technical content
- Focuses on career advice rather than algorithms

Write a new query that:
1. Preserves the aspects and keywords that led to RELEVANT documents
2. Avoids terms and phrasings that led to irrelevant results
3. Emphasizes what worked in the relevant documents
4. Captures the same user intent but more effectively
5. Uses different keywords/synonyms where needed
6. Is more specific or more general as appropriate

Focus on: What made the relevant documents relevant? Amplify those signals.
Avoid: What made other documents irrelevant? Don't repeat those patterns.

Respond with ONLY the rewritten query, nothing else.
```

## Expected Outcomes

### Old Implementation
- Query Transformer sees: What failed
- Result: "What should I avoid?"
- Transformed query: "machine learning algorithms" → "algorithm implementations" (generic)

### New Implementation
- Query Transformer sees: What worked AND what failed
- Result: "What should I preserve? What should I avoid?"
- Transformed query: "machine learning algorithms" → "supervised learning neural networks classification techniques" (specific)

## Line-by-Line Changes

| Aspect | Old | New | Benefit |
|--------|-----|-----|---------|
| Data extraction | 3 lines | 16 lines | Both categories + quality sorting |
| Feedback sections | 1 section | 2 sections | Balanced learning |
| Sort by quality | No | Yes | Better examples |
| Top docs | 3 irrelevant | 2 relevant + 3 irrelevant | More balanced |
| Prompt structure | 4 bullets | 6 bullets + focus statement | Clearer guidance |
| Logging info | Query only | Query + feedback counts | Better observability |

## Backward Compatibility Checklist

- [x] Function signature unchanged
- [x] Return type unchanged
- [x] No new dependencies
- [x] No breaking changes to state
- [x] Handles edge cases (no relevant docs, no irrelevant docs)
- [x] Works with batch and individual grading modes
- [x] Default behavior with missing keys (get with default)
- [x] Python syntax validated

## Integration Points

The enhancement integrates with:
- `state["document_grades"]` - reads grade list
- `state.get("original_query", "")` - reads query
- `self.llm.invoke()` - same LLM invoke as before
- `ENABLE_REFLECTION` - existing feature flag
- `ENABLE_QUERY_TRANSFORMATION` - existing feature flag
- `REFLECTION_SHOW_STATUS` - existing logging flag
- Return state - same structure as before

## Code Quality

- **Comments**: Clear section headers and explanations
- **Variable names**: Self-documenting (relevant_docs_sorted, positive_feedback)
- **Logic flow**: Separated into logical sections
- **Error handling**: Preserves existing try/except
- **Style**: Consistent with existing codebase
- **Performance**: No performance degradation
- **Maintainability**: Easy to adjust thresholds (2 relevant, 3 irrelevant)

## Testing Approach

```python
# Test Case 1: Mixed results
document_grades = [
    {"relevant": True, "score": 0.9, "reasoning": "Specific algorithm details"},
    {"relevant": True, "score": 0.85, "reasoning": "Relevant comparison"},
    {"relevant": False, "score": 0.3, "reasoning": "Too historical"},
    {"relevant": False, "score": 0.2, "reasoning": "Off-topic entirely"},
]
# Expected: Both positive and negative feedback in prompt

# Test Case 2: All relevant
document_grades = [
    {"relevant": True, "score": 0.9, "reasoning": "Excellent match"},
    {"relevant": True, "score": 0.88, "reasoning": "Very relevant"},
]
# Expected: Only positive feedback, empty negative_feedback string

# Test Case 3: All irrelevant
document_grades = [
    {"relevant": False, "score": 0.3, "reasoning": "Wrong topic"},
    {"relevant": False, "score": 0.2, "reasoning": "Unrelated"},
]
# Expected: Only negative feedback, empty positive_feedback string

# Test Case 4: Empty grades
document_grades = []
# Expected: Both empty strings, still generates transformed query
```

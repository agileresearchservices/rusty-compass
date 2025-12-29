# Query Transformer Enhancement: Positive and Negative Feedback

## Overview
This enhancement upgrades the `query_transformer_node` function to use both positive feedback (from relevant documents) and negative feedback (from irrelevant documents) to intelligently rewrite search queries.

## Current Implementation Location
File: `/Users/kevin/github/personal/rusty-compass/langchain_agent/main.py`
Lines: 1148-1206

## Current Limitations
The current implementation only learns from negative feedback:
- Extracts reasoning from documents where `grade.relevant = False`
- Tells the LLM what NOT to do
- Misses opportunities to reinforce what IS working

## Enhanced Implementation

Replace the entire `query_transformer_node` method with this implementation:

```python
def query_transformer_node(self, state: CustomAgentState) -> Dict[str, Any]:
    """
    Transform/rewrite the query for better retrieval results.

    Called when document grading fails and retry is allowed.
    Uses both positive and negative feedback to learn from success and failure.
    """
    if not ENABLE_REFLECTION or not ENABLE_QUERY_TRANSFORMATION:
        return {}

    original_query = state.get("original_query", "")
    document_grades = state.get("document_grades", [])

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

    try:
        response = self.llm.invoke(transform_prompt)
        transformed = response.content.strip()
    except Exception as e:
        print(f"[Query Transformer] Error: {e}")
        transformed = original_query  # Fallback to original

    if REFLECTION_SHOW_STATUS:
        feedback_info = f"({len(relevant_docs)} relevant, {len(irrelevant_docs)} irrelevant)"
        print(f"[Query Transformer] {feedback_info} '{original_query}' → '{transformed}'")

    # Increment iteration count
    new_iteration = state.get("iteration_count", 0) + 1

    # Create new user message with transformed query for retry
    retry_message = HumanMessage(
        content=f"[Retry with transformed query] {transformed}"
    )

    return {
        "transformed_query": transformed,
        "iteration_count": new_iteration,
        "messages": [retry_message],
        "lambda_mult": 0.5,  # Reset to balanced for retry
        "retrieved_documents": [],  # Clear previous documents
        "document_grades": [],
        "document_grade_summary": {}
    }
```

## Key Changes Explained

### 1. Extract Both Relevant and Irrelevant Documents
```python
relevant_docs = [g for g in document_grades if g["relevant"]]
irrelevant_docs = [g for g in document_grades if not g["relevant"]]
```
- Separates documents into two categories based on the `relevant` boolean flag
- Enables learning from both successes and failures

### 2. Sort by Score to Get Best Examples
```python
relevant_docs_sorted = sorted(
    relevant_docs,
    key=lambda x: x.get("score", 0.5),
    reverse=True
)
```
- Sorts both categories by relevance score (highest first)
- Ensures the LLM learns from the highest-quality examples
- Uses top 2 relevant docs and top 3 irrelevant docs

### 3. Build Positive Feedback Context
```python
positive_feedback = "DOCUMENTS THAT WERE RELEVANT:\n"
for doc in top_relevant:
    positive_feedback += f"- {doc['reasoning']}\n"
```
- Extracts and formats the reasoning from successful document retrievals
- Shows the LLM what aspects of the query led to good results
- Encourages the transformer to preserve and amplify these patterns

### 4. Build Negative Feedback Context
```python
negative_feedback = "DOCUMENTS THAT WERE NOT RELEVANT:\n"
for doc in top_irrelevant:
    negative_feedback += f"- {doc['reasoning']}\n"
```
- Extracts reasoning from irrelevant documents
- Shows what didn't work
- Helps the transformer avoid repeating failures

### 5. Enhanced Prompt Instructions
The new prompt includes:
- **Instruction 1**: Preserve aspects from relevant documents
- **Instruction 2**: Avoid irrelevant patterns
- **Instruction 3**: Emphasize what worked
- **Instructions 4-6**: Original intent preservation

The prompt now explicitly tells the LLM to:
- Look at what made relevant docs work
- Amplify those successful signals
- Don't repeat patterns that led to irrelevance

### 6. Enhanced Logging
```python
feedback_info = f"({len(relevant_docs)} relevant, {len(irrelevant_docs)} irrelevant)"
print(f"[Query Transformer] {feedback_info} '{original_query}' → '{transformed}'")
```
- Shows both counts in debug output
- Example: `[Query Transformer] (2 relevant, 5 irrelevant) 'original' → 'transformed'`

## Benefits of This Enhancement

1. **Learns from Success**: The transformer doesn't just learn what failed; it learns what worked
2. **Balanced Perspective**: Combines positive and negative examples for more nuanced transformations
3. **Quality-Weighted**: Uses highest-scoring documents as examples, not random ones
4. **Interpretable**: The feedback shown to the LLM clearly shows why documents were/weren't relevant
5. **Better Query Rewrites**: More likely to preserve beneficial keywords while removing detrimental ones

## Example Usage Scenario

**Original Query**: "machine learning algorithms"

**Retrieved Documents**:
- Doc 1: Relevant (score 0.9) - "Discusses supervised learning algorithms"
- Doc 2: Relevant (score 0.85) - "Covers neural networks and deep learning"
- Doc 3: Not Relevant (score 0.4) - "History of computers and technology trends"
- Doc 4: Not Relevant (score 0.35) - "Marketing strategies for tech companies"
- Doc 5: Not Relevant (score 0.3) - "General overview of machine learning without technical depth"

**Positive Feedback Given to LLM**:
```
DOCUMENTS THAT WERE RELEVANT:
- Discusses supervised learning algorithms
- Covers neural networks and deep learning
```

**Negative Feedback Given to LLM**:
```
DOCUMENTS THAT WERE NOT RELEVANT:
- History of computers and technology trends
- Marketing strategies for tech companies
- General overview of machine learning without technical depth
```

**Transformer's Learned Instruction**:
"Transform the query to focus more on specific algorithm types (supervised learning, neural networks, deep learning) and less on general/historical perspectives or marketing content"

**Transformed Query**: "supervised learning algorithms neural networks deep learning technical implementation"

## Implementation Steps

1. Open `/Users/kevin/github/personal/rusty-compass/langchain_agent/main.py`
2. Locate the `query_transformer_node` method (around line 1148)
3. Replace the entire method with the enhanced version above
4. Test by running queries that trigger document grading and query transformation
5. Check the logs to see the new feedback format: `[Query Transformer] (X relevant, Y irrelevant) ...`

## Compatibility

- No changes to function signature or return type
- No new imports required
- Fully compatible with existing state management
- No changes to document grade structure needed
- Works with both individual and batch document grading modes

## Testing Recommendations

1. Test with queries that have some relevant results
2. Test with queries that have mostly irrelevant results
3. Monitor transformed queries to ensure they improve retrieval
4. Check debug output to verify feedback counts are correct
5. Compare transformed query quality before/after enhancement

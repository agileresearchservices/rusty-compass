"""
Agent state types for LangGraph custom agent.

Contains TypedDict definitions for:
- DocumentGrade: Individual document relevance grade
- ReflectionResult: Grading operation result
- CustomAgentState: Full agent state schema with reflection support
"""

from typing import Sequence, List, Optional, Annotated
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from langgraph.graph import add_messages


class DocumentGrade(TypedDict):
    """Grade for a single retrieved document."""
    source: str
    relevant: bool
    score: float
    reasoning: str


class ReflectionResult(TypedDict):
    """Result of a grading operation."""
    grade: str  # "pass" or "fail"
    score: float  # 0.0 - 1.0
    reasoning: str  # Explanation


class CustomAgentState(TypedDict):
    """
    State schema for custom agent graph with dynamic lambda_mult and reflection.

    This extends the default agent state to include query analysis,
    dynamic search parameter adjustment, and reflection loop state.
    """
    # Core message state
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Query evaluation state
    lambda_mult: float
    query_analysis: str
    optimized_query: Optional[str]  # Pre-optimized query from evaluator

    # Reflection state
    iteration_count: int                          # Track retrieval iterations (0, 1, or 2)
    response_retry_count: int                     # Track response regeneration attempts
    retrieved_documents: List[Document]           # Raw documents from retrieval
    document_grades: List[DocumentGrade]          # Individual document grades
    document_grade_summary: ReflectionResult      # Overall document relevance
    response_grade: ReflectionResult              # Quality of final response
    original_query: str                           # Preserve original for transformation
    transformed_query: Optional[str]              # Rewritten query if docs were poor
    force_retrieval_retry: Optional[bool]         # Force agent to call knowledge_base after transformation

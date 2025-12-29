"""
Pydantic models for WebSocket events.

These events are streamed in real-time as the agent executes,
providing full observability into every step and decision.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ============================================================================
# BASE EVENT
# ============================================================================


class BaseEvent(BaseModel):
    """Base class for all WebSocket events."""

    type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    node: Optional[str] = None  # Current graph node name

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# ============================================================================
# CONNECTION EVENTS
# ============================================================================


class ConnectionEstablished(BaseEvent):
    """Sent when WebSocket connection is established."""

    type: Literal["connection_established"] = "connection_established"
    thread_id: str
    existing_messages: int = 0


class ConnectionError(BaseEvent):
    """Sent when connection fails."""

    type: Literal["connection_error"] = "connection_error"
    error: str


# ============================================================================
# NODE LIFECYCLE EVENTS
# ============================================================================


class NodeStartEvent(BaseEvent):
    """Emitted when a graph node starts execution."""

    type: Literal["node_start"] = "node_start"
    node: str
    input_summary: Optional[str] = None


class NodeEndEvent(BaseEvent):
    """Emitted when a graph node completes execution."""

    type: Literal["node_end"] = "node_end"
    node: str
    duration_ms: float
    output_summary: Optional[str] = None


# ============================================================================
# QUERY EVALUATOR EVENTS
# ============================================================================


class QueryEvaluationEvent(BaseEvent):
    """Emitted when query is evaluated for search strategy."""

    type: Literal["query_evaluation"] = "query_evaluation"
    node: Literal["query_evaluator"] = "query_evaluator"
    query: str
    lambda_mult: float  # 0.0 (lexical) to 1.0 (semantic)
    query_analysis: str  # LLM's reasoning
    search_strategy: str  # "lexical-heavy", "balanced", "semantic-heavy"


# ============================================================================
# HYBRID SEARCH EVENTS
# ============================================================================


class HybridSearchStartEvent(BaseEvent):
    """Emitted when hybrid search begins."""

    type: Literal["hybrid_search_start"] = "hybrid_search_start"
    node: Literal["tools"] = "tools"
    query: str
    lambda_mult: float
    fetch_k: int


class SearchCandidate(BaseModel):
    """A single search candidate before reranking."""

    source: str
    snippet: str
    full_content: Optional[str] = None
    vector_score: Optional[float] = None
    text_score: Optional[float] = None
    rrf_score: Optional[float] = None


class HybridSearchResultEvent(BaseEvent):
    """Emitted when hybrid search completes with candidates."""

    type: Literal["hybrid_search_result"] = "hybrid_search_result"
    node: Literal["tools"] = "tools"
    candidate_count: int
    candidates: List[SearchCandidate]


# ============================================================================
# RERANKER EVENTS
# ============================================================================


class RerankerStartEvent(BaseEvent):
    """Emitted when reranking begins."""

    type: Literal["reranker_start"] = "reranker_start"
    node: Literal["tools"] = "tools"
    model: str
    candidate_count: int


class RerankedDocument(BaseModel):
    """A document after reranking with its new score and rank."""

    source: str
    score: float  # Cross-encoder score (0.0-1.0)
    rank: int  # New rank after reranking
    original_rank: int  # Rank before reranking
    snippet: str
    rank_change: int = 0  # How much the rank changed


class RerankerResultEvent(BaseEvent):
    """Emitted when reranking completes with scored documents."""

    type: Literal["reranker_result"] = "reranker_result"
    node: Literal["tools"] = "tools"
    results: List[RerankedDocument]
    reranking_changed_order: bool = False


# ============================================================================
# DOCUMENT GRADING EVENTS
# ============================================================================


class DocumentGradingStartEvent(BaseEvent):
    """Emitted when document grading begins."""

    type: Literal["document_grading_start"] = "document_grading_start"
    node: Literal["document_grader"] = "document_grader"
    document_count: int


class DocumentGradeEvent(BaseEvent):
    """Emitted for each document that is graded."""

    type: Literal["document_grade"] = "document_grade"
    node: Literal["document_grader"] = "document_grader"
    source: str
    relevant: bool
    score: float  # 0.0-1.0
    reasoning: str


class DocumentGradingSummaryEvent(BaseEvent):
    """Emitted when all document grading is complete."""

    type: Literal["document_grading_summary"] = "document_grading_summary"
    node: Literal["document_grader"] = "document_grader"
    grade: str  # "pass" or "fail"
    relevant_count: int
    total_count: int
    average_score: float
    reasoning: str


# ============================================================================
# QUERY TRANSFORMATION EVENTS
# ============================================================================


class QueryTransformationEvent(BaseEvent):
    """Emitted when query is transformed for retry."""

    type: Literal["query_transformation"] = "query_transformation"
    node: Literal["query_transformer"] = "query_transformer"
    original_query: str
    transformed_query: str
    iteration: int
    max_iterations: int
    reasons: List[str]  # Why documents failed


# ============================================================================
# LLM RESPONSE EVENTS
# ============================================================================


class LLMReasoningStartEvent(BaseEvent):
    """Emitted when LLM starts generating reasoning."""

    type: Literal["llm_reasoning_start"] = "llm_reasoning_start"
    node: Literal["agent"] = "agent"


class LLMReasoningChunkEvent(BaseEvent):
    """Emitted for each chunk of LLM reasoning (streamed)."""

    type: Literal["llm_reasoning_chunk"] = "llm_reasoning_chunk"
    node: Literal["agent"] = "agent"
    content: str
    is_complete: bool = False


class LLMResponseStartEvent(BaseEvent):
    """Emitted when LLM starts generating response."""

    type: Literal["llm_response_start"] = "llm_response_start"
    node: Literal["agent"] = "agent"


class LLMResponseChunkEvent(BaseEvent):
    """Emitted for each chunk of LLM response (streamed)."""

    type: Literal["llm_response_chunk"] = "llm_response_chunk"
    node: Literal["agent"] = "agent"
    content: str
    is_complete: bool = False


class ToolCallEvent(BaseEvent):
    """Emitted when agent decides to call a tool."""

    type: Literal["tool_call"] = "tool_call"
    node: Literal["agent"] = "agent"
    tool_name: str
    tool_args: Dict[str, Any]


# ============================================================================
# RESPONSE GRADING EVENTS
# ============================================================================


class ResponseGradingEvent(BaseEvent):
    """Emitted when response quality is evaluated."""

    type: Literal["response_grading"] = "response_grading"
    node: Literal["response_grader"] = "response_grader"
    grade: str  # "pass" or "fail"
    score: float  # 0.0-1.0
    reasoning: str
    retry_count: int
    max_retries: int


# ============================================================================
# RESPONSE IMPROVEMENT EVENTS
# ============================================================================


class ResponseImprovementEvent(BaseEvent):
    """Emitted when response improvement is triggered."""

    type: Literal["response_improvement"] = "response_improvement"
    node: Literal["response_improver"] = "response_improver"
    feedback: str
    retry_count: int


# ============================================================================
# COMPLETION EVENTS
# ============================================================================


class AgentCompleteEvent(BaseEvent):
    """Emitted when agent execution completes successfully."""

    type: Literal["agent_complete"] = "agent_complete"
    thread_id: str
    total_duration_ms: float
    final_response: str
    iterations: int = 0  # Number of retrieval iterations
    response_retries: int = 0  # Number of response retries
    documents_used: int = 0
    title: Optional[str] = None  # Generated conversation title


class AgentErrorEvent(BaseEvent):
    """Emitted when agent execution fails."""

    type: Literal["agent_error"] = "agent_error"
    error: str
    node: Optional[str] = None
    recoverable: bool = False


# ============================================================================
# TOKEN BUDGET EVENTS
# ============================================================================


class TokenBudgetEvent(BaseEvent):
    """Emitted when token usage is tracked against budget."""

    type: Literal["token_budget"] = "token_budget"
    total_tokens_used: int
    token_budget: int
    budget_exceeded: bool
    warning_threshold_hit: bool


# ============================================================================
# CACHE HIT EVENTS
# ============================================================================


class CacheHitEvent(BaseEvent):
    """Emitted when a cache hit occurs."""

    type: Literal["cache_hit"] = "cache_hit"
    node: Literal["query_evaluator"] = "query_evaluator"
    query: str
    cached_result: Dict[str, Any]  # lambda_mult + query_analysis


# ============================================================================
# CONFIDENCE SCORE EVENTS
# ============================================================================


class ConfidenceScoreEvent(BaseEvent):
    """Emitted when confidence scoring is performed."""

    type: Literal["confidence_score"] = "confidence_score"
    node: str  # response_grader, document_grader, etc.
    score: float  # 0.0-1.0
    confidence: float  # 0.0-1.0
    early_stop_triggered: bool


# ============================================================================
# METRICS EVENT
# ============================================================================


class MetricsEvent(BaseEvent):
    """Emitted with timing and performance metrics."""

    type: Literal["metrics"] = "metrics"
    query_evaluation_ms: Optional[float] = None
    retrieval_ms: Optional[float] = None
    reranking_ms: Optional[float] = None
    document_grading_ms: Optional[float] = None
    llm_generation_ms: Optional[float] = None
    response_grading_ms: Optional[float] = None
    total_ms: float


# ============================================================================
# UNION TYPE FOR ALL EVENTS
# ============================================================================

AgentEvent = (
    ConnectionEstablished
    | ConnectionError
    | NodeStartEvent
    | NodeEndEvent
    | QueryEvaluationEvent
    | HybridSearchStartEvent
    | HybridSearchResultEvent
    | RerankerStartEvent
    | RerankerResultEvent
    | DocumentGradingStartEvent
    | DocumentGradeEvent
    | DocumentGradingSummaryEvent
    | QueryTransformationEvent
    | LLMReasoningStartEvent
    | LLMReasoningChunkEvent
    | LLMResponseStartEvent
    | LLMResponseChunkEvent
    | ToolCallEvent
    | ResponseGradingEvent
    | ResponseImprovementEvent
    | AgentCompleteEvent
    | AgentErrorEvent
    | TokenBudgetEvent
    | CacheHitEvent
    | ConfidenceScoreEvent
    | MetricsEvent
)

/**
 * TypeScript types for WebSocket events from the agent API.
 * These mirror the Pydantic models in api/schemas/events.py
 */

// Base event interface
export interface BaseEvent {
  type: string
  timestamp: string
  node?: string
}

// Connection events
export interface ConnectionEstablished extends BaseEvent {
  type: 'connection_established'
  thread_id: string
  existing_messages: number
}

export interface ConnectionError extends BaseEvent {
  type: 'connection_error'
  error: string
}

// Node lifecycle events
export interface NodeStartEvent extends BaseEvent {
  type: 'node_start'
  node: string
  input_summary?: string
}

export interface NodeEndEvent extends BaseEvent {
  type: 'node_end'
  node: string
  duration_ms: number
  output_summary?: string
}

// Query evaluator events
export interface QueryEvaluationEvent extends BaseEvent {
  type: 'query_evaluation'
  node: 'query_evaluator'
  query: string
  lambda_mult: number
  query_analysis: string
  search_strategy: 'lexical-heavy' | 'balanced' | 'semantic-heavy'
}

// Hybrid search events
export interface SearchCandidate {
  source: string
  snippet: string
  full_content?: string
  vector_score?: number
  text_score?: number
  rrf_score?: number
}

export interface HybridSearchStartEvent extends BaseEvent {
  type: 'hybrid_search_start'
  node: 'tools'
  query: string
  lambda_mult: number
  fetch_k: number
}

export interface HybridSearchResultEvent extends BaseEvent {
  type: 'hybrid_search_result'
  node: 'tools'
  candidate_count: number
  candidates: SearchCandidate[]
}

// Reranker events
export interface RerankerStartEvent extends BaseEvent {
  type: 'reranker_start'
  node: 'tools'
  model: string
  candidate_count: number
}

export interface RerankedDocument {
  source: string
  score: number
  rank: number
  original_rank: number
  snippet: string
  rank_change: number
  // Optional component scores (may be included from hybrid search)
  vector_score?: number
  text_score?: number
  rrf_score?: number
  page_content?: string
}

export interface RerankerResultEvent extends BaseEvent {
  type: 'reranker_result'
  node: 'tools'
  results: RerankedDocument[]
  reranking_changed_order: boolean
}

// Document grading events
export interface DocumentGradingStartEvent extends BaseEvent {
  type: 'document_grading_start'
  node: 'document_grader'
  document_count: number
}

export interface DocumentGradeEvent extends BaseEvent {
  type: 'document_grade'
  node: 'document_grader'
  source: string
  relevant: boolean
  score: number
  reasoning: string
}

export interface DocumentGradingSummaryEvent extends BaseEvent {
  type: 'document_grading_summary'
  node: 'document_grader'
  grade: 'pass' | 'fail'
  relevant_count: number
  total_count: number
  average_score: number
  reasoning: string
}

// Query transformation events
export interface QueryTransformationEvent extends BaseEvent {
  type: 'query_transformation'
  node: 'query_transformer'
  original_query: string
  transformed_query: string
  iteration: number
  max_iterations: number
  reasons: string[]
}

// LLM response events
export interface LLMReasoningStartEvent extends BaseEvent {
  type: 'llm_reasoning_start'
  node: 'agent'
}

export interface LLMReasoningChunkEvent extends BaseEvent {
  type: 'llm_reasoning_chunk'
  node: 'agent'
  content: string
  is_complete: boolean
}

export interface LLMResponseStartEvent extends BaseEvent {
  type: 'llm_response_start'
  node: 'agent'
}

export interface LLMResponseChunkEvent extends BaseEvent {
  type: 'llm_response_chunk'
  node: 'agent'
  content: string
  is_complete: boolean
}

export interface ToolCallEvent extends BaseEvent {
  type: 'tool_call'
  node: 'agent'
  tool_name: string
  tool_args: Record<string, unknown>
}

// Response grading events
export interface ResponseGradingEvent extends BaseEvent {
  type: 'response_grading'
  node: 'response_grader'
  grade: 'pass' | 'fail'
  score: number
  reasoning: string
  retry_count: number
  max_retries: number
}

// Response improvement events
export interface ResponseImprovementEvent extends BaseEvent {
  type: 'response_improvement'
  node: 'response_improver'
  feedback: string
  retry_count: number
}

// Completion events
export interface AgentCompleteEvent extends BaseEvent {
  type: 'agent_complete'
  thread_id: string
  total_duration_ms: number
  final_response: string
  iterations: number
  response_retries: number
  documents_used: number
  title?: string
}

export interface AgentErrorEvent extends BaseEvent {
  type: 'agent_error'
  error: string
  node?: string
  recoverable: boolean
}

// Token budget events
export interface TokenBudgetEvent extends BaseEvent {
  type: 'token_budget'
  total_tokens_used: number
  token_budget: number
  budget_exceeded: boolean
  warning_threshold_hit: boolean
}

// Cache hit events
export interface CacheHitEvent extends BaseEvent {
  type: 'cache_hit'
  node: 'query_evaluator'
  query: string
  cached_result: Record<string, unknown>
}

// Confidence score events
export interface ConfidenceScoreEvent extends BaseEvent {
  type: 'confidence_score'
  node: string
  score: number
  confidence: number
  early_stop_triggered: boolean
}

// Metrics event
export interface MetricsEvent extends BaseEvent {
  type: 'metrics'
  query_evaluation_ms?: number
  retrieval_ms?: number
  reranking_ms?: number
  document_grading_ms?: number
  llm_generation_ms?: number
  response_grading_ms?: number
  total_ms: number
}

// Union type of all events
export type AgentEvent =
  | ConnectionEstablished
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

// Helper type guards
export function isQueryEvaluation(event: AgentEvent): event is QueryEvaluationEvent {
  return event.type === 'query_evaluation'
}

export function isDocumentGradingSummary(event: AgentEvent): event is DocumentGradingSummaryEvent {
  return event.type === 'document_grading_summary'
}

export function isResponseGrading(event: AgentEvent): event is ResponseGradingEvent {
  return event.type === 'response_grading'
}

export function isAgentComplete(event: AgentEvent): event is AgentCompleteEvent {
  return event.type === 'agent_complete'
}

export function isAgentError(event: AgentEvent): event is AgentErrorEvent {
  return event.type === 'agent_error'
}

// Node names for routing
export type NodeName =
  | 'query_evaluator'
  | 'agent'
  | 'tools'
  | 'document_grader'
  | 'query_transformer'
  | 'response_grader'
  | 'response_improver'

// Node status for UI
export type NodeStatus = 'idle' | 'running' | 'complete' | 'error'

// Step representation for the observability panel
export interface ObservabilityStep {
  id: string
  node: NodeName
  status: NodeStatus
  startTime: Date
  endTime?: Date
  durationMs?: number
  events: AgentEvent[]
  summary?: string
}

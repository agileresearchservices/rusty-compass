"""
Observable Agent Service - Wrapper for LangChainAgent with event emission.

This service wraps the existing LangChainAgent and emits WebSocket events
during execution, providing full observability into the agent's workflow.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Any, Callable, Coroutine, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage

import sys
sys.path.insert(0, '/Users/kevin/github/personal/rusty-compass/langchain_agent')

from main import LangChainAgent
from config import (
    ENABLE_REFLECTION,
    ENABLE_DOCUMENT_GRADING,
    ENABLE_RESPONSE_GRADING,
    ENABLE_QUERY_TRANSFORMATION,
    ENABLE_RERANKING,
    RERANKER_MODEL,
    REFLECTION_MAX_ITERATIONS,
)

from api.schemas.events import (
    BaseEvent,
    NodeStartEvent,
    NodeEndEvent,
    QueryEvaluationEvent,
    HybridSearchStartEvent,
    HybridSearchResultEvent,
    SearchCandidate,
    RerankerStartEvent,
    RerankerResultEvent,
    RerankedDocument,
    DocumentGradingStartEvent,
    DocumentGradeEvent,
    DocumentGradingSummaryEvent,
    QueryTransformationEvent,
    LLMReasoningStartEvent,
    LLMResponseStartEvent,
    LLMResponseChunkEvent,
    ToolCallEvent,
    ResponseGradingEvent,
    ResponseImprovementEvent,
    AgentCompleteEvent,
    AgentErrorEvent,
    MetricsEvent,
)


# Type alias for emit callback
EmitCallback = Callable[[BaseEvent], Coroutine[Any, Any, None]]


class ObservableAgentService:
    """
    Observable wrapper for LangChainAgent that emits events during execution.

    This service provides the same functionality as LangChainAgent but emits
    structured events at each step, enabling real-time observability in the UI.
    """

    def __init__(self):
        """Initialize the observable agent service (lazy loading)."""
        self._agent: Optional[LangChainAgent] = None
        self._initialized = False
        self._lock = asyncio.Lock()

    async def ensure_initialized(self):
        """Initialize the agent if not already done."""
        async with self._lock:
            if not self._initialized:
                await self._initialize_agent()
                self._initialized = True

    async def _initialize_agent(self):
        """Initialize the underlying LangChainAgent."""
        # Run synchronous initialization in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._sync_init_agent)

    def _sync_init_agent(self):
        """Synchronous agent initialization."""
        self._agent = LangChainAgent()
        # Skip prerequisite verification in API mode
        # (health endpoint handles this)
        self._agent.initialize_components()
        self._agent.create_agent_graph()

    async def process_message(
        self,
        message: str,
        thread_id: str,
        emit: EmitCallback,
    ) -> Optional[str]:
        """
        Process a user message through the agent with observability.

        Args:
            message: The user's message
            thread_id: Conversation thread ID for persistence
            emit: Callback to emit events to the WebSocket

        Returns:
            The agent's final response text, or None if failed.
        """
        start_time = time.time()
        metrics: Dict[str, float] = {}

        try:
            # Set thread for conversation persistence
            self._agent.set_thread_id(thread_id)

            # Build initial state
            initial_state = {
                "messages": [HumanMessage(content=message)],
                "lambda_mult": 0.25,
                "query_analysis": "",
                "iteration_count": 0,
                "response_retry_count": 0,
                "retrieved_documents": [],
                "document_grades": [],
                "document_grade_summary": {},
                "response_grade": {},
                "original_query": message,
                "transformed_query": None,
            }

            config = {"configurable": {"thread_id": thread_id}}

            # Track metrics timing
            node_start_times: Dict[str, float] = {}
            final_response: Optional[str] = None
            iterations = 0
            response_retries = 0
            documents_used = 0

            # Stream through the graph
            async for event in self._astream_graph(initial_state, config, emit, node_start_times, metrics):
                # Extract final response from agent completions
                if isinstance(event, dict):
                    if "messages" in event:
                        for msg in event.get("messages", []):
                            if isinstance(msg, AIMessage) and msg.content:
                                if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                                    final_response = msg.content

                    # Track iterations and retries
                    if "iteration_count" in event:
                        iterations = event["iteration_count"]
                    if "response_retry_count" in event:
                        response_retries = event["response_retry_count"]
                    if "retrieved_documents" in event:
                        documents_used = len(event["retrieved_documents"])

            # Calculate total duration
            total_duration_ms = (time.time() - start_time) * 1000

            # Generate conversation title
            title = await self._generate_title(thread_id, message, final_response)

            # Emit completion event
            await emit(AgentCompleteEvent(
                thread_id=thread_id,
                total_duration_ms=total_duration_ms,
                final_response=final_response or "No response generated",
                iterations=iterations,
                response_retries=response_retries,
                documents_used=documents_used,
                title=title,
            ))

            # Emit metrics
            await emit(MetricsEvent(
                query_evaluation_ms=metrics.get("query_evaluator"),
                retrieval_ms=metrics.get("tools"),
                document_grading_ms=metrics.get("document_grader"),
                llm_generation_ms=metrics.get("agent"),
                response_grading_ms=metrics.get("response_grader"),
                total_ms=total_duration_ms,
            ))

            return final_response

        except Exception as e:
            await emit(AgentErrorEvent(
                error=str(e),
                recoverable=False,
            ))
            return None

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
        """
        loop = asyncio.get_event_loop()

        # Use the compiled app's stream method
        # We need to run this synchronously and emit events
        def run_sync():
            results = []
            for event in self._agent.app.stream(initial_state, config):
                results.append(event)
            return results

        # Run the graph
        events = await loop.run_in_executor(None, run_sync)

        # Process events and emit observability data
        for event in events:
            # Each event is a dict with node_name: output_dict
            for node_name, output in event.items():
                # Emit node start
                node_start = time.time()
                node_start_times[node_name] = node_start

                await emit(NodeStartEvent(
                    node=node_name,
                    input_summary=self._summarize_input(node_name, output),
                ))

                # Emit node-specific events
                await self._emit_node_events(node_name, output, emit)

                # Calculate duration
                duration_ms = (time.time() - node_start) * 1000
                metrics[node_name] = duration_ms

                # Emit node end
                await emit(NodeEndEvent(
                    node=node_name,
                    duration_ms=duration_ms,
                    output_summary=self._summarize_output(node_name, output),
                ))

                yield output

    async def _emit_node_events(
        self,
        node_name: str,
        output: Dict[str, Any],
        emit: EmitCallback,
    ):
        """Emit detailed events for specific nodes."""

        if node_name == "query_evaluator":
            await emit(QueryEvaluationEvent(
                query=output.get("original_query", ""),
                lambda_mult=output.get("lambda_mult", 0.25),
                query_analysis=output.get("query_analysis", ""),
                search_strategy=self._get_search_strategy(output.get("lambda_mult", 0.25)),
            ))

        elif node_name == "tools":
            # Emit search events
            documents = output.get("retrieved_documents", [])
            if documents:
                await emit(HybridSearchResultEvent(
                    candidate_count=len(documents),
                    candidates=[
                        SearchCandidate(
                            source=doc.metadata.get("source", "unknown"),
                            snippet=doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        )
                        for doc in documents[:10]  # Limit to 10 for UI
                    ],
                ))

                # Emit reranker events if enabled
                if ENABLE_RERANKING:
                    await emit(RerankerStartEvent(
                        model=RERANKER_MODEL,
                        candidate_count=len(documents),
                    ))

        elif node_name == "document_grader":
            grades = output.get("document_grades", [])
            summary = output.get("document_grade_summary", {})

            await emit(DocumentGradingStartEvent(
                document_count=len(grades),
            ))

            # Emit individual grades
            for grade in grades:
                await emit(DocumentGradeEvent(
                    source=grade.get("source", "unknown"),
                    relevant=grade.get("relevant", False),
                    score=grade.get("score", 0.0),
                    reasoning=grade.get("reasoning", ""),
                ))

            # Emit summary
            await emit(DocumentGradingSummaryEvent(
                grade=summary.get("grade", "unknown"),
                relevant_count=sum(1 for g in grades if g.get("relevant")),
                total_count=len(grades),
                average_score=summary.get("score", 0.0),
                reasoning=summary.get("reasoning", ""),
            ))

        elif node_name == "query_transformer":
            await emit(QueryTransformationEvent(
                original_query=output.get("original_query", ""),
                transformed_query=output.get("transformed_query", ""),
                iteration=output.get("iteration_count", 0),
                max_iterations=REFLECTION_MAX_ITERATIONS,
                reasons=[],  # Could extract from document grades
            ))

        elif node_name == "agent":
            # Emit LLM events
            messages = output.get("messages", [])
            for msg in messages:
                if isinstance(msg, AIMessage):
                    # Check for tool calls
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            await emit(ToolCallEvent(
                                tool_name=tool_call["name"],
                                tool_args=tool_call["args"],
                            ))
                    elif msg.content:
                        # Emit response (could be chunked for streaming)
                        await emit(LLMResponseStartEvent())
                        await emit(LLMResponseChunkEvent(
                            content=msg.content,
                            is_complete=True,
                        ))

        elif node_name == "response_grader":
            grade = output.get("response_grade", {})
            await emit(ResponseGradingEvent(
                grade=grade.get("grade", "unknown"),
                score=grade.get("score", 0.0),
                reasoning=grade.get("reasoning", ""),
                retry_count=output.get("response_retry_count", 0),
                max_retries=REFLECTION_MAX_ITERATIONS,
            ))

        elif node_name == "response_improver":
            await emit(ResponseImprovementEvent(
                feedback=output.get("feedback", ""),
                retry_count=output.get("response_retry_count", 0),
            ))

    def _get_search_strategy(self, lambda_mult: float) -> str:
        """Convert lambda_mult to human-readable search strategy."""
        if lambda_mult < 0.3:
            return "lexical-heavy"
        elif lambda_mult < 0.7:
            return "balanced"
        else:
            return "semantic-heavy"

    def _summarize_input(self, node_name: str, output: Dict[str, Any]) -> str:
        """Generate a brief summary of node input."""
        if node_name == "query_evaluator":
            return "Evaluating query type for optimal search strategy"
        elif node_name == "agent":
            return "Processing with LLM"
        elif node_name == "tools":
            return "Executing knowledge base search"
        elif node_name == "document_grader":
            return f"Grading {len(output.get('retrieved_documents', []))} documents"
        elif node_name == "query_transformer":
            return "Transforming query for retry"
        elif node_name == "response_grader":
            return "Evaluating response quality"
        elif node_name == "response_improver":
            return "Adding feedback for response improvement"
        return ""

    def _summarize_output(self, node_name: str, output: Dict[str, Any]) -> str:
        """Generate a brief summary of node output."""
        if node_name == "query_evaluator":
            return f"lambda={output.get('lambda_mult', 0.25):.2f}"
        elif node_name == "agent":
            messages = output.get("messages", [])
            if messages:
                msg = messages[-1]
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    return f"Tool calls: {len(msg.tool_calls)}"
                return "Response generated"
            return ""
        elif node_name == "tools":
            docs = output.get("retrieved_documents", [])
            return f"{len(docs)} documents retrieved"
        elif node_name == "document_grader":
            summary = output.get("document_grade_summary", {})
            return f"{summary.get('grade', 'unknown').upper()}"
        elif node_name == "query_transformer":
            return f"Query transformed (iteration {output.get('iteration_count', 0)})"
        elif node_name == "response_grader":
            grade = output.get("response_grade", {})
            return f"{grade.get('grade', 'unknown').upper()} ({grade.get('score', 0):.2f})"
        elif node_name == "response_improver":
            return f"Retry #{output.get('response_retry_count', 0)}"
        return ""

    async def _generate_title(
        self,
        thread_id: str,
        user_message: str,
        response: Optional[str],
    ) -> Optional[str]:
        """Generate and save a conversation title."""
        try:
            loop = asyncio.get_event_loop()
            title = await loop.run_in_executor(
                None,
                lambda: self._agent.generate_conversation_title(thread_id)
            )
            if title:
                await loop.run_in_executor(
                    None,
                    lambda: self._agent.update_conversation_title(thread_id, title)
                )
            return title
        except Exception as e:
            print(f"Error generating title: {e}")
            return None

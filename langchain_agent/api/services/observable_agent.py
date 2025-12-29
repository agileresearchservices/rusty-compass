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
    ENABLE_ASYNC_STREAMING,
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
    LLMReasoningChunkEvent,
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
        self._use_async_streaming = ENABLE_ASYNC_STREAMING

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

        This method supports two modes controlled by ENABLE_ASYNC_STREAMING:

        - When False (default, backward compatible):
          Runs entire graph in executor, collects all timing info after completion.
          More blocking but stable timing measurements.

        - When True (experimental, improved streaming):
          Streams events incrementally as they complete, emitting NodeStartEvent
          immediately and NodeEndEvent with accurate timing. This prevents blocking
          the async event loop and improves UI responsiveness.
          TRADEOFF: Timing may be slightly less accurate but better reactivity.
        """
        if self._use_async_streaming:
            # IMPROVED STREAMING MODE: Emit events incrementally as they occur
            async for output in self._astream_graph_improved(
                initial_state, config, emit, node_start_times, metrics
            ):
                yield output
        else:
            # BACKWARD COMPATIBLE MODE: Collect all results, then emit
            async for output in self._astream_graph_legacy(
                initial_state, config, emit, node_start_times, metrics
            ):
                yield output

    async def _astream_graph_legacy(
        self,
        initial_state: Dict[str, Any],
        config: Dict[str, Any],
        emit: EmitCallback,
        node_start_times: Dict[str, float],
        metrics: Dict[str, float],
    ):
        """
        Legacy streaming mode (ENABLE_ASYNC_STREAMING = False).

        Runs entire agent graph in thread executor, collecting all node
        executions and their timing before emitting events. This provides
        stable, accurate timing but blocks the async event loop during
        the entire graph execution.
        """
        loop = asyncio.get_event_loop()

        # Use the compiled app's stream method
        # Track timestamps for each node execution during synchronous execution
        def run_sync_with_timing():
            results = []
            # List of (node_name, start_time, end_time, output) for each execution
            node_executions: List[Dict[str, Any]] = []
            prev_time = time.time()

            for event in self._agent.app.stream(initial_state, config):
                current_time = time.time()

                # Each event is a dict with node_name: output_dict
                for node_name, output in event.items():
                    # Record this execution with timing
                    node_executions.append({
                        "node": node_name,
                        "start": prev_time,
                        "end": current_time,
                        "output": output,
                    })
                    prev_time = current_time

            return node_executions

        # Run the graph
        node_executions = await loop.run_in_executor(None, run_sync_with_timing)

        # Process events and emit observability data
        for execution in node_executions:
            node_name = execution["node"]
            output = execution["output"]
            start_time = execution["start"]
            end_time = execution["end"]

            # Calculate duration
            duration_ms = max((end_time - start_time) * 1000, 1.0)  # At least 1ms

            # Store for metrics (accumulate for repeated nodes)
            if node_name not in node_start_times:
                node_start_times[node_name] = start_time
            if node_name in metrics:
                metrics[node_name] += duration_ms
            else:
                metrics[node_name] = duration_ms

            # Emit node start
            await emit(NodeStartEvent(
                node=node_name,
                input_summary=self._summarize_input(node_name, output),
            ))

            # Emit node-specific events
            await self._emit_node_events(node_name, output, emit)

            # Emit node end with actual duration
            await emit(NodeEndEvent(
                node=node_name,
                duration_ms=duration_ms,
                output_summary=self._summarize_output(node_name, output),
            ))

            yield output

    async def _astream_graph_improved(
        self,
        initial_state: Dict[str, Any],
        config: Dict[str, Any],
        emit: EmitCallback,
        node_start_times: Dict[str, float],
        metrics: Dict[str, float],
    ):
        """
        Improved streaming mode (ENABLE_ASYNC_STREAMING = True).

        Streams events incrementally from the graph as they complete, emitting
        NodeStartEvent immediately and processing each event before emitting
        NodeEndEvent. This prevents blocking the async event loop and provides
        better UI responsiveness.

        TRADEOFF: Timing between events may be slightly less precise since we
        measure wall-clock time between async event emissions rather than
        collecting execution data. However, actual node execution time is
        captured accurately via context manager or event metadata.
        """
        loop = asyncio.get_event_loop()

        def run_sync_stream():
            """Run the agent graph and yield events as they occur."""
            for event in self._agent.app.stream(initial_state, config):
                # Each event is a dict with node_name: output_dict
                for node_name, output in event.items():
                    yield {
                        "node": node_name,
                        "output": output,
                    }

        # Stream events from the graph executor
        # We use a separate thread for the generator to avoid blocking async
        async def consume_stream():
            """Consume the sync stream in a separate executor."""
            gen = run_sync_stream()
            while True:
                try:
                    # Get next event in executor
                    event = await loop.run_in_executor(
                        None,
                        next,
                        gen,
                    )
                    yield event
                except StopIteration:
                    break

        # Process events incrementally
        async for event in consume_stream():
            node_name = event["node"]
            output = event["output"]

            # Record node start time (when we receive the event)
            node_start_time = time.time()

            # Emit node start event immediately (before processing)
            await emit(NodeStartEvent(
                node=node_name,
                input_summary=self._summarize_input(node_name, output),
            ))

            # Emit node-specific events
            await self._emit_node_events(node_name, output, emit)

            # Record node end time (after processing)
            node_end_time = time.time()
            duration_ms = max((node_end_time - node_start_time) * 1000, 1.0)

            # Store for metrics (accumulate for repeated nodes)
            if node_name not in node_start_times:
                node_start_times[node_name] = node_start_time
            if node_name in metrics:
                metrics[node_name] += duration_ms
            else:
                metrics[node_name] = duration_ms

            # Emit node end with processing duration
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

                # Emit reranking events if enabled
                if ENABLE_RERANKING:
                    # Emit reranker start event with candidate count
                    await emit(RerankerStartEvent(
                        model=RERANKER_MODEL,
                        candidate_count=len(documents),
                    ))

                    # Emit reranker result event with detailed document information
                    reranked_docs = self._compute_reranked_documents(documents)
                    reranking_changed_order = self._check_if_order_changed(documents, reranked_docs)

                    await emit(RerankerResultEvent(
                        results=reranked_docs,
                        reranking_changed_order=reranking_changed_order,
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
                    # Check for reasoning in additional_kwargs
                    reasoning = None
                    if hasattr(msg, "additional_kwargs") and msg.additional_kwargs:
                        reasoning = msg.additional_kwargs.get("reasoning")

                    # Emit reasoning if present
                    if reasoning:
                        await emit(LLMReasoningStartEvent())
                        await emit(LLMReasoningChunkEvent(
                            content=reasoning,
                            is_complete=True,
                        ))

                    # Check for tool calls
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            await emit(ToolCallEvent(
                                tool_name=tool_call["name"],
                                tool_args=tool_call["args"],
                            ))
                    elif msg.content:
                        # For responses without tool calls, check if content needs parsing
                        # (handles Ollama models that format reasoning as "Reasoning: ... \nAnswer: ...")
                        reasoning_extracted, response_content = self._parse_ollama_response(msg.content)

                        # Emit reasoning if extracted from content
                        if reasoning_extracted and not reasoning:
                            await emit(LLMReasoningStartEvent())
                            await emit(LLMReasoningChunkEvent(
                                content=reasoning_extracted,
                                is_complete=True,
                            ))

                        # Emit response (could be chunked for streaming)
                        await emit(LLMResponseStartEvent())
                        await emit(LLMResponseChunkEvent(
                            content=response_content,
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

    def _parse_ollama_response(self, content: str) -> tuple[Optional[str], str]:
        """
        Parse Ollama-formatted responses that may contain reasoning.

        Ollama models sometimes format responses as:
        "Reasoning: <reasoning text>
         Answer: <answer text>"

        Returns:
            A tuple of (reasoning_text, response_text)
            If no structured format is found, returns (None, content)
        """
        if not content:
            return None, content

        # Look for the pattern "Reasoning:" and "Answer:"
        reasoning_pattern = "Reasoning:"
        answer_pattern = "Answer:"

        if reasoning_pattern in content and answer_pattern in content:
            try:
                reasoning_start = content.find(reasoning_pattern) + len(reasoning_pattern)
                reasoning_end = content.find(answer_pattern)

                if reasoning_end > reasoning_start:
                    reasoning_text = content[reasoning_start:reasoning_end].strip()
                    answer_start = reasoning_end + len(answer_pattern)
                    answer_text = content[answer_start:].strip()

                    return reasoning_text, answer_text
            except Exception:
                # If parsing fails, return the original content
                pass

        return None, content

    def _compute_reranked_documents(self, documents: List) -> List:
        """
        Compute RerankedDocument objects from retrieved documents.

        Since the documents in output["retrieved_documents"] are already reranked
        by the tools_node before reaching this method, we construct RerankedDocument
        objects using their current positions and extract scores from metadata.

        Args:
            documents: List of LangChain Document objects (already reranked)

        Returns:
            List of RerankedDocument objects with ranking information
        """
        reranked_docs = []

        for rank, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "unknown")
            # Extract reranker score if available, otherwise use 0.0
            score = doc.metadata.get("reranker_score", 0.0)
            # Extract original rank if available, otherwise estimate based on position
            original_rank = doc.metadata.get("original_rank", rank)
            # Calculate rank change (positive = moved up)
            rank_change = original_rank - rank

            snippet = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content

            reranked_docs.append(RerankedDocument(
                source=source,
                score=score,
                rank=rank,
                original_rank=original_rank,
                snippet=snippet,
                rank_change=rank_change,
            ))

        return reranked_docs

    def _check_if_order_changed(self, documents: List, reranked_docs: List) -> bool:
        """
        Determine if reranking changed the document order.

        This is a heuristic check based on whether any document moved from its
        original position. Since we don't have the pre-reranking order directly,
        we check if any document has a non-zero rank_change value.

        Args:
            documents: List of LangChain Document objects (reranked)
            reranked_docs: List of RerankedDocument objects with rank info

        Returns:
            Boolean indicating if any document's rank changed
        """
        return any(doc.rank_change != 0 for doc in reranked_docs)

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
            # update_conversation_title() uses the internally set thread_id
            # and handles both generation and database update
            await loop.run_in_executor(
                None,
                self._agent.update_conversation_title
            )
            # Return a generated title from the user message for the WebSocket event
            # (the actual title is saved to DB by update_conversation_title)
            return user_message[:50].strip() if user_message else None
        except Exception as e:
            print(f"Error generating title: {e}")
            return None

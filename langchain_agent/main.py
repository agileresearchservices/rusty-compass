#!/usr/bin/env python3
"""
LangChain Agent with Real-Time Streaming, Local Knowledge Base, and Persistent Memory

A production-grade ReAct agent with the following features:
- Real-time character-by-character streaming of agent thinking and final responses
- Local knowledge base using ChromaDB with semantic search
- Persistent conversation memory using PostgreSQL
- Intelligent tool usage for knowledge retrieval
- Multi-turn conversations with context preservation

Powered by:
- LLM: Ollama (gpt-oss:20b)
- Embeddings: Ollama (nomic-embed-text:latest)
- Vector Store: ChromaDB
- Memory: PostgreSQL with LangGraph checkpointer
- Framework: LangGraph with ReAct agent pattern
"""

import sys
import uuid
import warnings
import time
import psycopg
from pathlib import Path
from typing import Sequence, Tuple, List, Optional
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import create_react_agent
from psycopg_pool import ConnectionPool
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, BaseMessage, AIMessage, HumanMessage
import httpx

# Suppress the LangGraphDeprecatedSinceV10 warning about create_react_agent migration.
# The recommended replacement (langchain.agents.create_react_agent) doesn't exist yet in 1.2.0.
# This warning is from an incomplete migration path and will be resolved in a future update.
# TODO: Switch to langchain.agents.create_react_agent once the migration is complete.
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*create_react_agent.*")

from config import (
    LLM_MODEL,
    LLM_TEMPERATURE,
    EMBEDDINGS_MODEL,
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_NAME,
    DATABASE_URL,
    DB_CONNECTION_KWARGS,
    DB_POOL_MAX_SIZE,
    RETRIEVER_TOOL_NAME,
    RETRIEVER_TOOL_DESCRIPTION,
    OLLAMA_BASE_URL,
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
    POSTGRES_DB,
    ENABLE_COMPACTION,
    MAX_CONTEXT_TOKENS,
    COMPACTION_THRESHOLD_PCT,
    MESSAGES_TO_KEEP_FULL,
    MIN_MESSAGES_FOR_COMPACTION,
    TOKEN_CHAR_RATIO,
    REASONING_ENABLED,
    REASONING_EFFORT,
)


class LangChainAgent:
    """
    Main agent class that manages the LLM, tools, and conversation state.

    Handles:
    - Real-time streaming of agent thinking and responses
    - Integration with local knowledge base (ChromaDB)
    - Persistent conversation memory (PostgreSQL)
    - Interactive multi-turn conversations
    """

    def __init__(self):
        """Initialize the agent and all its components"""
        self.llm = None
        self.embeddings = None
        self.vector_store = None
        self.pool = None
        self.checkpointer = None
        self.app = None
        self.thread_id = None

    def verify_prerequisites(self):
        """Verify that all required services are running"""
        print("Verifying prerequisites...")
        print()

        # Check Postgres connection
        try:
            with psycopg.connect(DATABASE_URL) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    print("✓ Postgres is accessible")
        except Exception as e:
            print(f"✗ Cannot connect to Postgres: {e}")
            print(f"  Connection string: {DATABASE_URL}")
            sys.exit(1)

        # Check ChromaDB persistence
        chroma_path = Path(CHROMA_DB_PATH)
        if not chroma_path.exists():
            print(f"✗ ChromaDB not initialized: {CHROMA_DB_PATH}")
            print("  Run: python load_sample_data.py")
            sys.exit(1)
        print("✓ ChromaDB is initialized")

        # Check for sample documents (check for sqlite3 or parquet files)
        chroma_sqlite = chroma_path / "chroma.sqlite3"
        chroma_parquet = chroma_path / "chroma.parquet"
        if not chroma_sqlite.exists() and not chroma_parquet.exists():
            print("✗ No data in ChromaDB")
            print("  Run: python load_sample_data.py")
            sys.exit(1)

        # Check Ollama connection
        try:
            with httpx.Client() as client:
                response = client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
                if response.status_code == 200:
                    print("✓ Ollama is accessible")
                else:
                    print(f"✗ Ollama returned unexpected status: {response.status_code}")
                    print(f"  URL: {OLLAMA_BASE_URL}")
                    sys.exit(1)
        except httpx.ConnectError:
            print(f"✗ Cannot connect to Ollama at {OLLAMA_BASE_URL}")
            print("  Make sure Ollama is running: ollama serve")
            print("  Or check that the base URL is correct in config.py")
            sys.exit(1)
        except Exception as e:
            print(f"✗ Error checking Ollama: {e}")
            print(f"  URL: {OLLAMA_BASE_URL}")
            sys.exit(1)

        print()

    def initialize_components(self):
        """Initialize all LLM and storage components"""
        print("Initializing components...")
        print()

        # Initialize LLM with streaming enabled
        print(f"Loading LLM: {LLM_MODEL}")
        self.llm = ChatOllama(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            base_url=OLLAMA_BASE_URL,
            streaming=True,
            num_predict=1024,  # Allow longer thinking/reasoning
            reasoning=REASONING_ENABLED,
            reasoning_effort=REASONING_EFFORT if REASONING_ENABLED else None,
        )
        print("✓ LLM initialized")

        # Initialize Embeddings
        print(f"Loading embeddings: {EMBEDDINGS_MODEL}")
        self.embeddings = OllamaEmbeddings(
            model=EMBEDDINGS_MODEL,
            base_url=OLLAMA_BASE_URL,
        )
        print("✓ Embeddings initialized")

        # Initialize Vector Store
        print(f"Loading ChromaDB: {CHROMA_DB_PATH}")
        self.vector_store = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=CHROMA_DB_PATH,
        )
        print("✓ Vector store initialized")

        # Initialize Postgres connection pool
        print("Connecting to Postgres checkpoint store...")
        connection_kwargs = DB_CONNECTION_KWARGS.copy()
        self.pool = ConnectionPool(
            conninfo=DATABASE_URL,
            max_size=DB_POOL_MAX_SIZE,
            kwargs=connection_kwargs
        )
        self.checkpointer = PostgresSaver(self.pool)
        print("✓ Postgres checkpoint store initialized")

        print()

    def create_agent_graph(self):
        """Create the ReAct agent graph with tools and memory"""
        print("Creating agent graph...")

        # Create the retriever tool
        retriever = self.vector_store.as_retriever()

        @tool
        def knowledge_base(query: str) -> str:
            """Search for information in the local document knowledge base.

            Use this tool to find relevant information about Python programming,
            machine learning concepts, and web development topics stored in the
            local document index using semantic search.

            Args:
                query: The search query to find relevant documents.

            Returns:
                Relevant document content from the knowledge base, or a message
                if no relevant documents are found.
            """
            results = retriever.invoke(query)
            if results:
                return "\n\n".join([doc.page_content for doc in results])
            return "No relevant information found."

        # Create the agent graph
        self.app = create_react_agent(
            model=self.llm,
            tools=[knowledge_base],
            checkpointer=self.checkpointer,
        )

        print("✓ Agent graph created and compiled")
        print()

    def generate_thread_id(self):
        """Generate a unique thread ID for conversation persistence"""
        self.thread_id = f"conversation_{uuid.uuid4().hex[:8]}"

    def set_thread_id(self, thread_id: str):
        """Set a specific thread ID to resume a conversation"""
        self.thread_id = thread_id

    def list_conversations(self):
        """List available previous conversations from PostgreSQL with titles"""
        try:
            with psycopg.connect(DATABASE_URL) as conn:
                with conn.cursor() as cur:
                    # Query the metadata table for conversations with titles
                    cur.execute("""
                        SELECT thread_id, title, created_at
                        FROM conversation_metadata
                        ORDER BY created_at DESC
                        LIMIT 20
                    """)
                    conversations = cur.fetchall()
                    return conversations
        except Exception as e:
            print(f"Error listing conversations: {e}")
            return []

    def clear_all_conversations(self):
        """Clear all previous conversations from the database"""
        try:
            with psycopg.connect(DATABASE_URL) as conn:
                conn.autocommit = True
                with conn.cursor() as cur:
                    # Delete all conversation metadata
                    cur.execute("DELETE FROM conversation_metadata")
                    metadata_count = cur.rowcount

                    # Delete all checkpoints (conversation history)
                    cur.execute("DELETE FROM checkpoints")
                    checkpoint_count = cur.rowcount

                    # Delete checkpoint blobs if they exist
                    try:
                        cur.execute("DELETE FROM checkpoint_blobs")
                    except psycopg.Error:
                        pass  # Table may not exist, which is acceptable

                    return metadata_count, checkpoint_count
        except Exception as e:
            print(f"Error clearing conversations: {e}")
            return 0, 0

    def save_conversation_title(self, user_message: str):
        """Generate and save a title for the current conversation from the first user message"""
        try:
            # Create a short title from the first message (first 50 chars)
            title = user_message[:60].strip()
            if not title:
                title = "Untitled Conversation"

            with psycopg.connect(DATABASE_URL) as conn:
                conn.autocommit = True
                with conn.cursor() as cur:
                    # Insert or update conversation metadata
                    cur.execute("""
                        INSERT INTO conversation_metadata (thread_id, title)
                        VALUES (%s, %s)
                        ON CONFLICT (thread_id)
                        DO UPDATE SET updated_at = CURRENT_TIMESTAMP
                    """, (self.thread_id, title))
        except Exception as e:
            # Don't fail the conversation if metadata save fails
            pass

    def estimate_token_count(self, messages: Sequence[BaseMessage]) -> int:
        """
        Estimate token count for a list of messages.
        Uses 1 token ≈ 4 characters heuristic (conservative for English).

        Args:
            messages: Sequence of BaseMessage objects to estimate token count for.

        Returns:
            Estimated token count based on character length.
        """
        try:
            total_chars = 0
            for msg in messages:
                if hasattr(msg, "content") and msg.content:
                    total_chars += len(str(msg.content))
            return total_chars // TOKEN_CHAR_RATIO
        except Exception:
            return 0

    def summarize_messages(self, messages_to_summarize: Sequence[BaseMessage]) -> str:
        """
        Use LLM to create a concise summary of older messages.
        Preserves key facts and context while being brief.

        Args:
            messages_to_summarize: Sequence of messages to summarize.

        Returns:
            A concise summary of the message content.
        """
        try:
            if not messages_to_summarize:
                return "No earlier context"

            # Build context of messages to summarize
            context = ""
            for msg in messages_to_summarize:
                if hasattr(msg, "content") and msg.content:
                    # Determine role from message type
                    if hasattr(msg, "type"):
                        role = "User" if msg.type == "human" else "Assistant"
                    else:
                        role = "Assistant" if "assistant" in str(type(msg)).lower() else "User"
                    context += f"{role}: {msg.content}\n\n"

            if not context.strip():
                return "No earlier context"

            # Prompt LLM to summarize
            summary_prompt = f"""Summarize the following conversation concisely in 1-2 paragraphs, preserving key facts and context:

{context}

Summary:"""

            # Invoke LLM for summary (direct, not through agent)
            response = self.llm.invoke(summary_prompt)
            return response.content if hasattr(response, "content") else str(response)

        except Exception as e:
            return f"[Unable to summarize {len(messages_to_summarize)} messages]"

    def compact_conversation_if_needed(self, messages: Sequence[BaseMessage]) -> Tuple[Sequence[BaseMessage], bool, int]:
        """
        Check if conversation needs compaction and compact if necessary.

        Args:
            messages: Sequence of messages to check for compaction.

        Returns:
            Tuple of (compacted_messages, was_compacted, num_compacted) where:
            - compacted_messages: The potentially compacted message sequence
            - was_compacted: Boolean indicating if compaction occurred
            - num_compacted: Number of messages that were compacted
        """
        if not ENABLE_COMPACTION or not messages:
            return messages, False, 0

        if len(messages) < MIN_MESSAGES_FOR_COMPACTION:
            return messages, False, 0

        # Estimate token count
        token_count = self.estimate_token_count(messages)
        threshold = int(MAX_CONTEXT_TOKENS * COMPACTION_THRESHOLD_PCT)

        if token_count < threshold:
            return messages, False, 0  # No compaction needed

        # Perform compaction
        messages_to_keep = messages[-MESSAGES_TO_KEEP_FULL:]
        messages_to_compact = messages[:-MESSAGES_TO_KEEP_FULL]

        # Generate summary
        summary_text = self.summarize_messages(messages_to_compact)

        # Create summary message
        summary_msg = SystemMessage(
            content=f"[Earlier conversation summary]: {summary_text}"
        )

        # Return compacted messages
        compacted = [summary_msg] + messages_to_keep
        return compacted, True, len(messages_to_compact)

    def run_conversation(self):
        """Run the interactive conversation loop"""
        print("=" * 70)
        print("LangChain Agent - Local Knowledge Base & Memory")
        print("=" * 70)
        print()
        print("Agent is ready! You can ask questions about:")
        print("  - Python programming basics")
        print("  - Machine learning concepts")
        print("  - Web development")
        print()
        print("Commands:")
        print("  - Type your question and press Enter")
        print("  - Type 'new' to start a new conversation")
        print("  - Type 'list' to see previous conversations")
        print("  - Type 'load <id>' to resume a conversation")
        print("  - Type 'clear' to delete all conversations")
        print("  - Type 'quit' or 'exit' to stop")
        print()
        print("=" * 70)
        print()

        self.generate_thread_id()
        print(f"Conversation ID: {self.thread_id}")
        print("(Title will be generated from your first message)")
        print()

        first_message = True

        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                # Save conversation title from first message
                if first_message and not user_input.lower().startswith(("list", "load", "new", "quit", "exit")):
                    self.save_conversation_title(user_input)
                    first_message = False

                # Handle special commands
                if user_input.lower() == "quit" or user_input.lower() == "exit":
                    print("\nGoodbye!")
                    break

                if user_input.lower() == "new":
                    self.generate_thread_id()
                    print(f"\n✓ New conversation started")
                    print(f"Conversation ID: {self.thread_id}")
                    print()
                    continue

                if user_input.lower() == "list":
                    print("\n📋 Previous Conversations:")
                    conversations = self.list_conversations()
                    if conversations:
                        for i, (thread_id, title, created_at) in enumerate(conversations, 1):
                            # Format the date nicely
                            date_str = created_at.strftime("%Y-%m-%d %H:%M") if created_at else "Unknown"
                            print(f"  {i}. {title}")
                            print(f"     ID: {thread_id} | {date_str}")
                        print("\nUse 'load <id>' to resume a conversation")
                    else:
                        print("  No previous conversations found")
                    print()
                    continue

                if user_input.lower().startswith("load "):
                    thread_id = user_input[5:].strip()
                    if thread_id:
                        self.set_thread_id(thread_id)
                        print(f"\n✓ Loaded conversation: {thread_id}")
                        print()
                    else:
                        print("\n✗ Please provide a conversation ID: load <id>")
                        print()
                    continue

                if user_input.lower() == "clear":
                    # Confirm before clearing
                    confirm = input("\n⚠️  This will delete ALL conversations and history. Continue? (yes/no): ").strip().lower()
                    if confirm == "yes":
                        metadata_count, checkpoint_count = self.clear_all_conversations()
                        print(f"\n✓ Cleared {metadata_count} conversation(s) and {checkpoint_count} checkpoint record(s)")
                    else:
                        print("✗ Clear cancelled")
                    print()
                    continue

                # Process the input through the agent
                print()
                self._invoke_agent(user_input)
                print()

            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n✗ Error: {e}")
                print("Try again or type 'quit' to exit\n")

    def _invoke_agent(self, user_input: str):
        """
        Invoke the agent with user input and stream intermediate reasoning steps.

        This method uses modern LangGraph streaming to show:
        1. Agent reasoning and decision-making steps
        2. Tool calls to the knowledge base with intermediate results
        3. Final response streamed character-by-character for real-time feedback

        Args:
            user_input: The user's question or command
        """
        try:
            # Prepare input for the agent
            input_data = {"messages": [("user", user_input)]}

            # Try to apply compaction to conversation if needed
            try:
                checkpoint_state = self.checkpointer.get({"configurable": {"thread_id": self.thread_id}})
                if checkpoint_state and "messages" in checkpoint_state:
                    current_messages = checkpoint_state["messages"]
                    compacted_msgs, was_compacted, num_compacted = self.compact_conversation_if_needed(current_messages)
                    if was_compacted:
                        print(f"[🗜️  Compacted {num_compacted} older messages to maintain context]")
            except Exception:
                # If compaction fails, just continue without it
                pass

            final_response = ""
            reasoning_content = ""

            # Get the current message count before invoking
            try:
                checkpoint_before = self.checkpointer.get({"configurable": {"thread_id": self.thread_id}})
                messages_before_count = len(checkpoint_before.get("messages", [])) if checkpoint_before else 0
            except Exception:
                messages_before_count = 0

            # Invoke the agent to get the complete response
            result = self.app.invoke(
                input_data,
                config={"configurable": {"thread_id": self.thread_id}},
            )

            # Extract final response and reasoning from result
            if "messages" in result:
                messages = result["messages"]
                # Only look at messages added in this turn (after the user message)
                # We need to find the assistant message that came after the last user input
                new_messages = messages[messages_before_count:] if messages_before_count < len(messages) else []

                # Find the last assistant message in the new messages (final response)
                for msg in reversed(new_messages):
                    if hasattr(msg, "content") and msg.content:
                        content = str(msg.content)
                        # Skip messages that are tool calls
                        if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                            final_response = content

                            # Extract reasoning content if available
                            if hasattr(msg, "additional_kwargs") and msg.additional_kwargs:
                                reasoning_content = msg.additional_kwargs.get("reasoning_content", "")
                            break

            # Display reasoning if available
            if REASONING_ENABLED and reasoning_content:
                print("Agent (thinking):")
                self._stream_text(reasoning_content)
                print()

            # Display the final response with streaming
            if final_response:
                print("Agent (response):")
                self._stream_text(final_response)
            else:
                print("Agent: Processing complete")

        except httpx.ConnectError as e:
            print(f"✗ Cannot connect to Ollama at {OLLAMA_BASE_URL}")
            print(f"  Error: {e}")
            print(f"\n  To fix:")
            print(f"  1. Make sure Ollama is running")
            print(f"  2. Check that Ollama base URL is correct: {OLLAMA_BASE_URL}")
            print(f"  3. Verify the models exist: 'ollama list'")
            print(f"  4. Try restarting Ollama")
        except Exception as e:
            print(f"✗ Error invoking agent: {e}")
            import traceback
            traceback.print_exc()

    def _stream_text(self, text: str, chunk_size: int = 1) -> None:
        """
        Stream text output character by character for real-time streaming effect.

        Creates an engaging user experience by displaying text as it's "generated",
        with a small delay between characters for visual feedback.

        Args:
            text: The text to stream to the console.
            chunk_size: Not used in current implementation (kept for compatibility).
        """
        # Stream character by character with minimal delay for real-time feel
        for char in text:
            print(char, end="", flush=True)
            time.sleep(0.005)  # Small delay between characters
        print()  # Final newline

    def run(self):
        """Main entry point for the agent"""
        try:
            self.verify_prerequisites()
            self.initialize_components()
            self.create_agent_graph()
            self.run_conversation()
        except KeyboardInterrupt:
            print("\n\nShutdown requested.")
        except Exception as e:
            print(f"\n✗ Fatal error: {e}")
            sys.exit(1)
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        if self.pool:
            self.pool.close()


def main():
    """Main function"""
    agent = LangChainAgent()
    agent.run()


if __name__ == "__main__":
    main()

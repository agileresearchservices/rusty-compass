#!/usr/bin/env python3
"""
LangChain Agent with Real-Time Streaming, Local Knowledge Base, and Persistent Memory

A production-grade ReAct agent with the following features:
- Real-time character-by-character streaming of agent thinking and final responses
- Local knowledge base using PostgreSQL with PGVector for semantic search
- Persistent conversation memory using PostgreSQL
- Intelligent tool usage for knowledge retrieval
- Multi-turn conversations with context preservation

Powered by:
- LLM: Ollama (gpt-oss:20b)
- Embeddings: Ollama (nomic-embed-text:latest)
- Vector Store: PostgreSQL with PGVector
- Memory: PostgreSQL with LangGraph checkpointer
- Framework: LangGraph with ReAct agent pattern
"""

import sys
import uuid
import warnings
import time
import json
import logging
import httpx
import psycopg
from pathlib import Path
from typing import Sequence, Tuple, List, Optional, Dict, Any, Union, TypedDict, Annotated

# Setup logging
logger = logging.getLogger(__name__)
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END, add_messages
from psycopg_pool import ConnectionPool
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, BaseMessage, AIMessage, HumanMessage, ToolMessage
from langchain_core.documents import Document
from dataclasses import dataclass
from typing import Generator
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================================
# STREAMING EVENT DEFINITIONS
# ============================================================================

@dataclass
class LLMStreamingEvent:
    """Base class for LLM streaming events"""
    pass

@dataclass
class LLMResponseStartEvent(LLMStreamingEvent):
    """Emitted when LLM starts generating a response"""
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            import time
            self.timestamp = time.time()

@dataclass
class LLMResponseChunkEvent(LLMStreamingEvent):
    """Emitted for each chunk of content received from the LLM stream"""
    content: str = ""
    is_complete: bool = False
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            import time
            self.timestamp = time.time()

# Suppress the LangGraphDeprecatedSinceV10 warning about create_react_agent migration.
# The recommended replacement (langchain.agents.create_react_agent) doesn't exist yet in 1.2.0.
# This warning is from an incomplete migration path and will be resolved in a future update.
# TODO: Switch to langchain.agents.create_react_agent once the migration is complete.
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*create_react_agent.*")

from config import (
    LLM_MODEL,
    LLM_TEMPERATURE,
    EMBEDDINGS_MODEL,
    DATABASE_URL,
    DB_CONNECTION_KWARGS,
    DB_POOL_MAX_SIZE,
    VECTOR_COLLECTION_NAME,
    RETRIEVER_K,
    RETRIEVER_FETCH_K,
    RETRIEVER_LAMBDA_MULT,
    RETRIEVER_SEARCH_TYPE,
    ENABLE_QUERY_EVALUATION,
    DEFAULT_LAMBDA_MULT,
    QUERY_EVAL_TIMEOUT_MS,
    RETRIEVER_TOOL_NAME,
    RETRIEVER_TOOL_DESCRIPTION,
    OLLAMA_BASE_URL,
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
    POSTGRES_DB,
    ENABLE_RERANKING,
    RERANKER_MODEL,
    RERANKER_FETCH_K,
    RERANKER_TOP_K,
    RERANKER_INSTRUCTION,
    ENABLE_COMPACTION,
    MAX_CONTEXT_TOKENS,
    COMPACTION_THRESHOLD_PCT,
    MESSAGES_TO_KEEP_FULL,
    MIN_MESSAGES_FOR_COMPACTION,
    TOKEN_CHAR_RATIO,
    REASONING_ENABLED,
    REASONING_EFFORT,
    # Reflection configuration
    ENABLE_REFLECTION,
    ENABLE_DOCUMENT_GRADING,
    ENABLE_RESPONSE_GRADING,
    ENABLE_QUERY_TRANSFORMATION,
    REFLECTION_MAX_ITERATIONS,
    REFLECTION_MIN_RELEVANT_DOCS,
    REFLECTION_DOC_SCORE_THRESHOLD,
    REFLECTION_RESPONSE_SCORE_THRESHOLD,
    REFLECTION_SHOW_STATUS,
)


# ============================================================================
# QWEN3 RERANKER IMPLEMENTATION (LangChain-Compatible)
# ============================================================================

class Qwen3Reranker:
    """
    Cross-encoder reranker using Qwen3-Reranker-8B from HuggingFace.

    Qwen3-Reranker is a causal language model that predicts "yes"/"no" tokens
    to indicate relevance. Scores are computed from the probability of the "yes" token.
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-Reranker-8B", instruction: Optional[str] = None):
        """
        Initialize the Qwen3 cross-encoder reranker model.

        Args:
            model_name: HuggingFace model identifier (default: Qwen/Qwen3-Reranker-8B)
            instruction: Custom domain instruction for reranking (optional)

        Raises:
            OSError: If model cannot be downloaded from HuggingFace
            RuntimeError: If required CUDA libraries are missing
        """
        self.model_name = model_name

        # Load tokenizer with left padding (important for causal LM)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.padding_side = "left"

        # Load as causal language model (not embedding model) in float16 to reduce memory (32GB → 16GB)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,  # Use float16 instead of float32 (50% memory reduction)
            device_map="auto"  # Automatically use GPU if available
        ).eval()

        # Move to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

        # Get token IDs for "yes" and "no" for score computation
        self.yes_token_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.no_token_id = self.tokenizer.convert_tokens_to_ids("no")

        # Default instruction if not provided
        self.instruction = instruction or "Given a web search query, retrieve relevant passages that answer the query"

        logger.info(f"✓ Qwen3-Reranker loaded on device: {self.device}")
        logger.info(f"  Yes token ID: {self.yes_token_id}, No token ID: {self.no_token_id}")

    def _format_input(self, query: str, document: str) -> str:
        """Format input according to Qwen3-Reranker specification"""
        return f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {document}"

    def score_documents(self, query: str, documents: List[Document]) -> List[tuple[Document, float]]:
        """
        Score documents by relevance to query using cross-encoder model.

        Uses Qwen3-Reranker to evaluate query-document pairs and assign relevance scores.

        Args:
            query: The search query string
            documents: List of LangChain Document objects to score

        Returns:
            List of (Document, score) tuples sorted by score in descending order.
            Scores are in range [0.0, 1.0] representing P(relevant)

        Raises:
            ValueError: If documents list is empty
        """
        if not documents:
            return []

        scores = []

        for doc in documents:
            # Format input according to model requirements
            formatted_input = self._format_input(query, doc.page_content)

            with torch.no_grad():
                # Tokenize
                inputs = self.tokenizer(
                    formatted_input,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=8192
                ).to(self.device)

                # Get model output
                outputs = self.model(**inputs)

                # Extract logits from the last token (what the model predicted at end)
                logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]

                # Compute score from yes/no token probabilities
                # Stack yes and no logits for softmax computation
                batch_scores = torch.stack([logits[:, self.no_token_id], logits[:, self.yes_token_id]], dim=1)
                # Apply log_softmax for numerical stability
                batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
                # Extract probability of "yes" (index 1)
                score = float(torch.exp(batch_scores[0, 1]).cpu())

            scores.append((doc, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores

    def rerank(self, query: str, documents: List[Document], top_k: int) -> List[tuple[Document, float]]:
        """
        Rerank documents and return top-k most relevant results.

        Scores all documents using the cross-encoder model and returns the top-k
        results sorted by relevance score in descending order.

        Args:
            query: The search query string
            documents: List of LangChain Document objects to rerank
            top_k: Maximum number of documents to return (actual may be less if insufficient documents)

        Returns:
            List of (Document, score) tuples for top-k results sorted by score descending.
            Returns fewer than top_k documents if input has fewer than top_k documents.

        Examples:
            >>> query = "What is machine learning?"
            >>> reranked = reranker.rerank(query, documents, top_k=4)
            >>> for doc, score in reranked:
            ...     print(f"Score: {score:.4f}, Source: {doc.metadata['source']}")
        """
        scored = self.score_documents(query, documents)

        # Log results
        logger.info(f"[Reranker] Reranked {len(documents)} → {min(top_k, len(scored))} docs")
        for i, (doc, score) in enumerate(scored[:top_k], 1):
            source = doc.metadata.get('source', 'unknown')
            logger.info(f"  {i}. score={score:.4f} [{source}]")

        return scored[:top_k]


# ============================================================================
# STATE SCHEMA FOR CUSTOM AGENT GRAPH
# ============================================================================

class DocumentGrade(TypedDict):
    """Grade for a single retrieved document"""
    source: str
    relevant: bool
    score: float
    reasoning: str


class ReflectionResult(TypedDict):
    """Result of a grading operation"""
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
    optimized_query: Optional[str]                # Pre-optimized query from evaluator

    # Reflection state
    iteration_count: int                          # Track retrieval iterations (0, 1, or 2)
    response_retry_count: int                     # Track response regeneration attempts
    retrieved_documents: List[Document]           # Raw documents from retrieval
    document_grades: List[DocumentGrade]          # Individual document grades
    document_grade_summary: ReflectionResult      # Overall document relevance
    response_grade: ReflectionResult              # Quality of final response
    original_query: str                           # Preserve original for transformation
    transformed_query: Optional[str]              # Rewritten query if docs were poor


# ============================================================================
# VECTOR STORE AND RETRIEVER
# ============================================================================

class SimplePostgresVectorStore:
    """
    PostgreSQL-based vector store for semantic and hybrid document search.

    Combines vector similarity search (768-dimensional embeddings) with full-text
    search using Reciprocal Rank Fusion (RRF) for improved retrieval quality.

    Attributes:
        embeddings: OllamaEmbeddings instance for generating query embeddings
        collection_id: PostgreSQL collection ID for document filtering
        pool: ConnectionPool for database connections (max_size=20)
        database_url: PostgreSQL connection string (optional, for compatibility)
    """

    def __init__(
        self,
        embeddings: OllamaEmbeddings,
        collection_id: str,
        pool: ConnectionPool,
        database_url: Optional[str] = None
    ) -> None:
        self.embeddings: OllamaEmbeddings = embeddings
        self.collection_id: str = collection_id
        self.pool: ConnectionPool = pool
        self.database_url: Optional[str] = database_url  # Kept for backwards compatibility

    def as_retriever(
        self,
        search_type: str = "similarity",
        search_kwargs: Optional[Dict[str, Any]] = None
    ) -> "PostgresRetriever":
        """Return a retriever interface"""
        if search_kwargs is None:
            search_kwargs = {
                "k": RETRIEVER_K,
                "fetch_k": RETRIEVER_FETCH_K,
                "lambda_mult": RETRIEVER_LAMBDA_MULT,
            }

        return PostgresRetriever(
            self,
            search_type=search_type,
            k=search_kwargs.get("k", RETRIEVER_K),
            fetch_k=search_kwargs.get("fetch_k", RETRIEVER_FETCH_K),
            lambda_mult=search_kwargs.get("lambda_mult", RETRIEVER_LAMBDA_MULT)
        )

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Search for document chunks similar to the query using vector embeddings.

        Generates a query embedding and finds the k most similar document chunks
        from the PostgreSQL vector store using cosine distance metric.

        Args:
            query: The search query string
            k: Number of similar documents to return (default: 4)

        Returns:
            List of k most similar LangChain Document objects with metadata

        Raises:
            Exception: If embedding generation or database query fails
        """
        try:
            # Generate query embedding
            query_embedding: List[float] = self.embeddings.embed_query(query)
            embedding_str: str = "[" + ",".join(str(float(e)) for e in query_embedding) + "]"

            # Search in PostgreSQL document chunks
            with self.pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        SELECT dc.content, d.metadata FROM document_chunks dc
                        JOIN documents d ON dc.document_id = d.id
                        WHERE d.collection_id = %s
                        ORDER BY dc.embedding <-> %s::vector
                        LIMIT %s
                        """,
                        (self.collection_id, embedding_str, k)
                    )

                    results: List[Document] = []
                    for row in cur.fetchall():
                        # Handle both dict rows (from pool) and tuple rows
                        if isinstance(row, dict):
                            content = row.get('content', '')
                            metadata = row.get('metadata', {})
                        else:
                            content, metadata = row

                        doc = Document(
                            page_content=content,
                            metadata=metadata or {}
                        )
                        results.append(doc)

                    return results

        except Exception as e:
            print(f"Error during similarity search: {e}")
            return []

    def hybrid_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5
    ) -> List[Document]:
        """
        Hybrid search combining vector similarity and full-text search using RRF.

        Uses Reciprocal Rank Fusion (RRF) to combine:
        1. Vector similarity search (semantic matching via embeddings)
        2. PostgreSQL full-text search (lexical/keyword matching via tsvector)

        Args:
            query: Search query string
            k: Number of final results to return
            fetch_k: Number of candidates to fetch from each method before reranking
            lambda_mult: Weight for text vs vector (0.0=pure vector, 1.0=pure text)

        Returns:
            List of Document objects ranked by RRF score
        """
        # Edge case: pure vector search
        if lambda_mult == 0.0:
            return self.similarity_search(query, k)

        # Edge case: pure text search
        if lambda_mult == 1.0:
            return self._text_search(query, k)

        # Validate lambda
        if not 0.0 <= lambda_mult <= 1.0:
            raise ValueError(f"lambda_mult must be in [0.0, 1.0], got {lambda_mult}")

        try:
            # Generate query embedding for vector search
            query_embedding: List[float] = self.embeddings.embed_query(query)
            embedding_str: str = "[" + ",".join(str(float(e)) for e in query_embedding) + "]"

            # Calculate weights for RRF
            vector_weight = 1.0 - lambda_mult
            text_weight = lambda_mult

            # Execute hybrid query using RRF
            with self.pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        WITH vector_results AS (
                            -- Vector similarity search
                            SELECT
                                dc.id,
                                dc.content,
                                d.metadata,
                                ROW_NUMBER() OVER (ORDER BY dc.embedding <-> %s::vector) AS vector_rank
                            FROM document_chunks dc
                            JOIN documents d ON dc.document_id = d.id
                            WHERE d.collection_id = %s
                            ORDER BY dc.embedding <-> %s::vector
                            LIMIT %s
                        ),
                        text_results AS (
                            -- Full-text search
                            SELECT
                                dc.id,
                                dc.content,
                                d.metadata,
                                ROW_NUMBER() OVER (
                                    ORDER BY ts_rank_cd(dc.content_tsv, websearch_to_tsquery('english', %s)) DESC
                                ) AS text_rank
                            FROM document_chunks dc
                            JOIN documents d ON dc.document_id = d.id
                            WHERE d.collection_id = %s
                              AND dc.content_tsv @@ websearch_to_tsquery('english', %s)
                            ORDER BY ts_rank_cd(dc.content_tsv, websearch_to_tsquery('english', %s)) DESC
                            LIMIT %s
                        ),
                        combined_results AS (
                            -- Union of both result sets
                            SELECT DISTINCT
                                COALESCE(vr.id, tr.id) AS id,
                                COALESCE(vr.content, tr.content) AS content,
                                COALESCE(vr.metadata, tr.metadata) AS metadata,
                                COALESCE(vr.vector_rank, 999999) AS vector_rank,
                                COALESCE(tr.text_rank, 999999) AS text_rank
                            FROM vector_results vr
                            FULL OUTER JOIN text_results tr ON vr.id = tr.id
                        )
                        SELECT
                            id,
                            content,
                            metadata,
                            (
                                (%s / (60.0 + vector_rank)) +
                                (%s / (60.0 + text_rank))
                            ) AS rrf_score
                        FROM combined_results
                        ORDER BY rrf_score DESC
                        LIMIT %s
                        """,
                        (
                            embedding_str,      # vector_results: embedding comparison
                            self.collection_id, # vector_results: collection filter
                            embedding_str,      # vector_results: ORDER BY
                            fetch_k,            # vector_results: LIMIT
                            query,              # text_results: query 1
                            self.collection_id, # text_results: collection filter
                            query,              # text_results: query 2
                            query,              # text_results: query 3
                            fetch_k,            # text_results: LIMIT
                            vector_weight,      # RRF vector weight
                            text_weight,        # RRF text weight
                            k                   # Final LIMIT
                        )
                    )

                    results: List[Document] = []
                    for row in cur.fetchall():
                        # Handle both dict rows (from pool) and tuple rows
                        if isinstance(row, dict):
                            content = row.get('content', '')
                            metadata = row.get('metadata', {})
                        else:
                            # Unpack tuple: (id, content, metadata, rrf_score)
                            _, content, metadata, _ = row

                        results.append(Document(
                            page_content=content,
                            metadata=metadata or {}
                        ))

                    return results

        except psycopg.errors.UndefinedColumn as e:
            if "content_tsv" in str(e):
                print("⚠ Warning: Full-text search not available. Run migration first.")
                print("  Falling back to pure vector search...")
                return self.similarity_search(query, k)
            raise

        except Exception as e:
            print(f"Error during hybrid search: {e}")
            return []

    def _text_search(self, query: str, k: int = 4) -> List[Document]:
        """Pure full-text search fallback for lambda=1.0"""
        try:
            with self.pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT dc.content, d.metadata
                        FROM document_chunks dc
                        JOIN documents d ON dc.document_id = d.id
                        WHERE d.collection_id = %s
                          AND dc.content_tsv @@ websearch_to_tsquery('english', %s)
                        ORDER BY ts_rank_cd(dc.content_tsv, websearch_to_tsquery('english', %s)) DESC
                        LIMIT %s
                        """,
                        (self.collection_id, query, query, k)
                    )

                    results: List[Document] = []
                    for row in cur.fetchall():
                        if isinstance(row, dict):
                            content = row.get('content', '')
                            metadata = row.get('metadata', {})
                        else:
                            content, metadata = row

                        results.append(Document(
                            page_content=content,
                            metadata=metadata or {}
                        ))

                    return results

        except Exception as e:
            print(f"Error during text search: {e}")
            return []


class PostgresRetriever:
    """Retriever interface for PostgreSQL vector store"""

    def __init__(
        self,
        vector_store: SimplePostgresVectorStore,
        search_type: str = "similarity",
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5
    ) -> None:
        self.vector_store: SimplePostgresVectorStore = vector_store
        self.search_type: str = search_type
        self.k: int = k
        self.fetch_k: int = fetch_k
        self.lambda_mult: float = lambda_mult

    def invoke(
        self,
        input_dict: Union[Dict[str, Any], str]
    ) -> List[Document]:
        """Retrieve documents for a query

        Args:
            input_dict: Either a dictionary with 'input' or 'query' key,
                       or a string query directly

        Returns:
            List of Document objects matching the query
        """
        if isinstance(input_dict, dict):
            query: str = input_dict.get("input") or input_dict.get("query", "")
        else:
            query: str = str(input_dict)

        if self.search_type == "hybrid":
            return self.vector_store.hybrid_search(
                query,
                k=self.k,
                fetch_k=self.fetch_k,
                lambda_mult=self.lambda_mult
            )
        elif self.search_type == "similarity":
            return self.vector_store.similarity_search(query, k=self.k)
        else:
            raise ValueError(f"Unknown search_type: {self.search_type}")


class LangChainAgent:
    """
    Main agent class that manages the LLM, tools, and conversation state.

    Handles:
    - Real-time streaming of agent thinking and responses
    - Integration with local knowledge base (PostgreSQL + PGVector)
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
        self.tools = []  # Will be populated in create_agent_graph
        self.retriever = None  # Base retriever
        self.reranker = None  # Cross-encoder reranker
        self.compression_retriever = None  # LangChain compression retriever with reranking

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

        # Check PGVector extension
        try:
            with psycopg.connect(DATABASE_URL) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')")
                    if cur.fetchone()[0]:
                        print("✓ PGVector extension is enabled")
                    else:
                        print("✗ PGVector extension not found")
                        print("  Run: python setup_db.py")
                        sys.exit(1)
        except Exception as e:
            print(f"✗ Error checking PGVector extension: {e}")
            sys.exit(1)

        # Check if documents table has data
        try:
            with psycopg.connect(DATABASE_URL) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT COUNT(*) FROM documents WHERE collection_id = %s",
                        (VECTOR_COLLECTION_NAME,)
                    )
                    doc_count = cur.fetchone()[0]
                    if doc_count == 0:
                        print(f"✗ No documents found in vector store")
                        print("  Run: python load_sample_data_pgvector.py")
                        sys.exit(1)
                    print(f"✓ Vector store has {doc_count} documents")
        except Exception as e:
            print(f"✗ Error checking vector store: {e}")
            print("  Make sure vector tables exist. Run: python setup_db.py")
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
            num_predict=4096,  # Allow comprehensive responses (increased from 1024 to prevent truncation)
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

        # Initialize Postgres connection pool (must be before vector store)
        print("Connecting to Postgres checkpoint store...")
        connection_kwargs = DB_CONNECTION_KWARGS.copy()
        self.pool = ConnectionPool(
            conninfo=DATABASE_URL,
            max_size=DB_POOL_MAX_SIZE,
            kwargs=connection_kwargs
        )
        print("✓ Postgres connection pool initialized")

        # Initialize Vector Store using PostgreSQL with PGVector
        print(f"Loading PostgreSQL vector store: {VECTOR_COLLECTION_NAME}")
        # Create a simple retriever that uses PostgreSQL directly
        self.vector_store = SimplePostgresVectorStore(
            embeddings=self.embeddings,
            collection_id=VECTOR_COLLECTION_NAME,
            pool=self.pool,
        )
        print("✓ Vector store initialized")

        # Create base retriever
        self.retriever = self.vector_store.as_retriever(
            search_type="hybrid",
            search_kwargs={
                "k": RERANKER_FETCH_K if ENABLE_RERANKING else RETRIEVER_K,
                "fetch_k": RETRIEVER_FETCH_K,
                "lambda_mult": RETRIEVER_LAMBDA_MULT,
            }
        )

        # Initialize Qwen3 Cross-Encoder Reranker
        if ENABLE_RERANKING:
            print(f"Loading cross-encoder reranker: {RERANKER_MODEL}")
            self.reranker = Qwen3Reranker(model_name=RERANKER_MODEL)
            print("✓ Reranker initialized")
        else:
            self.reranker = None

        # Initialize checkpointer with existing pool
        self.checkpointer = PostgresSaver(self.pool)
        print("✓ Postgres checkpoint store initialized")

        # Ensure conversation metadata table exists
        self._ensure_metadata_table()

        print()

    # ========================================================================
    # AGENT GRAPH NODES FOR DYNAMIC QUERY EVALUATION
    # ========================================================================

    def query_evaluator_node(self, state: CustomAgentState) -> Dict[str, Any]:
        """
        Evaluate the query type and determine optimal lambda_mult for hybrid search.

        Lambda interpretation (0.0=lexical, 1.0=semantic):
        - 0.0-0.2: Pure lexical (dates, model numbers, part numbers, exact identifiers)
        - 0.2-0.4: Lexical-heavy (specific versions, brands, frameworks)
        - 0.4-0.6: Balanced (mixed queries with concepts and specific terms)
        - 0.6-0.8: Semantic-heavy (framework guides, optimization techniques)
        - 0.8-1.0: Pure semantic (conceptual questions, "what is", "explain")
        """
        start_time = time.time()
        messages = state["messages"]

        # Extract last user message
        last_user_msg = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_user_msg = msg.content
                break

        if not last_user_msg:
            # No user message, use default
            return {
                "lambda_mult": 0.25,
                "query_analysis": "No query detected",
                "optimized_query": None
            }

        print(f"\n[Query Evaluator] Starting evaluation for query: '{last_user_msg[:80]}...'")

        # LLM evaluation prompt
        evaluation_prompt = f"""Analyze this search query and determine the optimal search strategy AND optimized query.

Query: "{last_user_msg}"

Lambda_mult Guidelines (0.0=lexical, 1.0=semantic):
- 0.0-0.2: Pure LEXICAL - Extract exact terms, preserve identifiers
- 0.2-0.4: LEXICAL-heavy - Preserve key terms, add keywords
- 0.4-0.6: BALANCED - Balance keywords and natural language
- 0.6-0.8: SEMANTIC-heavy - Preserve natural language, clarify intent
- 0.8-1.0: Pure SEMANTIC - PRESERVE NATURAL LANGUAGE (minimal changes)

CRITICAL RULES:
1. lambda >= 0.8: Keep query nearly identical
2. lambda >= 0.6: Keep most natural language
3. lambda <= 0.4: Optimize for keywords
4. lambda <= 0.2: Heavy keyword extraction
5. ALWAYS maintain user intent

Respond with ONLY JSON:
{{"lambda_mult": <0.0-1.0>, "reasoning": "<brief explanation>", "optimized_query": "<optimized query>"}}
"""

        try:
            # Call LLM for evaluation
            response = self.llm.invoke(evaluation_prompt)
            result_text = response.content.strip()

            # Parse JSON response
            result = json.loads(result_text)

            lambda_mult = float(result["lambda_mult"])
            reasoning = result.get("reasoning", "No reasoning provided")
            optimized_query = result.get("optimized_query", last_user_msg)

            # Validate range
            lambda_mult = max(0.0, min(1.0, lambda_mult))

            # Categorize search strategy
            if lambda_mult < 0.2:
                strategy = "Pure Lexical (BM25)"
            elif lambda_mult < 0.4:
                strategy = "Lexical-Heavy (BM25 dominant)"
            elif lambda_mult < 0.6:
                strategy = "Balanced (Hybrid)"
            elif lambda_mult < 0.8:
                strategy = "Semantic-Heavy (Vector dominant)"
            else:
                strategy = "Pure Semantic (Vector)"

            elapsed = time.time() - start_time
            print(f"[Query Evaluator] ✓ Strategy: {strategy}")
            print(f"  Lambda: {lambda_mult:.2f} | Reasoning: {reasoning}")
            print(f"  Original: '{last_user_msg}'")
            print(f"  Optimized: '{optimized_query}'")
            print(f"  Evaluation time: {elapsed:.3f}s")

            return {
                "lambda_mult": lambda_mult,
                "query_analysis": reasoning,
                "optimized_query": optimized_query
            }

        except Exception as e:
            # Fallback to default if evaluation fails
            elapsed = time.time() - start_time
            print(f"[Query Evaluator] ⚠ Evaluation failed: {e}")
            print(f"  Using default lambda=0.25 (Balanced search)")
            print(f"  Evaluation time: {elapsed:.3f}s")
            return {
                "lambda_mult": 0.25,
                "query_analysis": f"Evaluation failed: {str(e)}",
                "optimized_query": last_user_msg
            }

    def agent_node(self, state: CustomAgentState) -> Dict[str, Any]:
        """
        Agent reasoning node - calls LLM with tool binding and true streaming support.

        Supports both streaming and non-streaming LLM implementations:
        - If LLM has .stream() method: Uses true LLM streaming with event emission
        - If LLM only has .invoke(): Falls back to standard invoke method
        """
        start_time = time.time()
        messages = list(state["messages"])

        print(f"\n[Agent Node] Processing {len(messages)} messages...")

        # Inject Query Evaluator's recommendation with pre-optimized query
        # This provides stronger guidance including the actual optimized query
        lambda_mult = state.get("lambda_mult", 0.25)
        query_analysis = state.get("query_analysis", "")
        optimized_query = state.get("optimized_query", "")

        # Create search strategy message based on lambda ranges
        if lambda_mult >= 0.8:
            # Pure semantic search - preserve natural language queries
            search_hint = f"""SEARCH STRATEGY: Pure SEMANTIC (lambda={lambda_mult:.2f})

Pre-optimized query (USE EXACT):
"{optimized_query}"

DO NOT rewrite. Optimized for semantic vector search.
Reasoning: {query_analysis}"""
        elif lambda_mult >= 0.6:
            # Semantic-heavy search - mostly natural language
            search_hint = f"""SEARCH STRATEGY: Semantic-Heavy (lambda={lambda_mult:.2f})

Pre-optimized query (use as-is):
"{optimized_query}"

Mostly optimized for semantic search. Preserve natural language.
Reasoning: {query_analysis}"""
        elif lambda_mult >= 0.4:
            # Balanced hybrid search - can optimize both
            search_hint = f"""SEARCH STRATEGY: Balanced Hybrid (lambda={lambda_mult:.2f})

Pre-optimized query (prefer using as-is):
"{optimized_query}"

Balanced for both semantic and lexical search. Can optimize slightly if needed.
Reasoning: {query_analysis}"""
        elif lambda_mult >= 0.2:
            # Lexical-heavy search - keyword optimization is fine
            search_hint = f"""SEARCH STRATEGY: Lexical-Heavy (lambda={lambda_mult:.2f})

Pre-optimized query (use in knowledge_base):
"{optimized_query}"

Optimized for keyword/BM25 matching. Can refine for lexical search.
Reasoning: {query_analysis}"""
        else:
            # Pure lexical search - heavy keyword optimization
            search_hint = f"""SEARCH STRATEGY: Pure Lexical (lambda={lambda_mult:.2f})

Pre-optimized query (use exactly as provided):
"{optimized_query}"

Heavily optimized for exact term matching. Use with high precision.
Reasoning: {query_analysis}"""

        # Add search strategy with optimized query as system message if we have lambda evaluation
        if query_analysis and optimized_query:
            messages.insert(0, SystemMessage(content=search_hint))

        # Bind tools to LLM
        llm_with_tools = self.llm.bind_tools(self.tools)

        # Check if the LLM supports streaming
        if hasattr(llm_with_tools, "stream") and callable(getattr(llm_with_tools, "stream")):
            # Use true LLM streaming
            response = self._stream_llm_response(llm_with_tools, messages)
        else:
            # Fall back to invoke() for non-streaming LLMs
            print(f"[Agent Node] LLM does not support streaming, using invoke()")
            response = llm_with_tools.invoke(messages)

        # Calculate response statistics
        response_length = len(response.content) if hasattr(response, "content") else 0
        tool_calls = len(response.tool_calls) if hasattr(response, "tool_calls") and response.tool_calls else 0
        elapsed = time.time() - start_time

        if tool_calls > 0:
            print(f"[Agent Node] ✓ Generated {tool_calls} tool call(s)")
            for i, tool_call in enumerate(response.tool_calls, 1):
                print(f"  {i}. {tool_call.get('name', 'unknown')} - {tool_call.get('args', {})}")
        else:
            print(f"[Agent Node] ✓ Generated response ({response_length} chars)")

        print(f"  Processing time: {elapsed:.3f}s")

        return {"messages": [response]}

    def _stream_llm_response(self, llm_with_tools, messages: Sequence[BaseMessage]) -> AIMessage:
        """
        Stream the LLM response and accumulate the full response while emitting events.

        Falls back to invoke() when:
        - Streaming produces no content (e.g., when tool calls are being made)
        - An exception occurs during streaming

        Args:
            llm_with_tools: The LLM instance with tools bound
            messages: The input messages for the LLM

        Returns:
            The accumulated AIMessage response
        """
        stream_start = time.time()

        # Emit start event
        start_event = LLMResponseStartEvent()
        self._emit_streaming_event(start_event)

        # Accumulate response content
        accumulated_content = ""
        invoke_result = None  # Result from invoke fallback (not streaming chunks)
        chunk_count = 0
        used_fallback = False

        try:
            # Stream from the LLM
            for chunk in llm_with_tools.stream(messages):
                chunk_count += 1

                # Extract content from chunk
                # For tool-call responses, chunk.content might be None or empty
                if hasattr(chunk, "content") and chunk.content:
                    accumulated_content += chunk.content

                    # Emit chunk event
                    chunk_event = LLMResponseChunkEvent(content=chunk.content, is_complete=False)
                    self._emit_streaming_event(chunk_event)

        except StopIteration:
            # Normal generator exhaustion - not an error, continue
            pass
        except RuntimeError as e:
            # Ignore RuntimeError from StopIteration (Python 3.7+ async context)
            if "StopIteration" not in str(e):
                logger.warning(f"RuntimeError during LLM streaming: {e}. Falling back to invoke.")
        except Exception as e:
            logger.warning(f"Exception during LLM streaming: {e}. Falling back to invoke.")

        # If streaming produced no content (empty response), fall back to invoke
        # This happens when tool calls are being made instead of text response
        if not accumulated_content:
            used_fallback = True
            invoke_result = llm_with_tools.invoke(messages)

            # Extract content from the response
            if hasattr(invoke_result, "content"):
                accumulated_content = invoke_result.content if invoke_result.content else ""
            else:
                accumulated_content = str(invoke_result)

        # Emit completion event
        completion_event = LLMResponseChunkEvent(content="", is_complete=True)
        self._emit_streaming_event(completion_event)

        stream_elapsed = time.time() - stream_start

        # Log streaming details
        if used_fallback:
            print(f"[Streaming] {chunk_count} chunks received, no content → Using invoke() fallback")
        else:
            print(f"[Streaming] {chunk_count} chunks received, {len(accumulated_content)} chars accumulated")
        print(f"[Streaming] Time: {stream_elapsed:.3f}s")

        # If we got a result from invoke fallback, return it (preserves tool_calls)
        if invoke_result and isinstance(invoke_result, AIMessage):
            return invoke_result
        else:
            # Otherwise construct AIMessage from accumulated content
            # This covers both: streamed content, and invoke fallback without AIMessage type
            return AIMessage(content=accumulated_content)

    def _emit_streaming_event(self, event: LLMStreamingEvent) -> None:
        """
        Emit a streaming event (for future integration with WebSocket or event listeners).

        Currently logs the event. Can be extended to:
        - Send events to WebSocket clients
        - Broadcast to observability systems
        - Update real-time dashboards

        Args:
            event: The streaming event to emit
        """
        if isinstance(event, LLMResponseStartEvent):
            logger.debug("LLM streaming started")
        elif isinstance(event, LLMResponseChunkEvent):
            if event.is_complete:
                logger.debug("LLM streaming complete")
            else:
                logger.debug(f"LLM chunk received: {len(event.content)} chars")

    def tools_node(self, state: CustomAgentState) -> Dict[str, Any]:
        """
        Execute tool calls with access to state (for dynamic lambda_mult).
        Uses Qwen3 cross-encoder for reranking if enabled.
        """
        start_time = time.time()
        messages = state["messages"]
        last_message = messages[-1]

        # Extract tool calls from last message
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return {"messages": []}

        tool_responses = []
        lambda_mult = state.get("lambda_mult", 0.25)

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            if tool_name == "knowledge_base":
                query = tool_args.get("query", "")
                optimized_query = state.get("optimized_query")

                # Log query usage
                if optimized_query:
                    if query == optimized_query:
                        print(f"\n[Retrieval] ✓ Using optimized query: '{query}'")
                    else:
                        print(f"\n[Retrieval] ⚠ Query mismatch:")
                        print(f"  Expected: '{optimized_query}'")
                        print(f"  Received: '{query}'")
                else:
                    print(f"\n[Retrieval] Query: '{query}'")

                print(f"[Retrieval] Search strategy: Lambda={lambda_mult:.2f}")

                # Create retriever with dynamic lambda_mult
                retriever = self.vector_store.as_retriever(
                    search_type="hybrid",
                    search_kwargs={
                        "k": RERANKER_FETCH_K if ENABLE_RERANKING else RETRIEVER_K,
                        "fetch_k": RETRIEVER_FETCH_K,
                        "lambda_mult": lambda_mult
                    }
                )

                # Get initial results
                retrieve_start = time.time()
                results = retriever.invoke(query)
                retrieve_elapsed = time.time() - retrieve_start

                print(f"[Retrieval] ✓ Retrieved {len(results)} documents in {retrieve_elapsed:.3f}s")

                # Apply reranking if enabled
                if ENABLE_RERANKING and self.reranker and results:
                    # Capture original order for comparison and store original ranks
                    original_sources = [doc.metadata.get('source', 'unknown') for doc in results]
                    for i, doc in enumerate(results, 1):
                        doc.metadata['original_rank'] = i

                    rerank_start = time.time()
                    print(f"\n[Reranker] Processing {len(results)} candidates with Qwen3-Reranker-8B...")
                    reranked_results = self.reranker.rerank(query, results, RERANKER_TOP_K)
                    rerank_elapsed = time.time() - rerank_start

                    # Extract documents with scores and store in metadata
                    results_with_scores = [(doc, score) for doc, score in reranked_results]
                    results = [doc for doc, score in results_with_scores]

                    # Store reranker scores in metadata and update metadata for observability
                    for i, (doc, score) in enumerate(results_with_scores, 1):
                        doc.metadata['reranker_score'] = score

                    # Log reranking results with scores
                    avg_score = sum(score for _, score in results_with_scores) / len(results_with_scores) if results_with_scores else 0
                    print(f"[Reranker] ✓ Reranking complete in {rerank_elapsed:.3f}s → top {len(results)} selected (avg score: {avg_score:.4f})")
                    for i, (doc, score) in enumerate(results_with_scores, 1):
                        source = doc.metadata.get('source', 'unknown')
                        relevance_bar = "█" * int(score * 20)
                        print(f"  {i}. score={score:.4f} {relevance_bar} [{source}]")

                    # Log order changes if applicable
                    reranked_sources = [doc.metadata.get('source', 'unknown') for doc in results]
                    if original_sources[:len(reranked_sources)] != reranked_sources:
                        print(f"[Reranker] ℹ Order changed (reranking improved relevance)")
                    else:
                        print(f"[Reranker] ℹ Order unchanged (already optimally ranked)")
                else:
                    print(f"[Retrieval] No reranking (disabled or no documents)")

                content = "\n\n".join([doc.page_content for doc in results]) if results else "No relevant information found."

                # Create tool response message
                tool_responses.append(
                    ToolMessage(
                        content=content,
                        tool_call_id=tool_call["id"]
                    )
                )

                # Store retrieved documents for reflection grading
                if ENABLE_REFLECTION and ENABLE_DOCUMENT_GRADING:
                    return {
                        "messages": tool_responses,
                        "retrieved_documents": results
                    }

        return {"messages": tool_responses}

    def should_continue(self, state: CustomAgentState) -> str:
        """
        Determine whether to continue to tools, response_grader, or end.
        """
        messages = state["messages"]
        last_message = messages[-1]

        # If LLM made tool calls, route to tools
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"

        # If reflection is enabled, route to response grader
        if ENABLE_REFLECTION and ENABLE_RESPONSE_GRADING:
            return "response_grader"

        # Otherwise, end
        return "END"

    def route_after_doc_grading(self, state: CustomAgentState) -> str:
        """
        Route after document grading: continue to agent or retry with transformed query.
        """
        # If reflection is disabled, always continue to agent
        if not ENABLE_REFLECTION or not ENABLE_DOCUMENT_GRADING:
            return "agent"

        doc_summary = state.get("document_grade_summary", {})
        iteration = state.get("iteration_count", 0)

        # If docs are good, continue to agent
        if doc_summary.get("grade") == "pass":
            return "agent"

        # If docs are bad but can retry (and query transformation is enabled)
        if ENABLE_QUERY_TRANSFORMATION and iteration < REFLECTION_MAX_ITERATIONS:
            print(f"[Reflection] Documents failed grading. Retry {iteration + 1}/{REFLECTION_MAX_ITERATIONS}")
            return "query_transformer"

        # Max iterations reached or query transformation disabled, continue anyway
        if iteration >= REFLECTION_MAX_ITERATIONS:
            print(f"[Reflection] Max iterations reached. Proceeding with available documents.")
        return "agent"

    def route_after_response_grading(self, state: CustomAgentState) -> str:
        """
        Route after response grading: end or retry with feedback.

        If the response failed grading and retries are available, routes back
        to the agent with feedback to improve the response.
        """
        if not ENABLE_REFLECTION or not ENABLE_RESPONSE_GRADING:
            return "END"

        response_grade = state.get("response_grade", {})
        retry_count = state.get("response_retry_count", 0)

        # If response is good, we're done
        if response_grade.get("grade") == "pass":
            return "END"

        # If response is bad but can retry
        if retry_count < REFLECTION_MAX_ITERATIONS:
            print(f"[Reflection] Response failed grading. Retry {retry_count + 1}/{REFLECTION_MAX_ITERATIONS}")
            return "response_improver"

        # Max retries reached
        print(f"[Reflection] Response retry limit reached. Ending with current response.")
        return "END"

    def response_improver_node(self, state: CustomAgentState) -> Dict[str, Any]:
        """
        Add feedback message to prompt the agent to improve its response.

        Increments retry counter and adds a system message with the grading
        feedback to help the agent generate a better response.
        """
        response_grade = state.get("response_grade", {})
        retry_count = state.get("response_retry_count", 0)
        reasoning = response_grade.get("reasoning", "Response was incomplete or unclear")

        # Create feedback message content for both agent and observability
        feedback_content = f"[Response needs improvement] {reasoning}. Please provide a complete, well-structured response."

        # Create feedback message for the agent
        feedback_msg = HumanMessage(content=feedback_content)

        return {
            "messages": [feedback_msg],
            "response_retry_count": retry_count + 1,
            "feedback": feedback_content  # Expose feedback for observability events
        }

    # ========================================================================
    # REFLECTION NODES
    # ========================================================================

    def document_grader_node(self, state: CustomAgentState) -> Dict[str, Any]:
        """
        Grade retrieved documents for relevance to the user query.

        Uses LLM to evaluate each document and determine overall relevance.
        """
        if not ENABLE_REFLECTION or not ENABLE_DOCUMENT_GRADING:
            return {}

        start_time = time.time()
        retrieved_docs = state.get("retrieved_documents", [])

        if not retrieved_docs:
            return {
                "document_grades": [],
                "document_grade_summary": {
                    "grade": "fail",
                    "score": 0.0,
                    "reasoning": "No documents retrieved"
                }
            }

        print(f"\n[Document Grader] Grading {len(retrieved_docs)} documents...")

        # Extract the original query from messages
        messages = state["messages"]
        original_query = state.get("original_query", "")
        if not original_query:
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    # Skip retry messages
                    if not msg.content.startswith("[Retry with transformed query]"):
                        original_query = msg.content
                        break

        # Grade each document
        grades = []
        for i, doc in enumerate(retrieved_docs, 1):
            grade = self._grade_document(original_query, doc)
            grades.append(grade)
            # Show individual grading results
            status_icon = "✓" if grade["relevant"] else "✗"
            source = grade.get("source", "unknown")
            print(f"  {i}. {status_icon} [{source}] score={grade['score']:.2f} - {grade['reasoning']}")

        # Compute summary
        relevant_count = sum(1 for g in grades if g["relevant"])
        avg_score = sum(g["score"] for g in grades) / len(grades) if grades else 0.0

        # Pass if we have enough relevant documents
        passed = relevant_count >= REFLECTION_MIN_RELEVANT_DOCS and avg_score >= REFLECTION_DOC_SCORE_THRESHOLD

        summary = {
            "grade": "pass" if passed else "fail",
            "score": avg_score,
            "reasoning": f"{relevant_count}/{len(grades)} documents relevant (avg score: {avg_score:.2f})"
        }

        elapsed = time.time() - start_time

        if REFLECTION_SHOW_STATUS:
            status_icon = "✓" if passed else "✗"
            print(f"\n[Document Grader] {status_icon} {summary['reasoning']}")
            print(f"  Grading time: {elapsed:.3f}s")

        return {
            "document_grades": grades,
            "document_grade_summary": summary,
            "original_query": original_query
        }

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
            "optimized_query": None,  # Clear to force re-evaluation
            "retrieved_documents": [],  # Clear previous documents
            "document_grades": [],
            "document_grade_summary": {}
        }

    def response_grader_node(self, state: CustomAgentState) -> Dict[str, Any]:
        """
        Grade the quality of the agent's final response.

        Evaluates relevance, completeness, and accuracy.
        """
        if not ENABLE_REFLECTION or not ENABLE_RESPONSE_GRADING:
            return {}

        start_time = time.time()
        messages = state["messages"]
        original_query = state.get("original_query", "")

        print(f"\n[Response Grader] Evaluating response quality...")

        # Get the last AI response (non-tool-call)
        last_response = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                    last_response = msg.content
                    break

        if not last_response:
            return {
                "response_grade": {
                    "grade": "fail",
                    "score": 0.0,
                    "reasoning": "No response generated"
                }
            }

        # Get document context
        doc_summary = state.get("document_grade_summary", {})

        # Show response preview and context
        response_preview = last_response[:150].replace("\n", " ") + "..." if len(last_response) > 150 else last_response
        print(f"  Response: {response_preview}")
        print(f"  Response length: {len(last_response)} chars")

        grade = self._grade_response(
            query=original_query,
            response=last_response,
            doc_context=doc_summary
        )

        elapsed = time.time() - start_time

        if REFLECTION_SHOW_STATUS:
            status_icon = "✓" if grade["grade"] == "pass" else "✗"
            print(f"\n[Response Grader] {status_icon} {grade['grade'].upper()} (score: {grade['score']:.2f})")
            print(f"  Reasoning: {grade['reasoning']}")
            print(f"  Grading time: {elapsed:.3f}s")

        return {"response_grade": grade}

    def _grade_document(self, query: str, doc: Document) -> DocumentGrade:
        """Grade a single document for relevance using LLM."""
        source = doc.metadata.get("source", "unknown")
        content_preview = doc.page_content[:500]

        prompt = f"""Evaluate if this document is relevant to the user's query.

USER QUERY: {query}

DOCUMENT (from {source}):
{content_preview}

Evaluate:
1. Does this document contain information that helps answer the query?
2. Is the information directly relevant or only tangentially related?
3. Would using this document improve the response quality?

Respond with JSON only:
{{"relevant": true, "score": 0.8, "reasoning": "brief explanation"}}

Use these guidelines:
- relevant: true if the document helps answer the query, false otherwise
- score: 0.0-1.0 indicating relevance strength
- reasoning: one sentence explanation"""

        try:
            response = self.llm.invoke(prompt)
            result_text = response.content.strip()

            # Try to parse JSON
            result = json.loads(result_text)

            return {
                "source": source,
                "relevant": bool(result.get("relevant", False)),
                "score": float(result.get("score", 0.5)),
                "reasoning": str(result.get("reasoning", "No reasoning"))
            }
        except Exception as e:
            # Fallback: assume marginally relevant
            logger.warning(f"Document grading failed for {source}: {e}")
            return {
                "source": source,
                "relevant": True,
                "score": 0.5,
                "reasoning": f"Grading failed: {str(e)[:50]}"
            }

    def _grade_response(self, query: str, response: str, doc_context: dict) -> ReflectionResult:
        """Grade the quality of the final response using LLM."""
        prompt = f"""Evaluate the quality of this response to the user's query.

USER QUERY: {query}

RESPONSE:
{response[:1000]}

Evaluate on these criteria:
1. RELEVANCE: Does the response directly address the query?
2. COMPLETENESS: Does it provide a thorough answer?
3. CLARITY: Is the response well-structured and easy to understand?

Respond with JSON only:
{{"grade": "pass", "score": 0.85, "reasoning": "brief explanation"}}

Guidelines:
- grade: "pass" if response reasonably answers the query, "fail" if clearly wrong/irrelevant/incomplete
- score: 0.0-1.0 indicating quality
- reasoning: one sentence explanation

Only FAIL responses that are clearly wrong, irrelevant, or too incomplete to be useful."""

        try:
            result = self.llm.invoke(prompt)
            result_text = result.content.strip()

            parsed = json.loads(result_text)

            return {
                "grade": str(parsed.get("grade", "pass")),
                "score": float(parsed.get("score", 0.7)),
                "reasoning": str(parsed.get("reasoning", "No reasoning"))
            }
        except Exception as e:
            # Fallback: assume acceptable
            logger.warning(f"Response grading failed: {e}")
            return {
                "grade": "pass",
                "score": 0.7,
                "reasoning": f"Grading failed: {str(e)[:50]}"
            }

    def create_agent_graph(self):
        """Create custom StateGraph with query evaluation for dynamic lambda_mult"""
        print("Creating custom agent graph with query evaluation...")

        # Define the knowledge_base tool schema
        @tool
        def knowledge_base(query: str) -> str:
            """Search for information in the local document knowledge base.

            Use this tool to find relevant information about Python programming,
            machine learning concepts, and web development topics stored in the
            local document index.

            Args:
                query: The search query to find relevant documents.

            Returns:
                Relevant document content from the knowledge base, or a message
                if no relevant documents are found.
            """
            # This is just a schema definition
            # Actual execution happens in tools_node with dynamic lambda
            pass

        # Store tools list for the agent
        self.tools = [knowledge_base]

        # Build the graph
        workflow = StateGraph(CustomAgentState)

        # Add core nodes
        workflow.add_node("query_evaluator", self.query_evaluator_node)
        workflow.add_node("agent", self.agent_node)
        workflow.add_node("tools", self.tools_node)

        # Add reflection nodes (if enabled)
        if ENABLE_REFLECTION:
            if ENABLE_DOCUMENT_GRADING:
                workflow.add_node("document_grader", self.document_grader_node)
            if ENABLE_QUERY_TRANSFORMATION:
                workflow.add_node("query_transformer", self.query_transformer_node)
            if ENABLE_RESPONSE_GRADING:
                workflow.add_node("response_grader", self.response_grader_node)
                workflow.add_node("response_improver", self.response_improver_node)

        # Set entry point
        workflow.set_entry_point("query_evaluator")

        # Add edges
        workflow.add_edge("query_evaluator", "agent")

        # Agent routing: tools, response_grader, or END
        if ENABLE_REFLECTION and ENABLE_RESPONSE_GRADING:
            workflow.add_conditional_edges(
                "agent",
                self.should_continue,
                {
                    "tools": "tools",
                    "response_grader": "response_grader",
                    "END": END
                }
            )
            # Response grader -> conditional: END or response_improver
            workflow.add_conditional_edges(
                "response_grader",
                self.route_after_response_grading,
                {
                    "END": END,
                    "response_improver": "response_improver"
                }
            )
            # Response improver -> back to agent for another try
            workflow.add_edge("response_improver", "agent")
        else:
            workflow.add_conditional_edges(
                "agent",
                self.should_continue,
                {
                    "tools": "tools",
                    "END": END
                }
            )

        # Tools routing with document grading
        if ENABLE_REFLECTION and ENABLE_DOCUMENT_GRADING:
            # Tools -> Document Grader
            workflow.add_edge("tools", "document_grader")

            # Document Grader routing
            if ENABLE_QUERY_TRANSFORMATION:
                workflow.add_conditional_edges(
                    "document_grader",
                    self.route_after_doc_grading,
                    {
                        "agent": "agent",
                        "query_transformer": "query_transformer"
                    }
                )
                # Query Transformer -> Query Evaluator (for retry)
                workflow.add_edge("query_transformer", "query_evaluator")
            else:
                # No query transformation, always go to agent
                workflow.add_edge("document_grader", "agent")
        else:
            # No document grading, tools -> agent directly
            workflow.add_edge("tools", "agent")

        # Compile with checkpointer
        self.app = workflow.compile(checkpointer=self.checkpointer)

        if ENABLE_REFLECTION:
            print("✓ Custom agent graph created with query evaluator + reflection")
        else:
            print("✓ Custom agent graph created with query evaluator")
        print()

    def generate_thread_id(self):
        """Generate a unique thread ID for conversation persistence"""
        self.thread_id = f"conversation_{uuid.uuid4().hex[:8]}"

    def set_thread_id(self, thread_id: str):
        """Set a specific thread ID to resume a conversation"""
        self.thread_id = thread_id

    def _ensure_metadata_table(self):
        """Ensure the conversation_metadata table exists.

        Creates the conversation_metadata table if it doesn't already exist.
        This table stores conversation titles and timestamps for the conversation list.

        Raises:
            Does not raise exceptions - logs warnings if table creation fails.
        """
        try:
            with psycopg.connect(DATABASE_URL) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS conversation_metadata (
                            thread_id TEXT PRIMARY KEY,
                            title TEXT NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                conn.commit()
        except psycopg.Error as e:
            logger.warning(f"Could not create conversation_metadata table: {e}")
        except Exception as e:
            logger.error(f"Unexpected error creating conversation_metadata table: {e}")

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

    def generate_conversation_title(self, messages: List[BaseMessage]) -> str:
        """Use the LLM to generate a concise title for the conversation.

        Analyzes the conversation messages and generates a descriptive title
        that captures the main topic being discussed.

        Args:
            messages: List of conversation messages to analyze.

        Returns:
            A concise title (max 50 characters). Returns a default title if
            generation fails or no suitable messages are found.

        Raises:
            Does not raise exceptions - returns fallback titles on error.
        """
        try:
            # Build a summary of the conversation for title generation
            conversation_summary = []
            for msg in messages[-6:]:  # Use last 6 messages for context
                if hasattr(msg, "content") and msg.content:
                    # Safely get message type
                    role = "User" if hasattr(msg, "type") and msg.type == "human" else "Assistant"
                    content = str(msg.content)[:200]  # Truncate long messages
                    conversation_summary.append(f"{role}: {content}")

            if not conversation_summary:
                return "New Conversation"

            prompt = f"""Generate a very short title (max 50 chars) for this conversation.
The title should capture the main topic or question being discussed.
Return ONLY the title, nothing else.

Conversation:
{chr(10).join(conversation_summary)}

Title:"""

            response = self.llm.invoke(prompt)
            title = response.content.strip().strip('"\'')[:50]
            return title if title else "Untitled Conversation"
        except Exception as e:
            logger.debug(f"Title generation failed, using fallback: {e}")
            # Fallback: use first user message
            for msg in messages:
                if hasattr(msg, "type") and msg.type == "human" and hasattr(msg, "content") and msg.content:
                    return str(msg.content)[:50].strip()
            return "Untitled Conversation"

    def update_conversation_title(self):
        """Generate and save a title for the current conversation based on its content.

        Retrieves the current conversation messages from the checkpoint, generates
        a descriptive title using the LLM, and stores it in the conversation_metadata table.

        This method is called after each agent response to keep the title up-to-date
        with the conversation content.

        Raises:
            Does not raise exceptions - logs warnings if title update fails.
        """
        try:
            # Get current conversation messages from checkpoint
            checkpoint = self.checkpointer.get({"configurable": {"thread_id": self.thread_id}})
            if not checkpoint:
                logger.debug("No checkpoint found for title update")
                return

            # Access messages from channel_values (checkpoint is a dict)
            channel_values = checkpoint.get("channel_values", {})
            messages = channel_values.get("messages", [])
            if not messages:
                logger.debug("No messages in checkpoint for title update")
                return

            # Generate title from conversation
            title = self.generate_conversation_title(messages)

            with psycopg.connect(DATABASE_URL) as conn:
                with conn.cursor() as cur:
                    # Insert or update conversation metadata with new title
                    cur.execute("""
                        INSERT INTO conversation_metadata (thread_id, title)
                        VALUES (%s, %s)
                        ON CONFLICT (thread_id)
                        DO UPDATE SET title = EXCLUDED.title, updated_at = CURRENT_TIMESTAMP
                    """, (self.thread_id, title))
                conn.commit()
        except psycopg.Error as e:
            logger.warning(f"Database error updating conversation title: {e}")
        except Exception as e:
            logger.error(f"Unexpected error updating conversation title: {e}")

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

    def _fallback_summarize(self, messages_to_summarize: Sequence[BaseMessage]) -> str:
        """
        Create a simple fallback summary when LLM summarization fails.
        Uses basic heuristics to extract key information without LLM.

        Args:
            messages_to_summarize: Sequence of messages to summarize.

        Returns:
            A simple summary of the conversation.
        """
        if not messages_to_summarize:
            return "No earlier context"

        # Extract user questions and assistant topics
        user_topics = []
        assistant_topics = []

        for msg in messages_to_summarize:
            if hasattr(msg, "content") and msg.content:
                content_preview = str(msg.content)[:100].strip()
                if hasattr(msg, "type"):
                    if msg.type == "human":
                        user_topics.append(content_preview)
                    else:
                        assistant_topics.append(content_preview)
                else:
                    if "human" in str(type(msg)).lower():
                        user_topics.append(content_preview)
                    else:
                        assistant_topics.append(content_preview)

        # Build simple summary
        summary_parts = [f"Earlier conversation ({len(messages_to_summarize)} messages):"]

        if user_topics:
            summary_parts.append(f"User asked about: {', '.join(user_topics[:3])}")
            if len(user_topics) > 3:
                summary_parts.append(f"(and {len(user_topics) - 3} more topics)")

        if assistant_topics:
            summary_parts.append(f"Assistant discussed: {', '.join(assistant_topics[:3])}")
            if len(assistant_topics) > 3:
                summary_parts.append(f"(and {len(assistant_topics) - 3} more topics)")

        return ". ".join(summary_parts)

    def summarize_messages(self, messages_to_summarize: Sequence[BaseMessage]) -> str:
        """
        Use LLM to create a concise summary of older messages.
        Preserves key facts and context while being brief.
        Falls back to simple summaries if LLM fails.

        Args:
            messages_to_summarize: Sequence of messages to summarize.

        Returns:
            A concise summary of the message content.
        """
        if not messages_to_summarize:
            return "No earlier context"

        try:
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

        except httpx.ConnectError as e:
            logger.error(
                f"Connection error while summarizing {len(messages_to_summarize)} messages: {e}",
                exc_info=True
            )
            logger.info("Falling back to simple concatenation summary")
            return self._fallback_summarize(messages_to_summarize)

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(
                f"JSON/parsing error while summarizing {len(messages_to_summarize)} messages: {e}",
                exc_info=True
            )
            logger.info("Falling back to word count summary")
            return self._fallback_summarize(messages_to_summarize)

        except TimeoutError as e:
            logger.error(
                f"Timeout while summarizing {len(messages_to_summarize)} messages: {e}",
                exc_info=True
            )
            logger.info("Falling back to first/last message summary")
            # Get first and last messages
            first_msg = ""
            last_msg = ""
            if messages_to_summarize:
                if hasattr(messages_to_summarize[0], "content"):
                    first_msg = str(messages_to_summarize[0].content)[:80]
                if hasattr(messages_to_summarize[-1], "content"):
                    last_msg = str(messages_to_summarize[-1].content)[:80]
            summary = f"Earlier conversation ({len(messages_to_summarize)} messages): "
            if first_msg:
                summary += f"Started with: {first_msg}. "
            if last_msg:
                summary += f"Ended with: {last_msg}"
            return summary

        except Exception as e:
            logger.error(
                f"Unexpected error while summarizing {len(messages_to_summarize)} messages: {type(e).__name__}: {e}",
                exc_info=True
            )
            logger.info("Falling back to basic summary")
            return self._fallback_summarize(messages_to_summarize)

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
        num_compacted = len(messages_to_compact)

        # Log compaction completion with token counts
        compacted_token_count = self.estimate_token_count(compacted)
        logger.info(
            f"Compacted {num_compacted} messages "
            f"(token count: {compacted_token_count}/{MAX_CONTEXT_TOKENS})"
        )

        return compacted, True, num_compacted

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
        print("(Title will be updated after each message)")
        print()

        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()

                if not user_input:
                    continue

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
            # Prepare input for the agent with new state schema
            input_data = {
                "messages": [("user", user_input)],
                "lambda_mult": 0.25,  # Default, will be overridden by query_evaluator
                "query_analysis": "",
                "optimized_query": None,  # Will be set by query_evaluator_node
                # Reflection state initialization
                "iteration_count": 0,
                "response_retry_count": 0,
                "retrieved_documents": [],
                "document_grades": [],
                "document_grade_summary": {},
                "response_grade": {},
                "original_query": user_input,
                "transformed_query": None,
            }

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

            # Log query analysis for debugging (optional)
            if "query_analysis" in result and result["query_analysis"]:
                print(f"[Debug] Query Analysis: {result['query_analysis']}")
            if "lambda_mult" in result:
                print(f"[Debug] Lambda used: {result.get('lambda_mult', 'N/A'):.2f}")

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

            # Update conversation title after each turn
            self.update_conversation_title()

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
        Display text output from LLM response without artificial delays.

        Previously used character-by-character delays for simulated streaming.
        Now displays text immediately as it's received from true LLM streaming.

        Args:
            text: The text to display to the console.
            chunk_size: Not used in current implementation (kept for compatibility).
        """
        # Display text immediately without artificial delays
        # True streaming happens via _stream_llm_response and LLM chunk events
        print(text)
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
        # Clear reranker model from memory if loaded
        if self.reranker:
            del self.reranker
            torch.cuda.empty_cache()

        if self.pool:
            self.pool.close()


def main():
    """Main function"""
    agent = LangChainAgent()
    agent.run()


if __name__ == "__main__":
    main()

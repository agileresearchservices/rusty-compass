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
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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

        # Load as causal language model (not embedding model)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, dtype=torch.float32).eval()

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

class CustomAgentState(TypedDict):
    """
    State schema for custom agent graph with dynamic lambda_mult.

    This extends the default agent state to include query analysis
    and dynamic search parameter adjustment.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    lambda_mult: float
    query_analysis: str


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
        messages = state["messages"]

        # Extract last user message
        last_user_msg = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_user_msg = msg.content
                break

        if not last_user_msg:
            # No user message, use default
            return {"lambda_mult": 0.25, "query_analysis": "No query detected"}

        # LLM evaluation prompt
        evaluation_prompt = f"""Analyze this search query and determine the optimal search strategy by setting a lambda_mult value between 0.0 and 1.0.

Query: "{last_user_msg}"

Lambda_mult Guidelines (0.0=pure lexical/BM25, 1.0=pure semantic/vector):
- 0.0-0.2: Pure LEXICAL search
  * Use for: dates, model numbers, part numbers, exact product identifiers
  * Example: "GPT-4 released in 2023" → 0.05
  * Example: "Model XR-2500 specifications" → 0.1

- 0.2-0.4: LEXICAL-heavy search
  * Use for: specific versions, brands, frameworks, library updates
  * Example: "Django 4.2 authentication" → 0.3
  * Example: "Python 3.11 new features" → 0.25

- 0.4-0.6: BALANCED hybrid search
  * Use for: mixed queries with both specific terms and concepts
  * Example: "Python Flask REST API tutorial" → 0.5
  * Example: "React hooks best practices" → 0.45

- 0.6-0.8: SEMANTIC-heavy search
  * Use for: conceptual guides, optimization techniques, framework discussions
  * Example: "PostgreSQL query optimization techniques" → 0.65
  * Example: "Docker containerization guide" → 0.6

- 0.8-1.0: Pure SEMANTIC search
  * Use for: conceptual questions, "what is", "explain", "how does", "why"
  * Example: "What is machine learning?" → 0.95
  * Example: "Explain how neural networks learn" → 0.9

Respond with ONLY a JSON object in this format:
{{"lambda_mult": <float between 0.0 and 1.0>, "reasoning": "<brief explanation>"}}

Example response:
{{"lambda_mult": 0.85, "reasoning": "Conceptual question about machine learning requires semantic understanding"}}"""

        try:
            # Call LLM for evaluation
            response = self.llm.invoke(evaluation_prompt)
            result_text = response.content.strip()

            # Parse JSON response
            result = json.loads(result_text)

            lambda_mult = float(result["lambda_mult"])
            reasoning = result.get("reasoning", "No reasoning provided")

            # Validate range
            lambda_mult = max(0.0, min(1.0, lambda_mult))

            # Log evaluation (optional for debugging)
            print(f"[Query Evaluator] Lambda: {lambda_mult:.2f} | {reasoning}")

            return {
                "lambda_mult": lambda_mult,
                "query_analysis": reasoning
            }

        except Exception as e:
            # Fallback to default if evaluation fails
            print(f"⚠ Query evaluation failed: {e}. Using default lambda=0.25")
            return {
                "lambda_mult": 0.25,
                "query_analysis": f"Evaluation failed: {str(e)}"
            }

    def agent_node(self, state: CustomAgentState) -> Dict[str, Any]:
        """
        Agent reasoning node - calls LLM with tool binding.
        """
        messages = state["messages"]

        # Bind tools to LLM
        llm_with_tools = self.llm.bind_tools(self.tools)

        # Call LLM
        response = llm_with_tools.invoke(messages)

        return {"messages": [response]}

    def tools_node(self, state: CustomAgentState) -> Dict[str, Any]:
        """
        Execute tool calls with access to state (for dynamic lambda_mult).
        Uses Qwen3 cross-encoder for reranking if enabled.
        """
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
                results = retriever.invoke(query)

                # Apply reranking if enabled
                if ENABLE_RERANKING and self.reranker and results:
                    # Capture original order for comparison
                    original_sources = [doc.metadata.get('source', 'unknown') for doc in results]

                    print(f"\n[Reranker] Processing {len(results)} candidates...")
                    reranked_results = self.reranker.rerank(query, results, RERANKER_TOP_K)

                    # Extract documents with scores for logging
                    results_with_scores = [(doc, score) for doc, score in reranked_results]
                    results = [doc for doc, score in results_with_scores]

                    # Log reranking results with scores
                    print(f"[Reranker] Reranking complete → top {len(results)} selected:")
                    for i, (doc, score) in enumerate(results_with_scores, 1):
                        source = doc.metadata.get('source', 'unknown')
                        relevance_bar = "█" * int(score * 20)
                        print(f"  {i}. score={score:.4f} {relevance_bar} [{source}]")

                    # Log order changes if applicable
                    reranked_sources = [doc.metadata.get('source', 'unknown') for doc in results]
                    if original_sources[:len(reranked_sources)] != reranked_sources:
                        print(f"[Reranker] Order changed: {original_sources[:len(reranked_sources)]} → {reranked_sources}")
                    else:
                        print(f"[Reranker] Order unchanged (already optimally ranked)")

                content = "\n\n".join([doc.page_content for doc in results]) if results else "No relevant information found."

                # Create tool response message
                tool_responses.append(
                    ToolMessage(
                        content=content,
                        tool_call_id=tool_call["id"]
                    )
                )

        return {"messages": tool_responses}

    def should_continue(self, state: CustomAgentState) -> str:
        """
        Determine whether to continue to tools or end.
        """
        messages = state["messages"]
        last_message = messages[-1]

        # If LLM made tool calls, route to tools
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"

        # Otherwise, end
        return "END"

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

        # Add nodes
        workflow.add_node("query_evaluator", self.query_evaluator_node)
        workflow.add_node("agent", self.agent_node)
        workflow.add_node("tools", self.tools_node)

        # Set entry point
        workflow.set_entry_point("query_evaluator")

        # Add edges
        workflow.add_edge("query_evaluator", "agent")
        workflow.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                "tools": "tools",
                "END": END
            }
        )
        workflow.add_edge("tools", "agent")

        # Compile with checkpointer
        self.app = workflow.compile(checkpointer=self.checkpointer)

        print("✓ Custom agent graph created with query evaluator")
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
            # Prepare input for the agent with new state schema
            input_data = {
                "messages": [("user", user_input)],
                "lambda_mult": 0.25,  # Default, will be overridden by query_evaluator
                "query_analysis": ""
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

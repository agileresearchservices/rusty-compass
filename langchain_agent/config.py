"""
Configuration constants for LangChain Agent
"""

import os
from psycopg.rows import dict_row

__all__ = [
    # Ollama configuration
    "LLM_MODEL",
    "LLM_TEMPERATURE",
    "EMBEDDINGS_MODEL",
    "OLLAMA_BASE_URL",
    # PostgreSQL configuration
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "POSTGRES_DB",
    "DATABASE_URL",
    "DB_CONNECTION_KWARGS",
    "DB_POOL_MAX_SIZE",
    # PGVector configuration
    "VECTOR_DIMENSION",
    "VECTOR_INDEX_TYPE",
    "VECTOR_SIMILARITY_METRIC",
    "VECTOR_COLLECTION_NAME",
    # Retriever configuration
    "RETRIEVER_K",
    "RETRIEVER_FETCH_K",
    "RETRIEVER_LAMBDA_MULT",
    "RETRIEVER_SEARCH_TYPE",
    # Reranker configuration
    "ENABLE_RERANKING",
    "RERANKER_MODEL",
    "RERANKER_FETCH_K",
    "RERANKER_TOP_K",
    "RERANKER_INSTRUCTION",
    # Agent configuration
    "RETRIEVER_TOOL_NAME",
    "RETRIEVER_TOOL_DESCRIPTION",
    "AGENT_MODEL",
    "REASONING_ENABLED",
    "REASONING_EFFORT",
    # Query evaluation configuration
    "ENABLE_QUERY_EVALUATION",
    "DEFAULT_LAMBDA_MULT",
    "QUERY_EVAL_TIMEOUT_MS",
    # Project paths
    "BASE_DIR",
    "SAMPLE_DOCS_DIR",
    # LangChain documentation source
    "DOCS_REPO_URL",
    "DOCS_CACHE_DIR",
    "DOCS_SOURCE_DIRS",
    # Sample data
    "DEFAULT_THREAD_ID",
    # Conversation compaction
    "ENABLE_COMPACTION",
    "MAX_CONTEXT_TOKENS",
    "COMPACTION_THRESHOLD_PCT",
    "MESSAGES_TO_KEEP_FULL",
    "MIN_MESSAGES_FOR_COMPACTION",
    "TOKEN_CHAR_RATIO",
    # Reflection configuration
    "ENABLE_REFLECTION",
    "ENABLE_DOCUMENT_GRADING",
    "ENABLE_RESPONSE_GRADING",
    "ENABLE_QUERY_TRANSFORMATION",
    "REFLECTION_MAX_ITERATIONS",
    "REFLECTION_MIN_RELEVANT_DOCS",
    "REFLECTION_DOC_SCORE_THRESHOLD",
    "REFLECTION_RESPONSE_SCORE_THRESHOLD",
    "REFLECTION_SHOW_STATUS",
]

# ============================================================================
# OLLAMA CONFIGURATION
# ============================================================================

# LLM Model
LLM_MODEL = "gpt-oss:20b"
LLM_TEMPERATURE = 0

# Embeddings Model
EMBEDDINGS_MODEL = "nomic-embed-text:latest"

# Ollama Base URL (adjust if Ollama runs on different host/port)
OLLAMA_BASE_URL = "http://localhost:11434"

# ============================================================================
# POSTGRES CONFIGURATION
# ============================================================================

# Database connection details
POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = "postgres"
POSTGRES_HOST = "localhost"
POSTGRES_PORT = 5432
POSTGRES_DB = "langchain_agent"

# Full connection string for PostgresSaver
DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# Connection pool settings
DB_CONNECTION_KWARGS = {
    "autocommit": True,
    "prepare_threshold": 0,
    "row_factory": dict_row,  # Required for PostgresSaver
}
DB_POOL_MAX_SIZE = 20

# ============================================================================
# PGVECTOR CONFIGURATION
# ============================================================================

# Vector embedding dimension (nomic-embed-text produces 768-dimensional vectors)
VECTOR_DIMENSION = 768

# Vector index type: "ivfflat" (faster queries, slower indexing) or "hnsw" (slower queries, faster updates)
# Use IVFFlat for static knowledge bases where query speed is priority
VECTOR_INDEX_TYPE = "ivfflat"

# Vector similarity metric: "cosine", "l2", or "inner_product"
VECTOR_SIMILARITY_METRIC = "cosine"

# Collection name for vector storage
# Use "langchain_docs" for LangChain documentation, "local_knowledge" for sample docs
VECTOR_COLLECTION_NAME = "langchain_docs"

# ============================================================================
# RETRIEVER CONFIGURATION
# ============================================================================

# Number of documents to retrieve from vector store
RETRIEVER_K = 4

# Number of documents to fetch before filtering (for hybrid search)
# Increased to 30 to provide more candidates for reranking
RETRIEVER_FETCH_K = 30

# Lambda multiplier for hybrid search (0.0 = pure lexical, 1.0 = pure semantic/dense)
RETRIEVER_LAMBDA_MULT = 0.25

# Default search type: "similarity" (vector-only) or "hybrid" (vector + lexical using RRF)
RETRIEVER_SEARCH_TYPE = "hybrid"

# ============================================================================
# RERANKER CONFIGURATION (LangChain ContextualCompressionRetriever)
# ============================================================================

# Enable reranking of hybrid search results using cross-encoder model
ENABLE_RERANKING = True

# Cross-encoder reranker model from HuggingFace
# Options:
#   - "BAAI/bge-reranker-v2-m3" (recommended, fast and accurate)
#   - "BAAI/bge-reranker-v2-large" (more accurate but slower)
#   - "Qwen/Qwen3-Reranker-8B" (multilingual, state-of-the-art)
#   - "Qwen/Qwen3-Reranker-4B" (faster, smaller)
RERANKER_MODEL = "Qwen/Qwen3-Reranker-8B"

# Number of candidates to fetch before reranking
RERANKER_FETCH_K = 15

# Final number of documents to return after reranking
RERANKER_TOP_K = 4

# Custom instruction for reranker (domain-specific, affects 1-5% performance)
# If None, uses default instruction
RERANKER_INSTRUCTION = None

# ============================================================================
# QUERY EVALUATOR CONFIGURATION
# ============================================================================

# Enable intelligent query evaluation for dynamic lambda_mult adjustment
ENABLE_QUERY_EVALUATION = True

# Default lambda_mult when evaluation is disabled or fails
DEFAULT_LAMBDA_MULT = 0.25

# Query evaluation timeout (milliseconds) - max time to wait for LLM evaluation
QUERY_EVAL_TIMEOUT_MS = 3000  # 3 seconds max for LLM evaluation

# ============================================================================
# AGENT CONFIGURATION
# ============================================================================

# Retriever tool settings
RETRIEVER_TOOL_NAME = "knowledge_base"
RETRIEVER_TOOL_DESCRIPTION = "Search for information in the local document index."

# Agent settings
AGENT_MODEL = LLM_MODEL
REASONING_ENABLED = True
REASONING_EFFORT = "low"  # Options: "low", "medium", "high"

# ============================================================================
# PROJECT PATHS
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DOCS_DIR = os.path.join(os.path.dirname(BASE_DIR), "sample_docs")

# ============================================================================
# LANGCHAIN DOCUMENTATION SOURCE
# ============================================================================

# GitHub repository containing LangChain/LangGraph/LangSmith documentation
DOCS_REPO_URL = "https://github.com/langchain-ai/docs.git"

# Local cache directory for cloned documentation (outside langchain_agent/)
DOCS_CACHE_DIR = os.path.join(os.path.dirname(BASE_DIR), ".langchain_docs_cache")

# Source directories within the docs repo to process
DOCS_SOURCE_DIRS = ["src/oss", "src/langsmith"]

# ============================================================================
# SAMPLE DATA
# ============================================================================

# Default conversation thread ID (can be overridden per conversation)
DEFAULT_THREAD_ID = "default_thread"

# ============================================================================
# CONVERSATION COMPACTION (Smart Context Management)
# ============================================================================

# Enable automatic conversation compaction
ENABLE_COMPACTION = True

# Maximum estimated tokens in context (conservative estimate for gpt-oss:20b)
MAX_CONTEXT_TOKENS = 3000

# Trigger compaction at this percentage of max context (0.8 = 80%)
COMPACTION_THRESHOLD_PCT = 0.8

# Keep this many recent messages uncompacted (always preserved in full)
MESSAGES_TO_KEEP_FULL = 10

# Minimum number of messages before considering compaction
MIN_MESSAGES_FOR_COMPACTION = 20

# Token estimation (1 token ≈ 4 characters, conservative)
TOKEN_CHAR_RATIO = 4

# ============================================================================
# REFLECTION CONFIGURATION (Agent Self-Improvement)
# ============================================================================

# Enable reflection loop for document grading and response quality evaluation
ENABLE_REFLECTION = True

# Enable individual reflection components (only used if ENABLE_REFLECTION is True)
ENABLE_DOCUMENT_GRADING = True      # Grade retrieved documents for relevance
ENABLE_RESPONSE_GRADING = True      # Evaluate final response quality
ENABLE_QUERY_TRANSFORMATION = True  # Rewrite query if documents are poor

# Maximum number of retrieval iterations (1 = no retries, 2 = one retry)
REFLECTION_MAX_ITERATIONS = 2

# Minimum number of relevant documents required to pass grading
REFLECTION_MIN_RELEVANT_DOCS = 2

# Minimum score threshold for document relevance (0.0-1.0)
REFLECTION_DOC_SCORE_THRESHOLD = 0.5

# Minimum score threshold for response quality (0.0-1.0)
REFLECTION_RESPONSE_SCORE_THRESHOLD = 0.6

# Display reflection status in console output
REFLECTION_SHOW_STATUS = True

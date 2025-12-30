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
    "ENABLE_QUERY_EVAL_CACHE",
    "QUERY_EVAL_CACHE_MAX_SIZE",
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
    "DOCUMENT_GRADING_BATCH_SIZE",
    "DOCUMENT_GRADING_CONFIDENCE_THRESHOLD",
    # Token budget tracking
    "REFLECTION_MAX_TOKENS_TOTAL",
    "REFLECTION_TOKEN_WARNING_THRESHOLD",
    # Response grading confidence-based early stopping
    "RESPONSE_GRADING_HIGH_CONFIDENCE_THRESHOLD",
    "RESPONSE_GRADING_LOW_CONFIDENCE_RETRY",
    # Advanced reflection tuning (Tier 3)
    "REFLECTION_STRATEGY",
    "REFLECTION_ENABLE_CONFIDENCE_EXIT",
    "REFLECTION_TRACK_TRANSFORMATION_EFFECTIVENESS",
    # Observable agent streaming configuration
    "ENABLE_ASYNC_STREAMING",
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
#   - "BAAI/bge-reranker-v2-m3" (default, multilingual, ~2.3GB, fast)
#   - "BAAI/bge-reranker-v2-large" (more accurate, ~1.2GB)
#   - "BAAI/bge-reranker-base" (smaller, faster, ~440MB)
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Number of candidates to fetch before reranking
RERANKER_FETCH_K = 15

# Final number of documents to return after reranking
RERANKER_TOP_K = 4

# ============================================================================
# QUERY EVALUATOR CONFIGURATION
# ============================================================================

# Enable intelligent query evaluation for dynamic lambda_mult adjustment
ENABLE_QUERY_EVALUATION = True

# Default lambda_mult when evaluation is disabled or fails
DEFAULT_LAMBDA_MULT = 0.25

# Query evaluation timeout (milliseconds) - max time to wait for LLM evaluation
QUERY_EVAL_TIMEOUT_MS = 3000  # 3 seconds max for LLM evaluation

# Query evaluator caching configuration
ENABLE_QUERY_EVAL_CACHE = True
QUERY_EVAL_CACHE_MAX_SIZE = 100

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
REFLECTION_MIN_RELEVANT_DOCS = 1

# Minimum score threshold for document relevance (0.0-1.0)
REFLECTION_DOC_SCORE_THRESHOLD = 0.3

# Minimum score threshold for response quality (0.0-1.0)
REFLECTION_RESPONSE_SCORE_THRESHOLD = 0.6

# Display reflection status in console output
REFLECTION_SHOW_STATUS = True

# Batch size for document grading (process this many documents per LLM call)
# If more documents than this, they will be processed in batches
# Set to 1 to disable batch grading and grade documents individually
DOCUMENT_GRADING_BATCH_SIZE = 8

# Skip document grading (skip LLM calls) if average reranker confidence exceeds this threshold
# High reranker scores (>0.95) from Qwen3-Reranker already indicate relevance
# Set to 1.0 to always grade, 0.0 to never grade (optimization: ~6s savings on 60% of queries)
DOCUMENT_GRADING_CONFIDENCE_THRESHOLD = 0.95

# ============================================================================
# TOKEN BUDGET TRACKING (Prevent Runaway Costs)
# ============================================================================

# Hard limit on total tokens used in a conversation (prevents runaway costs)
# After reaching this limit, agent returns error and won't perform retries
REFLECTION_MAX_TOKENS_TOTAL = 50000

# Soft warning threshold - warns user when approaching limit (recommended: 80% of max)
# Allows agent to continue but with warnings
REFLECTION_TOKEN_WARNING_THRESHOLD = 40000

# ============================================================================
# RESPONSE GRADING CONFIDENCE-BASED EARLY STOPPING
# ============================================================================

# Confidence threshold for early stopping with passing grade (0.0-1.0)
# If grade is "pass" AND confidence > this value, skip retry loop and return response
# Example: 0.85 means "if we're 85% confident this is a good response, don't retry"
RESPONSE_GRADING_HIGH_CONFIDENCE_THRESHOLD = 0.85

# Confidence threshold for forced retry (0.0-1.0)
# If confidence < this value AND retry_count < max, always retry regardless of grade
# Example: 0.5 means "if we're less than 50% confident, force a retry"
RESPONSE_GRADING_LOW_CONFIDENCE_RETRY = 0.5

# ============================================================================
# ADVANCED REFLECTION TUNING (TIER 3 OPTIMIZATIONS)
# ============================================================================

# Reflection strategy: "strict" (high precision), "moderate" (balanced), "lenient" (fast)
# - "strict": Original thresholds (2/4 docs, 0.5+ score) - highest precision, more retries
# - "moderate": Balanced (1/4 docs, 0.3+ score) - good balance, fewer retries - RECOMMENDED
# - "lenient": Fast mode (1/4 docs, 0.2+ score) - fastest, may miss issues
REFLECTION_STRATEGY = "moderate"

# Enable confidence-based early exit (auto-pass if 2+ docs score 0.7+)
# Skips further retries when we have strong signal - saves time on queries with good matches
REFLECTION_ENABLE_CONFIDENCE_EXIT = True

# Track transformation effectiveness (stop if transformation makes things worse)
# Detects when query rewriting degrades results and avoids wasteful retries
REFLECTION_TRACK_TRANSFORMATION_EFFECTIVENESS = True

# ============================================================================
# OBSERVABLE AGENT STREAMING CONFIGURATION
# ============================================================================

# Enable incremental async streaming for improved responsiveness (EXPERIMENTAL)
# When False (default): Backward compatible behavior - waits for entire node completion
#   - Runs entire graph in executor, collects all timing info after completion
#   - More blocking but stable behavior
# When True: Improved streaming with incremental event emission
#   - Emits NodeStartEvent immediately when node begins execution
#   - Processes events as they complete instead of waiting for full node
#   - Emits NodeEndEvent with accurate timing after processing
#   - TRADEOFF: Timing may be slightly less accurate than legacy mode, but
#     provides better UI responsiveness and prevents async event loop blocking
ENABLE_ASYNC_STREAMING = True

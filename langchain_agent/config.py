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
    # ChromaDB configuration
    "CHROMA_DB_PATH",
    "CHROMA_COLLECTION_NAME",
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
    # Sample data
    "DEFAULT_THREAD_ID",
    # Conversation compaction
    "ENABLE_COMPACTION",
    "MAX_CONTEXT_TOKENS",
    "COMPACTION_THRESHOLD_PCT",
    "MESSAGES_TO_KEEP_FULL",
    "MIN_MESSAGES_FOR_COMPACTION",
    "TOKEN_CHAR_RATIO",
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
# CHROMADB CONFIGURATION
# ============================================================================

# ChromaDB persistence directory
CHROMA_DB_PATH = "./chroma_db"

# Collection name for storing documents
CHROMA_COLLECTION_NAME = "local_knowledge"

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

# Collection name for vector storage (same as ChromaDB for migration compatibility)
VECTOR_COLLECTION_NAME = "local_knowledge"

# ============================================================================
# RETRIEVER CONFIGURATION
# ============================================================================

# Number of documents to retrieve from vector store
RETRIEVER_K = 4

# Number of documents to fetch before filtering (for hybrid search)
RETRIEVER_FETCH_K = 20

# Lambda multiplier for hybrid search (0.0 = pure lexical, 1.0 = pure semantic/dense)
RETRIEVER_LAMBDA_MULT = 0.25

# Default search type: "similarity" (vector-only) or "hybrid" (vector + lexical using RRF)
RETRIEVER_SEARCH_TYPE = "hybrid"

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

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
    # Agent configuration
    "RETRIEVER_TOOL_NAME",
    "RETRIEVER_TOOL_DESCRIPTION",
    "AGENT_MODEL",
    "REASONING_ENABLED",
    "REASONING_EFFORT",
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
# AGENT CONFIGURATION
# ============================================================================

# Retriever tool settings
RETRIEVER_TOOL_NAME = "knowledge_base"
RETRIEVER_TOOL_DESCRIPTION = "Search for information in the local document index."

# Agent settings
AGENT_MODEL = LLM_MODEL
REASONING_ENABLED = True
REASONING_EFFORT = "medium"  # Options: "low", "medium", "high"

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

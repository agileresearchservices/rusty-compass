#!/usr/bin/env python3
"""
Unified Setup Script for LangChain Agent
Initializes PostgreSQL database, loads sample data, and pulls Ollama models
This is the single entry point for complete system setup from scratch
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import List, Tuple

import psycopg
from psycopg import sql
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_ollama import OllamaEmbeddings

from config import (
    DATABASE_URL,
    DB_CONNECTION_KWARGS,
    DB_POOL_MAX_SIZE,
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
    POSTGRES_DB,
    LLM_MODEL,
    EMBEDDINGS_MODEL,
    RERANKER_MODEL,
    OLLAMA_BASE_URL,
    VECTOR_COLLECTION_NAME,
    SAMPLE_DOCS_DIR,
)

# Document chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# ============================================================================
# STEP 1: POSTGRESQL DATABASE SETUP
# ============================================================================

def create_database():
    """Create the langchain_agent database if it doesn't exist"""
    print("\n[1/7] Creating database...")

    try:
        # Connect to the default postgres database to create our database
        admin_conn_string = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/postgres"

        with psycopg.connect(admin_conn_string) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                # Check if database exists
                cur.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s",
                    (POSTGRES_DB,)
                )
                if not cur.fetchone():
                    cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(POSTGRES_DB)))
                    print(f"      ✓ Database '{POSTGRES_DB}' created")
                else:
                    print(f"      ✓ Database '{POSTGRES_DB}' already exists")
    except Exception as e:
        print(f"      ✗ Error creating database: {e}")
        raise


def verify_connection():
    """Verify connection to the database"""
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version()")
                version = cur.fetchone()[0]
                postgres_version = version.split(',')[0]
                print(f"      ✓ Connected to: {postgres_version}")
    except Exception as e:
        print(f"      ✗ Error connecting to database: {e}")
        raise


def enable_pgvector_extension():
    """Enable the pgvector extension in the database"""
    print("\n[2/7] Enabling PGVector extension...")

    try:
        with psycopg.connect(DATABASE_URL) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                print("      ✓ PGVector extension enabled")
    except Exception as e:
        print(f"      ✗ Error enabling pgvector extension: {e}")
        raise


def create_vector_tables():
    """Create tables for vector storage with PGVector"""
    print("\n[3/7] Creating database tables...")

    try:
        with psycopg.connect(DATABASE_URL) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                # Create collections table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS collections (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        description TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Create documents table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        content TEXT NOT NULL,
                        embedding vector(768) NOT NULL,
                        metadata JSONB,
                        collection_id TEXT REFERENCES collections(id),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Create document_chunks table for semantic chunking
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS document_chunks (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                        chunk_index INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        embedding vector(768) NOT NULL,
                        content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                print("      ✓ Database tables created")
    except Exception as e:
        print(f"      ✗ Error creating tables: {e}")
        raise


def create_vector_indexes():
    """Create indexes for vector similarity search"""
    print("\n[4/7] Creating database indexes...")

    try:
        with psycopg.connect(DATABASE_URL) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                # Create IVFFlat index on documents table
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS documents_embedding_idx
                    ON documents USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """)

                # Create IVFFlat index on document_chunks table
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
                    ON document_chunks USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """)

                # Create GIN index for full-text search
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS document_chunks_content_tsv_idx
                    ON document_chunks USING GIN(content_tsv)
                """)

                # Create regular index on collection_id for filtering
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS documents_collection_id_idx
                    ON documents(collection_id)
                """)

                # Analyze table to update statistics
                cur.execute("ANALYZE document_chunks")
                print("      ✓ Indexes created (IVFFlat vector, GIN full-text)")
    except Exception as e:
        print(f"      ✗ Error creating indexes: {e}")
        raise


def init_checkpoint_tables():
    """Initialize the PostgresSaver checkpoint tables"""
    try:
        # Create connection pool
        connection_kwargs = DB_CONNECTION_KWARGS.copy()
        pool = ConnectionPool(
            conninfo=DATABASE_URL,
            max_size=DB_POOL_MAX_SIZE,
            kwargs=connection_kwargs
        )

        # Initialize the checkpointer (creates tables if they don't exist)
        checkpointer = PostgresSaver(pool)
        checkpointer.setup()
        pool.close()
        print("      ✓ Conversation checkpointer tables created")

    except Exception as e:
        print(f"      ✗ Error initializing checkpoint tables: {e}")
        raise


def init_metadata_table():
    """Initialize conversation metadata table"""
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_metadata (
                        thread_id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                print("      ✓ Conversation metadata table created")
    except Exception as e:
        print(f"      ✗ Error initializing metadata table: {e}")
        raise


# ============================================================================
# STEP 2: OLLAMA MODEL PULLING
# ============================================================================

def pull_ollama_model(model_name: str) -> bool:
    """Pull a model from Ollama (assumes ollama command is available)"""
    try:
        print(f"      Pulling {model_name}...")
        result = subprocess.run(
            ["ollama", "pull", model_name],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        if result.returncode == 0:
            print(f"      ✓ {model_name} pulled successfully")
            return True
        else:
            print(f"      ⚠ Ollama returned error code {result.returncode}")
            if result.stderr:
                print(f"         Error: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"      ✗ Timeout pulling {model_name} (model too large?)")
        return False
    except FileNotFoundError:
        print(f"      ✗ ollama command not found. Is Ollama installed?")
        return False
    except Exception as e:
        print(f"      ⚠ Error pulling {model_name}: {e}")
        return False


def setup_ollama_models():
    """Pull required Ollama models"""
    print("\n[5/7] Setting up Ollama models...")
    print(f"      Make sure Ollama is running: ollama serve")

    models_to_pull = [
        (LLM_MODEL, "LLM model"),
        (EMBEDDINGS_MODEL, "Embeddings model"),
        (RERANKER_MODEL, "Reranker model"),
    ]

    models_pulled = 0
    for model_name, description in models_to_pull:
        print(f"\n      [{description}] {model_name}")
        if pull_ollama_model(model_name):
            models_pulled += 1

    if models_pulled < len(models_to_pull):
        print(f"\n      ⚠ Only {models_pulled}/{len(models_to_pull)} models pulled")
        print("      You can manually pull missing models:")
        for model_name, description in models_to_pull:
            print(f"        ollama pull {model_name}")
        return False

    return True


# ============================================================================
# STEP 3: SAMPLE DATA LOADING
# ============================================================================

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start = end - overlap
        if end == len(text):
            break

    return chunks


def load_documents_from_directory(docs_dir: str) -> List[Tuple[str, str]]:
    """Load all text documents from a directory"""
    documents = []
    docs_path = Path(docs_dir)

    if not docs_path.exists():
        print(f"      ✗ Documents directory not found: {docs_dir}")
        return documents

    txt_files = list(docs_path.glob("*.txt"))
    if not txt_files:
        print(f"      ⚠ No .txt files found in {docs_dir}")
        return documents

    for file_path in txt_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                documents.append((file_path.name, content))
                print(f"      ✓ Loaded: {file_path.name} ({len(content)} chars)")
        except Exception as e:
            print(f"      ✗ Error loading {file_path.name}: {e}")

    return documents


def load_sample_data():
    """Load and chunk sample documents into PostgreSQL vector store"""
    print("\n[6/7] Loading sample documents...")

    # Load documents
    documents = load_documents_from_directory(SAMPLE_DOCS_DIR)

    if not documents:
        print(f"      ⚠ No documents found in {SAMPLE_DOCS_DIR}")
        print("      Sample data loading skipped")
        return 0

    print(f"\n      Loading {len(documents)} document(s) with embeddings...")

    try:
        # Initialize embeddings
        embeddings = OllamaEmbeddings(
            model=EMBEDDINGS_MODEL,
            base_url=OLLAMA_BASE_URL,
        )

        # Connect to PostgreSQL
        with psycopg.connect(DATABASE_URL) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                # Create collection
                cur.execute(
                    """
                    INSERT INTO collections (id, name, description)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (VECTOR_COLLECTION_NAME, VECTOR_COLLECTION_NAME, "Sample documents")
                )

                total_chunks = 0

                for filename, content in documents:
                    print(f"\n      Processing: {filename}")

                    # Insert full document
                    cur.execute(
                        """
                        INSERT INTO documents (content, embedding, metadata, collection_id)
                        VALUES (%s, %s, %s, %s)
                        RETURNING id
                        """,
                        (
                            content,
                            "[0" + ",0" * 767 + "]",  # Placeholder embedding
                            json.dumps({"source": filename}),
                            VECTOR_COLLECTION_NAME,
                        )
                    )
                    doc_id = cur.fetchone()[0]

                    # Chunk the document
                    chunks = chunk_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
                    print(f"        → Split into {len(chunks)} chunks")

                    # Generate all embeddings in batch
                    chunk_list: List[str] = list(chunks)
                    print(f"        → Generating embeddings...")

                    try:
                        # Batch embed all chunks at once
                        chunk_embeddings: List[List[float]] = embeddings.embed_documents(chunk_list)
                        print(f"        ✓ Generated {len(chunk_embeddings)} embeddings")

                        # Insert chunks with pre-generated embeddings
                        for chunk_idx, (chunk, embedding) in enumerate(zip(chunk_list, chunk_embeddings)):
                            embedding_str = "[" + ",".join(str(float(e)) for e in embedding) + "]"
                            cur.execute(
                                """
                                INSERT INTO document_chunks (document_id, chunk_index, content, embedding)
                                VALUES (%s, %s, %s, %s)
                                """,
                                (doc_id, chunk_idx, chunk, embedding_str)
                            )
                            total_chunks += 1

                    except Exception as e:
                        # Fallback to sequential embedding if batch fails
                        print(f"        ⚠ Batch embedding failed: {e}")
                        print(f"        Falling back to sequential...")
                        for chunk_idx, chunk in enumerate(chunk_list):
                            try:
                                embedding = embeddings.embed_query(chunk)
                                embedding_str = "[" + ",".join(str(float(e)) for e in embedding) + "]"
                                cur.execute(
                                    """
                                    INSERT INTO document_chunks (document_id, chunk_index, content, embedding)
                                    VALUES (%s, %s, %s, %s)
                                    """,
                                    (doc_id, chunk_idx, chunk, embedding_str)
                                )
                                total_chunks += 1
                            except Exception as inner_e:
                                print(f"        ⚠ Error embedding chunk {chunk_idx}: {inner_e}")

                    print(f"        ✓ Loaded {len(chunks)} chunks")

                print(f"\n      ✓ Successfully loaded {total_chunks} chunks")
                return total_chunks

    except Exception as e:
        print(f"      ✗ Error loading data: {e}")
        raise


def verify_data_load() -> bool:
    """Verify that documents and chunks were loaded"""
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                # Count documents
                cur.execute(
                    f"SELECT COUNT(*) FROM documents WHERE collection_id = '{VECTOR_COLLECTION_NAME}'"
                )
                doc_count = cur.fetchone()[0]

                # Count chunks
                cur.execute(
                    f"""
                    SELECT COUNT(*) FROM document_chunks
                    WHERE document_id IN (
                        SELECT id FROM documents WHERE collection_id = '{VECTOR_COLLECTION_NAME}'
                    )
                    """
                )
                chunk_count = cur.fetchone()[0]

                # Count chunks with embeddings
                cur.execute(
                    f"""
                    SELECT COUNT(*) FROM document_chunks
                    WHERE document_id IN (
                        SELECT id FROM documents WHERE collection_id = '{VECTOR_COLLECTION_NAME}'
                    ) AND embedding IS NOT NULL
                    """
                )
                embedding_count = cur.fetchone()[0]

                print(f"      ✓ Documents: {doc_count}")
                print(f"      ✓ Chunks: {chunk_count}")
                print(f"      ✓ Chunks with embeddings: {embedding_count}")

                return doc_count > 0 and chunk_count > 0 and embedding_count == chunk_count

    except Exception as e:
        print(f"      ✗ Error verifying load: {e}")
        return False


# ============================================================================
# MAIN SETUP ORCHESTRATION
# ============================================================================

def main():
    """Run complete setup process"""
    print("\n" + "=" * 70)
    print("LANGCHAIN AGENT - COMPLETE SETUP")
    print("=" * 70)
    print("\nThis script will:")
    print("  1. Create PostgreSQL database")
    print("  2. Initialize tables with vector indexes")
    print("  3. Pull Ollama models (LLM, embeddings, reranker)")
    print("  4. Load sample documents with embeddings")
    print("\n" + "=" * 70)

    try:
        # Step 1: PostgreSQL Setup
        create_database()
        verify_connection()
        enable_pgvector_extension()
        create_vector_tables()
        create_vector_indexes()
        init_checkpoint_tables()
        init_metadata_table()

        # Step 2: Ollama Models (optional - can fail if Ollama not running)
        models_ok = setup_ollama_models()

        # Step 3: Sample Data Loading
        chunk_count = load_sample_data()
        verify_data_load()

        # Summary
        print("\n" + "=" * 70)
        print("✓ SETUP COMPLETE!")
        print("=" * 70)
        print("\nYou can now run the agent:")
        print("  python main.py")
        print("\nExample queries:")
        print("  - What is Python programming?")
        print("  - How does machine learning work?")
        print("  - Tell me about web development")
        print("\n" + "=" * 70)

        return 0

    except Exception as e:
        print("\n" + "=" * 70)
        print(f"✗ SETUP FAILED: {e}")
        print("=" * 70)
        print("\nTroubleshooting:")
        print("1. PostgreSQL: Ensure Docker container is running")
        print("   docker compose up -d")
        print("2. Ollama: Start Ollama service")
        print("   ollama serve")
        print("3. Connection: Verify config.py settings")
        print("4. Models: If Ollama fails, manually pull:")
        print(f"   ollama pull {LLM_MODEL}")
        print(f"   ollama pull {EMBEDDINGS_MODEL}")
        print(f"   ollama pull {RERANKER_MODEL}")
        print("\n" + "=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())

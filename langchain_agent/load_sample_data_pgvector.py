#!/usr/bin/env python3
"""
Sample Data Loader for PostgreSQL with PGVector
Loads and chunks sample documents into PostgreSQL vector store
"""

import os
import sys
import json
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
import psycopg
from config import (
    EMBEDDINGS_MODEL,
    DATABASE_URL,
    VECTOR_COLLECTION_NAME,
    SAMPLE_DOCS_DIR,
    OLLAMA_BASE_URL,
)

# Chunk size - split documents into ~1000 character chunks with overlap
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_documents_from_directory(docs_dir: str) -> list[tuple[str, str]]:
    """
    Load all text documents from a directory.
    Returns list of (filename, content) tuples.
    """
    documents = []
    docs_path = Path(docs_dir)

    if not docs_path.exists():
        print(f"✗ Documents directory not found: {docs_dir}")
        return documents

    txt_files = list(docs_path.glob("*.txt"))
    if not txt_files:
        print(f"✗ No .txt files found in {docs_dir}")
        return documents

    for file_path in txt_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                documents.append((file_path.name, content))
                print(f"✓ Loaded: {file_path.name} ({len(content)} chars)")
        except Exception as e:
            print(f"✗ Error loading {file_path.name}: {e}")

    return documents


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks.
    """
    chunks = []
    start = 0

    while start < len(text):
        # Take chunk_size characters from start position
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]

        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk)

        # Move start position, accounting for overlap
        start = end - overlap

        # If we're at the end, break
        if end == len(text):
            break

    return chunks


def load_into_vector_store(documents: list[tuple[str, str]]) -> int:
    """
    Load and chunk documents into PostgreSQL vector store.
    Stores chunks in document_chunks table with embeddings.
    Returns the number of chunks loaded.
    """
    print(f"\nInitializing embeddings model: {EMBEDDINGS_MODEL}")
    print("(First run may take a moment...)\n")

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
                    (VECTOR_COLLECTION_NAME, VECTOR_COLLECTION_NAME, "Sample documents with chunks")
                )

                # Load and chunk documents
                total_chunks = 0

                for filename, content in documents:
                    print(f"\nProcessing: {filename}")

                    # Insert full document
                    cur.execute(
                        """
                        INSERT INTO documents (content, embedding, metadata, collection_id)
                        VALUES (%s, %s, %s, %s)
                        RETURNING id
                        """,
                        (
                            content,
                            "[0" + ",0" * 767 + "]",  # Placeholder embedding (will be filled by chunks)
                            json.dumps({"source": filename}),
                            VECTOR_COLLECTION_NAME,
                        )
                    )
                    doc_id = cur.fetchone()[0]

                    # Chunk the document
                    chunks = chunk_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
                    print(f"  → Split into {len(chunks)} chunks")

                    # Insert chunks with embeddings
                    for chunk_idx, chunk in enumerate(chunks):
                        try:
                            # Generate embedding for chunk
                            embedding = embeddings.embed_query(chunk)
                            embedding_str = "[" + ",".join(str(float(e)) for e in embedding) + "]"

                            # Insert chunk
                            cur.execute(
                                """
                                INSERT INTO document_chunks (document_id, chunk_index, content, embedding)
                                VALUES (%s, %s, %s, %s)
                                """,
                                (doc_id, chunk_idx, chunk, embedding_str)
                            )
                            total_chunks += 1

                            if (chunk_idx + 1) % 5 == 0:
                                print(f"    ✓ Embedded chunks {chunk_idx + 1}/{len(chunks)}")

                        except Exception as e:
                            print(f"    ⚠ Error embedding chunk {chunk_idx}: {e}")
                            continue

                    print(f"  ✓ Loaded {len(chunks)} chunks")

                print(f"\n✓ Successfully loaded {total_chunks} total chunks into PostgreSQL")
                return total_chunks

    except Exception as e:
        print(f"✗ Error loading into PostgreSQL: {e}")
        raise


def verify_load() -> bool:
    """
    Verify that documents and chunks were loaded.
    """
    print("\nVerifying data load...")

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

                print(f"✓ Documents in database: {doc_count}")
                print(f"✓ Chunks in database: {chunk_count}")
                print(f"✓ Chunks with embeddings: {embedding_count}")

                return doc_count > 0 and chunk_count > 0 and embedding_count == chunk_count

    except Exception as e:
        print(f"✗ Error verifying load: {e}")
        return False


def main():
    """Run the data loading process"""
    print("=" * 70)
    print("Sample Data Loader for PostgreSQL with PGVector")
    print("Chunking documents for semantic search")
    print("=" * 70)
    print()

    try:
        # Load documents
        print(f"Loading documents from: {SAMPLE_DOCS_DIR}\n")
        documents = load_documents_from_directory(SAMPLE_DOCS_DIR)

        if not documents:
            print("\n✗ No documents to load. Exiting.")
            sys.exit(1)

        print(f"\n✓ Loaded {len(documents)} documents")

        # Load and chunk into vector store
        chunk_count = load_into_vector_store(documents)

        # Verify load
        if verify_load():
            print()
            print("=" * 70)
            print("✓ Data loading complete!")
            print("=" * 70)
            print("\nNext steps:")
            print("1. Run the agent: python main.py")
            print("2. Try queries like:")
            print("   - 'Tell me about Python'")
            print("   - 'How does machine learning work?'")
            print("   - 'What is web development?'")
            return 0
        else:
            print("\n⚠ Verification warnings - some chunks may not have embeddings")
            return 1

    except Exception as e:
        print()
        print("=" * 70)
        print(f"✗ Loading failed: {e}")
        print("=" * 70)
        print("\nTroubleshooting:")
        print("1. Ensure PostgreSQL is running and accessible")
        print("2. Ensure tables exist: run `python setup_db.py`")
        print("3. Ensure Ollama is running: ollama serve")
        print(f"4. Verify the embedding model is available: ollama list")
        print(f"5. Check the Ollama endpoint: {OLLAMA_BASE_URL}")
        return 1


if __name__ == "__main__":
    exit(main())

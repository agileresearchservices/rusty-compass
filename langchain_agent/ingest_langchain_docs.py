#!/usr/bin/env python3
"""
LangChain Documentation Ingestion Script

Clones/updates the langchain-ai/docs repository and ingests markdown documentation
into the PostgreSQL vector store for use by the LangChain agent.

Supports:
- LangChain documentation (src/oss/python/langchain)
- LangGraph documentation (src/oss/python/langgraph)
- LangSmith documentation (src/langsmith)

Usage:
    python ingest_langchain_docs.py           # Full clone and ingest
    python ingest_langchain_docs.py --update  # Pull latest and re-ingest
    python ingest_langchain_docs.py --stats   # Show current stats only
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psycopg
from git import Repo, GitCommandError
from langchain_ollama import OllamaEmbeddings

from config import (
    DATABASE_URL,
    DOCS_CACHE_DIR,
    DOCS_REPO_URL,
    DOCS_SOURCE_DIRS,
    EMBEDDINGS_MODEL,
    OLLAMA_BASE_URL,
    VECTOR_COLLECTION_NAME,
)

# Document chunking settings (matching setup.py)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Collection name for LangChain docs
LANGCHAIN_COLLECTION_NAME = "langchain_docs"


def clone_or_update_repo(repo_url: str, cache_dir: str, update: bool = False) -> Path:
    """Clone the documentation repository or update it if it exists."""
    cache_path = Path(cache_dir)

    if cache_path.exists() and (cache_path / ".git").exists():
        if update:
            print(f"      Updating existing repo at {cache_dir}...")
            try:
                repo = Repo(cache_path)
                origin = repo.remotes.origin
                origin.pull()
                print(f"      ✓ Repository updated to latest")
            except GitCommandError as e:
                print(f"      ⚠ Git pull failed: {e}")
                print(f"      Using existing local copy")
        else:
            print(f"      ✓ Using existing repo at {cache_dir}")
        return cache_path
    else:
        print(f"      Cloning {repo_url}...")
        print(f"      This may take a few minutes...")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        Repo.clone_from(repo_url, cache_path, depth=1)
        print(f"      ✓ Repository cloned to {cache_dir}")
        return cache_path


def discover_markdown_files(base_dir: Path, source_dirs: List[str]) -> List[Path]:
    """Find all markdown files in the specified source directories."""
    markdown_files = []

    for source_dir in source_dirs:
        source_path = base_dir / source_dir
        if not source_path.exists():
            print(f"      ⚠ Source directory not found: {source_dir}")
            continue

        # Find all .md and .mdx files
        for pattern in ["**/*.md", "**/*.mdx"]:
            files = list(source_path.glob(pattern))
            markdown_files.extend(files)
            print(f"      Found {len(files)} {pattern.split('.')[-1]} files in {source_dir}")

    # Remove duplicates and sort
    markdown_files = sorted(set(markdown_files))
    return markdown_files


def is_binary_content(content: str) -> bool:
    """
    Detect if content appears to be binary/base64 encoded data.

    Returns True if content looks like binary data that shouldn't be ingested.
    """
    # Check for long stretches of base64-like characters (no spaces, repetitive)
    # Base64 uses A-Za-z0-9+/= characters
    base64_pattern = re.compile(r'[A-Za-z0-9+/=]{100,}')
    matches = base64_pattern.findall(content)

    if matches:
        # If we have large base64-like blocks, check their total size
        total_base64_chars = sum(len(m) for m in matches)
        if total_base64_chars > len(content) * 0.3:  # More than 30% is base64-like
            return True

    # Check for repetitive character patterns (common in encoded images)
    repetitive_pattern = re.compile(r'(.)\1{20,}')  # Same char repeated 20+ times
    if repetitive_pattern.search(content):
        # Check if significant portion is repetitive
        repetitive_matches = repetitive_pattern.findall(content)
        if len(repetitive_matches) > 5:
            return True

    # Check for very low text entropy (lots of repeated short sequences)
    if len(content) > 500:
        # Sample the content and check for unusual character distribution
        sample = content[:1000]
        unique_chars = len(set(sample))
        if unique_chars < 30:  # Very few unique characters suggests binary
            return True

    return False


def process_markdown(content: str, file_path: Path) -> str:
    """
    Clean markdown/MDX content for ingestion.

    - Strips MDX-specific syntax (imports, JSX components)
    - Preserves code blocks with language annotations
    - Cleans up excessive whitespace
    - Removes NUL bytes that PostgreSQL cannot handle
    - Removes base64 encoded images and binary data
    """
    # Remove NUL bytes (PostgreSQL text fields cannot contain them)
    content = content.replace('\x00', '')

    # Remove base64 encoded images (data:image/... patterns)
    content = re.sub(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+', '[image removed]', content)

    # Remove large base64 blocks (likely embedded images or binary data)
    content = re.sub(r'[A-Za-z0-9+/=]{200,}', '[binary data removed]', content)

    # Remove MDX import statements
    content = re.sub(r'^import\s+.*$', '', content, flags=re.MULTILINE)

    # Remove MDX export statements
    content = re.sub(r'^export\s+.*$', '', content, flags=re.MULTILINE)

    # Remove JSX-style components (self-closing)
    content = re.sub(r'<[A-Z][a-zA-Z]*\s*[^>]*/>', '', content)

    # Remove JSX-style component tags (opening and closing)
    content = re.sub(r'<[A-Z][a-zA-Z]*[^>]*>.*?</[A-Z][a-zA-Z]*>', '', content, flags=re.DOTALL)

    # Remove HTML-style comments
    content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)

    # Remove frontmatter (YAML between ---)
    content = re.sub(r'^---\s*\n.*?\n---\s*\n', '', content, flags=re.DOTALL)

    # Clean up multiple blank lines
    content = re.sub(r'\n{3,}', '\n\n', content)

    # Strip leading/trailing whitespace
    content = content.strip()

    return content


def generate_metadata(file_path: Path, repo_base: Path) -> Dict:
    """Generate rich metadata for a document."""
    relative_path = file_path.relative_to(repo_base)
    path_str = str(relative_path)

    # Determine section based on path
    if "langsmith" in path_str.lower():
        section = "langsmith"
    elif "langgraph" in path_str.lower():
        section = "langgraph"
    else:
        section = "langchain"

    # Determine doc type based on path
    doc_type = "reference"
    if "concepts" in path_str.lower() or "conceptual" in path_str.lower():
        doc_type = "concept"
    elif "how-to" in path_str.lower() or "how_to" in path_str.lower():
        doc_type = "how-to"
    elif "tutorial" in path_str.lower():
        doc_type = "tutorial"
    elif "quickstart" in path_str.lower() or "getting-started" in path_str.lower():
        doc_type = "quickstart"
    elif "api" in path_str.lower():
        doc_type = "api"

    # Generate docs URL (approximate mapping)
    url_path = path_str.replace("src/", "").replace(".mdx", "").replace(".md", "")
    url = f"https://docs.langchain.com/{url_path}"

    return {
        "source": path_str,
        "section": section,
        "doc_type": doc_type,
        "url": url,
        "filename": file_path.name,
    }


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks (matching setup.py implementation)."""
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


def clear_existing_docs(collection_name: str):
    """Clear existing documents from the specified collection."""
    print(f"      Clearing existing documents from collection '{collection_name}'...")

    with psycopg.connect(DATABASE_URL) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            # Delete chunks first (foreign key constraint)
            cur.execute(
                """
                DELETE FROM document_chunks
                WHERE document_id IN (
                    SELECT id FROM documents WHERE collection_id = %s
                )
                """,
                (collection_name,)
            )
            chunks_deleted = cur.rowcount

            # Delete documents
            cur.execute(
                "DELETE FROM documents WHERE collection_id = %s",
                (collection_name,)
            )
            docs_deleted = cur.rowcount

            print(f"      ✓ Cleared {docs_deleted} documents and {chunks_deleted} chunks")


def ensure_collection_exists(collection_name: str, description: str):
    """Ensure the collection exists in the database."""
    with psycopg.connect(DATABASE_URL) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO collections (id, name, description)
                VALUES (%s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET description = EXCLUDED.description
                """,
                (collection_name, collection_name, description)
            )


def ingest_documents(
    markdown_files: List[Path],
    repo_base: Path,
    collection_name: str,
    embeddings: OllamaEmbeddings,
    batch_size: int = 10
) -> Tuple[int, int]:
    """Ingest markdown documents into the vector store."""
    total_docs = 0
    total_chunks = 0

    with psycopg.connect(DATABASE_URL) as conn:
        conn.autocommit = True

        for i, file_path in enumerate(markdown_files):
            try:
                # Read and process content
                with open(file_path, "r", encoding="utf-8") as f:
                    raw_content = f.read()

                content = process_markdown(raw_content, file_path)

                # Skip empty documents
                if len(content.strip()) < 50:
                    continue

                # Skip documents that are mostly binary/base64 content
                if is_binary_content(content):
                    continue

                # Generate metadata
                metadata = generate_metadata(file_path, repo_base)

                with conn.cursor() as cur:
                    # Insert document
                    cur.execute(
                        """
                        INSERT INTO documents (content, embedding, metadata, collection_id)
                        VALUES (%s, %s, %s, %s)
                        RETURNING id
                        """,
                        (
                            content,
                            "[0" + ",0" * 767 + "]",  # Placeholder embedding
                            json.dumps(metadata),
                            collection_name,
                        )
                    )
                    doc_id = cur.fetchone()[0]
                    total_docs += 1

                    # Chunk the document
                    chunks = chunk_text(content)

                    if not chunks:
                        continue

                    # Generate embeddings in batch
                    try:
                        chunk_embeddings = embeddings.embed_documents(chunks)

                        # Insert chunks with embeddings
                        for chunk_idx, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
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
                        print(f"      ⚠ Error embedding {file_path.name}: {e}")
                        # Fallback to sequential
                        for chunk_idx, chunk in enumerate(chunks):
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
                                print(f"        ⚠ Chunk {chunk_idx} failed: {inner_e}")

                # Progress indicator
                if (i + 1) % 50 == 0:
                    print(f"      Processed {i + 1}/{len(markdown_files)} files...")

            except Exception as e:
                print(f"      ⚠ Error processing {file_path}: {e}")
                continue

    return total_docs, total_chunks


def show_stats():
    """Show current document statistics."""
    print("\n[Statistics]")

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            # Count by collection
            cur.execute("""
                SELECT collection_id, COUNT(*) as doc_count
                FROM documents
                GROUP BY collection_id
            """)
            collections = cur.fetchall()

            if not collections:
                print("      No documents found in database")
                return

            for collection_id, doc_count in collections:
                print(f"\n      Collection: {collection_id}")
                print(f"        Documents: {doc_count}")

                # Count chunks
                cur.execute("""
                    SELECT COUNT(*) FROM document_chunks
                    WHERE document_id IN (
                        SELECT id FROM documents WHERE collection_id = %s
                    )
                """, (collection_id,))
                chunk_count = cur.fetchone()[0]
                print(f"        Chunks: {chunk_count}")

                # Sample metadata
                cur.execute("""
                    SELECT metadata->>'section', COUNT(*)
                    FROM documents
                    WHERE collection_id = %s
                    GROUP BY metadata->>'section'
                """, (collection_id,))
                sections = cur.fetchall()
                if sections:
                    print(f"        Sections:")
                    for section, count in sections:
                        print(f"          - {section or 'unknown'}: {count} docs")


def main():
    """Main ingestion pipeline."""
    parser = argparse.ArgumentParser(description="Ingest LangChain documentation")
    parser.add_argument("--update", action="store_true", help="Pull latest docs and re-ingest")
    parser.add_argument("--stats", action="store_true", help="Show current statistics only")
    parser.add_argument("--keep-sample", action="store_true", help="Keep sample_docs collection")
    args = parser.parse_args()

    if args.stats:
        show_stats()
        return 0

    print("\n" + "=" * 70)
    print("LANGCHAIN DOCUMENTATION INGESTION")
    print("=" * 70)
    print("\nThis script will:")
    print("  1. Clone/update the langchain-ai/docs repository")
    print("  2. Process markdown files from src/oss and src/langsmith")
    print("  3. Generate embeddings and store in PostgreSQL")
    print("\n" + "=" * 70)

    try:
        # Step 1: Clone or update repository
        print("\n[1/4] Setting up documentation repository...")
        repo_path = clone_or_update_repo(DOCS_REPO_URL, DOCS_CACHE_DIR, update=args.update)

        # Step 2: Discover markdown files
        print("\n[2/4] Discovering markdown files...")
        markdown_files = discover_markdown_files(repo_path, DOCS_SOURCE_DIRS)
        print(f"      ✓ Found {len(markdown_files)} total markdown files")

        if not markdown_files:
            print("      ✗ No markdown files found. Check DOCS_SOURCE_DIRS in config.py")
            return 1

        # Step 3: Initialize embeddings and clear existing data
        print("\n[3/4] Preparing database...")
        embeddings = OllamaEmbeddings(
            model=EMBEDDINGS_MODEL,
            base_url=OLLAMA_BASE_URL,
        )

        # Clear existing LangChain docs
        clear_existing_docs(LANGCHAIN_COLLECTION_NAME)

        # Optionally clear sample docs too
        if not args.keep_sample:
            clear_existing_docs(VECTOR_COLLECTION_NAME)

        # Ensure collection exists
        ensure_collection_exists(
            LANGCHAIN_COLLECTION_NAME,
            "LangChain, LangGraph, and LangSmith documentation"
        )

        # Step 4: Ingest documents
        print("\n[4/4] Ingesting documents...")
        print(f"      Processing {len(markdown_files)} files...")
        print(f"      (This may take 10-30 minutes depending on your system)")

        total_docs, total_chunks = ingest_documents(
            markdown_files,
            repo_path,
            LANGCHAIN_COLLECTION_NAME,
            embeddings
        )

        # Summary
        print("\n" + "=" * 70)
        print("✓ INGESTION COMPLETE!")
        print("=" * 70)
        print(f"\n      Documents ingested: {total_docs}")
        print(f"      Total chunks: {total_chunks}")
        print(f"      Collection: {LANGCHAIN_COLLECTION_NAME}")

        # Show full stats
        show_stats()

        print("\n" + "=" * 70)
        print("\nYou can now run the agent:")
        print("  python main.py")
        print("\nExample queries:")
        print("  - What is LangGraph?")
        print("  - How do I create a ReAct agent in LangChain?")
        print("  - What is LangSmith tracing?")
        print("  - Explain the difference between LangChain and LangGraph")
        print("\n" + "=" * 70)

        return 0

    except Exception as e:
        print("\n" + "=" * 70)
        print(f"✗ INGESTION FAILED: {e}")
        print("=" * 70)
        print("\nTroubleshooting:")
        print("1. Ensure PostgreSQL is running: docker compose up -d")
        print("2. Ensure Ollama is running: ollama serve")
        print("3. Run setup.py first if database tables don't exist")
        print("4. Check network connectivity for git clone")
        print("\n" + "=" * 70)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

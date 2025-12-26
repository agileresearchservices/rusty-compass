#!/usr/bin/env python3
"""
Sample Data Loader
Loads sample documents into ChromaDB for the knowledge base
"""

import os
import sys
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from config import (
    EMBEDDINGS_MODEL,
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_NAME,
    SAMPLE_DOCS_DIR,
    OLLAMA_BASE_URL,
)


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


def create_vector_store(documents: list[tuple[str, str]]) -> Chroma:
    """
    Create a ChromaDB vector store and populate it with documents.
    """
    print(f"\nInitializing embeddings model: {EMBEDDINGS_MODEL}")
    print("(First run may take a moment to download the model...)\n")

    try:
        # Initialize embeddings
        embeddings = OllamaEmbeddings(
            model=EMBEDDINGS_MODEL,
            base_url=OLLAMA_BASE_URL,
        )

        # Create vector store with documents
        texts = [content for _, content in documents]
        metadatas = [{"source": filename} for filename, _ in documents]

        vector_store = Chroma.from_texts(
            texts=texts,
            metadatas=metadatas,
            embedding=embeddings,
            collection_name=CHROMA_COLLECTION_NAME,
            persist_directory=CHROMA_DB_PATH,
        )

        print(f"✓ Created vector store at: {CHROMA_DB_PATH}")
        print(f"✓ Collection: {CHROMA_COLLECTION_NAME}")
        print(f"✓ Indexed {len(documents)} documents")

        return vector_store

    except Exception as e:
        print(f"✗ Error creating vector store: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure Ollama is running: ollama serve")
        print(f"2. Verify the embedding model is available: ollama list")
        print(f"3. Check the Ollama endpoint: {OLLAMA_BASE_URL}")
        raise


def verify_retrieval(vector_store: Chroma, test_query: str = "Python"):
    """
    Test the vector store with a sample retrieval query.
    """
    print(f"\nTesting retrieval with query: '{test_query}'")

    try:
        results = vector_store.similarity_search(test_query, k=2)

        if results:
            print(f"✓ Found {len(results)} relevant documents:")
            for i, doc in enumerate(results, 1):
                source = doc.metadata.get("source", "unknown")
                preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                print(f"  {i}. {source}")
                print(f"     {preview}")
        else:
            print("✗ No results found")

    except Exception as e:
        print(f"✗ Error during retrieval test: {e}")
        raise


def main():
    """Run the data loading process"""
    print("=" * 60)
    print("Sample Data Loader for ChromaDB")
    print("=" * 60)
    print()

    try:
        # Load documents
        print(f"Loading documents from: {SAMPLE_DOCS_DIR}\n")
        documents = load_documents_from_directory(SAMPLE_DOCS_DIR)

        if not documents:
            print("\n✗ No documents to load. Exiting.")
            sys.exit(1)

        print(f"\n✓ Loaded {len(documents)} documents")

        # Create vector store
        vector_store = create_vector_store(documents)

        # Test retrieval
        verify_retrieval(vector_store)

        print()
        print("=" * 60)
        print("✓ Data loading complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Run the agent: python main.py")
        print("2. Try queries like: 'Tell me about Python', 'How does machine learning work?'")

    except Exception as e:
        print()
        print("=" * 60)
        print(f"✗ Loading failed: {e}")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Test hybrid search implementation"""

import time
from langchain_ollama import OllamaEmbeddings
from psycopg_pool import ConnectionPool
from main import SimplePostgresVectorStore
from config import (
    EMBEDDINGS_MODEL,
    DATABASE_URL,
    DB_CONNECTION_KWARGS,
    VECTOR_COLLECTION_NAME,
    RETRIEVER_K,
    RETRIEVER_FETCH_K,
    RETRIEVER_LAMBDA_MULT,
    OLLAMA_BASE_URL,
)

def test_search_methods():
    """Compare different search methods"""

    print("=" * 70)
    print("Testing Hybrid Search Implementation")
    print("=" * 70)

    # Initialize components
    print("\nInitializing components...")
    embeddings = OllamaEmbeddings(model=EMBEDDINGS_MODEL, base_url=OLLAMA_BASE_URL)

    connection_kwargs = DB_CONNECTION_KWARGS.copy()
    pool = ConnectionPool(
        conninfo=DATABASE_URL,
        max_size=5,
        kwargs=connection_kwargs
    )

    store = SimplePostgresVectorStore(
        embeddings=embeddings,
        collection_id=VECTOR_COLLECTION_NAME,
        pool=pool
    )

    print("✓ Components initialized")

    # Lambda interpretation: 0.0=pure lexical, 1.0=pure semantic
    test_queries = [
        # ===== Pure Lexical Queries (0.0-0.2) =====
        ("GPT-4 released in 2023", "Date + model number"),
        ("Model XR-2500 specifications", "Part number"),
        ("Intel Core i9-13900K specifications", "Product model specs"),
        ("RTX 4090 release date 2022", "Product + date"),
        ("SKU-XR-2500-A1 product manual", "Part number reference"),
        ("iPhone 15 Pro Max camera specs", "Device model details"),
        ("Google Pixel 8 Pro battery capacity mAh", "Device specs with unit"),
        ("Model RTX-3080-Ti VRAM 12GB", "Product identifier + specs"),

        # ===== Lexical-Heavy Queries (0.2-0.4) =====
        ("Django 4.2 authentication", "Version-specific"),
        ("Python 3.11 new features", "Version-specific features"),
        ("TensorFlow 2.13 API changes", "Library version updates"),
        ("Kubernetes 1.28 installation guide", "Tool version documentation"),
        ("Node.js version 20.5 release notes", "Runtime version info"),
        ("AWS Lambda pricing model 2024", "Service + date identifier"),

        # ===== Balanced Queries (0.4-0.6) =====
        ("MySQL 8.0 performance tuning", "Database version + optimization"),
        ("Python Flask REST API tutorial", "Balanced"),
        ("React hooks best practices", "Framework + technique"),
        ("Docker containerization guide", "Tool + process"),
        ("Express.js middleware configuration", "Framework + concept"),

        # ===== Semantic-Heavy Queries (0.6-0.9) =====
        ("Java Spring Boot microservices architecture", "Framework + architecture"),
        ("PostgreSQL query optimization techniques", "Database + method"),
        ("REST API design patterns", "Architecture + pattern"),
        ("Vue.js framework comparison", "Framework discussion"),

        # ===== Pure Semantic Queries (0.8-1.0) =====
        ("What is machine learning?", "Semantic/conceptual"),
        ("Explain how neural networks learn", "Conceptual/educational"),
        ("What is the difference between supervised and unsupervised learning?", "Conceptual comparison"),
        ("Describe the concept of overfitting in machine learning", "Conceptual explanation"),
        ("What is object-oriented programming?", "Programming paradigm"),
        ("Explain recursion in computer science", "Algorithm concept"),
        ("How do I learn Python programming?", "Educational question"),
        ("What are design patterns?", "Software design concept"),
    ]

    for query, description in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print(f"Type: {description}")
        print(f"{'='*70}")

        # Test pure vector (lambda=0.0)
        print("\n[1] Vector Only (lambda=0.0)")
        start = time.time()
        vector_results = store.hybrid_search(query, k=3, lambda_mult=0.0)
        vector_time = time.time() - start
        print(f"    Time: {vector_time*1000:.1f}ms | Results: {len(vector_results)}")
        for i, doc in enumerate(vector_results, 1):
            source = doc.metadata.get('source', 'unknown')
            preview = doc.page_content[:60].replace('\n', ' ') + "..."
            print(f"    {i}. [{source}] {preview}")

        # Test balanced hybrid (lambda=0.5)
        print("\n[2] Balanced Hybrid (lambda=0.5)")
        start = time.time()
        balanced_results = store.hybrid_search(query, k=3, lambda_mult=0.5)
        balanced_time = time.time() - start
        print(f"    Time: {balanced_time*1000:.1f}ms | Results: {len(balanced_results)}")
        for i, doc in enumerate(balanced_results, 1):
            source = doc.metadata.get('source', 'unknown')
            preview = doc.page_content[:60].replace('\n', ' ') + "..."
            print(f"    {i}. [{source}] {preview}")

        # Test configured hybrid (lambda from config)
        print(f"\n[3] Config Hybrid (lambda={RETRIEVER_LAMBDA_MULT})")
        start = time.time()
        config_results = store.hybrid_search(query, k=3, lambda_mult=RETRIEVER_LAMBDA_MULT)
        config_time = time.time() - start
        print(f"    Time: {config_time*1000:.1f}ms | Results: {len(config_results)}")
        for i, doc in enumerate(config_results, 1):
            source = doc.metadata.get('source', 'unknown')
            preview = doc.page_content[:60].replace('\n', ' ') + "..."
            print(f"    {i}. [{source}] {preview}")

        # Test pure text (lambda=1.0)
        print("\n[4] Text Only (lambda=1.0)")
        start = time.time()
        text_results = store.hybrid_search(query, k=3, lambda_mult=1.0)
        text_time = time.time() - start
        print(f"    Time: {text_time*1000:.1f}ms | Results: {len(text_results)}")
        for i, doc in enumerate(text_results, 1):
            source = doc.metadata.get('source', 'unknown')
            preview = doc.page_content[:60].replace('\n', ' ') + "..."
            print(f"    {i}. [{source}] {preview}")

    # Performance summary
    print(f"\n{'='*70}")
    print("Performance Summary:")
    print(f"{'='*70}")
    print(f"Text search (lambda=0.0): Full-text/lexical search only")
    print(f"Balanced hybrid (lambda=0.5): Equal weight to lexical and semantic")
    print(f"Config hybrid (lambda={RETRIEVER_LAMBDA_MULT}): {int((1-RETRIEVER_LAMBDA_MULT)*100)}% lexical, {int(RETRIEVER_LAMBDA_MULT*100)}% semantic")
    print(f"Vector search (lambda=1.0): Semantic/dense matching with embeddings")

    pool.close()
    print(f"\n{'='*70}")
    print("✓ Testing complete")
    print(f"{'='*70}")
    print("\nNext steps:")
    print("1. Review results from different search methods")
    print("2. Notice how results vary with lambda_mult parameter")
    print("3. Update config.py: RETRIEVER_SEARCH_TYPE = 'hybrid' to enable hybrid search")
    print("4. Run agent: python main.py")

if __name__ == "__main__":
    test_search_methods()

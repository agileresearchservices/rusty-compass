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
        ("LangChain 0.3.0 release notes", "Version + release"),
        ("langchain-openai package version", "Package version"),
        ("ChatOpenAI model_name parameter", "API parameter"),
        ("LangGraph StateGraph import", "Import statement"),
        ("LangSmith API key configuration", "Config parameter"),
        ("OPENAI_API_KEY environment variable", "Environment variable"),
        ("langchain_core.messages HumanMessage", "Module path"),
        ("RunnablePassthrough class reference", "Class reference"),

        # ===== Lexical-Heavy Queries (0.2-0.4) =====
        ("LangGraph checkpointer PostgreSQL", "Feature + technology"),
        ("LangChain LCEL pipe operator", "Framework syntax"),
        ("LangSmith tracing setup", "Tool configuration"),
        ("ChatPromptTemplate from_messages", "API method"),
        ("StructuredOutputParser JSON schema", "Parser configuration"),
        ("LangChain document loader PDF", "Loader type"),

        # ===== Balanced Queries (0.4-0.6) =====
        ("LangGraph state management patterns", "Framework + patterns"),
        ("LangChain retriever with reranking", "Feature + technique"),
        ("LangSmith evaluation dataset creation", "Tool + process"),
        ("RAG pipeline with LangChain", "Architecture + framework"),
        ("LangGraph human-in-the-loop workflow", "Framework + concept"),

        # ===== Semantic-Heavy Queries (0.6-0.9) =====
        ("How to build multi-agent systems with LangGraph", "Architecture question"),
        ("Best practices for LangChain memory management", "Best practices"),
        ("LangSmith debugging and observability strategies", "Strategies"),
        ("Comparing LangChain vs LangGraph for agents", "Comparison"),

        # ===== Pure Semantic Queries (0.8-1.0) =====
        ("What is LangGraph?", "Semantic/conceptual"),
        ("Explain how LangChain agents work", "Conceptual/educational"),
        ("What is the difference between chains and agents?", "Conceptual comparison"),
        ("Describe the concept of tool calling in LangChain", "Conceptual explanation"),
        ("What is retrieval augmented generation?", "RAG concept"),
        ("How does LangSmith help with LLM development?", "Tool purpose"),
        ("How do I get started with LangChain?", "Educational question"),
        ("What are the benefits of using LangGraph?", "Benefits question"),
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

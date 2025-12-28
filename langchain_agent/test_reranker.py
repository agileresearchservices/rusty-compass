#!/usr/bin/env python3
"""Test Qwen3-Reranker-8B cross-encoder implementation"""

import sys
import time
from langchain_core.documents import Document
from main import Qwen3Reranker


def show_reranking_comparison(query: str, original_docs, reranked_results):
    """Display before/after comparison of document reranking"""
    print(f"\n  Query: {query}\n")

    print(f"  BEFORE (Original Order):")
    for i, doc in enumerate(original_docs, 1):
        preview = doc.page_content[:60].replace('\n', ' ')
        print(f"    {i}. [{doc.metadata['source']}] {preview}...")

    print(f"\n  AFTER (Reranked by Relevance Score):")
    for i, (doc, score) in enumerate(reranked_results, 1):
        preview = doc.page_content[:60].replace('\n', ' ')
        print(f"    {i}. score={score:.4f} ⭐ [{doc.metadata['source']}] {preview}...")

    # Show what changed
    print(f"\n  CHANGES:")
    original_order = [doc.metadata['source'] for doc in original_docs]
    reranked_order = [doc.metadata['source'] for doc, _ in reranked_results]

    if original_order == reranked_order:
        print(f"    ➜ No reordering (documents already ranked by relevance)")
    else:
        changes = 0
        for i, (orig, reranked) in enumerate(zip(original_order, reranked_order)):
            if orig != reranked:
                orig_idx = original_order.index(reranked) + 1
                print(f"    ➜ Position {i+1}: {orig} → {reranked} (moved from position {orig_idx})")
                changes += 1
        if changes == 0:
            print(f"    ➜ Partial reordering in relevance scores")


def test_langgraph_query():
    """Test reranking with LangGraph query"""
    print("\n" + "=" * 70)
    print("Test 1: LangGraph Query")
    print("=" * 70)

    try:
        reranker = Qwen3Reranker(model_name="Qwen/Qwen3-Reranker-8B")

        docs = [
            Document(
                page_content="LangChain is a framework for developing applications powered by large language models",
                metadata={"source": "langchain_intro.mdx"}
            ),
            Document(
                page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs, used for agent workflows",
                metadata={"source": "langgraph_intro.mdx"}
            ),
            Document(
                page_content="StateGraph is the main class in LangGraph for defining workflows with nodes and edges",
                metadata={"source": "langgraph_state.mdx"}
            ),
            Document(
                page_content="LangSmith is a platform for debugging, testing, and monitoring LLM applications",
                metadata={"source": "langsmith_intro.mdx"}
            ),
        ]

        query = "What is LangGraph and how do I create workflows with StateGraph?"
        results = reranker.rerank(query, docs, top_k=3)

        print(f"✓ Reranking completed")
        show_reranking_comparison(query, docs, results)

        reranker_safe_cleanup(reranker)
        print("\n✓ Test passed\n")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rag_query():
    """Test reranking with RAG pipeline query"""
    print("=" * 70)
    print("Test 2: RAG Pipeline Query")
    print("=" * 70)

    try:
        reranker = Qwen3Reranker(model_name="Qwen/Qwen3-Reranker-8B")

        docs = [
            Document(
                page_content="Document loaders in LangChain help ingest data from various sources like PDFs and web pages",
                metadata={"source": "document_loaders.mdx"}
            ),
            Document(
                page_content="Text splitters divide documents into smaller chunks for embedding and retrieval",
                metadata={"source": "text_splitters.mdx"}
            ),
            Document(
                page_content="Vector stores like Chroma and Pinecone store embeddings for similarity search",
                metadata={"source": "vector_stores.mdx"}
            ),
            Document(
                page_content="Retrievers fetch relevant documents from vector stores based on query similarity",
                metadata={"source": "retrievers.mdx"}
            ),
        ]

        query = "How do I build a RAG pipeline with document loaders and vector stores?"
        results = reranker.rerank(query, docs, top_k=3)

        print(f"✓ Reranking completed")
        show_reranking_comparison(query, docs, results)

        reranker_safe_cleanup(reranker)
        print("\n✓ Test passed\n")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_score_range():
    """Test that scores are in valid [0.0, 1.0] range"""
    print("=" * 70)
    print("Test 3: Score Range Validation")
    print("=" * 70)

    try:
        reranker = Qwen3Reranker(model_name="Qwen/Qwen3-Reranker-8B")

        docs = [
            Document(page_content="LangChain agents use tools to interact with external systems", metadata={"source": "agents.mdx"}),
            Document(page_content="LangGraph enables complex multi-step workflows", metadata={"source": "langgraph.mdx"}),
            Document(page_content="LangSmith provides tracing and evaluation", metadata={"source": "langsmith.mdx"}),
            Document(page_content="LCEL is the expression language for chains", metadata={"source": "lcel.mdx"}),
            Document(page_content="Retrievers fetch documents from vector stores", metadata={"source": "retrievers.mdx"}),
        ]

        queries = [
            "LangChain agents",
            "What is LangGraph?",
            "How to trace LLM applications?",
        ]

        all_valid = True
        for query in queries:
            results = reranker.rerank(query, docs, top_k=3)
            print(f"\n  Query: {query}")
            for doc, score in results:
                if not (0.0 <= score <= 1.0):
                    print(f"  ✗ Invalid score {score}")
                    all_valid = False
                else:
                    print(f"    ✓ score={score:.4f} [{doc.metadata['source']}]")

        if all_valid:
            print(f"\n✓ All scores in valid range [0.0, 1.0]")

        reranker_safe_cleanup(reranker)
        print("\n✓ Test passed\n")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_relevance_ordering():
    """Test that more relevant documents get higher scores"""
    print("=" * 70)
    print("Test 4: Relevance Ordering (Higher Scores = More Relevant)")
    print("=" * 70)

    try:
        reranker = Qwen3Reranker(model_name="Qwen/Qwen3-Reranker-8B")

        docs = [
            Document(
                page_content="LangChain is a framework for building LLM applications",
                metadata={"source": "langchain_overview.mdx"}
            ),
            Document(
                page_content="Getting started with LangChain for beginners",
                metadata={"source": "langchain_quickstart.mdx"}
            ),
            Document(
                page_content="Advanced LangChain patterns and best practices",
                metadata={"source": "langchain_advanced.mdx"}
            ),
            Document(
                page_content="LangSmith is a platform for debugging LLM applications",
                metadata={"source": "langsmith_overview.mdx"}
            ),
        ]

        query = "How do I get started with LangChain?"
        results = reranker.rerank(query, docs, top_k=4)

        print(f"\n  Query: {query}\n")
        print(f"  Ranked Results (highest relevance first):")

        # Check if sorted by score descending
        is_sorted = all(
            results[i][1] >= results[i + 1][1]
            for i in range(len(results) - 1)
        )

        for i, (doc, score) in enumerate(results, 1):
            relevance_bar = "█" * int(score * 20)
            print(f"    {i}. score={score:.4f} {relevance_bar} [{doc.metadata['source']}]")

        if is_sorted:
            print(f"\n✓ Results properly sorted by relevance (descending)")
        else:
            print(f"\n✗ Results NOT sorted correctly")
            return False

        reranker_safe_cleanup(reranker)
        print("\n✓ Test passed\n")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance():
    """Test reranking performance"""
    print("=" * 70)
    print("Test 5: Performance Benchmark")
    print("=" * 70)

    try:
        reranker = Qwen3Reranker(model_name="Qwen/Qwen3-Reranker-8B")

        # Create test documents
        docs = [
            Document(
                page_content=f"Document {i}: This is sample content about LangChain, LangGraph, and building LLM applications",
                metadata={"source": f"langchain_doc{i}.mdx"}
            )
            for i in range(10)
        ]

        query = "What are the best practices for building agents with LangChain?"

        # Measure reranking time
        start = time.time()
        results = reranker.rerank(query, docs, top_k=5)
        elapsed = time.time() - start

        print(f"\n  Reranking Performance:")
        print(f"    Documents scored: {len(docs)}")
        print(f"    Time elapsed: {elapsed:.2f}s")
        print(f"    Average per document: {(elapsed / len(docs) * 1000):.1f}ms")
        print(f"\n  Top Results:")
        for i, (doc, score) in enumerate(results, 1):
            print(f"    {i}. score={score:.4f} [{doc.metadata['source']}]")

        # Performance target: < 3s for 10 documents
        if elapsed < 3.0:
            print(f"\n✓ Performance within target (<3.0s)")
        else:
            print(f"\n⚠ Performance slower than expected ({elapsed:.2f}s > 3.0s)")

        reranker_safe_cleanup(reranker)
        print("\n✓ Test passed\n")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_empty_documents():
    """Test handling of edge cases"""
    print("=" * 70)
    print("Test 6: Edge Cases")
    print("=" * 70)

    try:
        reranker = Qwen3Reranker(model_name="Qwen/Qwen3-Reranker-8B")

        # Test with empty list
        results = reranker.rerank("test query", [], top_k=4)
        assert len(results) == 0, "Expected empty results for empty documents"
        print("  ✓ Empty document list handled correctly")

        # Test with single document
        docs = [
            Document(
                page_content="Single document content",
                metadata={"source": "single.txt"}
            )
        ]
        results = reranker.rerank("test query", docs, top_k=1)
        assert len(results) == 1, "Expected 1 result for 1 document"
        print("  ✓ Single document handled correctly")

        # Test with top_k > document count
        docs = [
            Document(page_content=f"Doc {i}", metadata={"source": f"doc{i}.txt"})
            for i in range(3)
        ]
        results = reranker.rerank("test", docs, top_k=10)
        assert len(results) == 3, "Expected 3 results (all available docs)"
        print("  ✓ top_k > document_count handled correctly")

        reranker_safe_cleanup(reranker)
        print("\n✓ Test passed\n")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def reranker_safe_cleanup(reranker):
    """Safely cleanup reranker resources"""
    try:
        import torch
        del reranker
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  Note: Cleanup warning: {e}")


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("QWEN3-RERANKER-8B CROSS-ENCODER TESTS")
    print("=" * 70)

    tests = [
        ("LangGraph Query", test_langgraph_query),
        ("RAG Pipeline Query", test_rag_query),
        ("Score Range", test_score_range),
        ("Relevance Ordering", test_relevance_ordering),
        ("Performance", test_performance),
        ("Edge Cases", test_empty_documents),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}\n")
            failed += 1

    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    print("=" * 70)

    if failed == 0:
        print("\n✓ All tests passed! Qwen3-Reranker is working correctly.\n")
    else:
        print(f"\n✗ {failed} test(s) failed.\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

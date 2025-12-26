#!/usr/bin/env python3
"""Unit tests for the reranker module"""

import time
from langchain_core.documents import Document
from reranker import OllamaReranker
from config import OLLAMA_BASE_URL


def test_basic_reranking():
    """Test basic reranking functionality"""
    print("=" * 70)
    print("Test 1: Basic Reranking")
    print("=" * 70)

    try:
        reranker = OllamaReranker(base_url=OLLAMA_BASE_URL)

        # Create test documents
        docs = [
            Document(
                page_content="Python is a high-level programming language",
                metadata={"source": "python_guide.txt"}
            ),
            Document(
                page_content="Machine learning is a subset of artificial intelligence",
                metadata={"source": "ml_guide.txt"}
            ),
            Document(
                page_content="Web development involves building websites and applications",
                metadata={"source": "web_guide.txt"}
            ),
        ]

        # Rerank
        query = "What is machine learning?"
        results = reranker.rerank(query, docs, top_k=2)

        # Verify results
        assert len(results) == 2, f"Expected 2 results, got {len(results)}"
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results), "Results should be (doc, score) tuples"

        print(f"✓ Reranking completed")
        print(f"  Query: {query}")
        for i, (doc, score) in enumerate(results, 1):
            print(f"  {i}. score={score:.4f} [{doc.metadata['source']}]")

        reranker.close()
        print("\n✓ Test passed\n")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}\n")
        return False


def test_score_range():
    """Test that scores are in valid [0.0, 1.0] range"""
    print("=" * 70)
    print("Test 2: Score Range Validation")
    print("=" * 70)

    try:
        reranker = OllamaReranker(base_url=OLLAMA_BASE_URL)

        # Create test documents with various content
        docs = [
            Document(page_content="Python programming", metadata={"source": "doc1.txt"}),
            Document(page_content="JavaScript basics", metadata={"source": "doc2.txt"}),
            Document(page_content="Database design", metadata={"source": "doc3.txt"}),
            Document(page_content="Web frameworks", metadata={"source": "doc4.txt"}),
            Document(page_content="Machine learning", metadata={"source": "doc5.txt"}),
        ]

        queries = [
            "Python programming",
            "What is machine learning?",
            "How to build web applications?",
        ]

        all_valid = True
        for query in queries:
            results = reranker.rerank(query, docs, top_k=3)

            for doc, score in results:
                if not (0.0 <= score <= 1.0):
                    print(f"✗ Invalid score {score} for query '{query}'")
                    all_valid = False

        if all_valid:
            print(f"✓ All scores are in valid range [0.0, 1.0]")

        reranker.close()
        print("\n✓ Test passed\n")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}\n")
        return False


def test_top_k_selection():
    """Test that top_k parameter correctly limits results"""
    print("=" * 70)
    print("Test 3: Top-K Selection")
    print("=" * 70)

    try:
        reranker = OllamaReranker(base_url=OLLAMA_BASE_URL)

        # Create many test documents
        docs = [
            Document(page_content=f"Document {i} about Python", metadata={"source": f"doc{i}.txt"})
            for i in range(10)
        ]

        query = "Python programming"

        # Test different top_k values
        test_cases = [1, 3, 5, 10]
        results_valid = True

        for top_k in test_cases:
            results = reranker.rerank(query, docs, top_k=top_k)
            expected_len = min(top_k, len(docs))

            if len(results) != expected_len:
                print(f"✗ Expected {expected_len} results for top_k={top_k}, got {len(results)}")
                results_valid = False
            else:
                print(f"✓ top_k={top_k}: returned {len(results)} results")

        if results_valid:
            print(f"\n✓ Top-K selection working correctly")

        reranker.close()
        print("\n✓ Test passed\n")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}\n")
        return False


def test_sorted_by_score():
    """Test that results are sorted by score in descending order"""
    print("=" * 70)
    print("Test 4: Score Sorting")
    print("=" * 70)

    try:
        reranker = OllamaReranker(base_url=OLLAMA_BASE_URL)

        # Create test documents
        docs = [
            Document(
                page_content="Machine learning algorithms for classification",
                metadata={"source": "ml_classification.txt"}
            ),
            Document(
                page_content="Web development with Python and Flask",
                metadata={"source": "flask_guide.txt"}
            ),
            Document(
                page_content="Introduction to machine learning basics",
                metadata={"source": "ml_intro.txt"}
            ),
            Document(
                page_content="JavaScript for web browsers",
                metadata={"source": "js_guide.txt"}
            ),
        ]

        query = "What is machine learning?"
        results = reranker.rerank(query, docs, top_k=4)

        # Check if sorted by score descending
        is_sorted = all(
            results[i][1] >= results[i + 1][1]
            for i in range(len(results) - 1)
        )

        if is_sorted:
            print(f"✓ Results are sorted by score (descending)")
            for i, (doc, score) in enumerate(results, 1):
                print(f"  {i}. score={score:.4f} [{doc.metadata['source']}]")
        else:
            print(f"✗ Results are NOT sorted correctly")
            return False

        reranker.close()
        print("\n✓ Test passed\n")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}\n")
        return False


def test_empty_documents():
    """Test handling of empty document list"""
    print("=" * 70)
    print("Test 5: Empty Documents Handling")
    print("=" * 70)

    try:
        reranker = OllamaReranker(base_url=OLLAMA_BASE_URL)

        # Test with empty list
        results = reranker.rerank("test query", [], top_k=4)

        if len(results) == 0:
            print(f"✓ Empty document list handled correctly")
        else:
            print(f"✗ Expected empty results, got {len(results)}")
            return False

        reranker.close()
        print("\n✓ Test passed\n")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}\n")
        return False


def test_single_document():
    """Test reranking a single document"""
    print("=" * 70)
    print("Test 6: Single Document")
    print("=" * 70)

    try:
        reranker = OllamaReranker(base_url=OLLAMA_BASE_URL)

        # Test with single document
        docs = [
            Document(
                page_content="Python is a programming language",
                metadata={"source": "python.txt"}
            )
        ]

        results = reranker.rerank("Python", docs, top_k=1)

        if len(results) == 1:
            print(f"✓ Single document reranked correctly")
            doc, score = results[0]
            print(f"  score={score:.4f} [{doc.metadata['source']}]")
        else:
            print(f"✗ Expected 1 result, got {len(results)}")
            return False

        reranker.close()
        print("\n✓ Test passed\n")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}\n")
        return False


def test_performance():
    """Test reranking performance"""
    print("=" * 70)
    print("Test 7: Performance Benchmark")
    print("=" * 70)

    try:
        reranker = OllamaReranker(base_url=OLLAMA_BASE_URL)

        # Create test documents
        docs = [
            Document(
                page_content=f"Document {i}: {' '.join(['word'] * 50)}",
                metadata={"source": f"doc{i}.txt"}
            )
            for i in range(15)
        ]

        query = "What is this about?"

        # Measure reranking time
        start = time.time()
        results = reranker.rerank(query, docs, top_k=4)
        elapsed = time.time() - start

        print(f"Reranked {len(docs)} documents in {elapsed:.2f}s")
        print(f"Average per document: {(elapsed / len(docs) * 1000):.1f}ms")

        # Performance target: < 1.5s for 15 documents
        if elapsed < 1.5:
            print(f"✓ Performance target met (<1.5s)")
        else:
            print(f"⚠ Performance slower than target (expected <1.5s, got {elapsed:.2f}s)")

        reranker.close()
        print("\n✓ Test passed\n")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}\n")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("RERANKER UNIT TESTS")
    print("=" * 70 + "\n")

    tests = [
        test_basic_reranking,
        test_score_range,
        test_top_k_selection,
        test_sorted_by_score,
        test_empty_documents,
        test_single_document,
        test_performance,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}\n")
            failed += 1

    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    print("=" * 70 + "\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())

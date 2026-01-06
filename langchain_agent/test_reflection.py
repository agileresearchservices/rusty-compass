#!/usr/bin/env python3
"""
Test reflection loop functionality for the LangGraph agent.

Tests document grading, query transformation, and response grading.
"""

import sys
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

# Import configuration
from config import (
    ENABLE_REFLECTION,
    ENABLE_DOCUMENT_GRADING,
    ENABLE_RESPONSE_GRADING,
    ENABLE_QUERY_TRANSFORMATION,
    REFLECTION_MAX_ITERATIONS,
    REFLECTION_MIN_RELEVANT_DOCS,
    REFLECTION_DOC_SCORE_THRESHOLD,
)


def test_config_loaded():
    """Test that reflection configuration is properly loaded"""
    print("\n" + "=" * 70)
    print("Test 1: Configuration Loading")
    print("=" * 70)

    try:
        print(f"  ENABLE_REFLECTION: {ENABLE_REFLECTION}")
        print(f"  ENABLE_DOCUMENT_GRADING: {ENABLE_DOCUMENT_GRADING}")
        print(f"  ENABLE_RESPONSE_GRADING: {ENABLE_RESPONSE_GRADING}")
        print(f"  ENABLE_QUERY_TRANSFORMATION: {ENABLE_QUERY_TRANSFORMATION}")
        print(f"  REFLECTION_MAX_ITERATIONS: {REFLECTION_MAX_ITERATIONS}")
        print(f"  REFLECTION_MIN_RELEVANT_DOCS: {REFLECTION_MIN_RELEVANT_DOCS}")
        print(f"  REFLECTION_DOC_SCORE_THRESHOLD: {REFLECTION_DOC_SCORE_THRESHOLD}")

        assert isinstance(ENABLE_REFLECTION, bool), "ENABLE_REFLECTION should be bool"
        assert isinstance(REFLECTION_MAX_ITERATIONS, int), "REFLECTION_MAX_ITERATIONS should be int"
        assert REFLECTION_MAX_ITERATIONS >= 1, "REFLECTION_MAX_ITERATIONS should be >= 1"

        print("\n✓ Configuration loaded successfully")
        return True

    except Exception as e:
        print(f"\n✗ Configuration test failed: {e}")
        return False


def test_state_types():
    """Test that reflection state types are properly defined"""
    print("\n" + "=" * 70)
    print("Test 2: State Type Definitions")
    print("=" * 70)

    try:
        from agent_state import DocumentGrade, ReflectionResult, CustomAgentState

        # Test DocumentGrade
        doc_grade: DocumentGrade = {
            "source": "test.txt",
            "relevant": True,
            "score": 0.85,
            "reasoning": "Highly relevant to query"
        }
        print(f"  DocumentGrade: {doc_grade}")

        # Test ReflectionResult
        reflection_result: ReflectionResult = {
            "grade": "pass",
            "score": 0.9,
            "reasoning": "Good response quality"
        }
        print(f"  ReflectionResult: {reflection_result}")

        # Verify CustomAgentState has reflection fields
        from typing import get_type_hints
        hints = get_type_hints(CustomAgentState)

        required_fields = [
            "iteration_count",
            "retrieved_documents",
            "document_grades",
            "document_grade_summary",
            "response_grade",
            "original_query",
            "transformed_query"
        ]

        for field in required_fields:
            assert field in hints, f"CustomAgentState missing field: {field}"
            print(f"  ✓ CustomAgentState.{field} defined")

        print("\n✓ State types defined correctly")
        return True

    except Exception as e:
        print(f"\n✗ State type test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_graph_creation():
    """Test that the agent graph creates with reflection nodes"""
    print("\n" + "=" * 70)
    print("Test 3: Graph Creation with Reflection Nodes")
    print("=" * 70)

    try:
        from main import LangChainAgent

        print("  Initializing agent (this may take a moment)...")
        agent = LangChainAgent()

        # Call initialization methods (same as agent.run() but without conversation loop)
        print("  Verifying prerequisites...")
        agent.verify_prerequisites()
        print("  Initializing components...")
        agent.initialize_components()
        print("  Creating agent graph...")
        agent.create_agent_graph()

        # Check that graph was created
        assert agent.app is not None, "Agent graph not created"
        print("  ✓ Agent graph created")

        # Get the graph structure
        graph = agent.app.get_graph()
        node_names = list(graph.nodes.keys())

        print(f"  Graph nodes: {node_names}")

        # Check for required nodes
        required_nodes = ["query_evaluator", "agent", "tools"]
        for node in required_nodes:
            assert node in node_names, f"Missing required node: {node}"
            print(f"  ✓ Node '{node}' present")

        # Check for reflection nodes if enabled
        if ENABLE_REFLECTION:
            if ENABLE_DOCUMENT_GRADING:
                assert "document_grader" in node_names, "Missing document_grader node"
                print("  ✓ Node 'document_grader' present (reflection enabled)")

            if ENABLE_QUERY_TRANSFORMATION:
                assert "query_transformer" in node_names, "Missing query_transformer node"
                print("  ✓ Node 'query_transformer' present (reflection enabled)")

            if ENABLE_RESPONSE_GRADING:
                assert "response_grader" in node_names, "Missing response_grader node"
                print("  ✓ Node 'response_grader' present (reflection enabled)")

        print("\n✓ Graph creation test passed")
        return True

    except Exception as e:
        print(f"\n✗ Graph creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_initialized_agent():
    """Helper to create a fully initialized agent for testing"""
    from main import LangChainAgent
    agent = LangChainAgent()
    agent.verify_prerequisites()
    agent.initialize_components()
    agent.create_agent_graph()
    return agent


# Cache the agent for reuse across tests
_cached_agent = None


def get_test_agent():
    """Get or create a cached initialized agent for testing"""
    global _cached_agent
    if _cached_agent is None:
        _cached_agent = create_initialized_agent()
    return _cached_agent


def test_document_grader_standalone():
    """Test document grading logic with mock data"""
    print("\n" + "=" * 70)
    print("Test 4: Document Grader Logic (Standalone)")
    print("=" * 70)

    try:
        print("  Getting initialized agent...")
        agent = get_test_agent()

        # Create test documents
        test_docs = [
            Document(
                page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs.",
                metadata={"source": "langgraph_intro.mdx"}
            ),
            Document(
                page_content="LangChain provides tools for building applications powered by large language models.",
                metadata={"source": "langchain_intro.mdx"}
            ),
            Document(
                page_content="LangSmith is a platform for debugging, testing, and monitoring LLM applications.",
                metadata={"source": "langsmith_intro.mdx"}
            ),
        ]

        # Create mock state for testing
        mock_state = {
            "messages": [HumanMessage(content="What is LangGraph?")],
            "lambda_mult": 0.5,
            "query_analysis": "",
            "iteration_count": 0,
            "retrieved_documents": test_docs,
            "document_grades": [],
            "document_grade_summary": {},
            "response_grade": {},
            "original_query": "What is LangGraph?",
            "transformed_query": None,
        }

        print("  Running document_grader_node...")
        result = agent.document_grader_node(mock_state)

        # Check results
        assert "document_grades" in result, "document_grades not in result"
        assert "document_grade_summary" in result, "document_grade_summary not in result"

        grades = result["document_grades"]
        summary = result["document_grade_summary"]

        print(f"  Document grades: {len(grades)} documents graded")
        for grade in grades:
            status = "✓" if grade["relevant"] else "✗"
            print(f"    {status} {grade['source']}: score={grade['score']:.2f} - {grade['reasoning'][:50]}...")

        print(f"  Summary: {summary['grade']} (score: {summary['score']:.2f})")
        print(f"    {summary['reasoning']}")

        print("\n✓ Document grader test passed")
        return True

    except Exception as e:
        print(f"\n✗ Document grader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_response_grader_standalone():
    """Test response grading logic with mock data"""
    print("\n" + "=" * 70)
    print("Test 5: Response Grader Logic (Standalone)")
    print("=" * 70)

    try:
        print("  Getting initialized agent...")
        agent = get_test_agent()

        # Create mock state with a response
        mock_state = {
            "messages": [
                HumanMessage(content="What is LangGraph?"),
                AIMessage(content="LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends LangChain to enable complex workflows with cycles, persistence, and human-in-the-loop patterns. LangGraph uses a graph-based approach where nodes represent processing steps and edges define the flow between them."),
            ],
            "lambda_mult": 0.5,
            "query_analysis": "",
            "iteration_count": 0,
            "retrieved_documents": [],
            "document_grades": [],
            "document_grade_summary": {"grade": "pass", "score": 0.8, "reasoning": "Documents relevant"},
            "response_grade": {},
            "original_query": "What is LangGraph?",
            "transformed_query": None,
        }

        print("  Running response_grader_node...")
        result = agent.response_grader_node(mock_state)

        # Check results
        assert "response_grade" in result, "response_grade not in result"

        grade = result["response_grade"]
        print(f"  Response grade: {grade['grade']} (score: {grade['score']:.2f})")
        print(f"    {grade['reasoning']}")

        assert grade["grade"] in ["pass", "fail"], "Invalid grade value"
        assert 0.0 <= grade["score"] <= 1.0, "Score out of range"

        print("\n✓ Response grader test passed")
        return True

    except Exception as e:
        print(f"\n✗ Response grader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_query_transformer_standalone():
    """Test query transformation logic with mock data"""
    print("\n" + "=" * 70)
    print("Test 6: Query Transformer Logic (Standalone)")
    print("=" * 70)

    try:
        print("  Getting initialized agent...")
        agent = get_test_agent()

        # Create mock state with failed document grades
        mock_state = {
            "messages": [HumanMessage(content="LangGraph subgraph patterns")],
            "lambda_mult": 0.5,
            "query_analysis": "",
            "iteration_count": 0,
            "retrieved_documents": [],
            "document_grades": [
                {"source": "langchain.mdx", "relevant": False, "score": 0.2, "reasoning": "About LangChain basics, not subgraphs"},
                {"source": "langsmith.mdx", "relevant": False, "score": 0.1, "reasoning": "About LangSmith, not LangGraph"},
            ],
            "document_grade_summary": {"grade": "fail", "score": 0.15, "reasoning": "0/2 documents relevant"},
            "response_grade": {},
            "original_query": "LangGraph subgraph patterns",
            "transformed_query": None,
        }

        print("  Running query_transformer_node...")
        result = agent.query_transformer_node(mock_state)

        # Check results
        assert "transformed_query" in result, "transformed_query not in result"
        assert "iteration_count" in result, "iteration_count not in result"

        print(f"  Original query: '{mock_state['original_query']}'")
        print(f"  Transformed query: '{result['transformed_query']}'")
        print(f"  Iteration count: {result['iteration_count']}")

        assert result["iteration_count"] == 1, "Iteration count should be incremented"
        assert result["transformed_query"] != "", "Transformed query should not be empty"

        print("\n✓ Query transformer test passed")
        return True

    except Exception as e:
        print(f"\n✗ Query transformer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all reflection tests"""
    print("\n" + "=" * 70)
    print("REFLECTION LOOP TESTS")
    print("=" * 70)

    results = {
        "config": test_config_loaded(),
        "state_types": test_state_types(),
        "graph_creation": test_graph_creation(),
        "document_grader": test_document_grader_standalone(),
        "response_grader": test_response_grader_standalone(),
        "query_transformer": test_query_transformer_standalone(),
    }

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    for name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {name}: {status}")

    print()
    print(f"  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed! Reflection loop is working correctly.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())

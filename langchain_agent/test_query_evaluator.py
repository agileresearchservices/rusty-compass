#!/usr/bin/env python3
"""Test query evaluator lambda_mult determination"""

import sys
from main import LangChainAgent
from langchain_core.messages import HumanMessage


def test_query_evaluator():
    """Test different query types produce expected lambda values"""

    print("=" * 70)
    print("Testing Query Evaluator with Dynamic Lambda Adjustment")
    print("=" * 70)
    print()

    # Initialize agent
    print("Initializing agent...")
    agent = LangChainAgent()

    try:
        agent.verify_prerequisites()
    except SystemExit as e:
        print("❌ Prerequisites check failed")
        return 1

    agent.initialize_components()
    agent.create_agent_graph()

    print("✓ Agent initialized")
    print()

    # Test cases with expected lambda ranges
    test_cases = [
        ("What is machine learning?", 0.0, 0.3, "Semantic/conceptual"),
        ("Python Flask REST API tutorial", 0.3, 0.6, "Balanced"),
        ("Django 4.2 authentication", 0.5, 0.9, "Version-specific"),
        ("GPT-4 released in 2023", 0.8, 1.0, "Date + model number"),
        ("Model XR-2500 specifications", 0.9, 1.0, "Part number"),
    ]

    print("Testing Query Evaluator:")
    print("-" * 70)

    passed = 0
    failed = 0

    for query, min_lambda, max_lambda, description in test_cases:
        state = {
            "messages": [HumanMessage(content=query)],
            "lambda_mult": 0.0,
            "query_analysis": ""
        }

        result = agent.query_evaluator_node(state)
        lambda_mult = result["lambda_mult"]
        reasoning = result["query_analysis"]

        is_in_range = min_lambda <= lambda_mult <= max_lambda
        status = "✓" if is_in_range else "✗"

        print(f"\n{status} Query: {query}")
        print(f"  Type: {description}")
        print(f"  Expected range: {min_lambda:.1f}-{max_lambda:.1f}")
        print(f"  Got lambda_mult: {lambda_mult:.2f}")
        print(f"  Reasoning: {reasoning}")

        if is_in_range:
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(test_query_evaluator())

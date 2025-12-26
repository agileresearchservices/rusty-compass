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
    # Lambda interpretation: 0.0=pure lexical, 1.0=pure semantic
    test_cases = [
        # ===== Pure Lexical Queries (0.0-0.2) =====
        ("GPT-4 released in 2023", 0.0, 0.2, "Date + model number"),
        ("Model XR-2500 specifications", 0.0, 0.2, "Part number"),
        ("Intel Core i9-13900K specifications", 0.0, 0.2, "Product model specs"),
        ("RTX 4090 release date 2022", 0.0, 0.2, "Product + date"),
        ("SKU-XR-2500-A1 product manual", 0.0, 0.2, "Part number reference"),
        ("iPhone 15 Pro Max camera specs", 0.0, 0.2, "Device model details"),
        ("Google Pixel 8 Pro battery capacity mAh", 0.0, 0.2, "Device specs with unit"),
        ("Model RTX-3080-Ti VRAM 12GB", 0.0, 0.2, "Product identifier + specs"),

        # ===== Lexical-Heavy Queries (0.2-0.4) =====
        ("Django 4.2 authentication", 0.2, 0.4, "Version-specific"),
        ("Python 3.11 new features", 0.2, 0.4, "Version-specific features"),
        ("TensorFlow 2.13 API changes", 0.2, 0.4, "Library version updates"),
        ("Kubernetes 1.28 installation guide", 0.2, 0.4, "Tool version documentation"),
        ("Node.js version 20.5 release notes", 0.0, 0.3, "Runtime version info"),
        ("AWS Lambda pricing model 2024", 0.0, 0.15, "Service + version year"),

        # ===== Balanced Queries (0.4-0.6) =====
        ("MySQL 8.0 performance tuning", 0.4, 0.6, "Database version + optimization"),
        ("Python Flask REST API tutorial", 0.4, 0.6, "Balanced"),
        ("React hooks best practices", 0.4, 0.6, "Framework + technique"),
        ("Docker containerization guide", 0.4, 0.7, "Tool + process"),
        ("Express.js middleware configuration", 0.4, 0.6, "Framework + concept"),

        # ===== Semantic-Heavy Queries (0.6-0.8) =====
        ("Java Spring Boot microservices architecture", 0.6, 0.8, "Framework + architecture"),
        ("PostgreSQL query optimization techniques", 0.6, 0.8, "Database + method"),
        ("REST API design patterns", 0.6, 0.9, "Architecture + pattern"),
        ("Vue.js framework comparison", 0.6, 0.9, "Framework discussion"),

        # ===== Semantic/Conceptual Queries (0.8-1.0) =====
        ("What is machine learning?", 0.8, 1.0, "Semantic/conceptual"),
        ("Explain how neural networks learn", 0.8, 1.0, "Conceptual/educational"),
        ("What is the difference between supervised and unsupervised learning?", 0.8, 1.0, "Conceptual comparison"),
        ("Describe the concept of overfitting in machine learning", 0.8, 1.0, "Conceptual explanation"),
        ("What is object-oriented programming?", 0.8, 1.0, "Programming paradigm"),
        ("Explain recursion in computer science", 0.8, 1.0, "Algorithm concept"),
        ("How do I learn Python programming?", 0.8, 1.0, "Educational question"),
        ("What are design patterns?", 0.8, 1.0, "Software design concept"),
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

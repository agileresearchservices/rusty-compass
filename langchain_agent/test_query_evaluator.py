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
    # Note: Ranges are flexible as LLM evaluation can vary slightly
    test_cases = [
        # ===== Lexical-Heavy Queries (0.0-0.4) =====
        # These queries target specific identifiers, versions, or exact terms
        ("LangChain 0.3.0 release notes", 0.0, 0.4, "Version + release"),
        ("langchain-openai 0.2.0 changelog", 0.0, 0.4, "Package version"),
        ("ChatOpenAI model_name parameter", 0.0, 0.4, "API parameter"),
        ("LangGraph StateGraph class import", 0.0, 0.4, "Import statement"),
        ("LANGCHAIN_API_KEY environment variable", 0.0, 0.3, "Environment variable"),
        ("langchain_core.runnables RunnablePassthrough", 0.0, 0.4, "Module path"),
        ("LangSmith project ID configuration", 0.0, 0.4, "Config parameter"),
        ("BaseChatModel class signature", 0.0, 0.3, "Class reference"),

        # ===== Lexical-Leaning Queries (0.0-0.5) =====
        ("LangGraph checkpointer setup", 0.2, 0.5, "Feature setup"),
        ("LangChain LCEL syntax", 0.2, 0.7, "Framework syntax"),
        ("LangSmith tracing configuration", 0.2, 0.5, "Tool configuration"),
        ("ChatPromptTemplate from_messages method", 0.0, 0.5, "API method"),
        ("LangChain document loader types", 0.0, 0.7, "Loader types"),
        ("StructuredOutputParser schema format", 0.0, 0.5, "Parser format"),

        # ===== Balanced Queries (0.3-0.95) =====
        ("LangGraph state management patterns", 0.3, 0.7, "Framework + patterns"),
        ("LangChain RAG pipeline tutorial", 0.3, 0.7, "Balanced"),
        ("LangSmith evaluation best practices", 0.3, 0.95, "Tool + technique"),
        ("LangGraph workflow orchestration guide", 0.3, 0.8, "Tool + process"),
        ("LangChain memory configuration options", 0.3, 0.7, "Framework + concept"),

        # ===== Semantic-Heavy Queries (0.5-0.95) =====
        ("How to build multi-agent systems with LangGraph", 0.5, 0.95, "Architecture question"),
        ("LangChain retrieval optimization techniques", 0.5, 0.95, "Optimization + method"),
        ("LangSmith debugging strategies for agents", 0.5, 0.95, "Strategies"),
        ("Comparing chains and agents in LangChain", 0.5, 0.95, "Comparison"),

        # ===== Semantic/Conceptual Queries (0.7-1.0) =====
        ("What is LangGraph?", 0.7, 1.0, "Semantic/conceptual"),
        ("Explain how LangChain agents work", 0.7, 1.0, "Conceptual/educational"),
        ("What is the difference between LangChain and LangGraph?", 0.7, 1.0, "Conceptual comparison"),
        ("Describe the concept of tool calling in LangChain", 0.7, 1.0, "Conceptual explanation"),
        ("What is retrieval augmented generation?", 0.7, 1.0, "RAG concept"),
        ("How does LangSmith improve LLM development?", 0.7, 1.0, "Tool purpose"),
        ("How do I get started with LangChain?", 0.7, 1.0, "Educational question"),
        ("What are the benefits of using LangGraph for agents?", 0.7, 1.0, "Benefits question"),
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

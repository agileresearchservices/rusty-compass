#!/usr/bin/env python3
"""
Standalone Performance Test for BGE-Reranker
Expected timing: 10-15 seconds for 15 documents
"""
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple

def create_dummy_documents(count: int = 15) -> List[str]:
    """Create dummy documents"""
    documents = [
        "LangChain is a framework for developing applications powered by large language models",
        "LangGraph is a library for building stateful, multi-actor applications with LLMs",
        "StateGraph is the main class in LangGraph for defining workflows with nodes and edges",
        "LangSmith is a platform for debugging, testing, and monitoring LLM applications",
        "Document loaders in LangChain help ingest data from various sources",
        "Text splitters divide documents into smaller chunks for embedding",
        "Vector stores like Chroma and Pinecone store embeddings for similarity search",
        "Retrievers fetch relevant documents from vector stores based on query similarity",
        "LangChain agents use tools to interact with external systems",
        "LCEL is the expression language for building chains in LangChain",
        "Memory components help agents maintain conversation history",
        "Callbacks provide hooks for logging and monitoring LangChain applications",
        "Chains combine multiple components into reusable workflows",
        "Prompts are templates that format inputs for language models",
        "Output parsers extract structured data from LLM responses",
    ]
    return documents[:count]

def benchmark_reranker(batch_size: int = 8, max_length: int = 512):
    print("=" * 70)
    print(f"BGE-RERANKER PERFORMANCE BENCHMARK (max_length={max_length})")
    print("=" * 70)

    model_name = "BAAI/bge-reranker-v2-m3"
    query = "What is LangGraph and how do I create workflows with StateGraph?"

    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Batch size: {batch_size}")
    print(f"  Max length: {max_length}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # Load model
    print(f"\n[1/3] Loading model...")
    load_start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float16
    ).eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    load_time = time.time() - load_start
    print(f"  ✓ Model loaded in {load_time:.2f}s")

    # Create test data
    print(f"\n[2/3] Creating test documents...")
    documents = create_dummy_documents(15)
    print(f"  ✓ Created {len(documents)} documents")

    # Benchmark reranking
    print(f"\n[3/3] Benchmarking reranking...")
    rerank_start = time.time()
    scores = []

    for batch_idx in range(0, len(documents), batch_size):
        batch_docs = documents[batch_idx:batch_idx + batch_size]
        batch_start = time.time()

        # Format as pairs for BGE reranker
        pairs = [[query, doc] for doc in batch_docs]

        with torch.no_grad():
            inputs = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=max_length
            ).to(device)

            # Token stats
            token_lengths = [len(inp) for inp in inputs['input_ids']]
            avg_len = sum(token_lengths) / len(token_lengths)
            max_len = max(token_lengths)

            outputs = model(**inputs)
            logits = outputs.logits.view(-1).float()
            batch_scores = torch.sigmoid(logits).cpu().numpy()
            scores.extend(batch_scores)

        batch_time = time.time() - batch_start
        print(f"    Batch {batch_idx // batch_size + 1}: {len(batch_docs)} docs in {batch_time:.2f}s "
              f"(avg_tokens={avg_len:.0f}, max_tokens={max_len})")

    rerank_time = time.time() - rerank_start

    # Results
    print(f"\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nTiming:")
    print(f"  Reranking 15 docs: {rerank_time:.2f}s")
    print(f"  Average per doc: {rerank_time / 15:.3f}s")

    print(f"\nPerformance Assessment:")
    if rerank_time <= 15:
        print(f"  ✓ EXCELLENT: Within expected range (10-15s)")
    elif rerank_time <= 30:
        print(f"  ⚠ ACCEPTABLE: Slightly slower than expected")
    else:
        print(f"  ✗ CRITICAL: {rerank_time/15:.1f}x slower than expected!")
        print(f"     Expected: 10-15s, Got: {rerank_time:.2f}s")

    del model
    torch.cuda.empty_cache()
    print(f"\n" + "=" * 70)

    return rerank_time

if __name__ == "__main__":
    print("\nTesting BGE-Reranker with max_length=512:")
    time_512 = benchmark_reranker(batch_size=8, max_length=512)

    print("\n\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"\nmax_length=512: {time_512:.2f}s")

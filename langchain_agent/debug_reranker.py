#!/usr/bin/env python3
"""Debug script to inspect Qwen3-Reranker-8B model output format"""

import torch
from transformers import AutoTokenizer, AutoModel
from langchain_core.documents import Document

print("=" * 70)
print("QWEN3-RERANKER-8B OUTPUT FORMAT INSPECTION")
print("=" * 70)

# Load model
model_name = "Qwen/Qwen3-Reranker-8B"
print(f"\nLoading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True).eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"✓ Model loaded on: {device}")

# Test documents
query = "What is machine learning?"
docs = [
    Document(page_content="Machine learning is a subset of artificial intelligence", metadata={"source": "ml.txt"}),
    Document(page_content="Python is a programming language", metadata={"source": "python.txt"}),
]

print(f"\nQuery: {query}")
print(f"Documents: {len(docs)}")

# Test with each document
for i, doc in enumerate(docs, 1):
    print(f"\n--- Document {i}: {doc.metadata['source']} ---")

    # Prepare input
    pairs = [[query, doc.page_content]]

    with torch.no_grad():
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(device)

        print(f"Input shape: {inputs['input_ids'].shape}")

        # Get model output
        outputs = model(**inputs)

        # Inspect output attributes
        print(f"\nModel output type: {type(outputs)}")
        print(f"Output attributes: {outputs.keys() if hasattr(outputs, 'keys') else dir(outputs)}")

        # Check logits
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
            print(f"\n✓ Has logits attribute")
            print(f"  logits shape: {logits.shape}")
            print(f"  logits value: {logits}")
            print(f"  logits[0]: {logits[0]}")
            if logits.shape[0] >= 2:
                print(f"  Can apply softmax -> probs: {torch.softmax(logits[0], dim=-1)}")
        else:
            print(f"\n✗ No logits attribute")

        # Check last_hidden_state
        if hasattr(outputs, 'last_hidden_state'):
            lhs = outputs.last_hidden_state
            print(f"\n✓ Has last_hidden_state attribute")
            print(f"  shape: {lhs.shape}")
            print(f"  [CLS] token (first): {lhs[0, 0, :5]}...")  # Show first 5 dims
            print(f"  [CLS] shape: {lhs[0, 0].shape}")
        else:
            print(f"\n✗ No last_hidden_state attribute")

        # Try other common attributes
        for attr in ['score', 'scores', 'logit', 'embedding', 'embeddings']:
            if hasattr(outputs, attr):
                val = getattr(outputs, attr)
                print(f"\n✓ Found {attr}: shape={val.shape if hasattr(val, 'shape') else type(val)}")
                print(f"  value: {val}")

print("\n" + "=" * 70)
print("RECOMMENDATIONS:")
print("=" * 70)
print("""
1. If logits shape is (1, 2): Use softmax on logits[0] -> probs[1]
2. If logits shape is (1, 1): Use sigmoid on logits[0, 0]
3. If only last_hidden_state: May need to train/use pool output
4. Check HuggingFace model card for correct usage pattern

Based on typical cross-encoder patterns, scores should vary for different documents.
If all documents get the same score, the feature extraction is likely wrong.
""")

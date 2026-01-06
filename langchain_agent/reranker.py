"""
BGE Reranker implementation for cross-encoder document reranking.

Uses BAAI/bge-reranker-v2-m3 from HuggingFace for high-quality
relevance scoring of query-document pairs.
"""

import logging
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class BGEReranker:
    """
    Cross-encoder reranker using BAAI/bge-reranker-v2-m3 from HuggingFace.

    BGE Reranker is a sequence classification model that directly outputs
    relevance scores for query-document pairs. Scores are normalized to [0, 1]
    using sigmoid.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        """
        Initialize the BGE cross-encoder reranker model.

        Args:
            model_name: HuggingFace model identifier (default: BAAI/bge-reranker-v2-m3)

        Raises:
            OSError: If model cannot be downloaded from HuggingFace
            RuntimeError: If required CUDA libraries are missing
        """
        self.model_name = model_name

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load as sequence classification model in float16 for memory efficiency
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        ).eval()

        # Move to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

        logger.info(f"BGE-Reranker loaded on device: {self.device}")

    def score_documents(
        self,
        query: str,
        documents: List[Document],
        batch_size: int = 8
    ) -> List[Tuple[Document, float]]:
        """
        Score documents by relevance to query using batch processing.

        Args:
            query: The search query string
            documents: List of LangChain Document objects to score
            batch_size: Number of documents to process per batch (default: 8)

        Returns:
            List of (Document, score) tuples sorted by score descending.
            Scores are in range [0.0, 1.0] after sigmoid normalization.
        """
        if not documents:
            return []

        scores = []

        # Process documents in batches
        for batch_idx in range(0, len(documents), batch_size):
            batch_docs = documents[batch_idx:batch_idx + batch_size]

            # Create query-document pairs for tokenization
            pairs = [[query, doc.page_content] for doc in batch_docs]

            with torch.no_grad():
                # Tokenize all pairs in batch
                inputs = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512
                ).to(self.device)

                # Forward pass - get logits
                outputs = self.model(**inputs, return_dict=True)
                logits = outputs.logits.view(-1).float()

                # Apply sigmoid to normalize to [0, 1]
                batch_scores = torch.sigmoid(logits).cpu().numpy()

                # Pair each document with its score
                for doc, score in zip(batch_docs, batch_scores):
                    scores.append((doc, float(score)))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents and return top-k most relevant results.

        Args:
            query: The search query string
            documents: List of LangChain Document objects to rerank
            top_k: Maximum number of documents to return

        Returns:
            List of (Document, score) tuples for top-k results sorted by score descending.
        """
        scored_docs = self.score_documents(query, documents)
        return scored_docs[:top_k]

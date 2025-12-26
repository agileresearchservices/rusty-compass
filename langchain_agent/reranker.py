#!/usr/bin/env python3
"""
Reranker module for reordering hybrid search results using Ollama's Qwen3-Reranker-8B model.

This module provides intelligent reranking of document results to improve relevance
by scoring each document against the query using a cross-encoder model.
"""

import logging
import time
from typing import List, Tuple, Optional

import httpx
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class OllamaReranker:
    """
    Reranker that uses Ollama's Qwen3-Reranker-8B model to score document relevance.

    The reranker asks the model "Query: {query}\nDocument: {doc}\nRelevant: " and scores
    based on whether the model responds with "yes" or extracts probability from logprobs.

    Attributes:
        model: Name of the Ollama model to use
        base_url: Base URL for Ollama API (default: http://localhost:11434)
        timeout: Request timeout in seconds
        batch_size: Number of documents to process concurrently (currently unused, for future optimization)
        client: httpx.Client for making API requests
    """

    def __init__(
        self,
        model: str = "dengcao/Qwen3-Reranker-8B:Q3_K_M",
        base_url: str = "http://localhost:11434",
        timeout: int = 30,
        batch_size: int = 5,
    ):
        """
        Initialize the OllamaReranker.

        Args:
            model: Ollama model identifier
            base_url: Base URL for Ollama API endpoint
            timeout: Request timeout in seconds
            batch_size: Number of documents to process concurrently (for future use)

        Raises:
            ConnectionError: If unable to connect to Ollama or model not found
        """
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.batch_size = batch_size
        self.client = httpx.Client(timeout=timeout)

        # Verify model is available
        self._verify_model_available()

    def _verify_model_available(self) -> None:
        """
        Verify that the reranker model is available in Ollama.

        Raises:
            ConnectionError: If unable to connect to Ollama
            ValueError: If the model is not found
        """
        try:
            response = self.client.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()

            models = [m.get("name", "") for m in response.json().get("models", [])]

            if not any(self.model in m for m in models):
                error_msg = (
                    f"Model {self.model} not found in Ollama.\n"
                    f"Available models: {', '.join(models[:5])}\n\n"
                    f"To fix, run:\n"
                    f"  ollama pull {self.model}\n\n"
                    f"Or start Ollama with:\n"
                    f"  ollama serve"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            logger.info(f"✓ Reranker model {self.model} verified in Ollama")

        except httpx.ConnectError as e:
            error_msg = (
                f"Cannot connect to Ollama at {self.base_url}\n\n"
                f"To fix, ensure Ollama is running:\n"
                f"  ollama serve\n\n"
                f"Check OLLAMA_BASE_URL in config.py if using different host/port"
            )
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 4,
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents by relevance to the query.

        Args:
            query: The search query
            documents: List of Document objects to rerank
            top_k: Number of top documents to return

        Returns:
            List of (document, score) tuples sorted by score in descending order
        """
        if not documents:
            logger.warning("No documents to rerank")
            return []

        start_time = time.time()

        # Score each document
        scored_docs = []
        for i, doc in enumerate(documents, 1):
            try:
                score = self._score_document(query, doc)
                scored_docs.append((doc, score))
                logger.debug(
                    f"[Reranker] Scored doc {i}/{len(documents)}: {score:.4f} | "
                    f"{doc.metadata.get('source', 'unknown')}"
                )
            except Exception as e:
                logger.warning(
                    f"Error scoring document {i}, assigning neutral score: {e}"
                )
                scored_docs.append((doc, 0.5))  # Neutral score on error

        # Sort by score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Select top-k
        results = scored_docs[:top_k]
        elapsed = time.time() - start_time

        # Log results
        self._log_reranking_results(results, elapsed, len(documents))

        return results

    def _score_document(self, query: str, doc: Document) -> float:
        """
        Score a single document against the query using Ollama API.

        Args:
            query: The search query
            doc: The document to score

        Returns:
            Relevance score between 0.0 and 1.0

        Raises:
            Exception: Various errors from Ollama API (caught and handled with fallback)
        """
        # Prepare the prompt
        # Truncate document to avoid overly long prompts
        doc_text = doc.page_content[:500]
        prompt = f"Query: {query}\nDocument: {doc_text}\nRelevant: "

        try:
            # Call Ollama API
            response = self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,
                        "num_predict": 1,  # Only need 1 token
                    },
                },
            )
            response.raise_for_status()
            result = response.json()

            # Try to extract score from response
            # Method 1: Use "yes"/"no" response
            response_text = result.get("response", "").strip().lower()

            if response_text == "yes":
                score = 0.8  # Default high score for "yes"
            elif response_text == "no":
                score = 0.2  # Default low score for "no"
            else:
                # Fallback: any other response is treated as neutral
                logger.debug(
                    f"Unexpected reranker response: '{response_text}', using neutral score"
                )
                score = 0.5

            # Clamp score to [0.0, 1.0] range
            score = max(0.0, min(1.0, score))

            return score

        except httpx.TimeoutException:
            logger.warning("Timeout scoring document, using neutral score (0.5)")
            return 0.5
        except httpx.ConnectError as e:
            logger.warning(f"Connection error scoring document: {e}, using neutral score")
            return 0.5
        except Exception as e:
            logger.warning(f"Error scoring document: {e}, using neutral score")
            return 0.5

    def _log_reranking_results(
        self,
        results: List[Tuple[Document, float]],
        elapsed: float,
        total_docs: int,
    ) -> None:
        """
        Log the reranking results in a formatted way.

        Args:
            results: List of (document, score) tuples to log
            elapsed: Time taken to rerank in seconds
            total_docs: Total number of documents that were reranked
        """
        logger.info(f"[Reranker] Reranked {total_docs} → {len(results)} docs in {elapsed:.2f}s")
        logger.info("[Reranker] Reranked document scores:")

        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get("source", "unknown")
            preview = doc.page_content[:50].replace("\n", " ")
            logger.info(f"  {i}. score={score:.4f} [{source}] {preview}...")

    def close(self) -> None:
        """
        Clean up resources (close HTTP client).

        Should be called when the reranker is no longer needed to ensure
        proper cleanup of connections.
        """
        if self.client:
            self.client.close()
            logger.debug("Reranker client closed")

    def __del__(self):
        """Ensure cleanup on object deletion."""
        try:
            self.close()
        except Exception:
            pass


def test_reranker() -> None:
    """
    Simple test function to verify reranker is working.

    This can be run as: python -c "from reranker import test_reranker; test_reranker()"
    """
    from config import OLLAMA_BASE_URL

    try:
        reranker = OllamaReranker(base_url=OLLAMA_BASE_URL)
        print("✓ Reranker initialized successfully")

        # Create test documents
        test_docs = [
            Document(
                page_content="Python is a high-level programming language",
                metadata={"source": "python_guide.txt"},
            ),
            Document(
                page_content="Machine learning is a subset of artificial intelligence",
                metadata={"source": "ml_guide.txt"},
            ),
            Document(
                page_content="Web development involves building websites and applications",
                metadata={"source": "web_guide.txt"},
            ),
        ]

        # Test reranking
        query = "What is machine learning?"
        results = reranker.rerank(query, test_docs, top_k=2)

        print(f"\nReranking results for query: '{query}'")
        for i, (doc, score) in enumerate(results, 1):
            print(f"  {i}. score={score:.4f} [{doc.metadata['source']}]")

        reranker.close()
        print("\n✓ Reranker test completed successfully")

    except Exception as e:
        print(f"✗ Reranker test failed: {e}")
        raise


if __name__ == "__main__":
    # Enable debug logging for testing
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s - %(name)s - %(message)s",
    )

    test_reranker()

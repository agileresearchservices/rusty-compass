"""
PostgreSQL-based vector store with hybrid search capabilities.

Provides:
- SimplePostgresVectorStore: Main vector store with RRF-based hybrid search
- PostgresRetriever: LangChain-compatible retriever interface
"""

import logging
from typing import List, Optional, Dict, Any, Union

import psycopg
from psycopg_pool import ConnectionPool
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from config import RETRIEVER_K, RETRIEVER_FETCH_K, RETRIEVER_LAMBDA_MULT

logger = logging.getLogger(__name__)


class SimplePostgresVectorStore:
    """
    PostgreSQL-based vector store for semantic and hybrid document search.

    Combines vector similarity search (768-dimensional embeddings) with full-text
    search using Reciprocal Rank Fusion (RRF) for improved retrieval quality.

    Attributes:
        embeddings: OllamaEmbeddings instance for generating query embeddings
        collection_id: PostgreSQL collection ID for document filtering
        pool: ConnectionPool for database connections (max_size=20)
        database_url: PostgreSQL connection string (optional, for compatibility)
    """

    def __init__(
        self,
        embeddings: OllamaEmbeddings,
        collection_id: str,
        pool: ConnectionPool,
        database_url: Optional[str] = None
    ) -> None:
        self.embeddings: OllamaEmbeddings = embeddings
        self.collection_id: str = collection_id
        self.pool: ConnectionPool = pool
        self.database_url: Optional[str] = database_url  # Kept for backwards compatibility

    def as_retriever(
        self,
        search_type: str = "similarity",
        search_kwargs: Optional[Dict[str, Any]] = None
    ) -> "PostgresRetriever":
        """Return a retriever interface."""
        if search_kwargs is None:
            search_kwargs = {
                "k": RETRIEVER_K,
                "fetch_k": RETRIEVER_FETCH_K,
                "lambda_mult": RETRIEVER_LAMBDA_MULT,
            }

        return PostgresRetriever(
            self,
            search_type=search_type,
            k=search_kwargs.get("k", RETRIEVER_K),
            fetch_k=search_kwargs.get("fetch_k", RETRIEVER_FETCH_K),
            lambda_mult=search_kwargs.get("lambda_mult", RETRIEVER_LAMBDA_MULT)
        )

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Search for document chunks similar to the query using vector embeddings.

        Generates a query embedding and finds the k most similar document chunks
        from the PostgreSQL vector store using cosine distance metric.

        Args:
            query: The search query string
            k: Number of similar documents to return (default: 4)

        Returns:
            List of k most similar LangChain Document objects with metadata

        Raises:
            Exception: If embedding generation or database query fails
        """
        try:
            # Generate query embedding
            query_embedding: List[float] = self.embeddings.embed_query(query)
            embedding_str: str = "[" + ",".join(str(float(e)) for e in query_embedding) + "]"

            # Search in PostgreSQL document chunks
            with self.pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        SELECT dc.content, d.metadata FROM document_chunks dc
                        JOIN documents d ON dc.document_id = d.id
                        WHERE d.collection_id = %s
                        ORDER BY dc.embedding <-> %s::vector
                        LIMIT %s
                        """,
                        (self.collection_id, embedding_str, k)
                    )

                    results: List[Document] = []
                    for row in cur.fetchall():
                        # Handle both dict rows (from pool) and tuple rows
                        if isinstance(row, dict):
                            content = row.get('content', '')
                            metadata = row.get('metadata', {})
                        else:
                            content, metadata = row

                        doc = Document(
                            page_content=content,
                            metadata=metadata or {}
                        )
                        results.append(doc)

                    return results

        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []

    def hybrid_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5
    ) -> List[Document]:
        """
        Hybrid search combining vector similarity and full-text search using RRF.

        Uses Reciprocal Rank Fusion (RRF) to combine:
        1. Vector similarity search (semantic matching via embeddings)
        2. PostgreSQL full-text search (lexical/keyword matching via tsvector)

        Args:
            query: Search query string
            k: Number of final results to return
            fetch_k: Number of candidates to fetch from each method before reranking
            lambda_mult: Weight for text vs vector (0.0=pure vector, 1.0=pure text)

        Returns:
            List of Document objects ranked by RRF score
        """
        # Edge case: pure vector search
        if lambda_mult == 0.0:
            return self.similarity_search(query, k)

        # Edge case: pure text search
        if lambda_mult == 1.0:
            return self._text_search(query, k)

        # Validate lambda
        if not 0.0 <= lambda_mult <= 1.0:
            raise ValueError(f"lambda_mult must be in [0.0, 1.0], got {lambda_mult}")

        try:
            # Generate query embedding for vector search
            query_embedding: List[float] = self.embeddings.embed_query(query)
            embedding_str: str = "[" + ",".join(str(float(e)) for e in query_embedding) + "]"

            # Calculate weights for RRF
            vector_weight = 1.0 - lambda_mult
            text_weight = lambda_mult

            # Execute hybrid query using RRF
            with self.pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        WITH vector_results AS (
                            -- Vector similarity search
                            SELECT
                                dc.id,
                                dc.content,
                                d.metadata,
                                ROW_NUMBER() OVER (ORDER BY dc.embedding <-> %s::vector) AS vector_rank
                            FROM document_chunks dc
                            JOIN documents d ON dc.document_id = d.id
                            WHERE d.collection_id = %s
                            ORDER BY dc.embedding <-> %s::vector
                            LIMIT %s
                        ),
                        text_results AS (
                            -- Full-text search
                            SELECT
                                dc.id,
                                dc.content,
                                d.metadata,
                                ROW_NUMBER() OVER (
                                    ORDER BY ts_rank_cd(dc.content_tsv, websearch_to_tsquery('english', %s)) DESC
                                ) AS text_rank
                            FROM document_chunks dc
                            JOIN documents d ON dc.document_id = d.id
                            WHERE d.collection_id = %s
                              AND dc.content_tsv @@ websearch_to_tsquery('english', %s)
                            ORDER BY ts_rank_cd(dc.content_tsv, websearch_to_tsquery('english', %s)) DESC
                            LIMIT %s
                        ),
                        combined_results AS (
                            -- Union of both result sets
                            SELECT DISTINCT
                                COALESCE(vr.id, tr.id) AS id,
                                COALESCE(vr.content, tr.content) AS content,
                                COALESCE(vr.metadata, tr.metadata) AS metadata,
                                COALESCE(vr.vector_rank, 999999) AS vector_rank,
                                COALESCE(tr.text_rank, 999999) AS text_rank
                            FROM vector_results vr
                            FULL OUTER JOIN text_results tr ON vr.id = tr.id
                        )
                        SELECT
                            id,
                            content,
                            metadata,
                            (
                                (%s / (60.0 + vector_rank)) +
                                (%s / (60.0 + text_rank))
                            ) AS rrf_score
                        FROM combined_results
                        ORDER BY rrf_score DESC
                        LIMIT %s
                        """,
                        (
                            embedding_str,      # vector_results: embedding comparison
                            self.collection_id, # vector_results: collection filter
                            embedding_str,      # vector_results: ORDER BY
                            fetch_k,            # vector_results: LIMIT
                            query,              # text_results: query 1
                            self.collection_id, # text_results: collection filter
                            query,              # text_results: query 2
                            query,              # text_results: query 3
                            fetch_k,            # text_results: LIMIT
                            vector_weight,      # RRF vector weight
                            text_weight,        # RRF text weight
                            k                   # Final LIMIT
                        )
                    )

                    results: List[Document] = []
                    for row in cur.fetchall():
                        # Handle both dict rows (from pool) and tuple rows
                        if isinstance(row, dict):
                            content = row.get('content', '')
                            metadata = row.get('metadata', {})
                        else:
                            # Unpack tuple: (id, content, metadata, rrf_score)
                            _, content, metadata, _ = row

                        results.append(Document(
                            page_content=content,
                            metadata=metadata or {}
                        ))

                    return results

        except psycopg.errors.UndefinedColumn as e:
            if "content_tsv" in str(e):
                logger.warning("Full-text search not available. Run migration first.")
                logger.info("Falling back to pure vector search...")
                return self.similarity_search(query, k)
            raise

        except Exception as e:
            logger.error(f"Error during hybrid search: {e}")
            return []

    def _text_search(self, query: str, k: int = 4) -> List[Document]:
        """Pure full-text search fallback for lambda=1.0."""
        try:
            with self.pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT dc.content, d.metadata
                        FROM document_chunks dc
                        JOIN documents d ON dc.document_id = d.id
                        WHERE d.collection_id = %s
                          AND dc.content_tsv @@ websearch_to_tsquery('english', %s)
                        ORDER BY ts_rank_cd(dc.content_tsv, websearch_to_tsquery('english', %s)) DESC
                        LIMIT %s
                        """,
                        (self.collection_id, query, query, k)
                    )

                    results: List[Document] = []
                    for row in cur.fetchall():
                        if isinstance(row, dict):
                            content = row.get('content', '')
                            metadata = row.get('metadata', {})
                        else:
                            content, metadata = row

                        results.append(Document(
                            page_content=content,
                            metadata=metadata or {}
                        ))

                    return results

        except Exception as e:
            logger.error(f"Error during text search: {e}")
            return []


class PostgresRetriever:
    """Retriever interface for PostgreSQL vector store."""

    def __init__(
        self,
        vector_store: SimplePostgresVectorStore,
        search_type: str = "similarity",
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5
    ) -> None:
        self.vector_store: SimplePostgresVectorStore = vector_store
        self.search_type: str = search_type
        self.k: int = k
        self.fetch_k: int = fetch_k
        self.lambda_mult: float = lambda_mult

    def invoke(
        self,
        input_dict: Union[Dict[str, Any], str]
    ) -> List[Document]:
        """
        Retrieve documents for a query.

        Args:
            input_dict: Either a dictionary with 'input' or 'query' key,
                       or a string query directly

        Returns:
            List of Document objects matching the query
        """
        if isinstance(input_dict, dict):
            query: str = input_dict.get("input") or input_dict.get("query", "")
        else:
            query: str = str(input_dict)

        if self.search_type == "hybrid":
            return self.vector_store.hybrid_search(
                query,
                k=self.k,
                fetch_k=self.fetch_k,
                lambda_mult=self.lambda_mult
            )
        elif self.search_type == "similarity":
            return self.vector_store.similarity_search(query, k=self.k)
        else:
            raise ValueError(f"Unknown search_type: {self.search_type}")

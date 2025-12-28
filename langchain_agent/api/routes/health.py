"""
Health check endpoints for monitoring API and dependencies.
"""

import psycopg
from fastapi import APIRouter

import sys
sys.path.insert(0, '/Users/kevin/github/personal/rusty-compass/langchain_agent')
from config import DATABASE_URL, OLLAMA_BASE_URL

router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Check health of API and all dependencies.

    Returns:
        Health status of postgres, ollama, and overall system.
    """
    status = {
        "status": "ok",
        "postgres": False,
        "ollama": False,
        "vector_store": False,
    }

    # Check PostgreSQL
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                status["postgres"] = True

                # Check vector store has documents
                cur.execute("""
                    SELECT COUNT(*) FROM document_chunks
                    WHERE collection_id = (
                        SELECT id FROM collections WHERE name = 'langchain_docs'
                    )
                """)
                doc_count = cur.fetchone()[0]
                status["vector_store"] = doc_count > 0
                status["document_count"] = doc_count
    except Exception as e:
        status["postgres_error"] = str(e)

    # Check Ollama
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
            status["ollama"] = response.status_code == 200
    except Exception as e:
        status["ollama_error"] = str(e)

    # Overall status
    if not all([status["postgres"], status["ollama"]]):
        status["status"] = "degraded"

    return status


@router.get("/health/ready")
async def readiness_check():
    """
    Kubernetes-style readiness probe.
    Returns 200 if ready to accept traffic.
    """
    health = await health_check()
    if health["status"] == "ok":
        return {"ready": True}
    return {"ready": False, "reason": health}

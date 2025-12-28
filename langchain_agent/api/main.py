"""
FastAPI application with WebSocket support for real-time agent streaming.

This is the main entry point for the LangChain Agent API.
Run with: uvicorn api.main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import health, conversations, chat

app = FastAPI(
    title="LangChain Agent API",
    description="WebSocket-based API for observing agent execution with full observability",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",  # Alternative dev port
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register REST routes
app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(conversations.router, prefix="/api", tags=["conversations"])

# Register WebSocket route
app.include_router(chat.router, tags=["chat"])


@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    print("=" * 60)
    print("LangChain Agent API Starting...")
    print("=" * 60)
    print("  REST API:    http://localhost:8000/api")
    print("  WebSocket:   ws://localhost:8000/ws/chat")
    print("  API Docs:    http://localhost:8000/docs")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown."""
    print("LangChain Agent API shutting down...")

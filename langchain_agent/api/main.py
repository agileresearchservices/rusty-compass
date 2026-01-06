"""
FastAPI application with WebSocket support for real-time agent streaming.

This is the main entry point for the LangChain Agent API.
Run with: uvicorn api.main:app --reload --port 8000
"""

import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from config import RATE_LIMIT_ENABLED, API_KEY
from api.routes import health, conversations, chat
from api.middleware.auth import AuthConfigurationError
from logging_config import configure_logging, get_logger

# Configure structured logging
configure_logging()
logger = get_logger(__name__)

# Initialize rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    enabled=RATE_LIMIT_ENABLED,
)

app = FastAPI(
    title="LangChain Agent API",
    description="WebSocket-based API for observing agent execution with full observability",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add rate limiter to app state
app.state.limiter = limiter

# Register rate limit exceeded handler
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
    # Validate API_KEY is configured
    if not API_KEY:
        raise AuthConfigurationError(
            "API_KEY environment variable is not set. "
            "Authentication is required. Set API_KEY in your .env file."
        )

    logger.info(
        "api_started",
        rest_api="http://localhost:8000/api",
        websocket="ws://localhost:8000/ws/chat",
        docs="http://localhost:8000/docs",
        auth_required=True,
    )


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown."""
    logger.info("api_shutting_down")
    # Clean up the connection manager and agent service
    await chat.manager.shutdown()
    logger.info("api_shutdown_complete")

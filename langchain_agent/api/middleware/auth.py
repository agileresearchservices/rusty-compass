"""
API Key authentication middleware for FastAPI.

Provides:
- Header-based authentication (X-API-Key) for REST endpoints
- Query parameter authentication (api_key) for WebSocket endpoints
- Authentication is always required (API_KEY must be configured)
"""

import sys
from typing import Optional

from fastapi import Request, HTTPException, WebSocket, status
from fastapi.security import APIKeyHeader

# Add parent directory to path for config import
sys.path.insert(0, str(__file__).rsplit("/api/", 1)[0])
from config import API_KEY, API_KEY_HEADER, API_KEY_QUERY_PARAM

# API Key security scheme for OpenAPI documentation
api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


class AuthConfigurationError(Exception):
    """Raised when API_KEY is not configured."""
    pass


def validate_api_key_configured() -> None:
    """
    Validate that API_KEY is configured.

    Raises:
        AuthConfigurationError: If API_KEY environment variable is not set
    """
    if not API_KEY:
        raise AuthConfigurationError(
            "API_KEY environment variable is not set. "
            "Authentication is required. Set API_KEY in your .env file."
        )


async def verify_api_key(request: Request, api_key: Optional[str] = None) -> bool:
    """
    Verify API key from request header.

    Args:
        request: FastAPI request object
        api_key: Optional API key from dependency injection

    Returns:
        True if authentication passes

    Raises:
        HTTPException: If authentication fails (401 Unauthorized)
        AuthConfigurationError: If API_KEY is not configured
    """
    validate_api_key_configured()

    # Get API key from header
    provided_key = api_key or request.headers.get(API_KEY_HEADER)

    if not provided_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is required. Provide X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if provided_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return True


async def verify_websocket_api_key(websocket: WebSocket) -> bool:
    """
    Verify API key for WebSocket connections.

    WebSocket connections use query parameter authentication since
    headers cannot be set during WebSocket handshake in browsers.

    Args:
        websocket: FastAPI WebSocket object

    Returns:
        True if authentication passes, False if connection was closed

    Note:
        If authentication fails, the WebSocket connection is closed with
        code 4001 (custom unauthorized code) and this function returns False.
    """
    # Validate API_KEY is configured
    if not API_KEY:
        await websocket.close(
            code=4002,
            reason="Server misconfiguration: API_KEY not set"
        )
        return False

    # Get API key from query parameter
    provided_key = websocket.query_params.get(API_KEY_QUERY_PARAM)

    if not provided_key:
        await websocket.close(
            code=4001,
            reason="API key required. Provide api_key query parameter."
        )
        return False

    if provided_key != API_KEY:
        await websocket.close(code=4001, reason="Invalid API key")
        return False

    return True

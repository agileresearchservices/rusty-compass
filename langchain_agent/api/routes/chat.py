"""
WebSocket endpoint for real-time chat with agent observability.
"""

import asyncio
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter
from slowapi.util import get_remote_address

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import RATE_LIMIT_CHAT
from api.schemas.events import (
    AgentCompleteEvent,
    AgentErrorEvent,
    ConnectionEstablished,
    BaseEvent,
)
from api.middleware.auth import verify_api_key, verify_websocket_api_key
from logging_config import get_logger

logger = get_logger(__name__)

# Thread ID pattern: starts with letter, alphanumeric with underscores/hyphens, max 64 chars
THREAD_ID_PATTERN = re.compile(r'^[a-zA-Z][a-zA-Z0-9_-]{0,63}$')

# Initialize limiter (will use app.state.limiter)
limiter = Limiter(key_func=get_remote_address)

router = APIRouter()


# ============================================================================
# CONNECTION MANAGER
# ============================================================================


class ConnectionManager:
    """Manages WebSocket connections and message routing."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self._agent_service = None  # Lazy initialization

    @property
    def agent_service(self):
        """Lazy load agent service to avoid slow startup."""
        if self._agent_service is None:
            from api.services.observable_agent import ObservableAgentService
            self._agent_service = ObservableAgentService()
        return self._agent_service

    async def connect(self, websocket: WebSocket, thread_id: str) -> bool:
        """
        Accept WebSocket connection and register it.

        Args:
            websocket: The WebSocket connection
            thread_id: Conversation thread ID

        Returns:
            True if connection successful
        """
        await websocket.accept()
        self.active_connections[thread_id] = websocket
        return True

    async def disconnect(self, thread_id: str):
        """Remove connection from active connections."""
        if thread_id in self.active_connections:
            del self.active_connections[thread_id]

    async def emit_event(self, thread_id: str, event: BaseEvent):
        """
        Send an event to a specific connection.

        Args:
            thread_id: Target connection's thread ID
            event: Event to send
        """
        if thread_id in self.active_connections:
            websocket = self.active_connections[thread_id]
            try:
                await websocket.send_json(event.model_dump(mode="json"))
            except Exception as e:
                logger.error("websocket_send_error", thread_id=thread_id, error=str(e))

    def get_connection_count(self) -> int:
        """Return number of active connections."""
        return len(self.active_connections)

    async def shutdown(self):
        """
        Shutdown the connection manager and clean up resources.

        This should be called during application shutdown to properly
        release memory held by the agent service, including models and
        database connections.
        """
        # Close all active connections
        for thread_id in list(self.active_connections.keys()):
            try:
                websocket = self.active_connections[thread_id]
                await websocket.close()
            except Exception as e:
                logger.error("websocket_close_error", thread_id=thread_id, error=str(e))
        self.active_connections.clear()

        # Cleanup agent service if initialized
        if self._agent_service is not None:
            try:
                await self._agent_service.cleanup()
                self._agent_service = None
            except Exception as e:
                logger.error("agent_service_cleanup_error", error=str(e))


# Global connection manager instance
manager = ConnectionManager()


# ============================================================================
# REQUEST MODELS
# ============================================================================


class ChatMessage(BaseModel):
    """Incoming chat message from client with validation."""

    type: str = "chat_message"
    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="User message content (1-4000 characters)"
    )
    thread_id: Optional[str] = Field(
        None,
        description="Conversation thread ID (optional)"
    )

    @field_validator('thread_id')
    @classmethod
    def validate_thread_id(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if not THREAD_ID_PATTERN.match(v):
            raise ValueError(
                'thread_id must start with a letter and contain only '
                'alphanumeric characters, underscores, or hyphens (max 64 chars)'
            )
        return v

    @field_validator('message')
    @classmethod
    def validate_message_content(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError('message cannot be empty or whitespace only')
        return v


# ============================================================================
# WEBSOCKET ENDPOINT
# ============================================================================


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for chat with real-time observability.

    Protocol:
        1. Client connects with ?thread_id=xxx&api_key=xxx query params
        2. Server verifies API key authentication
        3. Server sends connection_established event
        4. Client sends chat_message events
        5. Server streams observability events as agent executes
        6. Server sends agent_complete or agent_error when done

    Query Parameters:
        thread_id: Conversation thread ID (optional, generated if not provided)
        api_key: API key for authentication (required)

    Client Message Format:
        {
            "type": "chat_message",
            "message": "What is LangGraph?",
            "thread_id": "conv_abc123"  // optional
        }

    Server Event Format:
        See api/schemas/events.py for all event types
    """
    # Verify API key authentication before accepting connection
    if not await verify_websocket_api_key(websocket):
        return  # Connection closed by verify function

    # Get or generate thread ID
    thread_id = websocket.query_params.get("thread_id")
    if not thread_id:
        thread_id = f"conversation_{uuid.uuid4().hex[:8]}"

    # Accept connection
    await manager.connect(websocket, thread_id)

    # Load existing message count from checkpoint
    existing_count = 0
    try:
        pool = manager.agent_service._agent.pool if manager.agent_service and manager.agent_service._agent else None

        if pool:
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    # Query checkpoint_blobs for existing messages
                    await cur.execute("""
                        SELECT blob, type
                        FROM checkpoint_blobs
                        WHERE thread_id = %s
                          AND channel = 'messages'
                        ORDER BY version DESC
                        LIMIT 1
                    """, (thread_id,))

                    blob_row = await cur.fetchone()

                    if blob_row and blob_row[0]:
                        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
                        blob, blob_type = blob_row
                        serializer = JsonPlusSerializer()
                        raw_messages = serializer.loads_typed((blob_type, blob))

                        # Count human and AI messages with content
                        existing_count = sum(
                            1 for msg in raw_messages
                            if hasattr(msg, 'type') and msg.type in ('human', 'ai')
                            and hasattr(msg, 'content') and msg.content
                        )
    except Exception as e:
        logger.warning("message_count_load_error", thread_id=thread_id, error=str(e))

    # Send connection established event
    await manager.emit_event(
        thread_id,
        ConnectionEstablished(
            thread_id=thread_id,
            existing_messages=existing_count,
        )
    )

    try:
        # Ensure agent service is initialized
        await manager.agent_service.ensure_initialized()

        while True:
            # Wait for client message
            data = await websocket.receive_json()

            if data.get("type") == "chat_message":
                message = data.get("message", "").strip()
                msg_thread_id = data.get("thread_id", thread_id)

                if not message:
                    continue

                # Create emit callback for this request
                async def emit_callback(event: BaseEvent):
                    await manager.emit_event(msg_thread_id, event)

                # Process message through agent
                try:
                    await manager.agent_service.process_message(
                        message=message,
                        thread_id=msg_thread_id,
                        emit=emit_callback,
                    )
                except Exception as e:
                    await manager.emit_event(
                        msg_thread_id,
                        AgentErrorEvent(
                            error=str(e),
                            recoverable=True,
                        )
                    )

    except WebSocketDisconnect:
        await manager.disconnect(thread_id)
    except Exception as e:
        logger.error("websocket_error", thread_id=thread_id, error=str(e))
        await manager.disconnect(thread_id)


# ============================================================================
# REST FALLBACK (for testing without WebSocket)
# ============================================================================


class ChatRequest(BaseModel):
    """REST chat request (non-streaming fallback) with validation."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="User message content (1-4000 characters)"
    )
    thread_id: Optional[str] = Field(
        None,
        description="Conversation thread ID (optional)"
    )

    @field_validator('thread_id')
    @classmethod
    def validate_thread_id(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if not THREAD_ID_PATTERN.match(v):
            raise ValueError(
                'thread_id must start with a letter and contain only '
                'alphanumeric characters, underscores, or hyphens (max 64 chars)'
            )
        return v

    @field_validator('message')
    @classmethod
    def validate_message_content(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError('message cannot be empty or whitespace only')
        return v


class ChatResponse(BaseModel):
    """REST chat response."""

    thread_id: str
    response: str
    duration_ms: float


@router.post("/api/chat", response_model=ChatResponse)
@limiter.limit(RATE_LIMIT_CHAT)
async def chat_rest(request: Request, chat_request: ChatRequest):
    """
    REST endpoint for chat (non-streaming fallback).

    For real-time streaming with observability, use the WebSocket endpoint.

    Requires X-API-Key header for authentication.

    Args:
        request: FastAPI request object (for auth and rate limiting)
        chat_request: Chat request with message and optional thread_id

    Returns:
        Chat response with agent's answer.
    """
    # Verify API key authentication
    await verify_api_key(request)

    thread_id = chat_request.thread_id or f"conversation_{uuid.uuid4().hex[:8]}"

    start_time = datetime.utcnow()

    # Collect events (we won't stream them in REST mode)
    events = []

    async def collect_event(event: BaseEvent):
        events.append(event)

    await manager.agent_service.ensure_initialized()

    try:
        final_response = await manager.agent_service.process_message(
            message=chat_request.message,
            thread_id=thread_id,
            emit=collect_event,
        )

        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        return ChatResponse(
            thread_id=thread_id,
            response=final_response or "No response generated",
            duration_ms=duration_ms,
        )
    except Exception as e:
        raise Exception(f"Agent error: {e}")

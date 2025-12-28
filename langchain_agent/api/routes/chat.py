"""
WebSocket endpoint for real-time chat with agent observability.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from api.schemas.events import (
    AgentCompleteEvent,
    AgentErrorEvent,
    ConnectionEstablished,
    BaseEvent,
)

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
                print(f"Error sending event to {thread_id}: {e}")

    def get_connection_count(self) -> int:
        """Return number of active connections."""
        return len(self.active_connections)


# Global connection manager instance
manager = ConnectionManager()


# ============================================================================
# REQUEST MODELS
# ============================================================================


class ChatMessage(BaseModel):
    """Incoming chat message from client."""

    type: str = "chat_message"
    message: str
    thread_id: Optional[str] = None


# ============================================================================
# WEBSOCKET ENDPOINT
# ============================================================================


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for chat with real-time observability.

    Protocol:
        1. Client connects with optional ?thread_id=xxx query param
        2. Server sends connection_established event
        3. Client sends chat_message events
        4. Server streams observability events as agent executes
        5. Server sends agent_complete or agent_error when done

    Client Message Format:
        {
            "type": "chat_message",
            "message": "What is LangGraph?",
            "thread_id": "conv_abc123"  // optional
        }

    Server Event Format:
        See api/schemas/events.py for all event types
    """
    # Get or generate thread ID
    thread_id = websocket.query_params.get("thread_id")
    if not thread_id:
        thread_id = f"conversation_{uuid.uuid4().hex[:8]}"

    # Accept connection
    await manager.connect(websocket, thread_id)

    # Send connection established event
    await manager.emit_event(
        thread_id,
        ConnectionEstablished(
            thread_id=thread_id,
            existing_messages=0,  # TODO: Load from checkpoint
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
        print(f"WebSocket error: {e}")
        await manager.disconnect(thread_id)


# ============================================================================
# REST FALLBACK (for testing without WebSocket)
# ============================================================================


class ChatRequest(BaseModel):
    """REST chat request (non-streaming fallback)."""

    message: str
    thread_id: Optional[str] = None


class ChatResponse(BaseModel):
    """REST chat response."""

    thread_id: str
    response: str
    duration_ms: float


@router.post("/api/chat", response_model=ChatResponse)
async def chat_rest(request: ChatRequest):
    """
    REST endpoint for chat (non-streaming fallback).

    For real-time streaming with observability, use the WebSocket endpoint.

    Args:
        request: Chat request with message and optional thread_id

    Returns:
        Chat response with agent's answer.
    """
    thread_id = request.thread_id or f"conversation_{uuid.uuid4().hex[:8]}"

    start_time = datetime.utcnow()

    # Collect events (we won't stream them in REST mode)
    events = []

    async def collect_event(event: BaseEvent):
        events.append(event)

    await manager.agent_service.ensure_initialized()

    try:
        final_response = await manager.agent_service.process_message(
            message=request.message,
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

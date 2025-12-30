"""
REST endpoints for managing conversations.
"""

from datetime import datetime
from typing import List, Optional

import psycopg
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import sys
sys.path.insert(0, '/Users/kevin/github/personal/rusty-compass/langchain_agent')
from config import DATABASE_URL

router = APIRouter()


# ============================================================================
# RESPONSE MODELS
# ============================================================================


class ConversationSummary(BaseModel):
    """Summary of a conversation for listing."""

    thread_id: str
    title: str
    created_at: datetime
    updated_at: Optional[datetime] = None


class ConversationDetail(BaseModel):
    """Full conversation details including messages."""

    thread_id: str
    title: str
    created_at: datetime
    message_count: int
    messages: List[dict]


class DeleteResponse(BaseModel):
    """Response after deleting conversations."""

    deleted_metadata: int
    deleted_checkpoints: int


# ============================================================================
# ENDPOINTS
# ============================================================================


@router.get("/conversations", response_model=List[ConversationSummary])
async def list_conversations(limit: int = 20):
    """
    List all previous conversations with titles and dates.

    Args:
        limit: Maximum number of conversations to return (default 20)

    Returns:
        List of conversation summaries ordered by most recent first.
    """
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT thread_id, title, created_at, updated_at
                    FROM conversation_metadata
                    ORDER BY COALESCE(updated_at, created_at) DESC
                    LIMIT %s
                """, (limit,))

                conversations = []
                for row in cur.fetchall():
                    conversations.append(ConversationSummary(
                        thread_id=row[0],
                        title=row[1],
                        created_at=row[2],
                        updated_at=row[3],
                    ))

                return conversations
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


@router.get("/conversations/{thread_id}", response_model=ConversationDetail)
async def get_conversation(thread_id: str):
    """
    Get full details of a specific conversation including messages.

    Args:
        thread_id: The conversation thread ID

    Returns:
        Full conversation details with message history.
    """
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                # Get metadata
                cur.execute("""
                    SELECT title, created_at
                    FROM conversation_metadata
                    WHERE thread_id = %s
                """, (thread_id,))

                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Conversation not found")

                title, created_at = row

                # Get checkpoint data for messages
                # Note: Messages are stored in checkpoint channel_values
                cur.execute("""
                    SELECT checkpoint
                    FROM checkpoints
                    WHERE thread_id = %s
                    ORDER BY checkpoint_id DESC
                    LIMIT 1
                """, (thread_id,))

                checkpoint_row = cur.fetchone()
                messages = []

                if checkpoint_row and checkpoint_row[0]:
                    # Parse checkpoint to extract messages
                    # Handles both dict and object message formats
                    checkpoint = checkpoint_row[0]
                    if isinstance(checkpoint, dict):
                        channel_values = checkpoint.get("channel_values", {})
                        raw_messages = channel_values.get("messages", [])

                        for msg in raw_messages:
                            # Handle dict format (from JSON serialization)
                            if isinstance(msg, dict):
                                content = msg.get("content", "")
                                msg_type = msg.get("type", "unknown")
                                # Skip tool messages and empty content
                                if content and msg_type in ("human", "ai"):
                                    messages.append({
                                        "type": msg_type,
                                        "content": content,
                                    })
                            # Handle object format (LangChain message objects)
                            elif hasattr(msg, "content"):
                                content = msg.content
                                msg_type = getattr(msg, "type", "unknown")
                                if content and msg_type in ("human", "ai"):
                                    messages.append({
                                        "type": msg_type,
                                        "content": content,
                                    })

                return ConversationDetail(
                    thread_id=thread_id,
                    title=title,
                    created_at=created_at,
                    message_count=len(messages),
                    messages=messages,
                )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


@router.delete("/conversations", response_model=DeleteResponse)
async def clear_all_conversations():
    """
    Delete all conversations and their history.

    WARNING: This is destructive and cannot be undone.

    Returns:
        Count of deleted metadata and checkpoint records.
    """
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                # Delete metadata
                cur.execute("DELETE FROM conversation_metadata")
                metadata_count = cur.rowcount

                # Delete checkpoints
                cur.execute("DELETE FROM checkpoints")
                checkpoint_count = cur.rowcount

                # Delete checkpoint blobs if they exist
                try:
                    cur.execute("DELETE FROM checkpoint_blobs")
                except psycopg.Error:
                    pass  # Table may not exist

                return DeleteResponse(
                    deleted_metadata=metadata_count,
                    deleted_checkpoints=checkpoint_count,
                )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


@router.delete("/conversations/{thread_id}")
async def delete_conversation(thread_id: str):
    """
    Delete a specific conversation.

    Args:
        thread_id: The conversation thread ID to delete

    Returns:
        Success status.
    """
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                # Delete metadata
                cur.execute(
                    "DELETE FROM conversation_metadata WHERE thread_id = %s",
                    (thread_id,)
                )

                if cur.rowcount == 0:
                    raise HTTPException(status_code=404, detail="Conversation not found")

                # Delete checkpoints
                cur.execute(
                    "DELETE FROM checkpoints WHERE thread_id = %s",
                    (thread_id,)
                )

                return {"status": "deleted", "thread_id": thread_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

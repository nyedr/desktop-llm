"""Utility functions for memory operations."""

import logging
import uuid
from typing import List, Dict
from datetime import datetime

from app.memory.manager import LightRAGManager
from app.models.chat import ChatRole

logger = logging.getLogger(__name__)


async def store_conversation_memory(
    request_id: str,
    messages: List[Dict],
    lightrag_manager: LightRAGManager,
    conversation_id: str,
    model: str
) -> None:
    """Store conversation memory in background.

    Args:
        request_id: ID of the current request
        messages: List of chat messages to store
        lightrag_manager: LightRAG manager instance
        conversation_id: ID of the conversation
        model: Model used for the conversation
    """
    try:
        # Filter out system messages and concatenate conversation
        conversation_parts = []
        for msg in messages:
            # Handle both dict and message objects
            if hasattr(msg, 'role'):
                role = msg.role
                content = msg.content
            else:
                role = msg.get('role')
                content = msg.get('content')

            if role and role != ChatRole.SYSTEM and content:
                conversation_parts.append(f"{role}: {content}")

        conversation = "\n".join(conversation_parts)

        if conversation:
            # Queue memory for background processing
            await lightrag_manager.queue_memory(
                text=conversation,
                metadata={
                    "conversation_id": conversation_id,
                    "request_id": request_id,
                    "model": model,
                    "timestamp": datetime.now().isoformat()
                }
            )
            logger.info(
                f"[{request_id}] Queued conversation memory for processing")

    except Exception as e:
        logger.warning(
            f"[{request_id}] Error storing conversation memory: {e}")

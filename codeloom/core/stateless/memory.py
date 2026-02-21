"""Database-backed conversation memory for stateless chat.

Provides memory utilities that work with the existing ConversationStore
but with a simpler interface optimized for the stateless query pattern.
"""

import logging
from typing import List, Dict, Optional, Any
from uuid import uuid4

logger = logging.getLogger(__name__)


def load_conversation_history(
    conversation_store: Any,
    project_id: str,
    user_id: str,
    max_history: int = 10,
) -> List[Dict[str, str]]:
    """Load conversation history from database.

    Retrieves the most recent conversation turns for context.

    Args:
        conversation_store: ConversationStore instance
        project_id: UUID of the project
        user_id: UUID of the user
        max_history: Maximum number of turns to retrieve

    Returns:
        List of {"role": str, "content": str} dictionaries

    Example:
        history = load_conversation_history(
            conversation_store=store,
            project_id=project_id,
            user_id=user_id,
            max_history=10,
        )
    """
    try:
        # Get conversation history (returns oldest first)
        messages = conversation_store.get_conversation_history(
            project_id=project_id,
            user_id=user_id,
            limit=max_history * 2,  # Each turn has 2 messages (user + assistant)
        )

        # Convert to simple format
        history = []
        for msg in messages:
            history.append({
                "role": msg["role"],
                "content": msg["content"],
            })

        # Return most recent max_history * 2 messages
        if len(history) > max_history * 2:
            history = history[-(max_history * 2):]

        logger.debug(f"Loaded {len(history)} history messages for project {project_id}")
        return history

    except Exception as e:
        logger.warning(f"Failed to load conversation history: {e}")
        return []


def save_conversation_turn(
    conversation_store: Any,
    project_id: str,
    user_id: str,
    user_message: str,
    assistant_response: str,
) -> bool:
    """Save a complete conversation turn (user + assistant) to database.

    Args:
        conversation_store: ConversationStore instance
        project_id: UUID of the project
        user_id: UUID of the user
        user_message: User's query
        assistant_response: Assistant's response

    Returns:
        True if saved successfully, False otherwise

    Example:
        save_conversation_turn(
            conversation_store=store,
            project_id=project_id,
            user_id=user_id,
            user_message="What are the key findings?",
            assistant_response="The key findings are...",
        )
    """
    try:
        # Save both messages in a batch
        conversation_store.save_messages(
            project_id=project_id,
            user_id=user_id,
            messages=[
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_response},
            ]
        )
        logger.debug(f"Saved conversation turn for project {project_id}")
        return True

    except Exception as e:
        logger.warning(f"Failed to save conversation turn: {e}")
        return False


def generate_session_id() -> str:
    """Generate a new session ID for conversation tracking.

    Returns:
        UUID string for session identification
    """
    return str(uuid4())


def format_history_for_context(
    history: List[Dict[str, str]],
    max_chars: int = 4000,
) -> str:
    """Format conversation history as context string.

    Truncates to fit within character limit while keeping most recent turns.

    Args:
        history: List of {"role": str, "content": str}
        max_chars: Maximum characters for formatted history

    Returns:
        Formatted history string
    """
    if not history:
        return ""

    formatted_turns = []
    for msg in history:
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "")
        formatted_turns.append(f"{role}: {content}")

    # Join and truncate
    full_history = "\n\n".join(formatted_turns)

    if len(full_history) <= max_chars:
        return full_history

    # If too long, keep the most recent turns
    while len(full_history) > max_chars and formatted_turns:
        formatted_turns.pop(0)  # Remove oldest
        full_history = "\n\n".join(formatted_turns)

    return full_history

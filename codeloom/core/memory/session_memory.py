"""
Dict-based Session Memory Service for cross-request conversation persistence.

Simple in-process storage using dict, keyed by (user_id, notebook_id).
No external dependencies required.
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

from llama_index.core.llms import ChatMessage, MessageRole

logger = logging.getLogger(__name__)


@dataclass
class SessionData:
    """Data stored for each session."""
    messages: List[Dict[str, str]] = field(default_factory=list)  # [{role, content}, ...]
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)


class SessionMemoryService:
    """
    In-process session memory using dict storage.

    Provides cross-request conversation persistence without external dependencies.
    Thread-safe with automatic TTL-based cleanup.
    """

    def __init__(
        self,
        max_messages_per_session: int = 100,
        session_ttl_hours: int = 24
    ):
        """
        Initialize session memory service.

        Args:
            max_messages_per_session: Maximum messages to retain per session (FIFO)
            session_ttl_hours: Hours before inactive sessions are eligible for cleanup
        """
        self._sessions: Dict[Tuple[str, str], SessionData] = {}
        self._lock = threading.Lock()
        self._max_messages = max_messages_per_session
        self._ttl_hours = session_ttl_hours
        logger.info(f"SessionMemoryService initialized: max_messages={max_messages_per_session}, ttl={session_ttl_hours}h")

    def _get_key(self, user_id: str, notebook_id: str) -> Tuple[str, str]:
        """Create session key from user_id and notebook_id."""
        return (str(user_id), str(notebook_id))

    def _get_or_create_session(self, user_id: str, notebook_id: str) -> SessionData:
        """Get existing session or create new one."""
        key = self._get_key(user_id, notebook_id)
        if key not in self._sessions:
            self._sessions[key] = SessionData()
            logger.debug(f"Created new session for user={user_id}, notebook={notebook_id}")
        return self._sessions[key]

    def add_message(
        self,
        user_id: str,
        notebook_id: str,
        role: str,
        content: str
    ) -> None:
        """
        Add a message to session history.

        Args:
            user_id: User identifier
            notebook_id: Notebook identifier
            role: Message role ('user' or 'assistant')
            content: Message content
        """
        with self._lock:
            session = self._get_or_create_session(user_id, notebook_id)
            session.messages.append({
                "role": role,
                "content": content
            })
            session.last_accessed = datetime.utcnow()

            # Enforce max messages limit (FIFO)
            if len(session.messages) > self._max_messages:
                removed = len(session.messages) - self._max_messages
                session.messages = session.messages[-self._max_messages:]
                logger.debug(f"Trimmed {removed} old messages from session")

    def add_exchange(
        self,
        user_id: str,
        notebook_id: str,
        user_message: str,
        assistant_message: str
    ) -> None:
        """
        Add a user-assistant exchange to session history.

        Args:
            user_id: User identifier
            notebook_id: Notebook identifier
            user_message: User's message
            assistant_message: Assistant's response
        """
        self.add_message(user_id, notebook_id, "user", user_message)
        self.add_message(user_id, notebook_id, "assistant", assistant_message)

    def get_history(
        self,
        user_id: str,
        notebook_id: str,
        limit: Optional[int] = None
    ) -> List[ChatMessage]:
        """
        Get chat history as LlamaIndex ChatMessage objects.

        Args:
            user_id: User identifier
            notebook_id: Notebook identifier
            limit: Maximum messages to return (None = all)

        Returns:
            List of ChatMessage objects for engine integration
        """
        with self._lock:
            key = self._get_key(user_id, notebook_id)
            if key not in self._sessions:
                return []

            session = self._sessions[key]
            session.last_accessed = datetime.utcnow()

            messages = session.messages
            if limit is not None:
                messages = messages[-limit:]

            # Convert to LlamaIndex ChatMessage format
            chat_messages = []
            for msg in messages:
                role = MessageRole.USER if msg["role"] == "user" else MessageRole.ASSISTANT
                chat_messages.append(ChatMessage(role=role, content=msg["content"]))

            return chat_messages

    def get_history_dicts(
        self,
        user_id: str,
        notebook_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        Get chat history as plain dicts.

        Args:
            user_id: User identifier
            notebook_id: Notebook identifier
            limit: Maximum messages to return (None = all)

        Returns:
            List of {role, content} dicts
        """
        with self._lock:
            key = self._get_key(user_id, notebook_id)
            if key not in self._sessions:
                return []

            session = self._sessions[key]
            session.last_accessed = datetime.utcnow()

            messages = session.messages.copy()
            if limit is not None:
                messages = messages[-limit:]

            return messages

    def clear_session(self, user_id: str, notebook_id: str) -> bool:
        """
        Clear all messages for a session.

        Args:
            user_id: User identifier
            notebook_id: Notebook identifier

        Returns:
            True if session existed and was cleared
        """
        with self._lock:
            key = self._get_key(user_id, notebook_id)
            if key in self._sessions:
                del self._sessions[key]
                logger.info(f"Cleared session for user={user_id}, notebook={notebook_id}")
                return True
            return False

    def get_message_count(self, user_id: str, notebook_id: str) -> int:
        """Get number of messages in session."""
        with self._lock:
            key = self._get_key(user_id, notebook_id)
            if key in self._sessions:
                return len(self._sessions[key].messages)
            return 0

    def get_context_string(
        self,
        user_id: str,
        notebook_id: str,
        limit: int = 10
    ) -> str:
        """
        Get recent history formatted as context string for LLM.

        Args:
            user_id: User identifier
            notebook_id: Notebook identifier
            limit: Maximum messages to include

        Returns:
            Formatted context string
        """
        messages = self.get_history_dicts(user_id, notebook_id, limit)

        if not messages:
            return ""

        context_parts = ["Recent conversation history:"]
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            # Truncate long messages for context
            content = msg["content"][:500] + "..." if len(msg["content"]) > 500 else msg["content"]
            context_parts.append(f"{role}: {content}")

        return "\n".join(context_parts)

    def cleanup_stale_sessions(self) -> int:
        """
        Remove sessions that haven't been accessed within TTL.

        Returns:
            Number of sessions removed
        """
        with self._lock:
            cutoff = datetime.utcnow() - timedelta(hours=self._ttl_hours)
            stale_keys = [
                key for key, session in self._sessions.items()
                if session.last_accessed < cutoff
            ]

            for key in stale_keys:
                del self._sessions[key]

            if stale_keys:
                logger.info(f"Cleaned up {len(stale_keys)} stale sessions")

            return len(stale_keys)

    def get_session_count(self) -> int:
        """Get total number of active sessions."""
        with self._lock:
            return len(self._sessions)

    def get_stats(self) -> Dict[str, Any]:
        """Get memory service statistics."""
        with self._lock:
            total_messages = sum(len(s.messages) for s in self._sessions.values())
            return {
                "active_sessions": len(self._sessions),
                "total_messages": total_messages,
                "max_messages_per_session": self._max_messages,
                "ttl_hours": self._ttl_hours
            }

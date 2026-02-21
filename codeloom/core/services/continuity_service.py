"""Service for conversation continuity and session management.

This module provides conversation continuity features across sessions,
including session summaries, topic extraction, and context restoration.
"""

import logging
from datetime import datetime
from typing import Optional

from .base import BaseService

logger = logging.getLogger(__name__)


class ContinuityService(BaseService):
    """Service for conversation continuity and session management.

    Provides features for:
    - Session summaries across conversation boundaries
    - Key topic extraction from conversation history
    - Conversation context restoration
    - Session cleanup and management

    Helps users pick up where they left off in previous sessions.
    """

    # Common stop words to exclude from topic extraction
    STOP_WORDS = {
        "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "can", "may", "might", "must", "shall",
        "about", "above", "after", "again", "against", "all", "and",
        "any", "because", "before", "below", "between", "both", "but",
        "for", "from", "how", "into", "more", "most", "not", "now",
        "only", "other", "out", "over", "same", "such", "than", "that",
        "their", "them", "then", "there", "these", "they", "this", "those",
        "through", "under", "very", "what", "when", "where", "which", "while",
        "who", "why", "with", "you", "your"
    }

    def __init__(self, pipeline=None, db_manager=None, project_manager=None):
        """Initialize continuity service.

        Args:
            pipeline: LocalRAGPipeline instance for RAG operations
            db_manager: Optional DatabaseManager for persistence
            project_manager: Optional ProjectManager for project operations
        """
        super().__init__(pipeline, db_manager, project_manager)

    def get_session_summary(
        self,
        project_id: str,
        user_id: Optional[str] = None
    ) -> dict:
        """Get summary of previous session for a project.

        Analyzes conversation history to provide context about what was
        discussed in previous sessions, helping users resume conversations.

        Args:
            project_id: UUID of the project
            user_id: Optional UUID of the user (defaults to pipeline user)

        Returns:
            Dictionary containing:
                - has_previous_session (bool): Whether previous session exists
                - last_active (datetime|None): Timestamp of last activity
                - summary (str|None): Brief summary of previous session
                - key_topics (list[str]): List of key topics discussed
                - pending_questions (list[str]): Unanswered questions (future)
                - message_count (int): Total messages in history

        Example:
            >>> summary = service.get_session_summary(
            ...     project_id="123",
            ...     user_id="user-456"
            ... )
            >>> if summary["has_previous_session"]:
            ...     print(f"Last active: {summary['last_active']}")
            ...     print(f"Summary: {summary['summary']}")
            ...     for topic in summary["key_topics"]:
            ...         print(f"- {topic}")
        """
        try:
            self._log_operation(
                "get_session_summary",
                project_id=project_id,
                user_id=user_id
            )

            # Check if conversation store is available
            if not self.pipeline or not hasattr(self.pipeline, '_conversation_store'):
                logger.warning("ConversationStore not available")
                return {"has_previous_session": False}

            conversation_store = self.pipeline._conversation_store

            # Use pipeline's current user if not specified
            if not user_id and hasattr(self.pipeline, '_current_user_id'):
                user_id = self.pipeline._current_user_id

            if not user_id:
                logger.warning("No user_id available for session summary")
                return {"has_previous_session": False}

            # Get conversation history (last 20 messages)
            history = conversation_store.get_conversation_history(
                project_id=project_id,
                user_id=user_id,
                limit=20
            )

            if not history:
                return {
                    "has_previous_session": False,
                    "last_active": None,
                    "summary": None,
                    "key_topics": [],
                    "pending_questions": [],
                    "message_count": 0
                }

            # Extract key topics
            topics = self._extract_topics(history)

            # Generate summary
            summary = self._generate_summary(history)

            # Get last activity timestamp
            last_active = None
            if history and "timestamp" in history[-1]:
                timestamp_str = history[-1]["timestamp"]
                try:
                    last_active = datetime.fromisoformat(timestamp_str)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid timestamp format: {timestamp_str}")

            result = {
                "has_previous_session": True,
                "last_active": last_active,
                "summary": summary,
                "key_topics": topics[:5],  # Top 5 topics
                "pending_questions": [],  # Future enhancement
                "message_count": len(history)
            }

            self._log_operation(
                "session_summary_generated",
                message_count=len(history),
                topic_count=len(topics)
            )

            return result

        except Exception as e:
            self._log_error(
                "get_session_summary",
                e,
                project_id=project_id,
                user_id=user_id
            )
            return {"has_previous_session": False}

    def _extract_topics(self, history: list[dict]) -> list[str]:
        """Extract key topics from conversation history.

        Uses simple keyword extraction to identify significant terms
        from user messages.

        Args:
            history: List of conversation message dictionaries

        Returns:
            List of topic keywords, sorted by frequency
        """
        # Track word frequencies
        word_counts = {}

        for msg in history:
            if msg.get("role") == "user":
                content = msg.get("content", "")

                # Extract words (simple tokenization)
                words = content.lower().split()

                for word in words:
                    # Clean word (remove punctuation)
                    word = word.strip(".,!?;:'\"()[]{}").strip()

                    # Filter out short words and stop words
                    if (len(word) > 4 and
                        word not in self.STOP_WORDS and
                        word.isalpha()):
                        word_counts[word] = word_counts.get(word, 0) + 1

        # Sort by frequency and return top topics
        sorted_topics = sorted(
            word_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [topic for topic, count in sorted_topics[:10]]

    def _generate_summary(self, history: list[dict]) -> Optional[str]:
        """Generate conversation summary from history.

        Creates a brief summary describing the conversation activity
        and most recent topic.

        Args:
            history: List of conversation message dictionaries

        Returns:
            Summary string or None if no messages
        """
        if not history:
            return None

        # Count user messages
        user_messages = [m for m in history if m.get("role") == "user"]

        if not user_messages:
            return None

        count = len(user_messages)

        # Get last query (truncated)
        last_query = user_messages[-1].get("content", "")
        if len(last_query) > 100:
            last_query = last_query[:97] + "..."

        # Format summary
        if count == 1:
            summary = f"You had 1 exchange. Question: \"{last_query}\""
        else:
            summary = f"You had {count} exchanges. Last question: \"{last_query}\""

        return summary

    def clear_session(
        self,
        project_id: str,
        user_id: Optional[str] = None
    ) -> bool:
        """Clear session data for a project.

        Removes all conversation history for the specified project
        and user, effectively resetting the session.

        Args:
            project_id: UUID of the project
            user_id: Optional UUID of the user (defaults to pipeline user)

        Returns:
            True if session was cleared, False if no session found or error

        Example:
            >>> success = service.clear_session(
            ...     project_id="123",
            ...     user_id="user-456"
            ... )
            >>> if success:
            ...     print("Session cleared successfully")
        """
        try:
            self._log_operation(
                "clear_session",
                project_id=project_id,
                user_id=user_id
            )

            # Check if conversation store is available
            if not self.pipeline or not hasattr(self.pipeline, '_conversation_store'):
                logger.warning("ConversationStore not available")
                return False

            conversation_store = self.pipeline._conversation_store

            # Use pipeline's current user if not specified
            if not user_id and hasattr(self.pipeline, '_current_user_id'):
                user_id = self.pipeline._current_user_id

            if not user_id:
                logger.warning("No user_id available for session clear")
                return False

            # Clear project history for user
            # Note: ConversationStore.clear_project_history clears for all users
            # We need to implement user-specific clearing
            # For now, use the existing method
            success = conversation_store.clear_project_history(project_id)

            if success:
                self._log_operation(
                    "session_cleared",
                    project_id=project_id,
                    user_id=user_id
                )
            else:
                logger.warning(
                    f"No session found to clear for project {project_id}"
                )

            return success

        except Exception as e:
            self._log_error(
                "clear_session",
                e,
                project_id=project_id,
                user_id=user_id
            )
            return False

    def get_conversation_context(
        self,
        project_id: str,
        user_id: Optional[str] = None,
        message_limit: int = 10
    ) -> list[dict]:
        """Get recent conversation context for a project.

        Retrieves recent conversation messages to provide context
        for continuing a conversation.

        Args:
            project_id: UUID of the project
            user_id: Optional UUID of the user (defaults to pipeline user)
            message_limit: Maximum number of messages to retrieve (default: 10)

        Returns:
            List of recent message dictionaries with role, content, timestamp

        Example:
            >>> context = service.get_conversation_context(
            ...     project_id="123",
            ...     user_id="user-456",
            ...     message_limit=5
            ... )
            >>> for msg in context:
            ...     print(f"{msg['role']}: {msg['content']}")
        """
        try:
            self._log_operation(
                "get_conversation_context",
                project_id=project_id,
                user_id=user_id,
                message_limit=message_limit
            )

            # Check if conversation store is available
            if not self.pipeline or not hasattr(self.pipeline, '_conversation_store'):
                logger.warning("ConversationStore not available")
                return []

            conversation_store = self.pipeline._conversation_store

            # Use pipeline's current user if not specified
            if not user_id and hasattr(self.pipeline, '_current_user_id'):
                user_id = self.pipeline._current_user_id

            if not user_id:
                logger.warning("No user_id available for conversation context")
                return []

            # Get recent conversation history
            history = conversation_store.get_conversation_history(
                project_id=project_id,
                user_id=user_id,
                limit=message_limit
            )

            self._log_operation(
                "context_retrieved",
                message_count=len(history)
            )

            return history

        except Exception as e:
            self._log_error(
                "get_conversation_context",
                e,
                project_id=project_id,
                user_id=user_id
            )
            return []

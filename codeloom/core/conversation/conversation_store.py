"""
Conversation Store for Persistent Conversation History

Provides conversation persistence with PostgreSQL for project-scoped chat history.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
from uuid import UUID, uuid4

from ..db import DatabaseManager
from ..db.models import Conversation

logger = logging.getLogger(__name__)


class ConversationStore:
    """
    Manages persistent conversation history with database persistence.

    Features:
    - Save and retrieve conversation messages
    - Project-scoped conversation history
    - User-specific conversation tracking
    - Automatic timestamp management
    - Message count and activity tracking
    """

    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize conversation store with database connection.

        Args:
            db_manager: DatabaseManager instance for database operations
        """
        self.db = db_manager
        logger.info("ConversationStore initialized")

    # =========================================================================
    # Message Persistence Operations
    # =========================================================================

    def save_message(
        self,
        project_id: str,
        user_id: str,
        role: str,
        content: str
    ) -> str:
        """
        Save a single conversation message.

        Args:
            project_id: UUID of the project
            user_id: UUID of the user
            role: Message role ('user' or 'assistant')
            content: Message content text

        Returns:
            conversation_id (UUID) of the saved message

        Raises:
            ValueError: If role is not 'user' or 'assistant'
            Exception: For database errors
        """
        if role not in ['user', 'assistant']:
            raise ValueError(f"Invalid role: {role}. Must be 'user' or 'assistant'")

        try:
            with self.db.get_session() as session:
                conversation = Conversation(
                    conversation_id=uuid4(),
                    project_id=UUID(project_id),
                    user_id=UUID(user_id),
                    role=role,
                    content=content
                )

                session.add(conversation)
                session.flush()

                logger.debug(
                    f"Saved message to project {project_id}: "
                    f"{role} - {len(content)} chars"
                )

                return str(conversation.conversation_id)

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to save message: {e}")
            raise

    def save_messages(
        self,
        project_id: str,
        user_id: str,
        messages: List[Dict]
    ) -> List[str]:
        """
        Save multiple conversation messages in batch.

        Args:
            project_id: UUID of the project
            user_id: UUID of the user
            messages: List of message dicts with 'role' and 'content' keys

        Returns:
            List of conversation_ids (UUIDs) of saved messages

        Raises:
            ValueError: If any message has invalid role
            Exception: For database errors
        """
        conversation_ids = []

        try:
            with self.db.get_session() as session:
                for msg in messages:
                    role = msg.get('role')
                    content = msg.get('content')

                    if role not in ['user', 'assistant']:
                        raise ValueError(f"Invalid role: {role}. Must be 'user' or 'assistant'")

                    conversation = Conversation(
                        conversation_id=uuid4(),
                        project_id=UUID(project_id),
                        user_id=UUID(user_id),
                        role=role,
                        content=content
                    )

                    session.add(conversation)
                    conversation_ids.append(str(conversation.conversation_id))

                session.flush()

                logger.info(
                    f"Saved {len(messages)} messages to project {project_id}"
                )

                return conversation_ids

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to save messages: {e}")
            raise

    # =========================================================================
    # Conversation Retrieval Operations
    # =========================================================================

    def get_conversation_history(
        self,
        project_id: str,
        user_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict]:
        """
        Retrieve conversation history for a project.

        Args:
            project_id: UUID of the project
            user_id: UUID of the user
            limit: Maximum number of messages to retrieve (default: 50)
            offset: Number of messages to skip (default: 0)

        Returns:
            List of conversation messages ordered by timestamp (oldest first)
            Each message is a dict with: conversation_id, role, content, timestamp
        """
        try:
            with self.db.get_session() as session:
                conversations = session.query(Conversation).filter(
                    Conversation.project_id == UUID(project_id),
                    Conversation.user_id == UUID(user_id)
                ).order_by(Conversation.timestamp.asc()).offset(offset).limit(limit).all()

                result = []
                for conv in conversations:
                    result.append({
                        "conversation_id": str(conv.conversation_id),
                        "role": conv.role,
                        "content": conv.content,
                        "timestamp": conv.timestamp.isoformat() if conv.timestamp else None
                    })

                logger.debug(
                    f"Retrieved {len(result)} messages from project {project_id}"
                )

                return result

        except Exception as e:
            logger.error(f"Failed to retrieve conversation history: {e}")
            raise

    # =========================================================================
    # History Management Operations
    # =========================================================================

    def clear_project_history(self, project_id: str) -> bool:
        """
        Clear all conversation history for a project.

        Args:
            project_id: UUID of the project

        Returns:
            True if cleared successfully, False if no messages found
        """
        try:
            with self.db.get_session() as session:
                deleted_count = session.query(Conversation).filter(
                    Conversation.project_id == UUID(project_id)
                ).delete()

                if deleted_count > 0:
                    logger.info(
                        f"Cleared {deleted_count} messages from project {project_id}"
                    )
                    return True
                else:
                    logger.warning(f"No conversation history found for project {project_id}")
                    return False

        except Exception as e:
            logger.error(f"Failed to clear project history: {e}")
            raise

    def clear_user_history(self, user_id: str) -> bool:
        """
        Clear all conversation history for a user across all projects.

        Args:
            user_id: UUID of the user

        Returns:
            True if cleared successfully, False if no messages found
        """
        try:
            with self.db.get_session() as session:
                deleted_count = session.query(Conversation).filter(
                    Conversation.user_id == UUID(user_id)
                ).delete()

                if deleted_count > 0:
                    logger.info(
                        f"Cleared {deleted_count} messages for user {user_id}"
                    )
                    return True
                else:
                    logger.warning(f"No conversation history found for user {user_id}")
                    return False

        except Exception as e:
            logger.error(f"Failed to clear user history: {e}")
            raise

    # =========================================================================
    # Statistics and Utilities
    # =========================================================================

    def get_message_count(self, project_id: str) -> int:
        """
        Get count of messages in a project.

        Args:
            project_id: UUID of the project

        Returns:
            Number of messages in the project
        """
        try:
            with self.db.get_session() as session:
                count = session.query(Conversation).filter(
                    Conversation.project_id == UUID(project_id)
                ).count()

                logger.debug(f"Project {project_id} has {count} messages")
                return count

        except Exception as e:
            logger.error(f"Failed to get message count: {e}")
            raise

    def get_last_activity(self, project_id: str) -> Optional[datetime]:
        """
        Get timestamp of last message in project.

        Args:
            project_id: UUID of the project

        Returns:
            Datetime of last message, or None if no messages
        """
        try:
            with self.db.get_session() as session:
                last_conversation = session.query(Conversation).filter(
                    Conversation.project_id == UUID(project_id)
                ).order_by(Conversation.timestamp.desc()).first()

                if last_conversation:
                    logger.debug(
                        f"Last activity in project {project_id}: "
                        f"{last_conversation.timestamp}"
                    )
                    return last_conversation.timestamp
                else:
                    logger.debug(f"No activity found in project {project_id}")
                    return None

        except Exception as e:
            logger.error(f"Failed to get last activity: {e}")
            raise

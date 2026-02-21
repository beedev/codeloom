"""Chat service implementation for DBProject.

This module implements the chat service layer, handling all chat-related
operations including streaming responses, conversation history, and context management.
"""

from typing import Iterator, Dict, Any, Optional, List

from .base import BaseService
from ..interfaces.services import IChatService


class ChatService(BaseService, IChatService):
    """Service for chat operations.

    Handles streaming chat responses, conversation history management,
    and project context switching. This service coordinates between
    the RAG pipeline, conversation store, and chat engine.
    """

    def stream_chat(
        self,
        query: str,
        project_id: str | None = None,
        user_id: str | None = None,
        conversation_id: str | None = None,
        mode: str = "chat",
        **kwargs
    ) -> Iterator[str]:
        """Stream chat response tokens.

        Args:
            query: User query text
            project_id: Optional project UUID to limit context
            user_id: Optional user UUID for multi-user support
            conversation_id: Optional conversation UUID for history
            mode: Chat mode ("chat" or "QA")
            **kwargs: Additional provider-specific parameters

        Yields:
            Response tokens/chunks as they are generated

        Raises:
            ValueError: If query is empty or invalid
        """
        # Validate input
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        self._log_operation(
            "stream_chat",
            project_id=project_id,
            user_id=user_id,
            mode=mode,
            query_length=len(query)
        )

        try:
            # Switch project context if specified
            if project_id:
                effective_user_id = user_id or "00000000-0000-0000-0000-000000000001"
                self.pipeline.switch_project(project_id, effective_user_id)
                self.logger.debug(f"Switched to project: {project_id}")

            # Get conversation history if needed (chatbot format)
            chatbot = kwargs.get("chatbot", [])

            # Stream response from pipeline
            # The pipeline handles mode internally (chat vs QA)
            response = self.pipeline.query(
                mode=mode,
                message=query,
                chatbot=chatbot
            )

            # Stream response tokens
            for token in response.response_gen:
                yield token

        except Exception as e:
            self._log_error("stream_chat", e, project_id=project_id, user_id=user_id)
            raise

    def get_context(
        self,
        project_id: str,
        top_k: int = 6
    ) -> Dict[str, Any]:
        """Get retrieval context for a project.

        Args:
            project_id: Project UUID
            top_k: Number of top chunks to retrieve

        Returns:
            Dictionary containing:
            - documents: List of source documents
            - node_count: Number of indexed nodes
            - retrieval_strategy: Current retrieval strategy

        Raises:
            ValueError: If project_id is invalid
        """
        if not project_id:
            raise ValueError("project_id is required")

        self._log_operation("get_context", project_id=project_id, top_k=top_k)

        try:
            # Get nodes for project from vector store
            nodes = self.pipeline._vector_store.get_nodes_by_project_sql(project_id)

            # Extract unique document sources
            documents = []
            seen_sources = set()

            for node in nodes:
                metadata = node.metadata or {}
                source_id = metadata.get("source_id")

                if source_id and source_id not in seen_sources:
                    seen_sources.add(source_id)
                    documents.append({
                        "source_id": source_id,
                        "file_name": metadata.get("file_name", "Unknown"),
                        "project_id": metadata.get("project_id")
                    })

            # Get retrieval strategy from settings
            retrieval_strategy = self.pipeline._settings.retrieval_strategy

            return {
                "documents": documents,
                "node_count": len(nodes),
                "retrieval_strategy": retrieval_strategy
            }

        except Exception as e:
            self._log_error("get_context", e, project_id=project_id)
            raise

    def set_chat_mode(self, mode: str) -> None:
        """Set the chat engine mode.

        Args:
            mode: Chat mode ("chat", "QA", "condense_question", "condense_plus_context")

        Raises:
            ValueError: If mode is invalid
        """
        valid_modes = ["chat", "QA", "condense_question", "condense_plus_context"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")

        self._log_operation("set_chat_mode", mode=mode)

        try:
            # The pipeline's set_chat_mode handles mode configuration
            self.pipeline.set_chat_mode()
            self.logger.info(f"Chat mode set to: {mode}")

        except Exception as e:
            self._log_error("set_chat_mode", e, mode=mode)
            raise

    def get_chat_history(
        self,
        conversation_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get conversation history.

        Args:
            conversation_id: Conversation UUID
            limit: Maximum number of messages to retrieve

        Returns:
            List of message dictionaries with role and content

        Raises:
            RuntimeError: If conversation store is not available
            ValueError: If conversation_id is invalid
        """
        if not conversation_id:
            raise ValueError("conversation_id is required")

        # Validate conversation store is available
        self._validate_database_available()

        if not self.pipeline._conversation_store:
            raise RuntimeError("Conversation persistence not available")

        self._log_operation("get_chat_history", conversation_id=conversation_id, limit=limit)

        try:
            # Get history from conversation store
            # Note: conversation_id is actually project_id in our architecture
            history = self.pipeline._conversation_store.get_conversation_history(
                project_id=conversation_id,
                user_id=self.pipeline._current_user_id,
                limit=limit
            )

            # Transform to standard format
            messages = []
            for entry in history:
                messages.append({
                    "role": entry.get("role", "user"),
                    "content": entry.get("content", ""),
                    "timestamp": entry.get("timestamp", "").isoformat()
                        if hasattr(entry.get("timestamp", ""), "isoformat")
                        else str(entry.get("timestamp", ""))
                })

            return messages

        except Exception as e:
            self._log_error("get_chat_history", e, conversation_id=conversation_id)
            raise

    def clear_chat_history(self, conversation_id: str) -> bool:
        """Clear conversation history.

        Args:
            conversation_id: Conversation UUID

        Returns:
            True if successful

        Raises:
            RuntimeError: If conversation store is not available
            ValueError: If conversation_id is invalid
        """
        if not conversation_id:
            raise ValueError("conversation_id is required")

        # Validate conversation store is available
        self._validate_database_available()

        if not self.pipeline._conversation_store:
            raise RuntimeError("Conversation persistence not available")

        self._log_operation("clear_chat_history", conversation_id=conversation_id)

        try:
            # Clear history from conversation store
            # Note: conversation_id is actually project_id in our architecture
            deleted_count = self.pipeline._conversation_store.clear_project_history(
                project_id=conversation_id,
                user_id=self.pipeline._current_user_id
            )

            self.logger.info(f"Cleared {deleted_count} messages from conversation {conversation_id}")
            return True

        except Exception as e:
            self._log_error("clear_chat_history", e, conversation_id=conversation_id)
            return False

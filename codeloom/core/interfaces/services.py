"""Abstract interfaces for service layer components.

This module defines the service layer interfaces following the service layer pattern.
Services encapsulate business logic and coordinate between the pipeline, database,
and other core components.
"""

from abc import ABC, abstractmethod
from typing import Iterator, Dict, Any, Optional, List
from pathlib import Path


class IChatService(ABC):
    """Interface for chat operations.

    Handles all chat-related business logic including streaming responses,
    context management, and conversation history.
    """

    @abstractmethod
    def stream_chat(
        self,
        query: str,
        notebook_id: str | None = None,
        user_id: str | None = None,
        conversation_id: str | None = None,
        mode: str = "chat",
        **kwargs
    ) -> Iterator[str]:
        """Stream chat response tokens.

        Args:
            query: User query text
            notebook_id: Optional notebook UUID to limit context
            user_id: Optional user UUID for multi-user support
            conversation_id: Optional conversation UUID for history
            mode: Chat mode ("chat" or "QA")
            **kwargs: Additional provider-specific parameters

        Yields:
            Response tokens/chunks as they are generated
        """
        pass

    @abstractmethod
    def get_context(
        self,
        notebook_id: str,
        top_k: int = 6
    ) -> Dict[str, Any]:
        """Get retrieval context for a notebook.

        Args:
            notebook_id: Notebook UUID
            top_k: Number of top chunks to retrieve

        Returns:
            Dictionary containing:
            - documents: List of source documents
            - node_count: Number of indexed nodes
            - retrieval_strategy: Current retrieval strategy
        """
        pass

    @abstractmethod
    def set_chat_mode(self, mode: str) -> None:
        """Set the chat engine mode.

        Args:
            mode: Chat mode ("chat", "QA", "condense_question", "condense_plus_context")
        """
        pass

    @abstractmethod
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
        """
        pass

    @abstractmethod
    def clear_chat_history(self, conversation_id: str) -> bool:
        """Clear conversation history.

        Args:
            conversation_id: Conversation UUID

        Returns:
            True if successful
        """
        pass


class IImageService(ABC):
    """Interface for image generation operations.

    Handles image generation requests and provider management.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        provider: str | None = None,
        num_images: int = 1,
        aspect_ratio: str = "1:1",
        **kwargs
    ) -> Dict[str, Any]:
        """Generate images from text prompt.

        Args:
            prompt: Text description of image to generate
            provider: Optional provider name override
            num_images: Number of images to generate (1-4)
            aspect_ratio: Aspect ratio (1:1, 16:9, 9:16, 4:3, 3:4)
            **kwargs: Provider-specific options

        Returns:
            Dictionary containing:
            - success: Boolean success status
            - images: List of file paths to generated images
            - provider: Provider name used
            - error: Optional error message
        """
        pass

    @abstractmethod
    def list_providers(self) -> List[str]:
        """List available image generation providers.

        Returns:
            List of provider name identifiers
        """
        pass

    @abstractmethod
    def get_provider_info(self, provider: str) -> Dict[str, Any]:
        """Get information about a specific provider.

        Args:
            provider: Provider name identifier

        Returns:
            Dictionary with provider capabilities and settings
        """
        pass

    @abstractmethod
    def list_generated_images(self) -> List[Dict[str, Any]]:
        """List all previously generated images.

        Returns:
            List of image metadata dictionaries
        """
        pass


class IDocumentService(ABC):
    """Interface for document management operations.

    Handles document uploads, processing, metadata management,
    and document lifecycle operations.
    """

    @abstractmethod
    def upload(
        self,
        file: bytes,
        filename: str,
        notebook_id: str,
        user_id: str | None = None,
        metadata: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """Upload and process a document.

        Args:
            file: File content as bytes
            filename: Original filename
            notebook_id: Notebook UUID to add document to
            user_id: Optional user UUID for ownership
            metadata: Optional document metadata (it_practice, offering_id, etc.)

        Returns:
            Dictionary containing:
            - success: Boolean success status
            - source_id: UUID of created source
            - filename: Processed filename
            - node_count: Number of nodes created
            - file_hash: MD5 hash for duplicate detection
            - error: Optional error message
        """
        pass

    @abstractmethod
    def delete(
        self,
        source_id: str,
        notebook_id: str,
        user_id: str | None = None
    ) -> bool:
        """Delete a document from notebook.

        Args:
            source_id: Source UUID to delete
            notebook_id: Notebook UUID containing the source
            user_id: Optional user UUID for authorization

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def toggle_active(
        self,
        source_id: str,
        active: bool,
        notebook_id: str | None = None
    ) -> bool:
        """Toggle document active status for RAG inclusion.

        Args:
            source_id: Source UUID to update
            active: New active status
            notebook_id: Optional notebook UUID for validation

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def list_documents(
        self,
        notebook_id: str,
        user_id: str | None = None,
        include_inactive: bool = True
    ) -> List[Dict[str, Any]]:
        """List documents in a notebook.

        Args:
            notebook_id: Notebook UUID
            user_id: Optional user UUID for filtering
            include_inactive: Include inactive documents

        Returns:
            List of document metadata dictionaries containing:
            - source_id: UUID
            - filename: Original filename
            - file_hash: MD5 hash
            - active: Active status
            - created_at: Upload timestamp
            - metadata: Custom metadata
        """
        pass

    @abstractmethod
    def get_document_info(
        self,
        source_id: str,
        notebook_id: str | None = None
    ) -> Dict[str, Any]:
        """Get detailed information about a document.

        Args:
            source_id: Source UUID
            notebook_id: Optional notebook UUID for validation

        Returns:
            Dictionary with document details and statistics
        """
        pass

    @abstractmethod
    def update_metadata(
        self,
        source_id: str,
        metadata: Dict[str, Any],
        notebook_id: str | None = None
    ) -> bool:
        """Update document metadata.

        Args:
            source_id: Source UUID
            metadata: New metadata dictionary
            notebook_id: Optional notebook UUID for validation

        Returns:
            True if successful
        """
        pass


class INotebookService(ABC):
    """Interface for notebook management operations.

    Handles notebook lifecycle, document organization, and
    NotebookLM-style document collections.
    """

    @abstractmethod
    def create_notebook(
        self,
        name: str,
        user_id: str,
        description: str | None = None
    ) -> Dict[str, Any]:
        """Create a new notebook.

        Args:
            name: Notebook name
            user_id: Owner user UUID
            description: Optional description

        Returns:
            Dictionary containing:
            - notebook_id: UUID of created notebook
            - name: Notebook name
            - created_at: Creation timestamp
        """
        pass

    @abstractmethod
    def get_notebook(
        self,
        notebook_id: str,
        user_id: str | None = None
    ) -> Dict[str, Any]:
        """Get notebook details.

        Args:
            notebook_id: Notebook UUID
            user_id: Optional user UUID for authorization

        Returns:
            Dictionary with notebook details and statistics
        """
        pass

    @abstractmethod
    def list_notebooks(
        self,
        user_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List notebooks for a user.

        Args:
            user_id: User UUID
            limit: Maximum number of notebooks to return
            offset: Offset for pagination

        Returns:
            List of notebook metadata dictionaries
        """
        pass

    @abstractmethod
    def update_notebook(
        self,
        notebook_id: str,
        name: str | None = None,
        description: str | None = None,
        user_id: str | None = None
    ) -> bool:
        """Update notebook metadata.

        Args:
            notebook_id: Notebook UUID
            name: Optional new name
            description: Optional new description
            user_id: Optional user UUID for authorization

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def delete_notebook(
        self,
        notebook_id: str,
        user_id: str | None = None
    ) -> bool:
        """Delete a notebook and all its documents.

        Args:
            notebook_id: Notebook UUID
            user_id: Optional user UUID for authorization

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def switch_notebook(
        self,
        notebook_id: str,
        user_id: str | None = None
    ) -> bool:
        """Switch active notebook context.

        Args:
            notebook_id: Notebook UUID to switch to
            user_id: Optional user UUID for authorization

        Returns:
            True if successful
        """
        pass


class IVisionService(ABC):
    """Interface for vision/image understanding operations.

    Handles image analysis, text extraction, and vision provider management.
    """

    @abstractmethod
    def analyze_image(
        self,
        image_path: str | Path,
        prompt: str | None = None,
        provider: str | None = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Analyze an image with vision AI.

        Args:
            image_path: Path to image file
            prompt: Optional prompt to guide analysis
            provider: Optional provider name override
            **kwargs: Provider-specific options

        Returns:
            Dictionary containing:
            - success: Boolean success status
            - description: Image description
            - text_content: Extracted text
            - provider: Provider name used
            - model: Model name used
            - error: Optional error message
        """
        pass

    @abstractmethod
    def extract_text(
        self,
        image_path: str | Path,
        provider: str | None = None,
        **kwargs
    ) -> str:
        """Extract text from an image (OCR-like).

        Args:
            image_path: Path to image file
            provider: Optional provider name override
            **kwargs: Provider-specific options

        Returns:
            Extracted text content
        """
        pass

    @abstractmethod
    def list_providers(self) -> List[str]:
        """List available vision providers.

        Returns:
            List of provider name identifiers
        """
        pass

    @abstractmethod
    def get_provider_info(self, provider: str) -> Dict[str, Any]:
        """Get information about a specific provider.

        Args:
            provider: Provider name identifier

        Returns:
            Dictionary with provider capabilities and settings
        """
        pass

"""Abstract interface for content processors."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from pathlib import Path

from llama_index.core.schema import Document


class ContentProcessor(ABC):
    """
    Abstract base class for content processors.

    Implementations handle specific file types:
    - PDF processor
    - DOCX processor
    - Image processor (OCR)
    - Audio processor (transcription)
    - Custom processors
    """

    @abstractmethod
    def process(
        self,
        content: bytes,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Process file content into documents.

        Args:
            content: Raw file content as bytes
            filename: Original filename for type detection
            metadata: Optional metadata to attach to documents

        Returns:
            List of Document objects with extracted text
        """
        pass

    @abstractmethod
    def process_file(
        self,
        file_path: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Process a file from disk into documents.

        Args:
            file_path: Path to the file
            metadata: Optional metadata to attach to documents

        Returns:
            List of Document objects with extracted text
        """
        pass

    @property
    @abstractmethod
    def supported_types(self) -> List[str]:
        """
        Return list of supported file extensions.

        Returns:
            List of file extensions (e.g., ['.pdf', '.PDF'])
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the processor name."""
        pass

    def can_process(self, filename: str) -> bool:
        """
        Check if this processor can handle the given file.

        Args:
            filename: Filename to check

        Returns:
            True if file type is supported
        """
        ext = Path(filename).suffix.lower()
        return ext in [s.lower() for s in self.supported_types]

    @property
    def description(self) -> str:
        """Return a description of what this processor handles."""
        types = ", ".join(self.supported_types)
        return f"{self.name} - handles: {types}"

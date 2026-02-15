"""Abstract interfaces for vision providers (image understanding)."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class VisionAnalysisResult:
    """Represents the result of image analysis."""
    description: str
    text_content: str
    provider: str
    model: str
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class VisionProvider(ABC):
    """Abstract base class for vision providers.

    All vision providers must implement this interface
    to be compatible with the plugin system.
    """

    @abstractmethod
    def analyze_image(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        **kwargs
    ) -> VisionAnalysisResult:
        """Analyze an image and extract description and text.

        Args:
            image_path: Path to the image file
            prompt: Optional prompt to guide the analysis
            **kwargs: Provider-specific options

        Returns:
            VisionAnalysisResult with description and text content
        """
        pass

    @abstractmethod
    def extract_text(
        self,
        image_path: str,
        **kwargs
    ) -> str:
        """Extract text from an image (OCR-like functionality).

        Args:
            image_path: Path to the image file
            **kwargs: Provider-specific options

        Returns:
            Extracted text content from the image
        """
        pass

    @abstractmethod
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about this provider.

        Returns:
            Dictionary with provider capabilities and settings
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the provider name identifier."""
        pass

    def validate(self) -> bool:
        """Validate provider configuration and API key availability.

        Returns:
            True if provider is properly configured
        """
        try:
            info = self.get_provider_info()
            return info.get("available", False)
        except Exception:
            return False

    def get_supported_formats(self) -> list:
        """Get list of supported image formats.

        Returns:
            List of supported file extensions
        """
        return [".png", ".jpg", ".jpeg", ".gif", ".webp"]

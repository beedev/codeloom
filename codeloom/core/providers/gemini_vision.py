"""Gemini Vision provider implementation for image understanding."""

import logging
import os
import base64
from pathlib import Path
from typing import Dict, Any, Optional

from ..interfaces.vision import VisionProvider, VisionAnalysisResult
from ...setting import get_settings, RAGSettings

logger = logging.getLogger(__name__)


class GeminiVisionProvider(VisionProvider):
    """
    Google Gemini Vision provider for image understanding.

    Uses Gemini's vision models to analyze images and extract text.

    Supported Models:
    - gemini-2.0-flash-exp: Fast, experimental (default)
    - gemini-1.5-pro-vision: High accuracy
    - gemini-1.5-flash: Fast, efficient
    """

    SUPPORTED_MODELS = {
        "gemini-2.0-flash-exp",       # Fast, experimental (default)
        "gemini-1.5-pro-vision",      # High accuracy
        "gemini-1.5-flash",           # Fast, efficient
        "gemini-1.5-pro",             # Pro model
    }

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        setting: Optional[RAGSettings] = None,
    ):
        """
        Initialize Gemini Vision provider.

        Args:
            model: Model name (default from GEMINI_VISION_MODEL env var)
            api_key: Google API key (uses GOOGLE_API_KEY env var if not provided)
            setting: RAG settings instance
        """
        self._setting = setting or get_settings()

        self._model = model or os.getenv(
            "GEMINI_VISION_MODEL",
            "gemini-2.0-flash-exp"
        )
        self._api_key = api_key or os.getenv("GOOGLE_API_KEY")

        if not self._api_key:
            logger.warning(
                "Google API key not found. Set GOOGLE_API_KEY environment variable."
            )
            self._client = None
        else:
            self._client = None
            self._initialize()

    def _initialize(self) -> None:
        """Initialize the Google Generative AI client."""
        try:
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
            logger.debug(f"Initialized Gemini Vision provider with model: {self._model}")
        except ImportError:
            raise ImportError(
                "google-genai not installed. "
                "Run: pip install google-genai"
            )

    def _load_image(self, image_path: str) -> tuple:
        """Load image from path and return base64 data with mime type."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Determine mime type
        suffix = path.suffix.lower()
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        mime_type = mime_types.get(suffix, "image/png")

        # Read and encode image
        image_bytes = path.read_bytes()
        image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

        return image_b64, mime_type

    def analyze_image(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        **kwargs
    ) -> VisionAnalysisResult:
        """
        Analyze an image and extract description and text.

        Args:
            image_path: Path to the image file
            prompt: Optional prompt to guide the analysis
            **kwargs: Additional options

        Returns:
            VisionAnalysisResult with description and text content
        """
        if self._client is None:
            raise RuntimeError("Gemini Vision client not initialized. Check API key.")

        try:
            from google.genai import types

            # Load image
            image_b64, mime_type = self._load_image(image_path)

            # Build analysis prompt
            analysis_prompt = prompt or """Analyze this image comprehensively. Provide:
1. A detailed description of what you see in the image
2. Any text visible in the image (transcribed exactly)
3. Key elements, objects, or information present

Format your response as:
DESCRIPTION: [detailed description]
TEXT CONTENT: [any visible text, or "No text found" if none]
KEY ELEMENTS: [list of key elements]"""

            # Create content with image
            content = [
                types.Part.from_text(text=analysis_prompt),
                types.Part.from_bytes(
                    data=base64.standard_b64decode(image_b64),
                    mime_type=mime_type
                ),
            ]

            # Generate analysis
            response = self._client.models.generate_content(
                model=self._model,
                contents=content,
            )

            # Parse response
            response_text = response.text if hasattr(response, 'text') else str(response)

            # Extract text content from response
            text_content = ""
            description = response_text

            if "TEXT CONTENT:" in response_text:
                parts = response_text.split("TEXT CONTENT:")
                if len(parts) > 1:
                    text_section = parts[1].split("KEY ELEMENTS:")[0] if "KEY ELEMENTS:" in parts[1] else parts[1]
                    text_content = text_section.strip()

            return VisionAnalysisResult(
                description=description,
                text_content=text_content,
                provider=self.name,
                model=self._model,
                metadata={"image_path": image_path}
            )

        except Exception as e:
            logger.error(f"Gemini Vision analysis failed: {e}")
            raise RuntimeError(f"Failed to analyze image: {e}") from e

    def extract_text(
        self,
        image_path: str,
        **kwargs
    ) -> str:
        """
        Extract text from an image (OCR-like functionality).

        Args:
            image_path: Path to the image file
            **kwargs: Additional options

        Returns:
            Extracted text content from the image
        """
        if self._client is None:
            raise RuntimeError("Gemini Vision client not initialized. Check API key.")

        try:
            from google.genai import types

            # Load image
            image_b64, mime_type = self._load_image(image_path)

            # OCR-focused prompt
            ocr_prompt = """Extract ALL text from this image.
Include:
- All visible text, headers, labels
- Text in charts, diagrams, or tables
- Any watermarks or small print

Return ONLY the extracted text, preserving the original layout as much as possible.
If no text is found, respond with "No text found"."""

            # Create content with image
            content = [
                types.Part.from_text(text=ocr_prompt),
                types.Part.from_bytes(
                    data=base64.standard_b64decode(image_b64),
                    mime_type=mime_type
                ),
            ]

            # Generate extraction
            response = self._client.models.generate_content(
                model=self._model,
                contents=content,
            )

            text = response.text if hasattr(response, 'text') else str(response)
            return text.strip()

        except Exception as e:
            logger.error(f"Gemini Vision text extraction failed: {e}")
            raise RuntimeError(f"Failed to extract text: {e}") from e

    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about this provider.

        Returns:
            Dictionary with provider capabilities and settings
        """
        return {
            "name": "gemini",
            "display_name": "Google Gemini Vision",
            "model": self._model,
            "available": self._client is not None,
            "supported_models": list(self.SUPPORTED_MODELS),
            "supported_formats": self.get_supported_formats(),
            "capabilities": [
                "image_analysis",
                "text_extraction",
                "object_detection",
                "scene_understanding",
            ],
        }

    @property
    def name(self) -> str:
        """Get the provider name identifier."""
        return "gemini"

    def validate(self) -> bool:
        """
        Validate provider configuration and connectivity.

        Returns:
            True if provider is properly configured
        """
        try:
            if self._client is None:
                return False
            return True
        except Exception as e:
            logger.warning(f"Gemini Vision validation failed: {e}")
            return False

    @classmethod
    def list_supported_models(cls) -> list:
        """Return list of supported Gemini vision models."""
        return sorted(cls.SUPPORTED_MODELS)

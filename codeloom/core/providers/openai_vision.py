"""OpenAI Vision provider implementation for image understanding."""

import logging
import os
import base64
from pathlib import Path
from typing import Dict, Any, Optional

from ..interfaces.vision import VisionProvider, VisionAnalysisResult
from ...setting import get_settings, RAGSettings

logger = logging.getLogger(__name__)


class OpenAIVisionProvider(VisionProvider):
    """
    OpenAI Vision provider for image understanding.

    Uses GPT-4V and GPT-4o models to analyze images and extract text.

    Supported Models:
    - gpt-4o: Latest multimodal model (default)
    - gpt-4-turbo: GPT-4 Turbo with vision
    - gpt-4-vision-preview: Original vision preview
    """

    SUPPORTED_MODELS = {
        "gpt-4o",                  # Latest multimodal (default)
        "gpt-4o-mini",             # Smaller, faster
        "gpt-4-turbo",             # GPT-4 Turbo with vision
        "gpt-4-vision-preview",    # Original vision preview
    }

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        setting: Optional[RAGSettings] = None,
    ):
        """
        Initialize OpenAI Vision provider.

        Args:
            model: Model name (default from OPENAI_VISION_MODEL env var)
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            setting: RAG settings instance
        """
        self._setting = setting or get_settings()

        self._model = model or os.getenv(
            "OPENAI_VISION_MODEL",
            "gpt-4o"
        )
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self._api_key:
            logger.warning(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
            )
            self._client = None
        else:
            self._client = None
            self._initialize()

    def _initialize(self) -> None:
        """Initialize the OpenAI client."""
        try:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key)
            logger.debug(f"Initialized OpenAI Vision provider with model: {self._model}")
        except ImportError:
            raise ImportError(
                "openai not installed. "
                "Run: pip install openai"
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
            **kwargs: Additional options (max_tokens, detail)

        Returns:
            VisionAnalysisResult with description and text content
        """
        if self._client is None:
            raise RuntimeError("OpenAI Vision client not initialized. Check API key.")

        try:
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

            # Get optional parameters
            max_tokens = kwargs.get("max_tokens", 1024)
            detail = kwargs.get("detail", "auto")  # low, high, or auto

            # Create message with image
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": analysis_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_b64}",
                                    "detail": detail,
                                },
                            },
                        ],
                    }
                ],
                max_tokens=max_tokens,
            )

            # Parse response
            response_text = response.choices[0].message.content or ""

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
                metadata={
                    "image_path": image_path,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    }
                }
            )

        except Exception as e:
            logger.error(f"OpenAI Vision analysis failed: {e}")
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
            **kwargs: Additional options (max_tokens, detail)

        Returns:
            Extracted text content from the image
        """
        if self._client is None:
            raise RuntimeError("OpenAI Vision client not initialized. Check API key.")

        try:
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

            # Get optional parameters
            max_tokens = kwargs.get("max_tokens", 2048)
            detail = kwargs.get("detail", "high")  # Use high detail for OCR

            # Create message with image
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": ocr_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_b64}",
                                    "detail": detail,
                                },
                            },
                        ],
                    }
                ],
                max_tokens=max_tokens,
            )

            text = response.choices[0].message.content or ""
            return text.strip()

        except Exception as e:
            logger.error(f"OpenAI Vision text extraction failed: {e}")
            raise RuntimeError(f"Failed to extract text: {e}") from e

    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about this provider.

        Returns:
            Dictionary with provider capabilities and settings
        """
        return {
            "name": "openai",
            "display_name": "OpenAI Vision (GPT-4V)",
            "model": self._model,
            "available": self._client is not None,
            "supported_models": list(self.SUPPORTED_MODELS),
            "supported_formats": self.get_supported_formats(),
            "capabilities": [
                "image_analysis",
                "text_extraction",
                "object_detection",
                "scene_understanding",
                "document_understanding",
            ],
        }

    @property
    def name(self) -> str:
        """Get the provider name identifier."""
        return "openai"

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
            logger.warning(f"OpenAI Vision validation failed: {e}")
            return False

    @classmethod
    def list_supported_models(cls) -> list:
        """Return list of supported OpenAI vision models."""
        return sorted(cls.SUPPORTED_MODELS)

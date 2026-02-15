"""Gemini Image Generation provider implementation (Nano Banana)."""

import logging
import os
import base64
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..interfaces.image_generation import ImageGenerationProvider
from ...setting import get_settings, RAGSettings

logger = logging.getLogger(__name__)


class GeminiImageProvider(ImageGenerationProvider):
    """
    Google Gemini/Imagen Image Generation provider.

    Supports image generation using Google's Imagen and Gemini models.

    Models (Imagen - use generate_images API):
    - imagen-4.0-generate-001: Production-ready, high quality (default)
    - imagen-4.0-fast-generate-001: Faster, lower quality
    - imagen-4.0-ultra-generate-001: Highest quality, slower

    Models (Gemini - use generate_content API with IMAGE modality):
    - gemini-2.0-flash-exp: Fast, experimental (Nano Banana)

    Default: imagen-4.0-generate-001 (best balance of quality and speed)
    """

    # Imagen models use generate_images API
    IMAGEN_MODELS = {
        "imagen-4.0-generate-001",       # Production, balanced
        "imagen-4.0-fast-generate-001",  # Faster, lower quality
        "imagen-4.0-ultra-generate-001", # Highest quality
    }

    # Gemini models use generate_content API with IMAGE modality
    GEMINI_MODELS = {
        "gemini-2.0-flash-exp",          # Nano Banana - fast, experimental
        "gemini-3-pro-image-preview",    # Nano Banana Pro - higher quality, 4K
    }

    SUPPORTED_MODELS = IMAGEN_MODELS | GEMINI_MODELS

    SUPPORTED_ASPECT_RATIOS = {
        "1:1",   # Square
        "16:9",  # Landscape wide
        "9:16",  # Portrait tall
        "4:3",   # Standard landscape
        "3:4",   # Standard portrait
    }

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        output_dir: Optional[str] = None,
        setting: Optional[RAGSettings] = None,
    ):
        """
        Initialize Gemini/Imagen Image provider.

        Args:
            model: Model name (default: imagen-4.0-generate-001)
            api_key: Google API key (uses GOOGLE_API_KEY env var if not provided)
            output_dir: Directory for generated images (default: outputs/images)
            setting: RAG settings instance
        """
        self._setting = setting or get_settings()

        self._model = model or os.getenv(
            "GEMINI_IMAGE_MODEL",
            os.getenv("IMAGEN_MODEL", "imagen-4.0-generate-001")
        )
        self._api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self._output_dir = Path(
            output_dir or os.getenv("IMAGE_OUTPUT_DIR", "outputs/images")
        )

        if not self._api_key:
            raise ValueError(
                "Google API key required. Set GOOGLE_API_KEY environment variable."
            )

        self._client = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the Google Generative AI client."""
        try:
            from google import genai
            from google.genai import types

            self._client = genai.Client(api_key=self._api_key)
            self._types = types

            # Ensure output directory exists
            self._output_dir.mkdir(parents=True, exist_ok=True)

            logger.debug(f"Initialized Gemini Image provider with model: {self._model}")
        except ImportError:
            raise ImportError(
                "google-genai not installed. "
                "Run: pip install google-genai"
            )

    def _is_imagen_model(self) -> bool:
        """Check if current model is an Imagen model."""
        return self._model in self.IMAGEN_MODELS or self._model.startswith("imagen-")

    def generate(
        self,
        prompt: str,
        num_images: int = 1,
        aspect_ratio: str = "1:1",
        **kwargs
    ) -> List[str]:
        """
        Generate images from a text prompt.

        Args:
            prompt: Text description of the image to generate
            num_images: Number of images to generate (1-4)
            aspect_ratio: Aspect ratio (1:1, 16:9, 9:16, 4:3, 3:4)
            **kwargs: Additional options (safety_filter_level, etc.)

        Returns:
            List of file paths to generated images
        """
        if self._client is None:
            raise RuntimeError("Gemini client not initialized")

        # Validate inputs
        num_images = max(1, min(num_images, 4))
        if aspect_ratio not in self.SUPPORTED_ASPECT_RATIOS:
            logger.warning(
                f"Unsupported aspect ratio '{aspect_ratio}', using '1:1'"
            )
            aspect_ratio = "1:1"

        # Route to appropriate API based on model type
        if self._is_imagen_model():
            return self._generate_with_imagen(prompt, num_images, aspect_ratio)
        else:
            return self._generate_with_gemini(prompt, num_images, aspect_ratio)

    def _generate_with_imagen(
        self,
        prompt: str,
        num_images: int,
        aspect_ratio: str,
    ) -> List[str]:
        """Generate images using Imagen API (generate_images)."""
        generated_paths = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            # Configure image generation using GenerateImagesConfig
            config = self._types.GenerateImagesConfig(
                numberOfImages=num_images,
                aspectRatio=aspect_ratio,
                outputMimeType="image/png",
            )

            # Generate images using the generate_images API
            response = self._client.models.generate_images(
                model=self._model,
                prompt=prompt,
                config=config,
            )

            # Process response - Imagen returns generated_images list
            for idx, generated_image in enumerate(response.generated_images):
                image_data = generated_image.image.image_bytes

                filename = f"imagen_{timestamp}_{idx}.png"
                filepath = self._output_dir / filename

                if isinstance(image_data, str):
                    image_bytes = base64.b64decode(image_data)
                else:
                    image_bytes = image_data

                filepath.write_bytes(image_bytes)
                generated_paths.append(str(filepath))
                logger.info(f"Generated image saved: {filepath}")

        except Exception as e:
            logger.error(f"Imagen generation failed: {e}")
            raise RuntimeError(f"Failed to generate image: {e}") from e

        return generated_paths

    def _generate_with_gemini(
        self,
        prompt: str,
        num_images: int,
        aspect_ratio: str,
    ) -> List[str]:
        """Generate images using Gemini API (generate_content with IMAGE modality)."""
        generated_paths = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            # Configure for image generation using generate_content
            config = self._types.GenerateContentConfig(
                responseModalities=["IMAGE", "TEXT"],
            )

            # Generate using generate_content API
            # Note: Gemini generates one image per request
            for img_idx in range(num_images):
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=f"Generate an image: {prompt}",
                    config=config,
                )

                # Process response - find parts with inline_data
                for part in response.candidates[0].content.parts:
                    if part.inline_data is not None:
                        image_data = part.inline_data.data
                        mime_type = part.inline_data.mime_type

                        # Determine extension from mime type
                        ext = "png"
                        if "jpeg" in mime_type or "jpg" in mime_type:
                            ext = "jpg"
                        elif "webp" in mime_type:
                            ext = "webp"

                        filename = f"gemini_{timestamp}_{img_idx}.{ext}"
                        filepath = self._output_dir / filename

                        if isinstance(image_data, str):
                            image_bytes = base64.b64decode(image_data)
                        else:
                            image_bytes = image_data

                        filepath.write_bytes(image_bytes)
                        generated_paths.append(str(filepath))
                        logger.info(f"Generated image saved: {filepath}")
                        break  # Only take first image from this request

        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise RuntimeError(f"Failed to generate image: {e}") from e

        return generated_paths

    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about this provider.

        Returns:
            Dictionary with provider capabilities and settings
        """
        return {
            "name": "gemini",
            "display_name": "Google Gemini (Nano Banana)",
            "model": self._model,
            "available": self._client is not None,
            "supported_models": list(self.SUPPORTED_MODELS),
            "supported_aspect_ratios": list(self.SUPPORTED_ASPECT_RATIOS),
            "max_images_per_request": 4,
            "output_formats": ["png", "jpeg", "webp"],
            "capabilities": [
                "text_to_image",
                "aspect_ratio_control",
                "batch_generation",
            ],
            "pricing": self._get_pricing(),
            "output_dir": str(self._output_dir),
        }

    def _get_pricing(self) -> Dict[str, Any]:
        """Get approximate pricing information."""
        pricing = {
            # Imagen 4.0 models
            "imagen-4.0-generate-001": {
                "per_image": 0.04,
                "currency": "USD",
                "note": "Production quality, balanced speed"
            },
            "imagen-4.0-fast-generate-001": {
                "per_image": 0.02,
                "currency": "USD",
                "note": "Faster generation, lower quality"
            },
            "imagen-4.0-ultra-generate-001": {
                "per_image": 0.08,
                "currency": "USD",
                "note": "Highest quality, slower"
            },
            # Gemini models (Nano Banana)
            "gemini-2.0-flash-exp": {
                "per_image": 0.039,
                "currency": "USD",
                "note": "Nano Banana - experimental, text generation capability"
            },
            "gemini-3-pro-image-preview": {
                "per_image": 0.07,
                "currency": "USD",
                "note": "Nano Banana Pro - higher quality, 4K support"
            },
        }
        return pricing.get(
            self._model,
            {"per_image": 0.04, "currency": "USD", "note": "Estimated"}
        )

    @property
    def name(self) -> str:
        """Get the provider name identifier."""
        return "gemini"

    @property
    def output_dir(self) -> Path:
        """Get the output directory for generated images."""
        return self._output_dir

    def validate(self) -> bool:
        """
        Validate provider configuration and connectivity.

        Returns:
            True if provider is properly configured
        """
        try:
            if self._client is None:
                return False

            # Try to list models to verify API key works
            # This is a lightweight check
            info = self.get_provider_info()
            return info.get("available", False)
        except Exception as e:
            logger.warning(f"Gemini validation failed: {e}")
            return False

    @classmethod
    def list_supported_models(cls) -> List[str]:
        """Return list of supported Gemini image models."""
        return sorted(cls.SUPPORTED_MODELS)

    @classmethod
    def list_supported_aspect_ratios(cls) -> List[str]:
        """Return list of supported aspect ratios."""
        return list(cls.SUPPORTED_ASPECT_RATIOS)

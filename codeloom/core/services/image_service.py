"""Image service implementation for DBProject.

This module implements the image service layer, handling all image generation
operations and provider management.
"""

from typing import Dict, Any, List
from pathlib import Path

from .base import BaseService
from ..interfaces.services import IImageService


class ImageService(BaseService, IImageService):
    """Service for image generation operations.

    Handles image generation requests using various providers (Gemini, etc.)
    and manages generated images. Coordinates with the plugin-based
    image provider system.
    """

    def __init__(self, pipeline, db_manager=None, project_manager=None):
        """Initialize image service.

        Args:
            pipeline: LocalRAGPipeline instance
            db_manager: Optional DatabaseManager
            project_manager: Optional ProjectManager
        """
        super().__init__(pipeline, db_manager, project_manager)

        # Initialize image provider via plugin system
        self._image_provider = None
        try:
            from ...core.plugins import get_configured_image_provider
            self._image_provider = get_configured_image_provider()
            self.logger.info(f"Initialized image provider: {self._image_provider.name}")
        except Exception as e:
            self.logger.warning(f"Image generation not available: {e}")

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

        Raises:
            ValueError: If prompt is empty or parameters are invalid
            RuntimeError: If image provider is not available
        """
        # Validate inputs
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        if num_images < 1 or num_images > 4:
            raise ValueError("num_images must be between 1 and 4")

        valid_ratios = ["1:1", "16:9", "9:16", "4:3", "3:4"]
        if aspect_ratio not in valid_ratios:
            raise ValueError(f"aspect_ratio must be one of {valid_ratios}")

        # Check if image provider is available
        if not self._image_provider:
            raise RuntimeError(
                "Image generation not configured. "
                "Set GOOGLE_API_KEY or configure another image provider."
            )

        self._log_operation(
            "generate",
            prompt_length=len(prompt),
            num_images=num_images,
            aspect_ratio=aspect_ratio,
            provider=provider or self._image_provider.name
        )

        try:
            # Generate images using plugin provider
            image_paths = self._image_provider.generate(
                prompt=prompt,
                num_images=num_images,
                aspect_ratio=aspect_ratio
            )

            if not image_paths:
                return {
                    "success": False,
                    "error": "No images were generated"
                }

            # Get image info for each generated image
            images_info = []
            for path in image_paths:
                try:
                    info = self._image_provider.get_image_info(path)
                    info["path"] = path
                    info["url"] = f"/image/{Path(path).name}"
                    images_info.append(info)
                except Exception as info_err:
                    self.logger.warning(f"Could not get info for image {path}: {info_err}")
                    # Add basic info
                    images_info.append({
                        "path": path,
                        "url": f"/image/{Path(path).name}"
                    })

            self.logger.info(
                f"Successfully generated {len(image_paths)} images "
                f"using {self._image_provider.name}"
            )

            return {
                "success": True,
                "images": images_info,
                "provider": self._image_provider.name,
                "count": len(image_paths)
            }

        except Exception as e:
            self._log_error("generate", e, prompt_length=len(prompt))
            return {
                "success": False,
                "error": str(e)
            }

    def list_providers(self) -> List[str]:
        """List available image generation providers.

        Returns:
            List of provider name identifiers

        Note:
            Currently returns a list with a single provider (the configured one).
            Future versions may support multiple providers.
        """
        self._log_operation("list_providers")

        try:
            if self._image_provider:
                return [self._image_provider.name]
            return []

        except Exception as e:
            self._log_error("list_providers", e)
            return []

    def get_provider_info(self, provider: str) -> Dict[str, Any]:
        """Get information about a specific provider.

        Args:
            provider: Provider name identifier

        Returns:
            Dictionary with provider capabilities and settings

        Raises:
            ValueError: If provider name is invalid
        """
        if not provider:
            raise ValueError("provider is required")

        self._log_operation("get_provider_info", provider=provider)

        try:
            if not self._image_provider:
                return {
                    "error": "No image provider configured"
                }

            if provider != self._image_provider.name:
                return {
                    "error": f"Provider {provider} not found"
                }

            # Return provider capabilities
            return {
                "name": self._image_provider.name,
                "supported_ratios": ["1:1", "16:9", "9:16", "4:3", "3:4"],
                "max_images": 4,
                "available": True
            }

        except Exception as e:
            self._log_error("get_provider_info", e, provider=provider)
            raise

    def list_generated_images(self) -> List[Dict[str, Any]]:
        """List all previously generated images.

        Returns:
            List of image metadata dictionaries containing:
            - path: File path to image
            - url: URL to access image
            - size: File size in bytes
            - created_at: Creation timestamp
            - dimensions: Image dimensions (if available)

        Raises:
            RuntimeError: If image provider is not available
        """
        # Check if image provider is available
        if not self._image_provider:
            raise RuntimeError("Image provider not configured")

        self._log_operation("list_generated_images")

        try:
            # Get list of generated images from provider
            image_paths = self._image_provider.list_generated_images()

            images_info = []
            for path in image_paths:
                try:
                    info = self._image_provider.get_image_info(path)
                    info["url"] = f"/image/{Path(path).name}"
                    images_info.append(info)
                except Exception as info_err:
                    self.logger.warning(f"Could not get info for image {path}: {info_err}")
                    # Add basic info
                    images_info.append({
                        "path": path,
                        "url": f"/image/{Path(path).name}"
                    })

            self.logger.info(f"Found {len(images_info)} generated images")
            return images_info

        except Exception as e:
            self._log_error("list_generated_images", e)
            raise

    def clear_images(self) -> int:
        """Clear all generated images.

        Returns:
            Number of images deleted

        Raises:
            RuntimeError: If image provider is not available
        """
        # Check if image provider is available
        if not self._image_provider:
            raise RuntimeError("Image provider not configured")

        self._log_operation("clear_images")

        try:
            # Clear images using provider
            deleted_count = self._image_provider.clear_images()

            self.logger.info(f"Deleted {deleted_count} generated images")
            return deleted_count

        except Exception as e:
            self._log_error("clear_images", e)
            raise

    def delete_image(self, image_path: str) -> bool:
        """Delete a specific generated image.

        Args:
            image_path: Path or filename of the image to delete

        Returns:
            True if successful

        Raises:
            ValueError: If image_path is invalid
        """
        if not image_path:
            raise ValueError("image_path is required")

        self._log_operation("delete_image", image_path=image_path)

        try:
            # Convert to Path object
            path = Path(image_path)

            # If only filename provided, construct full path
            if not path.is_absolute() and self._image_provider:
                path = self._image_provider.output_dir / path.name

            # Delete the file
            if path.exists():
                path.unlink()
                self.logger.info(f"Deleted image: {path}")
                return True
            else:
                self.logger.warning(f"Image not found: {path}")
                return False

        except Exception as e:
            self._log_error("delete_image", e, image_path=image_path)
            return False

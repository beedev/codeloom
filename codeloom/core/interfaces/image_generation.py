"""Abstract interface for image generation providers."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path


class ImageGenerationProvider(ABC):
    """Abstract base class for image generation providers.

    All image generation providers must implement this interface
    to be compatible with the plugin system.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        num_images: int = 1,
        aspect_ratio: str = "1:1",
        **kwargs
    ) -> List[str]:
        """Generate images from a text prompt.

        Args:
            prompt: Text description of the image to generate
            num_images: Number of images to generate (1-4)
            aspect_ratio: Aspect ratio (1:1, 16:9, 9:16, 4:3, 3:4)
            **kwargs: Provider-specific options

        Returns:
            List of file paths to generated images
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

    @property
    def output_dir(self) -> Path:
        """Get the output directory for generated images."""
        return Path("outputs/images")

    def validate(self) -> bool:
        """Validate provider configuration and connectivity.

        Returns:
            True if provider is properly configured
        """
        try:
            info = self.get_provider_info()
            return info.get("available", False)
        except Exception:
            return False

    def list_generated_images(self) -> List[str]:
        """List all previously generated images.

        Returns:
            List of file paths to generated images
        """
        if not self.output_dir.exists():
            return []

        image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
        return [
            str(f) for f in self.output_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ]

    def get_image_info(self, filepath: str) -> Dict[str, Any]:
        """Get metadata about a generated image.

        Args:
            filepath: Path to the image file

        Returns:
            Dictionary with image metadata
        """
        path = Path(filepath)
        if not path.exists():
            return {"error": "File not found"}

        stat = path.stat()
        return {
            "filename": path.name,
            "size_bytes": stat.st_size,
            "created": stat.st_ctime,
            "modified": stat.st_mtime,
        }

    def clear_images(self) -> int:
        """Clear all generated images from output directory.

        Returns:
            Number of files deleted
        """
        if not self.output_dir.exists():
            return 0

        count = 0
        image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
        for f in self.output_dir.iterdir():
            if f.suffix.lower() in image_extensions:
                f.unlink()
                count += 1
        return count

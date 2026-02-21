"""Image generation using Imagen (Google Vertex AI)."""
import os
import logging
import uuid
import base64
from datetime import datetime
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


class ImageGenerator:
    """Generate images using Imagen via Google Vertex AI."""

    def __init__(self, settings):
        """Initialize the image generator.

        Args:
            settings: RAGSettings instance containing image generation configuration
        """
        self._settings = settings.image_generation
        self._project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        self._location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

        # Create output directory (use absolute path)
        self._output_dir = Path(self._settings.output_dir).resolve()
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Vertex AI client (lazy loading)
        self._client = None

        logger.info(f"ImageGenerator initialized - Output dir: {self._output_dir}")

    def _get_client(self):
        """Lazy load the Vertex AI client."""
        if self._client is None:
            try:
                from google.cloud import aiplatform

                if not self._project_id:
                    raise ValueError("GOOGLE_CLOUD_PROJECT not set in environment")

                aiplatform.init(project=self._project_id, location=self._location)
                self._client = aiplatform
                logger.info(f"Vertex AI initialized - Project: {self._project_id}, Location: {self._location}")
            except ImportError:
                raise ImportError(
                    "google-cloud-aiplatform not installed. "
                    "Install it with: pip install google-cloud-aiplatform"
                )
        return self._client

    def generate_image(
        self,
        prompt: str,
        num_images: int = 1,
        aspect_ratio: str = "1:1"
    ) -> List[str]:
        """Generate images using Imagen via Vertex AI.

        Args:
            prompt: Text description of the image to generate
            num_images: Number of images to generate (1-4 for Imagen)
            aspect_ratio: Aspect ratio (1:1, 16:9, 9:16, 4:3, 3:4)

        Returns:
            List of file paths to generated images
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        if num_images < 1 or num_images > 4:
            raise ValueError("Number of images must be between 1 and 4")

        client = self._get_client()
        generated_files = []

        try:
            logger.info(f"Generating {num_images} image(s) with Imagen: {prompt[:50]}...")

            # Import Imagen model
            from vertexai.preview.vision_models import ImageGenerationModel

            # Initialize Imagen model
            model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")

            # Map aspect ratio to Imagen format
            aspect_ratio_map = {
                "1:1": "1:1",
                "16:9": "16:9",
                "9:16": "9:16",
                "4:3": "4:3",
                "3:4": "3:4",
            }
            imagen_aspect_ratio = aspect_ratio_map.get(aspect_ratio, "1:1")

            # Generate images using Imagen
            response = model.generate_images(
                prompt=prompt,
                number_of_images=num_images,
                aspect_ratio=imagen_aspect_ratio,
                safety_filter_level="block_some",
                person_generation="allow_adult",
            )

            # Save each generated image
            for i, image in enumerate(response.images):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_id = str(uuid.uuid4())[:8]
                filename = f"generated_{timestamp}_{unique_id}.png"
                filepath = self._output_dir / filename

                # Save image to file
                image.save(location=str(filepath))
                generated_files.append(str(filepath))
                logger.info(f"Image {i+1}/{num_images} saved: {filename}")

            logger.info(f"Successfully generated {len(generated_files)} image(s)")
            return generated_files

        except Exception as e:
            logger.error(f"Error generating images with Imagen: {e}")
            raise

    def list_generated_images(self) -> List[str]:
        """List all generated images in the output directory.

        Returns:
            List of file paths to generated images
        """
        try:
            supported_formats = self._settings.supported_formats.split(',')
            image_files = []

            for format_ext in supported_formats:
                format_ext = format_ext.strip()
                image_files.extend(self._output_dir.glob(f"*.{format_ext}"))

            # Sort by modification time (newest first)
            image_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            return [str(f) for f in image_files]

        except Exception as e:
            logger.error(f"Error listing images: {e}")
            return []

    def clear_output_dir(self) -> int:
        """Delete all generated images from the output directory.

        Returns:
            Number of files deleted
        """
        try:
            image_files = self.list_generated_images()
            deleted_count = 0

            for filepath in image_files:
                try:
                    Path(filepath).unlink()
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete {filepath}: {e}")

            logger.info(f"Cleared {deleted_count} image(s) from output directory")
            return deleted_count

        except Exception as e:
            logger.error(f"Error clearing output directory: {e}")
            return 0

    def add_text_overlay(self, image_path: str, text_elements: dict) -> str:
        """Add clean text overlays to a generated image.

        Args:
            image_path: Path to the base image
            text_elements: Dictionary with text to add
                {
                    "title": "Main Title Text",
                    "sections": [
                        {"heading": "Section 1", "content": "Description"},
                        {"heading": "Section 2", "content": "Description"}
                    ]
                }

        Returns:
            Path to the new image with text overlays
        """
        try:
            from PIL import Image, ImageDraw, ImageFont

            # Open the base image
            img = Image.open(image_path)
            draw = ImageDraw.Draw(img)

            # Load fonts (using default if custom fonts not available)
            try:
                title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 60)
                heading_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
                body_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
            except:
                # Fallback to default font
                title_font = ImageFont.load_default()
                heading_font = ImageFont.load_default()
                body_font = ImageFont.load_default()

            # Get image dimensions
            width, height = img.size

            # Add title at top
            if "title" in text_elements:
                title_text = text_elements["title"]
                # Center title
                bbox = draw.textbbox((0, 0), title_text, font=title_font)
                text_width = bbox[2] - bbox[0]
                x = (width - text_width) / 2
                y = 50

                # Add shadow for better readability
                draw.text((x+2, y+2), title_text, font=title_font, fill=(0, 0, 0, 180))
                draw.text((x, y), title_text, font=title_font, fill=(255, 255, 255, 255))

            # Add sections
            if "sections" in text_elements:
                y_offset = 180
                x_offset = 80
                section_spacing = 150

                for section in text_elements["sections"]:
                    # Section heading
                    if "heading" in section:
                        draw.text((x_offset+2, y_offset+2), section["heading"],
                                font=heading_font, fill=(0, 0, 0, 180))
                        draw.text((x_offset, y_offset), section["heading"],
                                font=heading_font, fill=(66, 135, 245, 255))  # Blue
                        y_offset += 60

                    # Section content
                    if "content" in section:
                        # Word wrap for long content
                        words = section["content"].split()
                        lines = []
                        current_line = []

                        for word in words:
                            test_line = ' '.join(current_line + [word])
                            bbox = draw.textbbox((0, 0), test_line, font=body_font)
                            if bbox[2] - bbox[0] < width - 160:
                                current_line.append(word)
                            else:
                                lines.append(' '.join(current_line))
                                current_line = [word]
                        if current_line:
                            lines.append(' '.join(current_line))

                        for line in lines[:3]:  # Max 3 lines per section
                            draw.text((x_offset+2, y_offset+2), line,
                                    font=body_font, fill=(0, 0, 0, 180))
                            draw.text((x_offset, y_offset), line,
                                    font=body_font, fill=(255, 255, 255, 255))
                            y_offset += 40

                    y_offset += section_spacing

            # Save the new image with overlay
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            new_filename = f"overlay_{timestamp}_{unique_id}.png"
            new_filepath = self._output_dir / new_filename

            img.save(str(new_filepath), "PNG")
            logger.info(f"Text overlay added, saved to: {new_filename}")

            return str(new_filepath)

        except Exception as e:
            logger.error(f"Error adding text overlay: {e}")
            return image_path  # Return original if overlay fails

    def get_image_info(self, filepath: str) -> dict:
        """Get information about a generated image.

        Args:
            filepath: Path to the image file

        Returns:
            Dictionary with image information
        """
        try:
            path = Path(filepath)
            if not path.exists():
                return {"error": "File not found"}

            stat = path.stat()
            size_mb = stat.st_size / (1024 * 1024)

            return {
                "filename": path.name,
                "size_mb": round(size_mb, 2),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            }
        except Exception as e:
            logger.error(f"Error getting image info: {e}")
            return {"error": str(e)}

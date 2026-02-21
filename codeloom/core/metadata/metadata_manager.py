"""Metadata management for IT Practices and Offerings.

This module provides centralized management for:
- IT Practices (Digital, CIS, Cloud, etc.)
- Offerings (services with descriptions and metadata)
- JSON-based persistence for configuration

"""
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class MetadataManager:
    """Manage IT practices and offerings with JSON persistence."""

    def __init__(self, config_dir: str = "data/config"):
        """Initialize metadata manager.

        Args:
            config_dir: Directory for storing configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.practices_file = self.config_dir / "practices.json"
        self.offerings_file = self.config_dir / "offerings.json"

        # Initialize default data structures
        self._practices: List[str] = []
        self._offerings: List[Dict] = []

        # Load existing data or create defaults
        self._load_or_initialize()

        logger.info(f"MetadataManager initialized - Config dir: {self.config_dir}")

    def _load_or_initialize(self) -> None:
        """Load existing configuration or create default structure."""
        # Load or create practices
        if self.practices_file.exists():
            try:
                with open(self.practices_file, "r") as f:
                    data = json.load(f)
                    self._practices = data.get("practices", [])
                logger.info(f"Loaded {len(self._practices)} IT practices")
            except Exception as e:
                logger.error(f"Error loading practices: {e}")
                self._create_default_practices()
        else:
            self._create_default_practices()

        # Load or create offerings
        if self.offerings_file.exists():
            try:
                with open(self.offerings_file, "r") as f:
                    data = json.load(f)
                    self._offerings = data.get("offerings", [])
                logger.info(f"Loaded {len(self._offerings)} offerings")
            except Exception as e:
                logger.error(f"Error loading offerings: {e}")
                self._create_default_offerings()
        else:
            self._create_default_offerings()

    def _create_default_practices(self) -> None:
        """Create default IT practices configuration."""
        default_practices = [
            "Digital Transformation",
            "Cloud Services",
            "Cybersecurity",
            "Data & Analytics",
            "Application Services",
            "Infrastructure Services",
            "Consulting & Advisory"
        ]

        practices_data = {
            "practices": default_practices,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }

        with open(self.practices_file, "w") as f:
            json.dump(practices_data, f, indent=2)

        self._practices = default_practices
        logger.info(f"Created default practices configuration with {len(default_practices)} practices")

    def _create_default_offerings(self) -> None:
        """Create default offerings configuration."""
        default_offerings = [
            {
                "id": str(uuid.uuid4()),
                "practice": "Cloud Services",
                "name": "Cloud Migration",
                "description": "End-to-end cloud migration services for legacy applications",
                "created_at": datetime.now().isoformat()
            },
            {
                "id": str(uuid.uuid4()),
                "practice": "Digital Transformation",
                "name": "Digital Experience Platform",
                "description": "Modern digital experience platform implementation",
                "created_at": datetime.now().isoformat()
            },
            {
                "id": str(uuid.uuid4()),
                "practice": "Cybersecurity",
                "name": "Security Assessment",
                "description": "Comprehensive security audit and vulnerability assessment",
                "created_at": datetime.now().isoformat()
            }
        ]

        offerings_data = {
            "offerings": default_offerings,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }

        with open(self.offerings_file, "w") as f:
            json.dump(offerings_data, f, indent=2)

        self._offerings = default_offerings
        logger.info(f"Created default offerings configuration with {len(default_offerings)} offerings")

    def _save_practices(self) -> None:
        """Save practices to JSON file."""
        practices_data = {
            "practices": self._practices,
            "created_at": self._get_created_at(self.practices_file),
            "updated_at": datetime.now().isoformat()
        }

        with open(self.practices_file, "w") as f:
            json.dump(practices_data, f, indent=2)

        logger.debug("Practices saved to disk")

    def _save_offerings(self) -> None:
        """Save offerings to JSON file."""
        offerings_data = {
            "offerings": self._offerings,
            "created_at": self._get_created_at(self.offerings_file),
            "updated_at": datetime.now().isoformat()
        }

        with open(self.offerings_file, "w") as f:
            json.dump(offerings_data, f, indent=2)

        logger.debug("Offerings saved to disk")

    def _get_created_at(self, file_path: Path) -> str:
        """Get the created_at timestamp from existing file or return current time."""
        try:
            if file_path.exists():
                with open(file_path, "r") as f:
                    data = json.load(f)
                    return data.get("created_at", datetime.now().isoformat())
        except Exception:
            pass
        return datetime.now().isoformat()

    # ========================================
    # IT Practice Management
    # ========================================

    def get_all_practices(self) -> List[str]:
        """Get all IT practices.

        Returns:
            List of practice names
        """
        return self._practices.copy()

    def add_practice(self, practice_name: str) -> bool:
        """Add a new IT practice.

        Args:
            practice_name: Name of the practice to add

        Returns:
            True if added successfully, False if already exists
        """
        if not practice_name or not practice_name.strip():
            logger.warning("Cannot add empty practice name")
            return False

        practice_name = practice_name.strip()

        if practice_name in self._practices:
            logger.warning(f"Practice already exists: {practice_name}")
            return False

        self._practices.append(practice_name)
        self._save_practices()

        logger.info(f"Added new practice: {practice_name}")
        return True

    def remove_practice(self, practice_name: str) -> bool:
        """Remove an IT practice.

        Args:
            practice_name: Name of the practice to remove

        Returns:
            True if removed successfully, False if not found
        """
        if practice_name not in self._practices:
            logger.warning(f"Practice not found: {practice_name}")
            return False

        # Check if any offerings use this practice
        offerings_with_practice = [
            o for o in self._offerings
            if o.get("practice") == practice_name
        ]

        if offerings_with_practice:
            logger.warning(
                f"Cannot remove practice '{practice_name}': "
                f"{len(offerings_with_practice)} offerings depend on it"
            )
            return False

        self._practices.remove(practice_name)
        self._save_practices()

        logger.info(f"Removed practice: {practice_name}")
        return True

    def practice_exists(self, practice_name: str) -> bool:
        """Check if a practice exists.

        Args:
            practice_name: Name of the practice to check

        Returns:
            True if practice exists
        """
        return practice_name in self._practices

    # ========================================
    # Offering Management
    # ========================================

    def get_all_offerings(self) -> List[Dict]:
        """Get all offerings.

        Returns:
            List of offering dictionaries
        """
        return self._offerings.copy()

    def get_offerings_by_practice(self, practice: str) -> List[Dict]:
        """Get all offerings for a specific practice.

        Args:
            practice: IT practice name

        Returns:
            List of offerings for the practice
        """
        return [
            offering for offering in self._offerings
            if offering.get("practice") == practice
        ]

    def get_offering_by_id(self, offering_id: str) -> Optional[Dict]:
        """Get offering by ID.

        Args:
            offering_id: Unique offering ID

        Returns:
            Offering dictionary or None if not found
        """
        for offering in self._offerings:
            if offering.get("id") == offering_id:
                return offering.copy()
        return None

    def add_offering(
        self,
        practice: str,
        offering_name: str,
        description: str = ""
    ) -> Optional[str]:
        """Add a new offering.

        Args:
            practice: IT practice name (must exist)
            offering_name: Name of the offering
            description: Offering description

        Returns:
            Offering ID if added successfully, None otherwise
        """
        # Validate practice exists
        if practice not in self._practices:
            logger.warning(f"Cannot add offering: practice '{practice}' does not exist")
            return None

        # Validate offering name
        if not offering_name or not offering_name.strip():
            logger.warning("Cannot add offering with empty name")
            return None

        offering_name = offering_name.strip()

        # Check for duplicate offering name within practice
        existing_offerings = self.get_offerings_by_practice(practice)
        if any(o.get("name") == offering_name for o in existing_offerings):
            logger.warning(
                f"Offering '{offering_name}' already exists in practice '{practice}'"
            )
            return None

        # Create offering
        offering_id = str(uuid.uuid4())
        new_offering = {
            "id": offering_id,
            "practice": practice,
            "name": offering_name,
            "description": description.strip(),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }

        self._offerings.append(new_offering)
        self._save_offerings()

        logger.info(f"Added offering: {offering_name} ({offering_id}) to practice '{practice}'")
        return offering_id

    def update_offering(
        self,
        offering_id: str,
        offering_name: Optional[str] = None,
        description: Optional[str] = None,
        practice: Optional[str] = None
    ) -> bool:
        """Update an existing offering.

        Args:
            offering_id: Offering ID to update
            offering_name: New name (optional)
            description: New description (optional)
            practice: New practice (optional, must exist)

        Returns:
            True if updated successfully, False otherwise
        """
        offering = None
        offering_index = -1

        for i, o in enumerate(self._offerings):
            if o.get("id") == offering_id:
                offering = o
                offering_index = i
                break

        if not offering:
            logger.warning(f"Offering not found: {offering_id}")
            return False

        # Validate practice if provided
        if practice and practice not in self._practices:
            logger.warning(f"Cannot update offering: practice '{practice}' does not exist")
            return False

        # Update fields
        if offering_name:
            offering["name"] = offering_name.strip()
        if description is not None:
            offering["description"] = description.strip()
        if practice:
            offering["practice"] = practice

        offering["updated_at"] = datetime.now().isoformat()

        self._offerings[offering_index] = offering
        self._save_offerings()

        logger.info(f"Updated offering: {offering_id}")
        return True

    def remove_offering(self, offering_id: str) -> bool:
        """Remove an offering.

        Args:
            offering_id: Offering ID to remove

        Returns:
            True if removed successfully, False if not found
        """
        offering_index = -1

        for i, o in enumerate(self._offerings):
            if o.get("id") == offering_id:
                offering_index = i
                break

        if offering_index == -1:
            logger.warning(f"Offering not found: {offering_id}")
            return False

        removed_offering = self._offerings.pop(offering_index)
        self._save_offerings()

        logger.info(f"Removed offering: {removed_offering.get('name')} ({offering_id})")
        return True

    def get_offerings_by_ids(self, offering_ids: List[str]) -> List[Dict]:
        """Get multiple offerings by their IDs.

        Args:
            offering_ids: List of offering IDs

        Returns:
            List of offering dictionaries
        """
        return [
            offering.copy() for offering in self._offerings
            if offering.get("id") in offering_ids
        ]

    # ========================================
    # Statistics and Reporting
    # ========================================

    def get_statistics(self) -> Dict:
        """Get metadata statistics.

        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_practices": len(self._practices),
            "total_offerings": len(self._offerings),
            "offerings_by_practice": {}
        }

        for practice in self._practices:
            offerings_count = len(self.get_offerings_by_practice(practice))
            stats["offerings_by_practice"][practice] = offerings_count

        return stats

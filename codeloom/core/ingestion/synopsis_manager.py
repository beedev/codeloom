"""Synopsis Manager for Offering-level Synopsis Generation and Storage."""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from llama_index.core.schema import BaseNode
from llama_index.core.llms.llm import LLM

logger = logging.getLogger(__name__)


class SynopsisManager:
    """
    Manages synopsis generation and storage for offerings.

    Synopses are generated at upload time and stored persistently,
    allowing faster problem-solving queries by using pre-generated summaries
    instead of generating them on-the-fly.
    """

    def __init__(self, storage_path: str = "data/offerings/synopses.json"):
        """
        Initialize Synopsis Manager.

        Args:
            storage_path: Path to JSON file storing synopses
        """
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._synopses: Dict[str, Dict] = self._load_synopses()
        logger.debug(f"SynopsisManager initialized with {len(self._synopses)} synopses")

    def _load_synopses(self) -> Dict[str, Dict]:
        """Load synopses from disk."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading synopses: {e}")
                return {}
        return {}

    def _save_synopses(self) -> None:
        """Save synopses to disk."""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(self._synopses, f, indent=2)
            logger.debug(f"Saved {len(self._synopses)} synopses to {self.storage_path}")
        except Exception as e:
            logger.error(f"Error saving synopses: {e}")

    def generate_synopsis(
        self,
        offering_id: str,
        offering_name: str,
        nodes: List[BaseNode],
        llm: LLM,
        file_list: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Generate synopsis for an offering from its document nodes.

        Args:
            offering_id: Unique offering identifier
            offering_name: Name of the offering
            nodes: Document nodes for this offering
            llm: Language model for synopsis generation
            file_list: List of files associated with this offering

        Returns:
            Generated synopsis text or None if generation fails
        """
        if not nodes:
            logger.warning(f"No nodes provided for offering {offering_name}")
            return None

        logger.info(f"Generating synopsis for offering: {offering_name} ({len(nodes)} nodes)")

        # Combine all node content (limit to prevent token overflow)
        all_content = "\n\n".join([node.get_content() for node in nodes])
        # Limit to ~8000 chars to keep LLM context manageable
        content_text = all_content[:8000]

        synopsis_prompt = f"""You are a solutions architect creating a comprehensive synopsis of a technology offering.

Offering Name: {offering_name}

Complete Documentation:
{content_text}

Your Task:
Create a comprehensive synopsis (300-500 words) that captures:

1. **Core Value Proposition**: What is this offering and what fundamental problem does it solve?
2. **Key Capabilities**: What are the primary technical capabilities and features?
3. **Target Use Cases**: What types of business problems or scenarios is this designed for?
4. **Industry Applications**: Which industries or domains benefit most from this?
5. **Technical Strengths**: What makes this offering unique or particularly effective?
6. **Integration & Deployment**: How does it fit into existing technology ecosystems?

Write a clear, comprehensive summary that would help someone understand if this offering is relevant to a specific customer problem. Focus on WHAT it does and WHY it's valuable, not just technical specifications.

Synopsis:"""

        try:
            response = llm.complete(synopsis_prompt)
            synopsis = response.text.strip()

            # Store synopsis
            self._synopses[offering_id] = {
                "offering_name": offering_name,
                "synopsis": synopsis,
                "created_at": datetime.now().isoformat(),
                "file_list": file_list or [],
                "node_count": len(nodes)
            }

            # Save to disk
            self._save_synopses()

            logger.info(f"Synopsis generated and stored for {offering_name}: {len(synopsis)} chars")
            return synopsis

        except Exception as e:
            logger.error(f"Error generating synopsis for {offering_name}: {e}")
            return None

    def get_synopsis(self, offering_id: str) -> Optional[Dict]:
        """
        Get stored synopsis for an offering.

        Args:
            offering_id: Unique offering identifier

        Returns:
            Synopsis data dictionary or None if not found
        """
        return self._synopses.get(offering_id)

    def get_all_synopses(self) -> Dict[str, Dict]:
        """Get all stored synopses."""
        return self._synopses.copy()

    def update_synopsis(
        self,
        offering_id: str,
        offering_name: str,
        nodes: List[BaseNode],
        llm: LLM,
        file_list: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Update synopsis for an existing offering.

        This is called when new documents are added to an existing offering.

        Args:
            offering_id: Unique offering identifier
            offering_name: Name of the offering
            nodes: All document nodes for this offering (including new ones)
            llm: Language model for synopsis generation
            file_list: Updated list of files

        Returns:
            Updated synopsis text or None if generation fails
        """
        logger.info(f"Updating synopsis for offering: {offering_name}")
        return self.generate_synopsis(offering_id, offering_name, nodes, llm, file_list)

    def delete_synopsis(self, offering_id: str) -> bool:
        """
        Delete synopsis for an offering.

        Args:
            offering_id: Unique offering identifier

        Returns:
            True if deleted, False if not found
        """
        if offering_id in self._synopses:
            del self._synopses[offering_id]
            self._save_synopses()
            logger.info(f"Deleted synopsis for offering ID: {offering_id}")
            return True
        return False

    def list_offerings(self) -> List[str]:
        """Get list of offering names with synopses."""
        return [data["offering_name"] for data in self._synopses.values()]

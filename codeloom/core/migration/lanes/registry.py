"""Migration lane registry.

Simple dict-based registry.  All lanes are registered at import time
via ``lanes/__init__.py``.  No plugin discovery, no entry points --
this is a monolith and all lanes ship together.
"""

import logging
from typing import Any, Dict, List, Optional

from .base import MigrationLane

logger = logging.getLogger(__name__)


class LaneRegistry:
    """Registry for migration lanes.

    Class-level store so the migration engine can call
    ``LaneRegistry.detect_lane(...)`` without holding an instance.
    """

    _lanes: Dict[str, MigrationLane] = {}

    @classmethod
    def register(cls, lane: MigrationLane) -> None:
        """Register a migration lane instance."""
        cls._lanes[lane.lane_id] = lane
        logger.info(
            "Registered migration lane: %s (%s)",
            lane.lane_id,
            lane.display_name,
        )

    @classmethod
    def detect_lane(
        cls,
        source_framework: str,
        target_stack: Dict[str, Any],
    ) -> Optional[MigrationLane]:
        """Auto-detect the best lane for a source/target combination.

        Iterates all registered lanes, scores applicability, and returns
        the highest-scoring lane above 0.0 -- or ``None``.
        """
        best_lane: Optional[MigrationLane] = None
        best_score = 0.0

        for lane in cls._lanes.values():
            if lane.deprecated:
                continue
            try:
                score = lane.detect_applicability(source_framework, target_stack)
                if score > best_score:
                    best_score = score
                    best_lane = lane
            except Exception:
                logger.warning(
                    "Lane %s applicability check failed", lane.lane_id, exc_info=True
                )

        if best_lane is not None:
            logger.info(
                "Detected migration lane: %s (score=%.2f)",
                best_lane.lane_id,
                best_score,
            )

        return best_lane

    @classmethod
    def get_lane(cls, lane_id: str) -> Optional[MigrationLane]:
        """Get a lane by ID.  Returns ``None`` if not found."""
        return cls._lanes.get(lane_id)

    @classmethod
    def list_lanes(cls) -> List[Dict[str, Any]]:
        """List all registered lanes with metadata."""
        return [
            {
                "lane_id": lane.lane_id,
                "display_name": lane.display_name,
                "source_frameworks": lane.source_frameworks,
                "target_frameworks": lane.target_frameworks,
                "version": lane.version,
                "deprecated": lane.deprecated,
                "min_source_version": lane.min_source_version,
                "max_source_version": lane.max_source_version,
            }
            for lane in cls._lanes.values()
        ]

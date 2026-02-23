"""Migration lanes -- framework-specific migration intelligence.

All built-in lanes are registered on import.  The :class:`LaneRegistry`
is the single entry point for the migration engine to discover and use
lanes.
"""

from .base import (
    GateDefinition,
    GateResult,
    MigrationLane,
    TransformResult,
    TransformRule,
)
from .registry import LaneRegistry

# ── Register built-in lanes ──────────────────────────────────────────

from .struts_to_springboot import StrutsToSpringBootLane
from .storedproc_to_orm import StoredProcToOrmLane

LaneRegistry.register(StrutsToSpringBootLane())
LaneRegistry.register(StoredProcToOrmLane())

__all__ = [
    "GateDefinition",
    "GateResult",
    "LaneRegistry",
    "MigrationLane",
    "StoredProcToOrmLane",
    "StrutsToSpringBootLane",
    "TransformResult",
    "TransformRule",
]

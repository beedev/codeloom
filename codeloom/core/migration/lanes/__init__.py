"""Migration lanes -- framework-specific migration intelligence.

All built-in lanes are registered on import.  The :class:`LaneRegistry`
is the single entry point for the migration engine to discover and use
lanes.
"""

from .base import (
    CONFIDENCE_HIGH,
    CONFIDENCE_STANDARD,
    GateCategory,
    GateDefinition,
    GateResult,
    MigrationLane,
    TransformResult,
    TransformRule,
    aggregate_confidence,
    confidence_tier,
)
from .registry import LaneRegistry

# ── Register built-in lanes ──────────────────────────────────────────

from .struts_to_springboot import StrutsToSpringBootLane
from .storedproc_to_orm import StoredProcToOrmLane
from .vbnet_to_dotnetcore import VbNetToDotNetCoreLane

LaneRegistry.register(StrutsToSpringBootLane())
LaneRegistry.register(StoredProcToOrmLane())
LaneRegistry.register(VbNetToDotNetCoreLane())

__all__ = [
    "CONFIDENCE_HIGH",
    "CONFIDENCE_STANDARD",
    "GateCategory",
    "GateDefinition",
    "GateResult",
    "LaneRegistry",
    "MigrationLane",
    "StoredProcToOrmLane",
    "StrutsToSpringBootLane",
    "TransformResult",
    "TransformRule",
    "VbNetToDotNetCoreLane",
    "aggregate_confidence",
    "confidence_tier",
]

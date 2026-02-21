"""Data contracts for the Deep Understanding Engine.

All structured types used across the understanding subsystem.
Kept as dataclasses (not ORM models) for transport between layers.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID


class EntryPointType(Enum):
    """Classification of detected entry points."""
    HTTP_ENDPOINT = "http_endpoint"
    MESSAGE_HANDLER = "message_handler"
    SCHEDULED_TASK = "scheduled_task"
    CLI_COMMAND = "cli_command"
    EVENT_LISTENER = "event_listener"
    STARTUP_HOOK = "startup_hook"
    PUBLIC_API = "public_api"
    UNKNOWN = "unknown"


class AnalysisTier(Enum):
    """Token budget tier for chain analysis."""
    TIER_1 = "tier_1"   # Full source, <=100K tokens
    TIER_2 = "tier_2"   # Depth-prioritized truncation, <=200K tokens
    TIER_3 = "tier_3"   # Summarization fallback, >200K tokens


@dataclass
class EvidenceRef:
    """Links an extracted fact back to source code.

    Every business rule, data entity, or integration discovered by the
    LLM must carry at least one EvidenceRef so reviewers can verify.
    """
    unit_id: str
    qualified_name: str
    file_path: str
    start_line: int
    end_line: int
    snippet: Optional[str] = None


@dataclass
class EntryPoint:
    """A detected entry point into the application."""
    unit_id: str
    name: str
    qualified_name: str
    file_path: str
    entry_type: EntryPointType
    language: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    detected_by: str = "heuristic"  # "heuristic" | "annotation" | "both"


@dataclass
class CallTreeNode:
    """A node in the traced call tree.

    Recursive structure: each node holds its children.
    The root node is the entry point itself.
    """
    unit_id: str
    name: str
    qualified_name: str
    unit_type: str
    language: str
    file_path: str
    start_line: int
    end_line: int
    source: Optional[str]
    depth: int
    edge_type: str = "calls"
    children: List["CallTreeNode"] = field(default_factory=list)
    token_count: int = 0


@dataclass
class DeepContextBundle:
    """Complete analysis output for one entry point.

    The full bundle shape is persisted in deep_analyses.result_json
    for forward-compatible reprocessing and auditability.
    """
    entry_point: EntryPoint
    tier: AnalysisTier
    total_units: int
    total_tokens: int

    # Extracted knowledge -- each item carries EvidenceRefs
    business_rules: List[Dict[str, Any]] = field(default_factory=list)
    data_entities: List[Dict[str, Any]] = field(default_factory=list)
    integrations: List[Dict[str, Any]] = field(default_factory=list)
    side_effects: List[Dict[str, Any]] = field(default_factory=list)
    cross_cutting_concerns: List[str] = field(default_factory=list)

    # Narrative summary (for chat injection)
    narrative: str = ""
    confidence: float = 0.0
    coverage: float = 0.0
    chain_truncated: bool = False

    # Schema version for forward compatibility
    schema_version: int = 1
    prompt_version: str = "v1.0"
    analyzed_at: Optional[str] = None


@dataclass
class AnalysisError:
    """Structured error from analysis pipeline."""
    phase: str          # "detection" | "tracing" | "analysis" | "storage"
    message: str
    unit_id: Optional[str] = None
    recoverable: bool = True

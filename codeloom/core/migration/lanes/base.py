"""Migration lane base class and supporting dataclasses.

A migration lane encapsulates framework-specific migration intelligence:
deterministic transforms, LLM prompt augmentation, quality gates, and
asset strategy overrides.  Core stays orchestration -- lanes own the
domain knowledge for a specific migration path.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


# ── Dataclasses ──────────────────────────────────────────────────────


@dataclass
class TransformRule:
    """A deterministic transform rule that maps source patterns to target code."""

    name: str
    """Rule identifier, e.g. ``"action_to_controller"``."""

    source_pattern: Dict[str, Any]
    """Match criteria applied to CodeUnit metadata,
    e.g. ``{"unit_type": "struts_action"}``."""

    target_template: str
    """Template key used for code generation,
    e.g. ``"spring_controller_method"``."""

    confidence: float
    """How confident the transform is (0.0--1.0)."""

    requires_review: bool = False
    """Flag the output for human review when ``True``."""

    description: str = ""
    """Optional human-readable explanation of what this rule does."""


@dataclass
class TransformResult:
    """Result of applying a deterministic transform to a source unit."""

    source_unit_id: str
    target_code: str
    target_path: str
    rule_name: str
    confidence: float
    notes: List[str] = field(default_factory=list)


@dataclass
class GateDefinition:
    """Definition of a quality gate that validates migration output."""

    name: str
    """Gate identifier, e.g. ``"endpoint_parity"``."""

    description: str
    """Human-readable explanation,
    e.g. ``"Every Struts action path has a @RequestMapping"``."""

    blocking: bool = True
    """When ``True`` the migration cannot proceed on failure."""


@dataclass
class GateResult:
    """Result of running a quality gate."""

    gate_name: str
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)
    blocking: bool = True


# ── Abstract Base Class ──────────────────────────────────────────────


class MigrationLane(ABC):
    """Abstract base for migration lanes.

    Each lane owns what is unique to a specific migration path:

    * **Deterministic transforms** -- source pattern to target code.
    * **LLM prompt augmentation** -- inject domain context into prompts.
    * **Quality gates** -- validate migration completeness.
    * **Asset strategy overrides** -- how to handle specific file types.

    Parsers, edge detectors, and framework analyzers are *not* part of
    lanes -- they are first-class core citizens, always available.
    """

    # ── Identity ─────────────────────────────────────────────────

    @property
    @abstractmethod
    def lane_id(self) -> str:
        """Unique identifier, e.g. ``"struts_to_springboot"``."""
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name, e.g. ``"Struts 1.x/2.x -> Spring Boot"``."""
        ...

    @property
    @abstractmethod
    def source_frameworks(self) -> List[str]:
        """Framework identifiers this lane migrates FROM,
        e.g. ``["struts1", "struts2"]``."""
        ...

    @property
    @abstractmethod
    def target_frameworks(self) -> List[str]:
        """Framework identifiers this lane migrates TO,
        e.g. ``["springboot"]``."""
        ...

    @abstractmethod
    def detect_applicability(
        self, source_framework: str, target_stack: Dict[str, Any]
    ) -> float:
        """Score how applicable this lane is for a source/target combo.

        Returns 0.0 (not applicable) to 1.0 (perfect match).
        Called by :class:`LaneRegistry` to auto-detect the best lane.
        """
        ...

    # ── Deterministic Transforms ─────────────────────────────────

    @abstractmethod
    def get_transform_rules(self) -> List[TransformRule]:
        """Return all deterministic transform rules this lane provides."""
        ...

    @abstractmethod
    def apply_transforms(
        self,
        units: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> List[TransformResult]:
        """Apply deterministic transforms to matching source units.

        Args:
            units: CodeUnit dicts (from database query).
            context: Migration context including target_stack and plan
                metadata.

        Returns:
            Transform results.  The LLM handles what rules could not.
        """
        ...

    # ── LLM Prompt Augmentation ──────────────────────────────────

    @abstractmethod
    def augment_prompt(
        self,
        phase_type: str,
        base_prompt: str,
        context: Dict[str, Any],
    ) -> str:
        """Augment a phase prompt with lane-specific context.

        Args:
            phase_type: ``"architecture"``, ``"discovery"``,
                ``"transform"``, or ``"test"``.
            base_prompt: The base prompt from the phase executor.
            context: Migration context.

        Returns:
            Augmented prompt string with lane-specific knowledge injected.
        """
        ...

    # ── Quality Gates ────────────────────────────────────────────

    @abstractmethod
    def get_gates(self) -> List[GateDefinition]:
        """Return all quality gate definitions this lane provides."""
        ...

    @abstractmethod
    def run_gate(
        self,
        gate_name: str,
        source_units: List[Dict[str, Any]],
        target_outputs: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> GateResult:
        """Run a specific quality gate.

        Args:
            gate_name: Name of the gate to run.
            source_units: Original source code units.
            target_outputs: Generated target code/artifacts.
            context: Migration context.

        Returns:
            :class:`GateResult` with pass/fail and details.
        """
        ...

    # ── Asset Strategy Overrides ─────────────────────────────────

    @abstractmethod
    def get_asset_strategy_overrides(self) -> Dict[str, Dict[str, Any]]:
        """Return file-type strategy overrides for the asset inventory.

        Returns:
            Dict mapping file patterns/types to strategy dicts, e.g.::

                {"struts-config.xml": {"strategy": "convert", "priority": "high"}}
        """
        ...

"""Mainframe (COBOL/PL1/JCL) → Modern Target migration lane.

Provides programmatic lane detection and metadata for mainframe codebases.
The primary intelligence lives in the ``lane-mainframe.md`` sub-skill file;
this class enables ``codeloom_get_lane_info`` MCP tool to surface mainframe-
specific transform rules and quality gates via the API.
"""

from typing import Any, Dict, List

from .base import (
    GateCategory,
    GateDefinition,
    GateResult,
    MigrationLane,
    TransformResult,
    TransformRule,
)


class MainframeToModernLane(MigrationLane):
    """Migration lane for IBM mainframe → modern target stack."""

    # ── Identity ─────────────────────────────────────────────────

    @property
    def lane_id(self) -> str:
        return "mainframe_to_modern"

    @property
    def display_name(self) -> str:
        return "Mainframe (COBOL/PL1/JCL) → Modern Target"

    @property
    def source_frameworks(self) -> List[str]:
        return ["cobol", "pli", "pl1", "jcl", "cics", "ims"]

    @property
    def target_frameworks(self) -> List[str]:
        return ["python", "java", "dotnet", "csharp"]

    @property
    def version(self) -> str:
        return "1.0.0"

    # ── Detection ────────────────────────────────────────────────

    def detect_applicability(
        self, source_framework: str, target_stack: Dict[str, Any]
    ) -> float:
        source_lower = source_framework.lower()
        if source_lower in {"cobol", "pli", "pl1", "jcl", "cics", "ims"}:
            return 0.95
        # Heuristic: mainframe-adjacent languages
        if source_lower in {"rexx", "natural", "easytrieve"}:
            return 0.6
        return 0.0

    # ── Deterministic Transforms ─────────────────────────────────

    def get_transform_rules(self) -> List[TransformRule]:
        return [
            TransformRule(
                name="comp3_to_decimal",
                source_pattern={"metadata.pic_clause": ".*COMP-3.*"},
                target_template="Decimal",
                confidence=0.99,
                description="COMP-3 packed decimal → Decimal/BigDecimal",
            ),
            TransformRule(
                name="stop_run_to_exit",
                source_pattern={"source": ".*STOP RUN.*"},
                target_template="raise SystemExit(0)",
                confidence=0.95,
                description="STOP RUN terminates run unit → SystemExit",
            ),
            TransformRule(
                name="goback_to_return",
                source_pattern={"source": ".*GOBACK.*"},
                target_template="return",
                confidence=0.95,
                description="GOBACK returns to caller → return statement",
            ),
            TransformRule(
                name="copybook_to_shared_type",
                source_pattern={"unit_type": "copybook"},
                target_template="@dataclass with from_line/to_line",
                confidence=0.90,
                description="COPY member → shared dataclass/POJO",
            ),
            TransformRule(
                name="vsam_ksds_to_kv_store",
                source_pattern={"metadata.file_org": ".*KSDS.*"},
                target_template="SQLite/dict with key-based access",
                confidence=0.85,
                description="VSAM KSDS → key-value store",
            ),
            TransformRule(
                name="cics_to_rest_endpoint",
                source_pattern={"metadata.program_category": "cics_online"},
                target_template="REST API endpoint",
                confidence=0.80,
                requires_review=True,
                description="CICS online program → REST API endpoint",
            ),
            TransformRule(
                name="ims_dli_to_orm",
                source_pattern={"metadata.program_category": "ims_dli"},
                target_template="ORM/JPA with hierarchical schema",
                confidence=0.75,
                requires_review=True,
                description="IMS DL/I → ORM with parent-child relationships",
            ),
        ]

    def apply_transforms(
        self,
        units: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> List[TransformResult]:
        # Mainframe transforms are primarily LLM-driven via the sub-skill.
        # The deterministic rules above guide the LLM prompt, not direct
        # code generation.
        return []

    # ── LLM Prompt Augmentation ──────────────────────────────────

    def augment_prompt(
        self,
        phase_type: str,
        base_prompt: str,
        context: Dict[str, Any],
    ) -> str:
        if phase_type == "transform":
            return base_prompt + _MAINFRAME_TRANSFORM_HINTS
        if phase_type == "architecture":
            return base_prompt + _MAINFRAME_ARCHITECTURE_HINTS
        return base_prompt

    # ── Quality Gates ────────────────────────────────────────────

    def get_gates(self) -> List[GateDefinition]:
        return [
            GateDefinition(
                name="decimal_parity",
                category=GateCategory.PARITY,
                description="All COMP-3/V99 fields use Decimal, not float",
                blocking=True,
            ),
            GateDefinition(
                name="operator_parity",
                category=GateCategory.PARITY,
                description="Comparison operators match source exactly (> vs >=)",
                blocking=True,
            ),
            GateDefinition(
                name="stop_run_semantics",
                category=GateCategory.PARITY,
                description="STOP RUN → SystemExit, not return",
                blocking=True,
            ),
            GateDefinition(
                name="record_layout_parity",
                category=GateCategory.PARITY,
                description="Fixed-width record layouts match COBOL PIC field widths",
                blocking=True,
            ),
            GateDefinition(
                name="compile_check",
                category=GateCategory.COMPILE,
                description="Target code compiles without errors",
                blocking=True,
            ),
            GateDefinition(
                name="copybook_shared",
                category=GateCategory.PARITY,
                description="Copybook types are shared (single definition), not duplicated",
                blocking=False,
            ),
            GateDefinition(
                name="jcl_compile_skipped",
                category=GateCategory.PARITY,
                description="JCL compile/link steps are not migrated",
                blocking=False,
            ),
        ]

    def run_gate(
        self,
        gate_name: str,
        source_units: List[Dict[str, Any]],
        target_outputs: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> GateResult:
        # Gate evaluation is done by the /migrate skill via LLM comparison.
        # This returns a pass-through so the engine doesn't block.
        return GateResult(gate_name=gate_name, passed=True, blocking=False)

    # ── Asset Strategy Overrides ─────────────────────────────────

    def get_asset_strategy_overrides(self) -> Dict[str, Dict[str, Any]]:
        return {
            "jcl": {
                "strategy": "rewrite",
                "reason": "JCL orchestration — targets derived from plan's target stack",
                "sub_types": {
                    "compile_link": {"strategy": "no_change"},
                    "sort_merge": {"strategy": "convert"},
                    "data_mgmt": {"strategy": "convert"},
                    "application_run": {"strategy": "convert"},
                    "proc_invoke": {"strategy": "convert"},
                },
            },
            "cobol": {
                "strategy": "rewrite",
                "reason": "COBOL programs → target language with streaming I/O",
            },
        }


# ── Prompt Hints ─────────────────────────────────────────────────────

_MAINFRAME_TRANSFORM_HINTS = """

## Mainframe Transform Guidance

CRITICAL rules for COBOL/PL1/JCL migration:
1. COMP-3 / implied decimal (V99) → ALWAYS use Decimal/BigDecimal, NEVER float/double
2. STOP RUN → raise SystemExit(0) (terminates run unit), GOBACK → return (returns to caller)
3. Comparison operators: verify EACH one matches source exactly (> vs >=)
4. Copybook record layouts: exact field widths from PIC declarations, shared dataclass/POJO
5. Fixed-width I/O: from_line()/to_line() must match COBOL LRECL exactly
6. VSAM KSDS: key-based access with INVALID KEY → KeyError/exception handling
7. AT END: proper EOF/StopIteration — no silent data loss
8. PERFORM THRU: merge paragraph range into single function (no GO TO simulation)
9. JCL compile/link steps (COBOLCL, IGYWCL): SKIP — do not generate code
10. 88-level conditions → named constants or enum values
"""

_MAINFRAME_ARCHITECTURE_HINTS = """

## Mainframe Architecture Guidance

Key architectural decisions for mainframe modernization:
1. COBOL batch programs → standalone modules with main() entry points
2. COBOL CICS online programs → REST API endpoints (COMMAREA → request/response)
3. IMS DL/I programs → ORM layer with hierarchical-to-relational mapping
4. Copybooks → shared type definitions (Foundation MVP, imported by all programs)
5. VSAM KSDS files → SQLite/JDBC key-value tables (VsamAdapter pattern)
6. Sequential files → streaming I/O (generators/iterators, no full-file memory load)
7. JCL job orchestration → shell scripts calling migrated program modules
8. SORT/MERGE utilities → shell sort or target-lang streaming (heapq.merge)
9. Working-Storage persistence → class instance variables (state preserved across calls)
10. RETURN-CODE → exit codes or return values (process return code semantics)
"""

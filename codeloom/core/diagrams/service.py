"""DiagramService — orchestrator for per-MVP diagram generation and caching.

Structural diagrams (class, package, component) are generated fresh from ASG data.

Behavioral diagrams use a two-tier strategy:
  - sequence, activity, usecase: DETERMINISTIC from ChainTracer call tree data
    (no LLM calls — every arrow corresponds to a real code path)
  - deployment: LLM-assisted but grounded with detected infrastructure from ASG

Falls back to LLM-based behavioral generation if call tree data is unavailable.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from . import queries, structural
from .behavioral import BEHAVIORAL_TYPES, generate_behavioral_diagram
from .behavioral_from_asg import (
    generate_activity_diagram,
    generate_sequence_diagram,
    generate_usecase_diagram,
)
from .renderer import render_puml_to_svg

logger = logging.getLogger(__name__)

STRUCTURAL_TYPES = {"class", "package", "component"}
# Sequence, activity, usecase are now deterministic from ASG call tree
ASG_BEHAVIORAL_TYPES = {"sequence", "activity", "usecase"}
# Deployment still uses LLM (with improved grounding)
LLM_BEHAVIORAL_TYPES = {"deployment"}
ALL_DIAGRAM_TYPES = STRUCTURAL_TYPES | BEHAVIORAL_TYPES

_DIAGRAM_METADATA = {
    "class": {"category": "structural", "label": "Class Diagram"},
    "package": {"category": "structural", "label": "Package Diagram"},
    "component": {"category": "structural", "label": "Component Diagram"},
    "sequence": {"category": "behavioral", "label": "Sequence Diagram"},
    "usecase": {"category": "behavioral", "label": "Use Case Diagram"},
    "activity": {"category": "behavioral", "label": "Activity Diagram"},
    "deployment": {"category": "behavioral", "label": "Deployment Diagram"},
}


class DiagramService:
    """Generates and caches UML diagrams scoped to a Functional MVP."""

    def __init__(self, db, pipeline=None):
        """Initialize DiagramService.

        Args:
            db: DatabaseManager instance
            pipeline: LocalRAGPipeline (needed for behavioral diagram LLM access)
        """
        self._db = db
        self._pipeline = pipeline

    def get_diagram(
        self,
        plan_id: str,
        mvp_id: int,
        diagram_type: str,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """Get or generate a diagram for an MVP.

        Args:
            plan_id: Migration plan UUID
            mvp_id: Functional MVP ID
            diagram_type: One of the 7 diagram types
            force_refresh: Force regeneration even if cached

        Returns:
            {diagram_type, category, puml, svg, title, cached, generated_at}
        """
        if diagram_type not in ALL_DIAGRAM_TYPES:
            raise ValueError(
                f"Unknown diagram type '{diagram_type}'. "
                f"Must be one of: {', '.join(sorted(ALL_DIAGRAM_TYPES))}"
            )

        mvp = self._get_mvp(plan_id, mvp_id)
        project_id = self._get_project_id(plan_id)
        meta = _DIAGRAM_METADATA[diagram_type]

        # Structural: always generated fresh
        if diagram_type in STRUCTURAL_TYPES:
            puml = self._generate_structural(diagram_type, mvp, project_id)
            try:
                svg = render_puml_to_svg(puml)
            except RuntimeError as e:
                logger.warning("PlantUML render failed for %s: %s", diagram_type, e)
                svg = self._error_svg(str(e))
            title = f"{meta['label']}: {mvp.get('name', 'MVP')}"
            return {
                "diagram_type": diagram_type,
                "category": meta["category"],
                "puml": puml,
                "svg": svg,
                "title": title,
                "cached": False,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

        # Behavioral: check cache first
        if not force_refresh:
            cached = self._get_cached_diagram(plan_id, mvp_id, diagram_type)
            if cached:
                return {
                    "diagram_type": diagram_type,
                    "category": meta["category"],
                    "puml": cached["puml"],
                    "svg": cached["svg"],
                    "title": cached.get("title", meta["label"]),
                    "cached": True,
                    "generated_at": cached.get("generated_at"),
                }

        # ASG-based behavioral diagrams (deterministic, no LLM)
        if diagram_type in ASG_BEHAVIORAL_TYPES:
            result = self._generate_behavioral_from_asg(diagram_type, mvp, project_id)
        else:
            # Deployment: LLM-assisted with detected infrastructure grounding
            mvp_context = self._build_mvp_context(mvp, project_id, plan_id)
            result = generate_behavioral_diagram(diagram_type, mvp_context, self._db, project_id)

        # Render SVG
        try:
            svg = render_puml_to_svg(result["puml"])
        except RuntimeError as e:
            logger.warning("PlantUML render failed for behavioral %s: %s", diagram_type, e)
            svg = self._error_svg(str(e))

        # Cache result
        now = datetime.now(timezone.utc).isoformat()
        cache_entry = {
            "puml": result["puml"],
            "svg": svg,
            "title": result["title"],
            "generated_at": now,
        }
        self._cache_diagram(plan_id, mvp_id, diagram_type, cache_entry)

        # Write diagram files to disk (fire-and-forget)
        try:
            self._write_diagram_to_disk(plan_id, mvp_id, diagram_type, result["puml"], svg)
        except Exception as e:
            logger.warning("Failed to write diagram to disk: %s", e)

        return {
            "diagram_type": diagram_type,
            "category": meta["category"],
            "puml": result["puml"],
            "svg": svg,
            "title": result["title"],
            "cached": False,
            "generated_at": now,
        }

    def list_available(self, plan_id: str, mvp_id: int) -> List[Dict[str, Any]]:
        """List availability and cache status for all 7 diagram types.

        Returns:
            List of {diagram_type, category, label, cached, generated_at}
        """
        self._get_mvp(plan_id, mvp_id)  # Validate existence
        cached_diagrams = self._get_all_cached(plan_id, mvp_id)

        result = []
        for dtype, meta in _DIAGRAM_METADATA.items():
            entry = {
                "diagram_type": dtype,
                "category": meta["category"],
                "label": meta["label"],
                "cached": False,
                "generated_at": None,
            }
            if dtype in cached_diagrams:
                entry["cached"] = True
                entry["generated_at"] = cached_diagrams[dtype].get("generated_at")
            result.append(entry)

        return result

    # ── Internal helpers ──────────────────────────────────────────────

    def _generate_behavioral_from_asg(
        self,
        diagram_type: str,
        mvp: Dict,
        project_id: str,
    ) -> Dict[str, str]:
        """Generate behavioral diagrams deterministically from ASG call tree data.

        Uses ChainTracer to trace real call paths. Falls back to LLM-based
        generation if call tree data is empty (e.g., analysis not yet run).
        """
        from ..understanding.chain_tracer import ChainTracer

        unit_ids = mvp.get("unit_ids", [])
        mvp_name = mvp.get("name", "Unnamed MVP")

        if not unit_ids:
            logger.warning("No unit_ids for MVP %s — cannot generate ASG behavioral diagram", mvp_name)
            return {
                "puml": f"@startuml\nnote \"No unit data for {mvp_name}\" as N\n@enduml",
                "title": f"{diagram_type.title()} Diagram: {mvp_name}",
                "description": "No data available",
            }

        tracer = ChainTracer(self._db)

        if diagram_type == "usecase":
            # Use case only needs entry point detection, not full tree tracing
            try:
                entry_points = tracer.detect_entry_points(project_id)
                result = generate_usecase_diagram(entry_points, mvp_name, mvp_unit_ids=unit_ids)
                if "No call tree data" not in result.get("description", ""):
                    return result
            except Exception as e:
                logger.warning("UseCase ASG generation failed: %s — falling back to LLM", e)

        elif diagram_type in ("sequence", "activity"):
            # Trace call trees from entry points within the MVP
            try:
                entry_points = tracer.detect_entry_points(project_id)
                mvp_set = set(unit_ids)
                mvp_entries = [ep for ep in entry_points if ep.unit_id in mvp_set]

                # If no explicit entries in MVP, try using MVP units as trace roots
                # Do NOT fall back to project-wide entry points — they're unrelated
                if not mvp_entries and unit_ids:
                    from ..understanding.models import EntryPoint, EntryPointType
                    for uid in unit_ids[:5]:
                        mvp_entries.append(EntryPoint(
                            unit_id=uid,
                            name="",
                            qualified_name="",
                            entry_type=EntryPointType.UNKNOWN,
                            evidence=[],
                        ))

                call_trees = []
                for ep in mvp_entries[:5]:  # Cap at 5 entry points for readability
                    tree = tracer.trace_call_tree(project_id, ep.unit_id, max_depth=6)
                    if tree and tree.children:
                        call_trees.append(tree)

                if call_trees:
                    if diagram_type == "sequence":
                        result = generate_sequence_diagram(call_trees, mvp_name)
                    else:
                        result = generate_activity_diagram(call_trees, mvp_name)

                    if "No call tree data" not in result.get("description", ""):
                        return result
                    logger.info("Call trees empty for %s — falling back", mvp_name)
                else:
                    logger.info("No call trees found for %s — falling back to LLM", mvp_name)

            except Exception as e:
                logger.warning("%s ASG generation failed: %s — falling back to LLM", diagram_type, e)

        # Fallback: use LLM-based generation (only reached when ASG tracing raises an exception,
        # meaning there IS code with behavioral potential but tracing failed)
        logger.info("Falling back to LLM-based %s diagram for %s", diagram_type, mvp_name)
        mvp_context = {**mvp, "target_stack": None, "detected_infra": ""}
        return generate_behavioral_diagram(diagram_type, mvp_context, self._db, project_id)

    def _generate_structural(self, diagram_type: str, mvp: Dict, project_id: str) -> str:
        """Generate a structural diagram's PlantUML from ASG data."""
        unit_ids = mvp.get("unit_ids", [])
        file_ids = mvp.get("file_ids", [])

        if diagram_type == "class":
            data = queries.get_mvp_class_data(self._db, project_id, unit_ids)
            return structural.generate_class_diagram(data)
        elif diagram_type == "package":
            data = queries.get_mvp_package_data(self._db, project_id, file_ids, unit_ids)
            return structural.generate_package_diagram(data)
        elif diagram_type == "component":
            data = queries.get_mvp_component_data(self._db, project_id, unit_ids, file_ids)
            return structural.generate_component_diagram(data)
        else:
            raise ValueError(f"Unknown structural type: {diagram_type}")

    def _get_mvp(self, plan_id: str, mvp_id: int) -> Dict:
        """Load MVP from database."""
        from ..db.models import FunctionalMVP
        pid = UUID(plan_id) if isinstance(plan_id, str) else plan_id

        with self._db.get_session() as session:
            mvp = session.query(FunctionalMVP).filter(
                FunctionalMVP.mvp_id == mvp_id,
                FunctionalMVP.plan_id == pid,
            ).first()
            if not mvp:
                raise ValueError(f"MVP {mvp_id} not found in plan {plan_id}")
            return {
                "mvp_id": mvp.mvp_id,
                "name": mvp.name,
                "description": mvp.description,
                "unit_ids": mvp.unit_ids or [],
                "file_ids": mvp.file_ids or [],
                "metrics": mvp.metrics or {},
                "analysis_output": mvp.analysis_output,
                "diagrams": mvp.diagrams,
            }

    def _get_project_id(self, plan_id: str) -> str:
        """Get the source project ID from a migration plan."""
        from ..db.models import MigrationPlan
        pid = UUID(plan_id) if isinstance(plan_id, str) else plan_id

        with self._db.get_session() as session:
            plan = session.query(MigrationPlan).filter(
                MigrationPlan.plan_id == pid,
            ).first()
            if not plan:
                raise ValueError(f"Plan {plan_id} not found")
            return str(plan.source_project_id)

    def _build_mvp_context(self, mvp: Dict, project_id: str, plan_id: Optional[str] = None) -> Dict:
        """Build full MVP context for behavioral diagram generation.

        Includes detected infrastructure from ASG for grounding deployment diagrams.
        """
        target_stack = None
        if plan_id:
            from ..db.models import MigrationPlan
            pid = UUID(plan_id) if isinstance(plan_id, str) else plan_id
            with self._db.get_session() as session:
                plan = session.query(MigrationPlan).filter(
                    MigrationPlan.plan_id == pid,
                ).first()
                target_stack = plan.target_stack if plan else None

        # Detect actual infrastructure from ASG imports
        unit_ids = mvp.get("unit_ids", [])
        detected_infra = queries.get_detected_infrastructure(self._db, project_id, unit_ids)

        return {
            **mvp,
            "target_stack": target_stack,
            "detected_infra": detected_infra,
        }

    def _get_cached_diagram(self, plan_id: str, mvp_id: int, diagram_type: str) -> Optional[Dict]:
        """Get a cached diagram from the MVP's diagrams JSONB."""
        from ..db.models import FunctionalMVP
        pid = UUID(plan_id) if isinstance(plan_id, str) else plan_id

        with self._db.get_session() as session:
            mvp = session.query(FunctionalMVP).filter(
                FunctionalMVP.mvp_id == mvp_id,
                FunctionalMVP.plan_id == pid,
            ).first()
            if not mvp or not mvp.diagrams:
                return None
            return mvp.diagrams.get(diagram_type)

    def _get_all_cached(self, plan_id: str, mvp_id: int) -> Dict:
        """Get all cached diagrams for an MVP."""
        from ..db.models import FunctionalMVP
        pid = UUID(plan_id) if isinstance(plan_id, str) else plan_id

        with self._db.get_session() as session:
            mvp = session.query(FunctionalMVP).filter(
                FunctionalMVP.mvp_id == mvp_id,
                FunctionalMVP.plan_id == pid,
            ).first()
            return mvp.diagrams if mvp and mvp.diagrams else {}

    def _cache_diagram(
        self, plan_id: str, mvp_id: int, diagram_type: str, entry: Dict,
    ) -> None:
        """Cache a diagram result on the MVP's diagrams JSONB."""
        from ..db.models import FunctionalMVP
        pid = UUID(plan_id) if isinstance(plan_id, str) else plan_id

        with self._db.get_session() as session:
            mvp = session.query(FunctionalMVP).filter(
                FunctionalMVP.mvp_id == mvp_id,
                FunctionalMVP.plan_id == pid,
            ).first()
            if not mvp:
                return

            diagrams = dict(mvp.diagrams) if mvp.diagrams else {}
            diagrams[diagram_type] = entry
            mvp.diagrams = diagrams
            session.commit()

    def _write_diagram_to_disk(
        self,
        plan_id: str,
        mvp_id: int,
        diagram_type: str,
        puml: str,
        svg: str,
    ) -> None:
        """Write diagram PlantUML and SVG to the MVP's disk folder."""
        from ..migration.engine import MigrationEngine

        # Resolve plan directory
        pid = UUID(plan_id) if isinstance(plan_id, str) else plan_id
        with self._db.get_session() as session:
            from ..db.models import MigrationPlan, Project
            plan = session.query(MigrationPlan).filter(
                MigrationPlan.plan_id == pid
            ).first()
            if not plan:
                return
            proj = session.query(Project).filter(
                Project.project_id == plan.source_project_id
            ).first() if plan.source_project_id else None
            project_name = proj.name if proj else None

        plan_dir = MigrationEngine._get_plan_dir(str(plan_id), project_name)
        diag_dir = os.path.join(plan_dir, "_plans", f"mvp-{mvp_id}", "diagrams")
        os.makedirs(diag_dir, exist_ok=True)

        if puml:
            with open(os.path.join(diag_dir, f"{diagram_type}.puml"), "w") as f:
                f.write(puml)
        if svg:
            with open(os.path.join(diag_dir, f"{diagram_type}.svg"), "w") as f:
                f.write(svg)

    @staticmethod
    def _error_svg(message: str) -> str:
        """Generate a minimal SVG showing an error message."""
        safe_msg = message[:200].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="100">'
            '<rect width="400" height="100" fill="#FFF3CD" stroke="#FFCA2C" rx="8"/>'
            f'<text x="20" y="35" fill="#664D03" font-size="13" font-family="sans-serif">'
            f'Diagram render error</text>'
            f'<text x="20" y="60" fill="#664D03" font-size="11" font-family="sans-serif">'
            f'{safe_msg}</text>'
            '</svg>'
        )

"""CodeLoom MCP Server implementation.

Exposes 16 tools covering project intelligence, MVP management, phase execution,
source code access, RAG search, ground truth validation, lane detection, and
full-autonomy migration (list_units, get_import_graph, save_plan, save_mvps,
complete_transform, start_transform).

All tool handlers are registered on the mcp.server.Server instance and communicate
via JSON-encoded TextContent responses.
"""

import json
import logging

from typing import Any, Dict, List, Optional
from uuid import UUID

from mcp.server import Server
from mcp.types import TextContent, Tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool input schemas
# ---------------------------------------------------------------------------

_TOOL_DEFINITIONS: List[Tool] = [
    Tool(
        name="codeloom_list_projects",
        description=(
            "List all CodeLoom projects. Returns project metadata including "
            "ID, name, creation date, and file count."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "page": {"type": "integer", "default": 1, "description": "Page number (1-based)"},
                "page_size": {"type": "integer", "default": 20, "description": "Results per page"},
            },
        },
    ),
    Tool(
        name="codeloom_get_project_intel",
        description=(
            "Get comprehensive intelligence for a project: AST stats, ASG edge counts, "
            "understanding engine status, and existing migration plan summaries."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string", "description": "Project UUID"},
            },
            "required": ["project_id"],
        },
    ),
    Tool(
        name="codeloom_list_mvps",
        description="List all MVPs for a migration plan with status, confidence, and cluster name.",
        inputSchema={
            "type": "object",
            "properties": {
                "plan_id": {"type": "string", "description": "Migration plan UUID"},
            },
            "required": ["plan_id"],
        },
    ),
    Tool(
        name="codeloom_get_mvp_context",
        description=(
            "Get rich context for an MVP: source unit summaries, lane info, "
            "deep analysis narratives, ground truth summary, and cross-boundary "
            "integration points (ASG edges crossing MVP boundary)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "plan_id": {"type": "string", "description": "Migration plan UUID"},
                "mvp_id": {"type": "integer", "description": "MVP ID"},
            },
            "required": ["plan_id", "mvp_id"],
        },
    ),
    Tool(
        name="codeloom_get_compiled_context",
        description=(
            "Get fully compiled migration context for an MVP phase. "
            "Returns dependency-ordered source code, ASG edges, deep analysis, "
            "and ground truth in a single bundle. "
            "Use instead of fetching units individually."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "plan_id": {"type": "string", "description": "Migration plan UUID"},
                "mvp_id": {"type": "integer", "description": "MVP ID"},
                "phase_type": {
                    "type": "string",
                    "enum": ["transform", "analyze", "design", "test", "architecture"],
                    "description": "Phase type determines token budget and context selection",
                },
                "token_budget": {
                    "type": "integer",
                    "description": "Override token budget (default: auto-scaled per phase type)",
                },
            },
            "required": ["plan_id", "mvp_id"],
        },
    ),
    Tool(
        name="codeloom_get_source_unit",
        description="Get full source code and metadata for a code unit by ID.",
        inputSchema={
            "type": "object",
            "properties": {
                "unit_id": {"type": "string", "description": "Code unit UUID"},
            },
            "required": ["unit_id"],
        },
    ),
    Tool(
        name="codeloom_search_codebase",
        description=(
            "Semantic search over an ingested codebase using RAG retrieval. "
            "Requires the RAG pipeline to be initialized."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string", "description": "Project UUID"},
                "query": {"type": "string", "description": "Natural language search query"},
                "top_k": {"type": "integer", "default": 5, "description": "Number of results"},
            },
            "required": ["project_id", "query"],
        },
    ),
    Tool(
        name="codeloom_validate_output",
        description=(
            "Run ground truth advisory validation on generated output text "
            "for a given phase type. Returns a list of advisory warnings."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string", "description": "Project UUID"},
                "phase_type": {
                    "type": "string",
                    "description": "Phase type: discovery|architecture|analyze|design|transform|test",
                },
                "output_text": {"type": "string", "description": "Output text to validate"},
            },
            "required": ["project_id", "phase_type", "output_text"],
        },
    ),
    Tool(
        name="codeloom_get_lane_info",
        description=(
            "Detect the migration lane for a source framework + target stack pair. "
            "Returns lane details, transform rules, and quality gates."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "source_framework": {
                    "type": "string",
                    "description": "Source framework name (e.g. 'struts', 'stored_proc', 'vbnet')",
                },
                "target_stack_json": {
                    "type": "string",
                    "description": "JSON string of target stack dict (e.g. '{\"framework\": \"spring_boot\"}')",
                },
            },
            "required": ["source_framework", "target_stack_json"],
        },
    ),
    # ── Full-autonomy migration tools ────────────────────────────────────
    Tool(
        name="codeloom_list_units",
        description=(
            "List all code units in a project for discovery. Paginated to handle large codebases. "
            "Filter by language or unit_type. Returns file_path for each unit."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string", "description": "Project UUID"},
                "language": {
                    "type": "string",
                    "description": "Optional filter: python, typescript, javascript, java, csharp, etc.",
                },
                "unit_type": {
                    "type": "string",
                    "description": "Optional filter: function, class, method, module",
                },
                "page": {"type": "integer", "default": 1, "description": "Page number (1-based)"},
                "page_size": {"type": "integer", "default": 50, "description": "Results per page (max 100)"},
            },
            "required": ["project_id"],
        },
    ),
    Tool(
        name="codeloom_save_plan",
        description=(
            "Save a Claude-generated migration plan (architecture + discovery docs) to CodeLoom DB "
            "for viewing inside the CodeLoom UI. Creates a new MigrationPlan with two completed "
            "plan-level phases (architecture + discovery). Returns plan_id for subsequent MVP saving."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string", "description": "Source project UUID"},
                "target_brief": {
                    "type": "string",
                    "description": "One-line description of the migration target (e.g. 'Migrate to TypeScript Express REST API')",
                },
                "target_stack": {
                    "type": "string",
                    "description": 'JSON string: {"languages": ["typescript"], "frameworks": ["expressjs"]}',
                },
                "architecture_doc": {
                    "type": "string",
                    "description": "Markdown: architecture decisions, target patterns, DI strategy, file structure",
                },
                "discovery_doc": {
                    "type": "string",
                    "description": "Markdown: codebase summary, module breakdown, migration strategy rationale",
                },
                "output_dir": {
                    "type": "string",
                    "description": "Target output directory for generated code files (e.g. migration-output/<project_id>/)",
                },
            },
            "required": ["project_id", "target_brief", "target_stack", "architecture_doc", "discovery_doc"],
        },
    ),
    Tool(
        name="codeloom_save_mvps",
        description=(
            "Save Claude-defined MVP cluster definitions to CodeLoom DB. "
            "Pass source_file_paths (relative paths from codeloom_list_units); the tool "
            "automatically resolves them to CodeFile and CodeUnit records. "
            "Creates Phase 3 (transform) and Phase 4 (test) rows for each MVP."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "plan_id": {"type": "string", "description": "Migration plan UUID from codeloom_save_plan"},
                "mvps": {
                    "type": "array",
                    "description": "List of MVP definitions",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Short MVP name (e.g. 'Foundation & Setup')"},
                            "description": {"type": "string", "description": "What this MVP covers"},
                            "priority": {
                                "type": "integer",
                                "description": "Migration order — lower = earlier (0-based)",
                            },
                            "source_file_paths": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Source file paths as returned by codeloom_list_units",
                            },
                            "depends_on_names": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Names of MVPs this one depends on (tool resolves to IDs)",
                            },
                        },
                        "required": ["name", "priority", "source_file_paths"],
                    },
                },
            },
            "required": ["plan_id", "mvps"],
        },
    ),
    Tool(
        name="codeloom_complete_transform",
        description=(
            "Record the outcome of an MVP's transform phase in CodeLoom. "
            "Pass status='complete' (default) to mark success: sets MVP to 'migrated'. "
            "Pass status='failed' when compile fails after all fix rounds: sets MVP to 'needs_review' "
            "so the CodeLoom UI shows it needs manual intervention. "
            "Records transform_summary and output file paths (no code content) for UI display."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "plan_id": {"type": "string", "description": "Migration plan UUID"},
                "mvp_id": {"type": "integer", "description": "MVP ID"},
                "transform_summary": {
                    "type": "string",
                    "description": "Markdown summary: files generated, transforms applied, compile result, errors",
                },
                "output_files": {
                    "type": "array",
                    "description": "List of generated output files (metadata only — code stays on disk)",
                    "items": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "language": {"type": "string"},
                            "source_path": {
                                "type": "string",
                                "description": "Original source file path (e.g. ORDCOMP.cbl) — enables diff viewer to pair source ↔ migrated",
                            },
                        },
                    },
                },
                "status": {
                    "type": "string",
                    "enum": ["complete", "failed"],
                    "default": "complete",
                    "description": "complete = success (MVP → migrated); failed = compile unresolved (MVP → needs_review)",
                },
            },
            "required": ["plan_id", "mvp_id", "transform_summary"],
        },
    ),
    Tool(
        name="codeloom_start_transform",
        description=(
            "Mark an MVP's transform phase as in-progress. Call this before starting file generation "
            "so the CodeLoom UI shows the MVP as actively being migrated. "
            "Sets MVP status to 'in_progress' and phase 3 status to 'in_progress'."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "plan_id": {"type": "string", "description": "Migration plan UUID"},
                "mvp_id": {"type": "integer", "description": "MVP ID"},
            },
            "required": ["plan_id", "mvp_id"],
        },
    ),
    Tool(
        name="codeloom_get_import_graph",
        description=(
            "Analyze import relationships between source files using ASG edges. "
            "Returns: (1) shared_files — files imported by `shared_threshold` or more other files, "
            "sorted by fan-in count descending. These MUST go into Foundation MVP during clustering. "
            "(2) import_edges — every source→target import pair (capped at 2000). "
            "Use during /migrate init before MVP clustering to identify shared infrastructure."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string", "description": "Project UUID"},
                "shared_threshold": {
                    "type": "integer",
                    "default": 3,
                    "description": "Min importer count to flag a file as shared infrastructure (default: 3)",
                },
            },
            "required": ["project_id"],
        },
    ),
    Tool(
        name="codeloom_save_accuracy_report",
        description=(
            "Persist migration accuracy report to CodeLoom DB after compare+fix completes. "
            "Stores pre/post-fix scores, fix counts, full markdown report, and per-MVP breakdown. "
            "Called automatically by /migrate compare at the end of every full run."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "plan_id": {"type": "string", "description": "Migration plan UUID"},
                "overall_score": {"type": "number", "description": "Accuracy score before auto-fixes (0–100)"},
                "fixed_score": {"type": "number", "description": "Accuracy score after auto-fixes (0–100)"},
                "fixes_applied": {"type": "integer", "description": "Number of surgical fixes successfully applied"},
                "fixes_pending": {"type": "integer", "description": "Number of issues requiring manual attention"},
                "report_markdown": {"type": "string", "description": "Full MIGRATION_ACCURACY.md content"},
                "per_mvp": {
                    "type": "array",
                    "description": "Per-MVP breakdown: [{mvp_name, score, fixed_score, programs, constructs, correct, gaps, bugs}]",
                    "items": {"type": "object"},
                },
            },
            "required": ["plan_id", "overall_score", "fixed_score", "fixes_applied", "fixes_pending", "report_markdown"],
        },
    ),
]


# ---------------------------------------------------------------------------
# CodeLoomMCPServer
# ---------------------------------------------------------------------------


class CodeLoomMCPServer:
    """MCP server wrapping CodeLoom's database and migration engine.

    Args:
        db_manager: DatabaseManager instance for DB access.
        pipeline: Optional LocalRAGPipeline for semantic search (codeloom_search_codebase).
    """

    def __init__(self, db_manager: Any, pipeline: Optional[Any] = None):
        self._db = db_manager
        self._pipeline = pipeline
        self.server = Server("codeloom")

        # Register handlers
        self._register_handlers()

    @staticmethod
    def _unescape_markdown(text: str) -> str:
        """Convert literal \\n sequences to real newlines.

        MCP tool arguments can arrive with escaped newlines (two chars: backslash + n)
        instead of actual newline characters depending on how the JSON-RPC message was
        constructed.  This normalises them before storing markdown in the DB so the
        frontend renders correctly.
        """
        if not text:
            return text
        return text.replace("\\n", "\n")

    # ── Handler Registration ────────────────────────────────────────────

    def _register_handlers(self) -> None:
        """Register list_tools and call_tool handlers on the MCP Server."""
        server = self.server

        @server.list_tools()
        async def list_tools() -> List[Tool]:
            return _TOOL_DEFINITIONS

        @server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            try:
                result = await self._dispatch(name, arguments or {})
            except Exception as exc:
                logger.exception("MCP tool %s raised: %s", name, exc)
                result = {"error": str(exc), "tool": name}
            return [TextContent(type="text", text=json.dumps(result, default=str))]

    # ── Dispatcher ──────────────────────────────────────────────────────

    async def _dispatch(self, name: str, args: Dict[str, Any]) -> Any:
        dispatch_table = {
            "codeloom_list_projects": self._list_projects,
            "codeloom_get_project_intel": self._get_project_intel,
            "codeloom_list_mvps": self._list_mvps,
            "codeloom_get_mvp_context": self._get_mvp_context,
            "codeloom_get_compiled_context": self._get_compiled_context,
            "codeloom_get_source_unit": self._get_source_unit,
            "codeloom_search_codebase": self._search_codebase,
            "codeloom_validate_output": self._validate_output,
            "codeloom_get_lane_info": self._get_lane_info,
            # Full-autonomy migration tools
            "codeloom_list_units": self._list_units,
            "codeloom_save_plan": self._save_plan,
            "codeloom_save_mvps": self._save_mvps,
            "codeloom_complete_transform": self._complete_transform,
            "codeloom_start_transform": self._start_transform,
            "codeloom_get_import_graph": self._get_import_graph,
            "codeloom_save_accuracy_report": self._save_accuracy_report,
        }
        handler = dispatch_table.get(name)
        if handler is None:
            return {"error": f"Unknown tool: {name}"}
        return await handler(args)

    # ── Tool implementations ────────────────────────────────────────────

    async def _list_projects(self, args: Dict) -> Dict:
        from codeloom.core.db.models import Project, CodeFile
        from sqlalchemy import func

        page = max(1, int(args.get("page", 1)))
        page_size = min(100, max(1, int(args.get("page_size", 20))))
        offset = (page - 1) * page_size

        with self._db.get_session() as session:
            total = session.query(func.count(Project.project_id)).scalar() or 0
            rows = (
                session.query(Project)
                .order_by(Project.created_at.desc())
                .offset(offset)
                .limit(page_size)
                .all()
            )
            projects = []
            for p in rows:
                file_count = (
                    session.query(func.count(CodeFile.file_id))
                    .filter(CodeFile.project_id == p.project_id)
                    .scalar()
                    or 0
                )
                projects.append(
                    {
                        "project_id": str(p.project_id),
                        "name": p.name,
                        "created_at": p.created_at.isoformat() if p.created_at else None,
                        "file_count": file_count,
                    }
                )

        return {"total": total, "page": page, "page_size": page_size, "projects": projects}

    async def _get_project_intel(self, args: Dict) -> Dict:
        from codeloom.core.db.models import (
            Project, CodeFile, CodeUnit, CodeEdge, MigrationPlan, FunctionalMVP,
        )
        from sqlalchemy import func, text

        project_id = args["project_id"]
        pid = UUID(project_id)

        with self._db.get_session() as session:
            project = session.query(Project).filter(Project.project_id == pid).first()
            if not project:
                return {"error": f"Project {project_id} not found"}

            # File stats
            file_rows = (
                session.query(CodeFile.language, func.count(CodeFile.file_id).label("cnt"))
                .filter(CodeFile.project_id == pid)
                .group_by(CodeFile.language)
                .all()
            )
            file_stats = {r.language or "unknown": r.cnt for r in file_rows}
            total_files = sum(file_stats.values())

            # Unit stats
            total_units = (
                session.query(func.count(CodeUnit.unit_id))
                .filter(CodeUnit.project_id == pid)
                .scalar()
                or 0
            )

            # ASG edge counts by type
            edge_rows = session.execute(
                text(
                    "SELECT edge_type, COUNT(*) AS cnt FROM code_edges "
                    "WHERE project_id = :pid GROUP BY edge_type"
                ),
                {"pid": pid},
            ).fetchall()
            edge_stats = {r.edge_type: r.cnt for r in edge_rows}

            # Understanding engine status (table may not exist in minimal deployments)
            understanding_status = {}
            try:
                analysis_rows = session.execute(
                    text("SELECT tier, COUNT(*) AS cnt FROM deep_analyses WHERE project_id = :pid GROUP BY tier"),
                    {"pid": pid},
                ).fetchall()
                understanding_status = {r.tier: r.cnt for r in analysis_rows}
            except Exception as _ue:
                logger.debug("deep_analyses unavailable: %s", _ue)
                session.rollback()  # Clear aborted txn so subsequent queries succeed

            # Migration plans
            plans = (
                session.query(MigrationPlan)
                .filter(MigrationPlan.source_project_id == pid)
                .order_by(MigrationPlan.created_at.desc())
                .all()
            )
            plan_summaries = []
            for plan in plans:
                mvp_count = (
                    session.query(func.count(FunctionalMVP.mvp_id))
                    .filter(FunctionalMVP.plan_id == plan.plan_id)
                    .scalar()
                    or 0
                )
                plan_summaries.append(
                    {
                        "plan_id": str(plan.plan_id),
                        "status": plan.status,
                        "target_brief": (plan.target_brief or "")[:200],
                        "pipeline_version": plan.pipeline_version,
                        "mvp_count": mvp_count,
                        "created_at": plan.created_at.isoformat() if plan.created_at else None,
                        "output_dir": (plan.discovery_metadata or {}).get("output_dir", ""),
                        "asset_strategies": plan.asset_strategies,
                    }
                )

        # Lane detection per source language
        from codeloom.core.migration.lanes.registry import LaneRegistry

        detected_lanes = []
        target_stack = {}
        if plan_summaries:
            # Use latest plan's target stack if available
            latest_plan_id = plan_summaries[-1].get("plan_id")
            if latest_plan_id:
                with self._db.get_session() as session:
                    latest_plan = session.get(MigrationPlan, UUID(latest_plan_id))
                    if latest_plan:
                        target_stack = latest_plan.target_stack or {}

        all_lanes = LaneRegistry.list_lanes()
        for lang in file_stats:
            for lane_info in all_lanes:
                lane = LaneRegistry.get_lane(lane_info["lane_id"])
                if lane and not lane.deprecated:
                    score = lane.detect_applicability(lang, target_stack)
                    if score > 0.0:
                        detected_lanes.append({
                            "source_language": lang,
                            "lane_id": lane_info["lane_id"],
                            "display_name": lane_info["display_name"],
                            "confidence": score,
                            "source_frameworks": lane_info["source_frameworks"],
                            "target_frameworks": lane_info["target_frameworks"],
                        })

        return {
            "project_id": project_id,
            "name": project.name,
            "total_files": total_files,
            "file_breakdown": file_stats,
            "total_units": total_units,
            "asg_edges": edge_stats,
            "understanding_analyses": understanding_status,
            "migration_plans": plan_summaries,
            "detected_lanes": detected_lanes,
        }

    async def _list_mvps(self, args: Dict) -> Dict:
        from codeloom.core.db.models import FunctionalMVP, MigrationPhase, CodeFile

        plan_id = args["plan_id"]
        pid = UUID(plan_id)

        with self._db.get_session() as session:
            mvps = (
                session.query(FunctionalMVP)
                .filter(FunctionalMVP.plan_id == pid)
                .order_by(FunctionalMVP.priority, FunctionalMVP.mvp_id)
                .all()
            )
            if not mvps:
                return {"plan_id": plan_id, "mvps": [], "total": 0}

            result = []
            for mvp in mvps:
                phases = (
                    session.query(MigrationPhase)
                    .filter(
                        MigrationPhase.plan_id == pid,
                        MigrationPhase.mvp_id == mvp.mvp_id,
                    )
                    .order_by(MigrationPhase.phase_number)
                    .all()
                )
                phase_summary = [
                    {"phase_number": p.phase_number, "phase_type": p.phase_type, "status": p.status}
                    for p in phases
                ]
                confidence = None
                if mvp.analysis_output and isinstance(mvp.analysis_output, dict):
                    confidence = mvp.analysis_output.get("confidence_score")

                # Resolve file_ids → file_paths so the run workflow can do import audits
                source_file_paths = []
                for fid in (mvp.file_ids or []):
                    cf = session.query(CodeFile).filter(CodeFile.file_id == fid).first()
                    if cf and cf.file_path:
                        source_file_paths.append(cf.file_path)

                result.append(
                    {
                        "mvp_id": mvp.mvp_id,
                        "name": mvp.name,
                        "description": (mvp.description or "")[:300],
                        "status": mvp.status,
                        "priority": mvp.priority,
                        "unit_ids": [str(uid) for uid in (mvp.unit_ids or [])],
                        "unit_count": len(mvp.unit_ids or []),
                        "source_file_paths": source_file_paths,
                        "file_count": len(mvp.file_ids or []),
                        "confidence": confidence,
                        "phases": phase_summary,
                    }
                )

        return {"plan_id": plan_id, "total": len(result), "mvps": result}

    async def _get_mvp_context(self, args: Dict) -> Dict:
        from codeloom.core.db.models import FunctionalMVP, MigrationPlan, CodeUnit, CodeFile
        from codeloom.core.migration.lanes.registry import LaneRegistry

        plan_id = args["plan_id"]
        mvp_id = int(args["mvp_id"])
        pid = UUID(plan_id)

        with self._db.get_session() as session:
            plan = session.query(MigrationPlan).filter(MigrationPlan.plan_id == pid).first()
            if not plan:
                return {"error": f"Plan {plan_id} not found"}

            mvp = (
                session.query(FunctionalMVP)
                .filter(FunctionalMVP.plan_id == pid, FunctionalMVP.mvp_id == mvp_id)
                .first()
            )
            if not mvp:
                return {"error": f"MVP {mvp_id} not found in plan {plan_id}"}

            # Source unit summaries
            unit_summaries = []
            for uid in (mvp.unit_ids or [])[:50]:
                cu = session.query(CodeUnit).filter(CodeUnit.unit_id == uid).first()
                if cu:
                    cf = session.query(CodeFile).filter(CodeFile.file_id == cu.file_id).first()
                    unit_summaries.append(
                        {
                            "unit_id": str(cu.unit_id),
                            "name": cu.name,
                            "unit_type": cu.unit_type,
                            "language": cu.language,
                            "file_path": (cf.file_path if cf else "") or "",
                            "signature": (cu.signature or "")[:200],
                        }
                    )

            # Lane detection
            target_stack = plan.target_stack or {}
            source_framework = (plan.discovery_metadata or {}).get("source_framework", "unknown")
            lane_info = None
            try:
                lane_match = LaneRegistry.detect_lane(source_framework, target_stack)
                if lane_match:
                    lane, confidence = lane_match
                    lane_info = {
                        "lane_id": lane.lane_id,
                        "confidence": confidence,
                        "source_frameworks": lane.source_frameworks,
                        "target_frameworks": lane.target_frameworks,
                    }
            except Exception as _le:
                logger.debug("Lane detection failed: %s", _le)

            # Deep analysis narratives
            deep_narratives = []
            try:
                from codeloom.core.migration.context_builder import MigrationContextBuilder
                project_id = str(plan.source_project_id)
                ctx_builder = MigrationContextBuilder(self._db, project_id)
                deep_ctx = ctx_builder.get_deep_analysis_context(
                    [str(uid) for uid in (mvp.unit_ids or [])],
                    max_narratives=3,
                )
                if deep_ctx:
                    deep_narratives = [deep_ctx]
            except Exception as _de:
                logger.debug("Deep analysis context skipped: %s", _de)

            # Ground truth summary
            gt_summary = None
            try:
                from codeloom.core.migration.ground_truth import CodebaseGroundTruth
                gt = CodebaseGroundTruth(self._db, str(plan.source_project_id))
                gt_summary = gt.format_layer_summary()
            except Exception as _ge:
                logger.debug("Ground truth summary skipped: %s", _ge)

            # Integration points: cross-boundary ASG edges + high fan-in/out units
            integration_points: Dict = {}
            try:
                from codeloom.core.migration.context_builder import MigrationContextBuilder as _MCB
                project_id = str(plan.source_project_id)
                _ctx = _MCB(self._db, project_id)
                uid_strings = [str(u) for u in (mvp.unit_ids or [])]
                cross_edges = _ctx._get_mvp_cross_edges(uid_strings, limit=30)
                boundary_units = _ctx._get_mvp_integration_points(uid_strings)
                integration_points = {
                    "cross_edges": cross_edges,
                    "boundary_units": boundary_units,
                }
            except Exception as _ip_err:
                logger.debug("Integration points skipped: %s", _ip_err)

            # Build return dict inside session scope to avoid DetachedInstanceError
            return {
                "plan_id": plan_id,
                "mvp_id": mvp_id,
                "name": mvp.name,
                "description": mvp.description or "",
                "status": mvp.status,
                "unit_count": len(mvp.unit_ids or []),
                "source_units": unit_summaries,
                "sp_references": list(mvp.sp_references or []),
                "lane_info": lane_info,
                "deep_narratives": deep_narratives,
                "ground_truth_summary": gt_summary,
                "integration_points": integration_points,
                "analysis_output": mvp.analysis_output,
            }

    async def _get_compiled_context(self, args: Dict) -> Dict:
        """Return fully assembled, dependency-ordered context for an MVP phase.

        Uses MigrationContextBuilder.build_phase_context() so the caller gets
        a single text block instead of N sequential codeloom_get_source_unit calls.
        """
        from codeloom.core.db.models import MigrationPlan, FunctionalMVP, MigrationPhase
        from codeloom.core.migration.context_builder import MigrationContextBuilder

        plan_id = args["plan_id"]
        mvp_id = args["mvp_id"]
        phase_type = args.get("phase_type", "transform")

        pid = UUID(plan_id)

        with self._db.get_session() as session:
            plan = session.query(MigrationPlan).filter(
                MigrationPlan.plan_id == pid
            ).first()
            if not plan:
                return {"error": f"Plan {plan_id} not found"}

            mvp = session.query(FunctionalMVP).filter(
                FunctionalMVP.plan_id == pid,
                FunctionalMVP.mvp_id == mvp_id,
            ).first()
            if not mvp:
                return {"error": f"MVP {mvp_id} not found in plan {plan_id}"}

            mvp_context = {
                "mvp_id": mvp.mvp_id,
                "name": mvp.name,
                "description": mvp.description or "",
                "unit_ids": list(mvp.unit_ids or []),
                "file_ids": list(mvp.file_ids or []),
            }

            # Gather previous phase outputs for context continuity
            previous_outputs: Dict[int, str] = {}
            completed_phases = session.query(MigrationPhase).filter(
                MigrationPhase.plan_id == pid,
                MigrationPhase.status == "complete",
            ).all()
            for p in completed_phases:
                if p.output:
                    previous_outputs[p.phase_number] = p.output

            project_id = str(plan.source_project_id)

        # Build context outside session — context_builder opens its own sessions
        ctx_builder = MigrationContextBuilder(self._db, project_id)

        kwargs: Dict[str, Any] = {
            "mvp_context": mvp_context,
            "context_type": phase_type,
        }
        if args.get("token_budget"):
            kwargs["token_budget"] = int(args["token_budget"])

        try:
            context_str = ctx_builder.build_phase_context(
                phase_number=3,
                previous_outputs=previous_outputs,
                **kwargs,
            )
        except Exception as exc:
            logger.error("build_phase_context failed: %s", exc)
            return {"error": f"Context build failed: {exc}"}

        return {
            "plan_id": plan_id,
            "mvp_id": mvp_id,
            "phase_type": phase_type,
            "context": context_str,
            "token_estimate": len(context_str) // 4,
        }

    async def _get_source_unit(self, args: Dict) -> Dict:
        from codeloom.core.db.models import CodeUnit, CodeFile

        unit_id = args["unit_id"]
        uid = UUID(unit_id)

        with self._db.get_session() as session:
            cu = session.query(CodeUnit).filter(CodeUnit.unit_id == uid).first()
            if not cu:
                return {"error": f"Code unit {unit_id} not found"}

            cf = session.query(CodeFile).filter(CodeFile.file_id == cu.file_id).first()
            return {
                "unit_id": str(cu.unit_id),
                "name": cu.name,
                "qualified_name": cu.qualified_name or "",
                "unit_type": cu.unit_type,
                "language": cu.language,
                "file_path": (cf.file_path if cf else "") or "",
                "signature": cu.signature or "",
                "source_code": cu.source or "",
                "docstring": cu.docstring or "",
                "metadata": cu.unit_metadata or {},
            }

    async def _search_codebase(self, args: Dict) -> Dict:
        if self._pipeline is None:
            return {
                "error": (
                    "RAG pipeline not available. Start CodeLoom with pipeline initialized "
                    "to enable semantic search."
                )
            }

        project_id = args["project_id"]
        query = args["query"]
        top_k = min(20, max(1, int(args.get("top_k", 5))))

        try:
            # stateless_query is synchronous; requires message/project_id/user_id.
            # Use nil UUID as MCP sentinel — avoids UUID parse error on conversation save.
            response = self._pipeline.stateless_query(
                message=query,
                project_id=project_id,
                user_id="00000000-0000-0000-0000-000000000000",
                max_sources=top_k,
                include_history=False,
            )
            # format_sources returns dicts with keys: filename, snippet, score
            sources = response.get("sources", [])
            results = []
            for src in sources:
                if isinstance(src, dict):
                    results.append({
                        "text": src.get("snippet", src.get("text", src.get("content", "")))[:500],
                        "score": src.get("score"),
                        "file_path": src.get("filename", ""),
                    })
                else:
                    # NodeWithScore or similar object
                    text = src.get_content() if hasattr(src, "get_content") else str(src)
                    results.append({
                        "text": text[:500],
                        "score": getattr(src, "score", None),
                        "file_path": getattr(getattr(src, "node", src), "metadata", {}).get("file_name", ""),
                    })
            return {"project_id": project_id, "query": query, "results": results}
        except Exception as exc:
            return {"error": f"Search failed: {exc}"}

    async def _validate_output(self, args: Dict) -> Dict:
        from codeloom.core.migration.ground_truth import CodebaseGroundTruth

        project_id = args["project_id"]
        phase_type = args["phase_type"]
        output_text = args["output_text"]

        try:
            gt = CodebaseGroundTruth(self._db, project_id)
            issues = gt.validate_phase_output(phase_type, output_text)
            return {
                "project_id": project_id,
                "phase_type": phase_type,
                "issue_count": len(issues),
                "issues": [
                    {
                        "issue_type": i.issue_type,
                        "severity": i.severity,
                        "message": i.message,
                    }
                    for i in issues
                ],
            }
        except Exception as exc:
            return {"error": f"Validation failed: {exc}"}

    async def _get_lane_info(self, args: Dict) -> Dict:
        from codeloom.core.migration.lanes.registry import LaneRegistry

        source_framework = args["source_framework"]
        target_stack_str = args.get("target_stack_json", "{}")
        try:
            target_stack = json.loads(target_stack_str)
        except json.JSONDecodeError:
            target_stack = {}

        try:
            match = LaneRegistry.detect_lane(source_framework, target_stack)
        except Exception as exc:
            return {"error": f"Lane detection failed: {exc}"}

        if not match:
            return {
                "source_framework": source_framework,
                "lane_detected": False,
                "message": "No lane matched for this source framework / target stack combination.",
            }

        lane, confidence = match

        # Collect transform rules
        rules = []
        try:
            for rule in lane.get_transform_rules():
                rules.append(
                    {
                        "name": rule.name,
                        "description": rule.description,
                        "pattern": rule.pattern,
                        "template": (rule.template or "")[:300],
                        "confidence": rule.confidence,
                        "requires_review": rule.requires_review,
                    }
                )
        except Exception as _re:
            logger.debug("Could not enumerate transform rules: %s", _re)

        # Collect quality gates
        gates = []
        try:
            for gate in lane.get_quality_gates():
                gates.append(
                    {
                        "name": gate.name,
                        "category": gate.category,
                        "blocking": gate.blocking,
                        "description": gate.description,
                    }
                )
        except Exception as _ge:
            logger.debug("Could not enumerate quality gates: %s", _ge)

        return {
            "source_framework": source_framework,
            "lane_detected": True,
            "lane_id": lane.lane_id,
            "confidence": confidence,
            "source_frameworks": lane.source_frameworks,
            "target_frameworks": lane.target_frameworks,
            "transform_rules": rules,
            "quality_gates": gates,
        }

    # ── Full-autonomy migration handlers ────────────────────────────────

    async def _list_units(self, args: Dict) -> Dict:
        from codeloom.core.db.models import CodeUnit, CodeFile
        from sqlalchemy import func

        project_id = args["project_id"]
        pid = UUID(project_id)
        language = args.get("language")
        unit_type = args.get("unit_type")
        page = max(1, int(args.get("page", 1)))
        page_size = min(100, max(1, int(args.get("page_size", 50))))
        offset = (page - 1) * page_size

        with self._db.get_session() as session:
            q = (
                session.query(CodeUnit, CodeFile.file_path)
                .join(CodeFile, CodeUnit.file_id == CodeFile.file_id)
                .filter(CodeUnit.project_id == pid)
            )
            if language:
                q = q.filter(CodeUnit.language == language)
            if unit_type:
                q = q.filter(CodeUnit.unit_type == unit_type)

            total = q.count()
            rows = q.order_by(CodeFile.file_path, CodeUnit.name).offset(offset).limit(page_size).all()

            units = [
                {
                    "unit_id": str(cu.unit_id),
                    "name": cu.name,
                    "qualified_name": cu.qualified_name or "",
                    "unit_type": cu.unit_type,
                    "language": cu.language,
                    "file_path": file_path or "",
                    "signature": (cu.signature or "")[:200],
                }
                for cu, file_path in rows
            ]

        return {
            "project_id": project_id,
            "total": total,
            "page": page,
            "page_size": page_size,
            "pages": (total + page_size - 1) // page_size,
            "units": units,
        }

    async def _save_plan(self, args: Dict) -> Dict:
        from codeloom.core.db.models import MigrationPlan, MigrationPhase, User
        import uuid as _uuid
        from datetime import datetime

        project_id = args["project_id"]
        target_brief = args["target_brief"]
        target_stack_str = args["target_stack"]
        architecture_doc = self._unescape_markdown(args["architecture_doc"])
        discovery_doc = self._unescape_markdown(args["discovery_doc"])
        output_dir = args.get("output_dir", f"migration-output/{project_id}/")

        try:
            target_stack = json.loads(target_stack_str)
        except json.JSONDecodeError:
            return {"error": f"target_stack is not valid JSON: {target_stack_str[:100]}"}

        pid = UUID(project_id)
        plan_id = _uuid.uuid4()

        with self._db.get_session() as session:
            # Resolve user_id — use first admin user or any user in the system
            user = session.query(User).order_by(User.created_at).first()
            if not user:
                return {"error": "No users found in DB — cannot create plan without a user_id"}
            user_id = user.user_id

            plan = MigrationPlan(
                plan_id=plan_id,
                user_id=user_id,
                source_project_id=pid,
                target_brief=target_brief,
                target_stack=target_stack,
                status="in_progress",
                pipeline_version=2,
                migration_type="framework_migration",
                discovery_metadata={
                    "output_dir": output_dir,
                    "orchestrator": "claude_code_cli",
                },
            )
            session.add(plan)
            session.flush()  # Get plan_id assigned

            # Phase 1: Architecture (plan-level, already complete)
            arch_phase = MigrationPhase(
                plan_id=plan_id,
                mvp_id=None,
                phase_number=1,
                phase_type="architecture",
                status="complete",
                output=architecture_doc,
                approved=True,
                approved_at=datetime.utcnow(),
            )
            session.add(arch_phase)

            # Phase 2: Discovery (plan-level, already complete)
            disc_phase = MigrationPhase(
                plan_id=plan_id,
                mvp_id=None,
                phase_number=2,
                phase_type="discovery",
                status="complete",
                output=discovery_doc,
                approved=True,
                approved_at=datetime.utcnow(),
            )
            session.add(disc_phase)
            session.commit()

        return {
            "plan_id": str(plan_id),
            "status": "in_progress",
            "output_dir": output_dir,
            "message": f"Plan created with 2 completed plan-level phases. View in CodeLoom UI at /migration.",
        }

    async def _save_mvps(self, args: Dict) -> Dict:
        from codeloom.core.db.models import FunctionalMVP, MigrationPhase, CodeFile, CodeUnit

        plan_id = args["plan_id"]
        mvps_input = args["mvps"]
        pid = UUID(plan_id)

        created = []
        name_to_id: Dict[str, int] = {}

        with self._db.get_session() as session:
            for mvp_def in mvps_input:
                name = mvp_def["name"]
                description = mvp_def.get("description", "")
                priority = int(mvp_def.get("priority", 0))
                source_file_paths = mvp_def.get("source_file_paths", [])

                # Resolve file paths → CodeFile records
                file_ids: List[str] = []
                unit_ids: List[str] = []
                for path in source_file_paths:
                    matches = (
                        session.query(CodeFile)
                        .filter(
                            CodeFile.file_path.endswith(path) | (CodeFile.file_path == path)
                        )
                        .all()
                    )
                    for cf in matches:
                        fid = str(cf.file_id)
                        if fid not in file_ids:
                            file_ids.append(fid)
                        # Collect all units in this file
                        units = (
                            session.query(CodeUnit.unit_id)
                            .filter(CodeUnit.file_id == cf.file_id)
                            .all()
                        )
                        for (uid,) in units:
                            uid_str = str(uid)
                            if uid_str not in unit_ids:
                                unit_ids.append(uid_str)

                mvp = FunctionalMVP(
                    plan_id=pid,
                    name=name,
                    description=description,
                    priority=priority,
                    status="discovered",
                    file_ids=file_ids,
                    unit_ids=unit_ids,
                )
                session.add(mvp)
                session.flush()  # Get mvp_id

                # Phase 3: Transform (pending)
                session.add(MigrationPhase(
                    plan_id=pid,
                    mvp_id=mvp.mvp_id,
                    phase_number=3,
                    phase_type="transform",
                    status="pending",
                ))
                # Phase 4: Test (pending)
                session.add(MigrationPhase(
                    plan_id=pid,
                    mvp_id=mvp.mvp_id,
                    phase_number=4,
                    phase_type="test",
                    status="pending",
                ))

                name_to_id[name] = mvp.mvp_id
                created.append({
                    "mvp_id": mvp.mvp_id,
                    "name": name,
                    "file_count": len(file_ids),
                    "unit_count": len(unit_ids),
                    "priority": priority,
                })

            # Resolve depends_on_names → mvp_ids
            for i, mvp_def in enumerate(mvps_input):
                depends_on_names = mvp_def.get("depends_on_names", [])
                if depends_on_names:
                    dep_ids = [name_to_id[n] for n in depends_on_names if n in name_to_id]
                    if dep_ids:
                        mvp_id_val = created[i]["mvp_id"]
                        mvp_obj = (
                            session.query(FunctionalMVP)
                            .filter(FunctionalMVP.mvp_id == mvp_id_val)
                            .first()
                        )
                        if mvp_obj:
                            mvp_obj.depends_on_mvp_ids = dep_ids

            session.commit()

        return {
            "plan_id": plan_id,
            "mvp_count": len(created),
            "mvps": created,
            "message": f"{len(created)} MVPs saved. Each has Phase 3 (transform) and Phase 4 (test) rows pending.",
        }

    async def _complete_transform(self, args: Dict) -> Dict:
        from codeloom.core.db.models import MigrationPhase, MigrationPlan, FunctionalMVP
        from datetime import datetime

        plan_id = args["plan_id"]
        mvp_id = int(args["mvp_id"])
        transform_summary = self._unescape_markdown(args["transform_summary"])
        output_files = args.get("output_files", [])
        status = args.get("status", "complete")  # "complete" | "failed"

        pid = UUID(plan_id)

        with self._db.get_session() as session:
            phase = (
                session.query(MigrationPhase)
                .filter(
                    MigrationPhase.plan_id == pid,
                    MigrationPhase.mvp_id == mvp_id,
                    MigrationPhase.phase_number == 3,
                )
                .first()
            )
            if not phase:
                return {"error": f"Phase 3 not found for plan {plan_id} mvp {mvp_id}"}

            # Store file metadata only (no code content — code stays on disk).
            # source_path is an optional hint for the diff viewer to match
            # migrated files back to their original source (e.g. ORDCOMP.cbl → compordm.py).
            files_meta = []
            for f in output_files:
                entry: dict = {"file_path": f.get("file_path", ""), "language": f.get("language", "")}
                if f.get("source_path"):
                    entry["source_path"] = f["source_path"]
                files_meta.append(entry)
            phase.output = transform_summary
            phase.output_files = files_meta

            # Persist output_dir in phase_metadata so get_diff_context can
            # resolve relative file paths back to disk for inline display.
            plan = session.query(MigrationPlan).filter(MigrationPlan.plan_id == pid).first()
            plan_output_dir = (plan.discovery_metadata or {}).get("output_dir", "") if plan else ""
            existing_meta = dict(phase.phase_metadata or {})
            if plan_output_dir and not existing_meta.get("output_path"):
                existing_meta["output_path"] = plan_output_dir
                phase.phase_metadata = existing_meta

            if status == "failed":
                phase.status = "failed"
                phase.approved = False
            else:
                phase.status = "complete"
                phase.approved = True
                phase.approved_at = datetime.utcnow()

            # Update MVP status
            mvp = (
                session.query(FunctionalMVP)
                .filter(FunctionalMVP.plan_id == pid, FunctionalMVP.mvp_id == mvp_id)
                .first()
            )
            if mvp:
                mvp.status = "needs_review" if status == "failed" else "migrated"

            session.commit()

        mvp_final_status = "needs_review" if status == "failed" else "migrated"
        return {
            "plan_id": plan_id,
            "mvp_id": mvp_id,
            "status": mvp_final_status,
            "phase": "transform",
            "complete": status != "failed",
            "output_files_count": len(output_files),
        }

    async def _start_transform(self, args: Dict) -> Dict:
        from codeloom.core.db.models import MigrationPhase, FunctionalMVP
        from datetime import datetime

        plan_id = args["plan_id"]
        mvp_id = int(args["mvp_id"])
        pid = UUID(plan_id)

        with self._db.get_session() as session:
            phase = (
                session.query(MigrationPhase)
                .filter(
                    MigrationPhase.plan_id == pid,
                    MigrationPhase.mvp_id == mvp_id,
                    MigrationPhase.phase_number == 3,
                )
                .first()
            )
            if not phase:
                return {"error": f"Phase 3 not found for plan {plan_id} mvp {mvp_id}"}

            phase.status = "in_progress"

            mvp = (
                session.query(FunctionalMVP)
                .filter(FunctionalMVP.plan_id == pid, FunctionalMVP.mvp_id == mvp_id)
                .first()
            )
            if mvp:
                mvp.status = "in_progress"
                mvp.updated_at = datetime.utcnow()

            session.commit()

        return {
            "plan_id": plan_id,
            "mvp_id": mvp_id,
            "status": "in_progress",
            "phase": "transform",
        }

    async def _get_import_graph(self, args: Dict) -> Dict:
        from codeloom.core.db.models import CodeEdge, CodeUnit, CodeFile
        from sqlalchemy.orm import aliased

        project_id = args["project_id"]
        shared_threshold = int(args.get("shared_threshold", 3))
        pid = UUID(project_id)

        with self._db.get_session() as session:
            SrcUnit = aliased(CodeUnit)
            TgtUnit = aliased(CodeUnit)
            SrcFile = aliased(CodeFile)
            TgtFile = aliased(CodeFile)

            rows = (
                session.query(
                    SrcFile.file_path.label("source_file"),
                    TgtFile.file_path.label("target_file"),
                )
                .select_from(CodeEdge)
                .join(SrcUnit, CodeEdge.source_unit_id == SrcUnit.unit_id)
                .join(TgtUnit, CodeEdge.target_unit_id == TgtUnit.unit_id)
                .join(SrcFile, SrcUnit.file_id == SrcFile.file_id)
                .join(TgtFile, TgtUnit.file_id == TgtFile.file_id)
                .filter(
                    CodeEdge.project_id == pid,
                    CodeEdge.edge_type == "imports",
                )
                .distinct()
                .all()
            )

            # Build fan-in map: target_file → set of unique importer file_paths
            fan_in: Dict[str, set] = {}
            edges = []
            for row in rows:
                src = row.source_file or ""
                tgt = row.target_file or ""
                if not src or not tgt or src == tgt:
                    continue
                edges.append({"source": src, "target": tgt})
                fan_in.setdefault(tgt, set()).add(src)

            shared_files = [
                {
                    "file_path": fp,
                    "importer_count": len(importers),
                    "imported_by": sorted(list(importers))[:10],
                }
                for fp, importers in fan_in.items()
                if len(importers) >= shared_threshold
            ]
            shared_files.sort(key=lambda x: x["importer_count"], reverse=True)

        return {
            "project_id": project_id,
            "total_import_edges": len(edges),
            "shared_files": shared_files,
            "import_edges": edges[:2000],
            "truncated": len(edges) > 2000,
        }

    async def _save_accuracy_report(self, args: Dict) -> Dict:
        """Persist migration accuracy report to DB after compare+fix completes.

        Also auto-completes all MVPs (status=migrated) and the plan (status=complete)
        since comparison is the final step — compilation was verified during transform.
        """
        from codeloom.core.db.models import MigrationPlan, FunctionalMVP, MigrationPhase
        from datetime import datetime
        plan_id = args["plan_id"]
        pid = UUID(plan_id)
        with self._db.get_session() as session:
            plan = session.query(MigrationPlan).filter(
                MigrationPlan.plan_id == pid
            ).first()
            if not plan:
                return {"error": f"Plan {plan_id} not found"}

            # Save accuracy data
            plan.accuracy_score         = float(args["overall_score"])
            plan.accuracy_fixed_score   = float(args["fixed_score"])
            plan.accuracy_fixes_applied = int(args["fixes_applied"])
            plan.accuracy_fixes_pending = int(args["fixes_pending"])
            plan.accuracy_report_md     = self._unescape_markdown(args["report_markdown"])
            plan.accuracy_per_mvp       = args.get("per_mvp", [])
            plan.accuracy_last_run      = datetime.utcnow()

            # Auto-complete all MVPs — compare is the final step
            mvps = session.query(FunctionalMVP).filter(
                FunctionalMVP.plan_id == pid
            ).all()
            mvp_count = len(mvps)
            for mvp in mvps:
                mvp.status = "migrated"
                # Auto-approve any pending/in-progress phases for this MVP
                phases = session.query(MigrationPhase).filter(
                    MigrationPhase.plan_id == pid,
                    MigrationPhase.mvp_id == mvp.mvp_id,
                ).all()
                for phase in phases:
                    if phase.status != "complete":
                        phase.status = "complete"
                    if not phase.approved:
                        phase.approved = True
                        phase.approved_at = datetime.utcnow()

            # Mark plan complete
            plan.status = "complete"

            session.commit()

        return {
            "plan_id": plan_id,
            "accuracy_score": args["overall_score"],
            "fixed_score": args["fixed_score"],
            "fixes_applied": args["fixes_applied"],
            "fixes_pending": args["fixes_pending"],
            "mvps_completed": mvp_count,
            "message": "Accuracy report saved. Migration marked complete.",
        }

    # ── Internal helpers ────────────────────────────────────────────────

    def _get_engine(self):
        """Create a MigrationEngine bound to our DB + pipeline."""
        from codeloom.core.migration.engine import MigrationEngine
        return MigrationEngine(db_manager=self._db, pipeline=self._pipeline)

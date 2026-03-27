"""Python-specific chapter overrides for reverse engineering documentation.

When the project is predominantly Python (FastAPI/Flask/Django/CLI), these
generators replace the generic chapters with deep, Python-aware functional
specifications:
  - Per-module class/function inventories with decorator analysis
  - FastAPI/Flask/Django route mapping with auth and middleware info
  - ORM model and Pydantic schema catalogs
  - Database access pattern classification (ORM vs raw SQL)
  - Call graphs showing request-response, pipeline, and agentic flows
  - pip dependency and environment variable inventories
  - Exception hierarchy and retry/resilience patterns

Each override function has the same signature as the generic generators:
    (db: DatabaseManager, project_id: str, pipeline=None) -> str
"""

import json
import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID

from sqlalchemy import text

from ..db import DatabaseManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _pid(project_id: str) -> UUID:
    return UUID(project_id) if isinstance(project_id, str) else project_id


def _safe_source(row) -> str:
    """Return source text or empty string if NULL."""
    return row.source or "" if hasattr(row, "source") else ""


def _safe_meta(row) -> dict:
    """Return metadata dict, handling NULL/string/dict."""
    raw = getattr(row, "metadata", None)
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return {}
    return {}


# ---------------------------------------------------------------------------
# Data-fetch helpers
# ---------------------------------------------------------------------------


def _fetch_classes(db: DatabaseManager, pid: UUID) -> list:
    """Fetch all class-level code_units for the project."""
    with db.get_session() as session:
        return session.execute(
            text("""
                SELECT u.unit_id, u.name, u.qualified_name, u.signature,
                       u.docstring, u.source, u.unit_type,
                       u.start_line, u.end_line,
                       f.file_path, u.metadata
                FROM code_units u
                JOIN code_files f ON u.file_id = f.file_id
                WHERE u.project_id = :pid AND u.unit_type = 'class'
                ORDER BY f.file_path, u.start_line
            """),
            {"pid": pid},
        ).fetchall()


def _fetch_functions(db: DatabaseManager, pid: UUID) -> list:
    """Fetch all top-level function code_units for the project."""
    with db.get_session() as session:
        return session.execute(
            text("""
                SELECT u.unit_id, u.name, u.qualified_name, u.signature,
                       u.docstring, u.source, u.unit_type,
                       u.start_line, u.end_line,
                       f.file_path, u.metadata
                FROM code_units u
                JOIN code_files f ON u.file_id = f.file_id
                WHERE u.project_id = :pid AND u.unit_type = 'function'
                ORDER BY f.file_path, u.start_line
            """),
            {"pid": pid},
        ).fetchall()


def _fetch_methods(db: DatabaseManager, pid: UUID) -> list:
    """Fetch all methods linked via 'contains' edges to their parent class."""
    with db.get_session() as session:
        return session.execute(
            text("""
                SELECT cu.unit_id, cu.name, cu.qualified_name, cu.signature,
                       cu.docstring, cu.source, cu.metadata,
                       cu.start_line, cu.end_line,
                       parent.name AS class_name, parent.unit_id AS class_unit_id,
                       f.file_path
                FROM code_edges e
                JOIN code_units cu ON e.target_unit_id = cu.unit_id
                JOIN code_units parent ON e.source_unit_id = parent.unit_id
                JOIN code_files f ON cu.file_id = f.file_id
                WHERE e.project_id = :pid AND e.edge_type = 'contains'
                  AND cu.unit_type = 'method'
                ORDER BY parent.name, cu.start_line
            """),
            {"pid": pid},
        ).fetchall()


def _fetch_calls(db: DatabaseManager, pid: UUID) -> list:
    """Fetch call edges."""
    with db.get_session() as session:
        return session.execute(
            text("""
                SELECT su.name AS caller, tu.name AS callee,
                       su.qualified_name AS caller_qn, tu.qualified_name AS callee_qn,
                       su.unit_type AS caller_type, tu.unit_type AS callee_type
                FROM code_edges e
                JOIN code_units su ON e.source_unit_id = su.unit_id
                JOIN code_units tu ON e.target_unit_id = tu.unit_id
                WHERE e.project_id = :pid AND e.edge_type = 'calls'
                ORDER BY su.name, tu.name
            """),
            {"pid": pid},
        ).fetchall()


def _fetch_imports(db: DatabaseManager, pid: UUID) -> list:
    """Fetch import edges."""
    with db.get_session() as session:
        return session.execute(
            text("""
                SELECT su.name AS source_module, tu.name AS imported,
                       sf.file_path AS from_file, tf.file_path AS to_file
                FROM code_edges e
                JOIN code_units su ON e.source_unit_id = su.unit_id
                JOIN code_units tu ON e.target_unit_id = tu.unit_id
                JOIN code_files sf ON su.file_id = sf.file_id
                JOIN code_files tf ON tu.file_id = tf.file_id
                WHERE e.project_id = :pid AND e.edge_type = 'imports'
                ORDER BY sf.file_path, tu.name
            """),
            {"pid": pid},
        ).fetchall()


def _fetch_deep_analyses(db: DatabaseManager, pid: UUID) -> list:
    """Fetch all deep analysis results for the project."""
    with db.get_session() as session:
        return session.execute(
            text("""
                SELECT a.analysis_id, a.entry_type, a.result_json, a.narrative,
                       a.confidence_score, a.coverage_pct,
                       u.name AS entry_name, u.unit_id AS entry_unit_id,
                       f.file_path
                FROM deep_analyses a
                JOIN code_units u ON a.entry_unit_id = u.unit_id
                JOIN code_files f ON u.file_id = f.file_id
                WHERE a.project_id = :pid
                ORDER BY u.name
            """),
            {"pid": pid},
        ).fetchall()


def _fetch_entry_points(db: DatabaseManager, pid: UUID) -> list:
    """Fetch detected entry points."""
    with db.get_session() as session:
        return session.execute(
            text("""
                SELECT ep.entry_point_id, ep.entry_type, ep.confidence,
                       u.name, u.qualified_name, u.signature, u.metadata,
                       f.file_path
                FROM entry_points ep
                JOIN code_units u ON ep.unit_id = u.unit_id
                JOIN code_files f ON u.file_id = f.file_id
                WHERE ep.project_id = :pid
                ORDER BY ep.entry_type, u.name
            """),
            {"pid": pid},
        ).fetchall()


# ---------------------------------------------------------------------------
# Source-text extraction helpers
# ---------------------------------------------------------------------------


def _extract_decorators(source: str) -> List[str]:
    """Extract decorator lines from Python source."""
    return re.findall(r"^\s*(@\w[\w.]*(?:\([^)]*\))?)", source, re.MULTILINE)


def _extract_route_info(source: str) -> List[Dict[str, str]]:
    """Extract FastAPI/Flask route decorators with method and path."""
    routes = []
    for m in re.finditer(
        r"@\w*\.(?:get|post|put|delete|patch|options|head)\s*\(\s*[\"']([^\"']+)",
        source,
        re.IGNORECASE,
    ):
        full = m.group(0)
        method_match = re.search(r"\.(get|post|put|delete|patch|options|head)", full, re.IGNORECASE)
        method = method_match.group(1).upper() if method_match else "GET"
        routes.append({"path": m.group(1), "method": method})
    return routes


def _extract_env_vars(source: str) -> List[str]:
    """Extract environment variable references from source."""
    return sorted(set(re.findall(
        r"os\.(?:environ|getenv)\s*[\[(]\s*[\"'](\w+)", source
    )))


def _extract_config_keys(source: str) -> List[str]:
    """Extract configuration key references."""
    return sorted(set(re.findall(
        r"(?:config|settings|cfg)\s*[\[.]\s*[\"']?(\w+)",
        source,
        re.IGNORECASE,
    )))


def _detect_framework(source_files: list) -> str:
    """Detect Python web framework from file content patterns.

    Returns one of: FastAPI, Flask, Django, CLI, Library, Unknown.
    """
    all_source = " ".join(getattr(r, "source", "") or "" for r in source_files[:50])
    if "FastAPI" in all_source or "from fastapi" in all_source.lower():
        return "FastAPI"
    if "Flask" in all_source or "from flask" in all_source.lower():
        return "Flask"
    if "django" in all_source.lower():
        return "Django"
    if "argparse" in all_source or "click" in all_source or "typer" in all_source:
        return "CLI"
    return "Library"


def _classify_architecture(unit_rows: list, edge_rows: list) -> str:
    """Classify architecture pattern from unit and edge distributions."""
    unit_types = {r.unit_type for r in unit_rows} if unit_rows else set()
    edge_count = len(edge_rows) if edge_rows else 0

    # Heuristic classification
    has_routes = any("route" in str(getattr(r, "metadata", "") or "").lower() for r in unit_rows)
    has_workers = any("worker" in (getattr(r, "name", "") or "").lower() for r in unit_rows)
    has_agent = any("agent" in (getattr(r, "name", "") or "").lower() for r in unit_rows)

    if has_agent:
        return "Agentic"
    if has_workers and has_routes:
        return "Pipeline + API"
    if has_routes:
        return "Monolith" if edge_count > 100 else "Microservice"
    return "Library"


def _module_from_path(file_path: str) -> str:
    """Convert file path to Python module notation."""
    # Strip leading directories up to the package root, convert / to .
    path = file_path.replace("\\", "/")
    # Remove .py extension
    if path.endswith(".py"):
        path = path[:-3]
    # Replace / with .
    return path.replace("/", ".")


# ---------------------------------------------------------------------------
# Chapter 1 Override: Module Inventory & Application Architecture
# ---------------------------------------------------------------------------


def generate_module_inventory(
    db: DatabaseManager, project_id: str, pipeline: Any = None,
) -> str:
    """Per-module catalog with class/function counts, framework detection,
    architecture classification, and package structure tree."""
    pid = _pid(project_id)
    lines = ["# 1. Module Inventory & Application Architecture\n"]

    try:
        with db.get_session() as session:
            # Project info
            proj = session.execute(
                text("SELECT name, primary_language, languages, file_count, total_lines "
                     "FROM projects WHERE project_id = :pid"),
                {"pid": pid},
            ).fetchone()

            if not proj:
                return "# 1. Module Inventory & Application Architecture\n\nProject not found."

            # Language breakdown
            lang_rows = session.execute(
                text("SELECT language, COUNT(*) AS cnt FROM code_files "
                     "WHERE project_id = :pid GROUP BY language ORDER BY cnt DESC"),
                {"pid": pid},
            ).fetchall()

            # All Python files with source for framework detection
            py_files = session.execute(
                text("SELECT file_path, source FROM code_files "
                     "WHERE project_id = :pid AND language = 'python' "
                     "ORDER BY file_path LIMIT 50"),
                {"pid": pid},
            ).fetchall()

            # Units grouped by file
            unit_rows = session.execute(
                text("""
                    SELECT u.unit_type, u.name, u.qualified_name, u.metadata,
                           f.file_path
                    FROM code_units u
                    JOIN code_files f ON u.file_id = f.file_id
                    WHERE u.project_id = :pid
                    ORDER BY f.file_path, u.start_line
                """),
                {"pid": pid},
            ).fetchall()

            # Edge counts
            edge_rows = session.execute(
                text("SELECT edge_type, COUNT(*) AS cnt FROM code_edges "
                     "WHERE project_id = :pid GROUP BY edge_type ORDER BY cnt DESC"),
                {"pid": pid},
            ).fetchall()

        # Framework and architecture detection
        framework = _detect_framework(py_files)
        architecture = _classify_architecture(unit_rows, edge_rows)

        # Detect entry point
        entry_point = "unknown"
        for pf in py_files:
            fp = pf.file_path or ""
            if fp.endswith("__main__.py"):
                entry_point = fp
                break
            if fp.endswith("app.py") or fp.endswith("main.py") or fp.endswith("manage.py"):
                entry_point = fp

        lines += [
            "## Application Type\n",
            f"- **Framework**: {framework}",
            f"- **Architecture**: {architecture}",
            f"- **Entry point**: `{entry_point}`",
            f"- **Primary Language**: {proj.primary_language or 'Python'}",
            f"- **Total Files**: {proj.file_count or 0}",
            f"- **Total Lines**: {proj.total_lines or 0}",
            "",
        ]

        # Module catalog -- group units by file
        modules: Dict[str, Dict[str, int]] = defaultdict(lambda: {
            "classes": 0, "functions": 0, "methods": 0, "lines": 0,
        })
        for u in unit_rows:
            fp = u.file_path or ""
            ut = u.unit_type or ""
            if ut == "class":
                modules[fp]["classes"] += 1
            elif ut == "function":
                modules[fp]["functions"] += 1
            elif ut == "method":
                modules[fp]["methods"] += 1

        lines += [
            "## Module Catalog\n",
            "| Module | Path | Classes | Functions | Methods |",
            "|--------|------|---------|-----------|---------|",
        ]
        for fp in sorted(modules.keys()):
            mod_name = _module_from_path(fp)
            m = modules[fp]
            lines.append(
                f"| {mod_name} | `{fp}` | {m['classes']} | {m['functions']} | {m['methods']} |"
            )
        lines.append("")

        # Package structure tree
        dirs: Set[str] = set()
        for fp in sorted(modules.keys()):
            parts = fp.replace("\\", "/").split("/")
            for i in range(1, len(parts)):
                dirs.add("/".join(parts[:i]))

        if dirs:
            lines.append("## Package Structure\n")
            lines.append("```")
            for d in sorted(dirs):
                depth = d.count("/")
                name = d.split("/")[-1]
                indent = "    " * depth
                prefix = "├── " if depth > 0 else ""
                lines.append(f"{indent}{prefix}{name}/")
            lines.append("```")
            lines.append("")

        # Language breakdown
        lines += [
            "## Language Breakdown\n",
            "| Language | Files |",
            "|----------|-------|",
        ]
        for row in lang_rows:
            lines.append(f"| {row.language or 'unknown'} | {row.cnt} |")
        lines.append("")

        # ASG edge summary
        lines += [
            "## ASG Relationship Summary\n",
            "| Edge Type | Count |",
            "|-----------|-------|",
        ]
        total_edges = 0
        for row in edge_rows:
            lines.append(f"| {row.edge_type} | {row.cnt} |")
            total_edges += row.cnt
        lines.append(f"| **Total** | **{total_edges}** |")
        lines.append("")

        # Processing pattern classification
        lines.append("## Processing Pattern Classification\n")
        pattern_counts: Dict[str, List[str]] = defaultdict(list)
        for u in unit_rows:
            meta = _safe_meta(u)
            name = u.name or ""
            modifiers = meta.get("modifiers", [])

            if any("route" in str(d).lower() or "get" in str(d).lower() or "post" in str(d).lower()
                   for d in modifiers):
                pattern_counts["Request-Response"].append(name)
            elif "worker" in name.lower() or "task" in name.lower():
                pattern_counts["Background Worker"].append(name)
            elif "pipeline" in name.lower() or "pipe" in name.lower():
                pattern_counts["Pipeline"].append(name)
            elif "agent" in name.lower() or "loop" in name.lower():
                pattern_counts["Agentic"].append(name)

        if pattern_counts:
            for pattern, names in sorted(pattern_counts.items()):
                lines.append(f"- **{pattern}** ({len(names)}): {', '.join(sorted(names)[:10])}")
        else:
            lines.append(f"- **{architecture}**: Standard {framework} application pattern")
        lines.append("")

    except Exception as e:
        logger.error("Python chapter 1 generation failed: %s", e, exc_info=True)
        lines.append(f"\n*Generation error: {e}*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chapter 3 Override: API Endpoints & Entry Points
# ---------------------------------------------------------------------------


def generate_api_endpoints(
    db: DatabaseManager, project_id: str, pipeline: Any = None,
) -> str:
    """Map all API routes, background workers, MCP tools, and CLI entry points."""
    pid = _pid(project_id)
    lines = [f"# 3. API Endpoints & Entry Points\n"]

    try:
        entry_points = _fetch_entry_points(db, pid)
        functions = _fetch_functions(db, pid)
        classes = _fetch_classes(db, pid)

        # Also scan source for route decorators that might not be in entry_points
        with db.get_session() as session:
            route_units = session.execute(
                text("""
                    SELECT u.name, u.qualified_name, u.signature, u.source,
                           u.metadata, f.file_path, u.start_line
                    FROM code_units u
                    JOIN code_files f ON u.file_id = f.file_id
                    WHERE u.project_id = :pid
                      AND u.unit_type IN ('function', 'method')
                    ORDER BY f.file_path, u.start_line
                """),
                {"pid": pid},
            ).fetchall()

        # ----- REST API Routes -----
        rest_routes: List[Dict[str, str]] = []
        for ru in route_units:
            source = _safe_source(ru)
            meta = _safe_meta(ru)
            modifiers = meta.get("modifiers", [])

            routes = _extract_route_info(source)
            if not routes:
                # Check modifiers for route decorators
                for mod in modifiers:
                    if isinstance(mod, str) and re.search(
                        r"(get|post|put|delete|patch)\s*\(", mod, re.IGNORECASE
                    ):
                        route_match = re.search(r"[\"']([^\"']+)[\"']", mod)
                        method_match = re.search(r"(get|post|put|delete|patch)", mod, re.IGNORECASE)
                        if route_match and method_match:
                            routes.append({
                                "path": route_match.group(1),
                                "method": method_match.group(1).upper(),
                            })

            for route in routes:
                # Determine auth requirement from decorators
                auth = "Unknown"
                for mod in modifiers:
                    mod_str = str(mod).lower()
                    if "depends" in mod_str and "auth" in mod_str:
                        auth = "Required"
                        break
                    if "public" in mod_str or "no_auth" in mod_str:
                        auth = "Public"
                        break

                rest_routes.append({
                    "method": route["method"],
                    "path": route["path"],
                    "handler": ru.name or "",
                    "file": ru.file_path or "",
                    "auth": auth,
                    "description": (ru.docstring or "")[:80],
                })

        if rest_routes:
            lines += [
                "## REST API Routes\n",
                "| Method | Path | Handler | File | Auth | Description |",
                "|--------|------|---------|------|------|-------------|",
            ]
            for r in rest_routes:
                lines.append(
                    f"| {r['method']} | `{r['path']}` | `{r['handler']}()` "
                    f"| `{r['file']}` | {r['auth']} | {r['description']} |"
                )
            lines.append("")

        # ----- Entry Points from understanding engine -----
        if entry_points:
            ep_by_type: Dict[str, list] = defaultdict(list)
            for ep in entry_points:
                ep_by_type[ep.entry_type or "unknown"].append(ep)

            for ep_type, eps in sorted(ep_by_type.items()):
                if ep_type in ("http_endpoint",) and rest_routes:
                    continue  # Already covered above
                lines += [
                    f"## {ep_type.replace('_', ' ').title()} Entry Points\n",
                    "| Name | Module | File | Confidence |",
                    "|------|--------|------|------------|",
                ]
                for ep in eps:
                    lines.append(
                        f"| `{ep.name}` | `{ep.qualified_name or ''}` "
                        f"| `{ep.file_path}` | {ep.confidence or 0:.0%} |"
                    )
                lines.append("")

        # ----- Background Workers -----
        workers = []
        for u in route_units:
            name = (u.name or "").lower()
            if any(kw in name for kw in ("worker", "task", "celery", "cron", "scheduler", "job")):
                workers.append(u)

        if workers:
            lines += [
                "## Background Workers\n",
                "| Worker | Module | File | Description |",
                "|--------|--------|------|-------------|",
            ]
            for w in workers:
                doc = (w.docstring or "")[:80] if hasattr(w, "docstring") else ""
                lines.append(
                    f"| `{w.name}` | `{w.qualified_name or ''}` "
                    f"| `{w.file_path}` | {doc} |"
                )
            lines.append("")

        # ----- MCP Tools -----
        mcp_tools = []
        for u in route_units:
            source = _safe_source(u)
            if "@mcp" in source.lower() or "tool" in (u.name or "").lower():
                mcp_tools.append(u)

        if mcp_tools:
            lines += [
                "## MCP Tools\n",
                "| Tool | Handler | File | Description |",
                "|------|---------|------|-------------|",
            ]
            for t in mcp_tools:
                doc = (t.docstring or "")[:80] if hasattr(t, "docstring") else ""
                lines.append(
                    f"| `{t.name}` | `{t.qualified_name or ''}()` "
                    f"| `{t.file_path}` | {doc} |"
                )
            lines.append("")

        # ----- CLI Entry Points -----
        cli_entries = []
        for u in route_units:
            source = _safe_source(u)
            if any(kw in source for kw in ("argparse", "click.command", "typer", "__main__")):
                cli_entries.append(u)

        if cli_entries:
            lines += [
                "## CLI Entry Points\n",
                "| Command | Module | File | Description |",
                "|---------|--------|------|-------------|",
            ]
            for c in cli_entries:
                doc = (c.docstring or "")[:80] if hasattr(c, "docstring") else ""
                lines.append(
                    f"| `{c.name}` | `{c.qualified_name or ''}` "
                    f"| `{c.file_path}` | {doc} |"
                )
            lines.append("")

        # Summary
        lines += [
            "## Summary\n",
            f"- **REST Routes**: {len(rest_routes)}",
            f"- **Entry Points**: {len(entry_points)}",
            f"- **Background Workers**: {len(workers)}",
            f"- **MCP Tools**: {len(mcp_tools)}",
            f"- **CLI Commands**: {len(cli_entries)}",
            "",
        ]

    except Exception as e:
        logger.error("Python chapter 3 generation failed: %s", e, exc_info=True)
        lines.append(f"\n*Generation error: {e}*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chapter 4 Override: Per-Module Functional Specifications
# ---------------------------------------------------------------------------


def generate_per_module_specs(
    db: DatabaseManager, project_id: str, pipeline: Any = None,
) -> str:
    """Per-class and per-module functional specifications with method signatures,
    business logic summaries, dependency maps, and configuration keys."""
    pid = _pid(project_id)
    lines = [f"# 4. Functional Specifications\n"]

    try:
        classes = _fetch_classes(db, pid)
        methods = _fetch_methods(db, pid)
        analyses = _fetch_deep_analyses(db, pid)

        # Build analysis lookup
        analysis_by_name: Dict[str, dict] = {}
        for a in analyses:
            rj = a.result_json if isinstance(a.result_json, dict) else {}
            if isinstance(rj, str):
                try:
                    rj = json.loads(rj)
                except Exception:
                    rj = {}
            analysis_by_name[a.entry_name] = {
                "result_json": rj,
                "narrative": a.narrative or "",
            }

        # Index methods by class name
        methods_by_class: Dict[str, list] = defaultdict(list)
        for m in methods:
            methods_by_class[m.class_name].append(m)

        # Fetch call edges for dependency analysis
        calls = _fetch_calls(db, pid)
        callee_by_caller: Dict[str, Set[str]] = defaultdict(set)
        for c in calls:
            callee_by_caller[c.caller_qn or c.caller].add(c.callee_qn or c.callee)

        # Generate spec per class
        for cls in classes:
            source = _safe_source(cls)
            meta = _safe_meta(cls)
            cls_name = cls.name
            qn = cls.qualified_name or cls_name

            lines.append(f"## {qn}\n")

            # Purpose from docstring or deep analysis
            docstring = cls.docstring or ""
            ainfo = analysis_by_name.get(cls_name, {})
            narrative = ainfo.get("narrative", "")

            purpose = docstring.split("\n")[0] if docstring else ""
            if not purpose and narrative:
                purpose = narrative[:200]
            if purpose:
                lines += [
                    "### Purpose\n",
                    purpose,
                    "",
                ]

            # Decorators
            decorators = _extract_decorators(source)
            if decorators:
                lines.append("### Decorators\n")
                for d in decorators[:10]:
                    lines.append(f"- `{d.strip()}`")
                lines.append("")

            # Method table
            cls_methods = methods_by_class.get(cls_name, [])
            if cls_methods:
                lines += [
                    f"### Class: {cls_name}\n",
                    "| Method | Signature | Async | Description |",
                    "|--------|-----------|-------|-------------|",
                ]
                for meth in cls_methods:
                    m_meta = _safe_meta(meth)
                    is_async = "Yes" if m_meta.get("is_async") else "No"
                    sig = (meth.signature or "")[:60]
                    doc = (meth.docstring or "")[:60] if hasattr(meth, "docstring") else ""
                    lines.append(
                        f"| `{meth.name}` | `{sig}` | {is_async} | {doc} |"
                    )
                lines.append("")

            # Key business logic from deep analysis
            rj = ainfo.get("result_json", {})
            business_rules = rj.get("business_rules", [])
            if business_rules:
                lines += ["### Key Business Logic\n"]
                for rule in business_rules[:10]:
                    if isinstance(rule, dict):
                        desc = rule.get("description", rule.get("rule", str(rule)))
                    else:
                        desc = str(rule)
                    lines.append(f"- {desc[:150]}")
                lines.append("")

            # Dependencies (what this class calls)
            cls_deps: Set[str] = set()
            for meth in cls_methods:
                meth_qn = meth.qualified_name or meth.name
                cls_deps.update(callee_by_caller.get(meth_qn, set()))
            # Also check class-level calls
            cls_deps.update(callee_by_caller.get(qn, set()))

            if cls_deps:
                lines += [
                    "### Dependencies\n",
                    "| Dependency | Type | Purpose |",
                    "|-----------|------|---------|",
                ]
                for dep in sorted(cls_deps)[:15]:
                    dep_type = "Called"
                    lines.append(f"| `{dep}` | {dep_type} | -- |")
                lines.append("")

            # Configuration / env vars
            env_vars = _extract_env_vars(source)
            config_keys = _extract_config_keys(source)
            if env_vars or config_keys:
                lines.append("### Configuration\n")
                if env_vars:
                    lines += [
                        "| Env Variable | Source |",
                        "|-------------|--------|",
                    ]
                    for ev in env_vars[:10]:
                        lines.append(f"| `{ev}` | `os.environ` / `os.getenv` |")
                    lines.append("")
                if config_keys:
                    for ck in config_keys[:10]:
                        lines.append(f"- Config key: `{ck}`")
                    lines.append("")

            lines.append("---\n")

        # If no classes, show top-level functions grouped by file
        if not classes:
            functions = _fetch_functions(db, pid)
            funcs_by_file: Dict[str, list] = defaultdict(list)
            for f in functions:
                funcs_by_file[f.file_path].append(f)

            for fp in sorted(funcs_by_file.keys()):
                mod_name = _module_from_path(fp)
                lines.append(f"## {mod_name}\n")
                lines += [
                    "| Function | Signature | Async | Description |",
                    "|----------|-----------|-------|-------------|",
                ]
                for func in funcs_by_file[fp]:
                    f_meta = _safe_meta(func)
                    is_async = "Yes" if f_meta.get("is_async") else "No"
                    sig = (func.signature or "")[:60]
                    doc = (func.docstring or "")[:60]
                    lines.append(
                        f"| `{func.name}` | `{sig}` | {is_async} | {doc} |"
                    )
                lines.append("")

    except Exception as e:
        logger.error("Python chapter 4 generation failed: %s", e, exc_info=True)
        lines.append(f"\n*Generation error: {e}*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chapter 5 Override: Data Models & ORM Schema
# ---------------------------------------------------------------------------


def generate_python_data_models(
    db: DatabaseManager, project_id: str, pipeline: Any = None,
) -> str:
    """Catalog SQLAlchemy models, Pydantic schemas, dataclasses, and typed dicts."""
    pid = _pid(project_id)
    lines = [f"# 5. Data Models & Schema\n"]

    try:
        classes = _fetch_classes(db, pid)
        methods = _fetch_methods(db, pid)

        # Classify classes by base class / decorator patterns
        orm_models: List[Any] = []
        pydantic_schemas: List[Any] = []
        dataclasses_list: List[Any] = []
        other_classes: List[Any] = []

        for cls in classes:
            source = _safe_source(cls)
            meta = _safe_meta(cls)
            sig = cls.signature or ""
            decorators = _extract_decorators(source)
            decorator_text = " ".join(decorators).lower()

            if any(base in sig for base in ("Base", "DeclarativeBase", "Model")) and \
               any(kw in source for kw in ("Column(", "mapped_column(", "__tablename__")):
                orm_models.append(cls)
            elif any(base in sig for base in ("BaseModel", "BaseSchema")) or \
                 "pydantic" in source.lower():
                pydantic_schemas.append(cls)
            elif "dataclass" in decorator_text:
                dataclasses_list.append(cls)
            else:
                other_classes.append(cls)

        # ----- ORM Models -----
        if orm_models:
            lines += [
                "## SQLAlchemy / ORM Models\n",
                "| Model | Table | File | Fields | Relationships |",
                "|-------|-------|------|--------|---------------|",
            ]
            for cls in orm_models:
                source = _safe_source(cls)
                meta = _safe_meta(cls)

                # Extract __tablename__
                table_match = re.search(r"__tablename__\s*=\s*[\"'](\w+)[\"']", source)
                table_name = table_match.group(1) if table_match else "?"

                # Count columns and relationships
                col_count = len(re.findall(r"(?:Column|mapped_column)\s*\(", source))
                rel_count = len(re.findall(r"relationship\s*\(", source))

                # Get class fields from enricher metadata
                fields = meta.get("fields", [])
                if fields and isinstance(fields, list):
                    col_count = max(col_count, len(fields))

                lines.append(
                    f"| `{cls.name}` | `{table_name}` | `{cls.file_path}` "
                    f"| {col_count} | {rel_count} |"
                )
            lines.append("")

            # Detailed model sections
            for cls in orm_models[:20]:
                source = _safe_source(cls)
                meta = _safe_meta(cls)
                fields = meta.get("fields", [])

                lines.append(f"### {cls.name}\n")

                if fields and isinstance(fields, list):
                    lines += [
                        "| Column | Type | Nullable | Description |",
                        "|--------|------|----------|-------------|",
                    ]
                    for f in fields[:30]:
                        if isinstance(f, dict):
                            fname = f.get("name", "?")
                            ftype = f.get("type", "?")
                            lines.append(f"| `{fname}` | `{ftype}` | -- | -- |")
                        elif isinstance(f, str):
                            lines.append(f"| `{f}` | -- | -- | -- |")
                    lines.append("")
                else:
                    # Fallback: parse Column() calls
                    col_matches = re.findall(
                        r"(\w+)\s*=\s*(?:Column|mapped_column)\s*\(([^)]*)\)",
                        source,
                    )
                    if col_matches:
                        lines += [
                            "| Column | Definition |",
                            "|--------|-----------|",
                        ]
                        for cname, cdef in col_matches[:30]:
                            lines.append(f"| `{cname}` | `{cdef[:60]}` |")
                        lines.append("")

        # ----- Pydantic Schemas -----
        if pydantic_schemas:
            lines += [
                "## Pydantic / DTO Schemas\n",
                "| Schema | File | Fields | Usage |",
                "|--------|------|--------|-------|",
            ]
            for cls in pydantic_schemas:
                meta = _safe_meta(cls)
                fields = meta.get("fields", [])
                field_count = len(fields) if isinstance(fields, list) else 0
                doc = (cls.docstring or "")[:60]
                lines.append(
                    f"| `{cls.name}` | `{cls.file_path}` | {field_count} | {doc} |"
                )
            lines.append("")

        # ----- Dataclasses -----
        if dataclasses_list:
            lines += [
                "## Dataclasses\n",
                "| Class | File | Fields | Description |",
                "|-------|------|--------|-------------|",
            ]
            for cls in dataclasses_list:
                meta = _safe_meta(cls)
                fields = meta.get("fields", [])
                field_count = len(fields) if isinstance(fields, list) else 0
                doc = (cls.docstring or "")[:60]
                lines.append(
                    f"| `{cls.name}` | `{cls.file_path}` | {field_count} | {doc} |"
                )
            lines.append("")

        # Summary
        lines += [
            "## Summary\n",
            f"- **ORM Models**: {len(orm_models)}",
            f"- **Pydantic Schemas**: {len(pydantic_schemas)}",
            f"- **Dataclasses**: {len(dataclasses_list)}",
            f"- **Other Classes**: {len(other_classes)}",
            "",
        ]

    except Exception as e:
        logger.error("Python chapter 5 generation failed: %s", e, exc_info=True)
        lines.append(f"\n*Generation error: {e}*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chapter 6 Override: Database Access Patterns
# ---------------------------------------------------------------------------


def generate_database_patterns(
    db: DatabaseManager, project_id: str, pipeline: Any = None,
) -> str:
    """Classify ORM usage, raw SQL patterns, connection management,
    and transaction patterns."""
    pid = _pid(project_id)
    lines = [f"# 6. Database Access Patterns\n"]

    try:
        with db.get_session() as session:
            # Fetch all units with source for SQL pattern scanning
            all_units = session.execute(
                text("""
                    SELECT u.name, u.qualified_name, u.source, u.unit_type,
                           u.metadata, f.file_path
                    FROM code_units u
                    JOIN code_files f ON u.file_id = f.file_id
                    WHERE u.project_id = :pid
                      AND u.unit_type IN ('function', 'method', 'class')
                    ORDER BY f.file_path, u.start_line
                """),
                {"pid": pid},
            ).fetchall()

        # Classify access patterns
        orm_usage: List[Dict[str, str]] = []
        raw_sql_usage: List[Dict[str, str]] = []
        session_patterns: List[Dict[str, str]] = []

        for unit in all_units:
            source = _safe_source(unit)
            if not source:
                continue

            name = unit.name or ""
            qn = unit.qualified_name or name
            fp = unit.file_path or ""

            # ORM pattern detection
            orm_ops = []
            if "session.add(" in source or ".add(" in source:
                orm_ops.append("CREATE")
            if "session.query(" in source or ".query(" in source or "select(" in source:
                orm_ops.append("READ")
            if "session.merge(" in source or ".merge(" in source:
                orm_ops.append("UPDATE")
            if "session.delete(" in source or ".delete(" in source:
                orm_ops.append("DELETE")
            if "session.execute(" in source:
                orm_ops.append("EXECUTE")
            if "bulk_insert" in source or "bulk_save" in source:
                orm_ops.append("BULK")

            if orm_ops:
                orm_usage.append({
                    "name": qn,
                    "file": fp,
                    "operations": ", ".join(sorted(set(orm_ops))),
                    "pattern": "Session-based",
                })

            # Raw SQL detection
            sql_patterns = re.findall(
                r"text\s*\(\s*[\"']{1,3}(.*?)[\"']{1,3}\s*\)",
                source,
                re.DOTALL,
            )
            for sql in sql_patterns:
                sql_upper = sql.strip().upper()
                query_type = "UNKNOWN"
                if sql_upper.startswith("SELECT"):
                    query_type = "SELECT"
                elif sql_upper.startswith("INSERT"):
                    query_type = "INSERT"
                elif sql_upper.startswith("UPDATE"):
                    query_type = "UPDATE"
                elif sql_upper.startswith("DELETE"):
                    query_type = "DELETE"
                elif "WITH" in sql_upper:
                    query_type = "CTE"

                purpose = sql.strip()[:80].replace("\n", " ")
                raw_sql_usage.append({
                    "location": f"{qn} (`{fp}`)",
                    "query_type": query_type,
                    "purpose": purpose,
                })

            # Session/connection management
            if "get_session" in source or "session_scope" in source or "SessionLocal" in source:
                session_patterns.append({
                    "name": qn,
                    "file": fp,
                    "pattern": "Context manager" if "with" in source else "Manual",
                })

        # ----- ORM Usage Table -----
        if orm_usage:
            lines += [
                "## ORM Usage\n",
                "| Module | Operations | File | Pattern |",
                "|--------|-----------|------|---------|",
            ]
            seen = set()
            for ou in orm_usage:
                key = ou["name"]
                if key in seen:
                    continue
                seen.add(key)
                lines.append(
                    f"| `{ou['name']}` | {ou['operations']} | `{ou['file']}` | {ou['pattern']} |"
                )
            lines.append("")

        # ----- Raw SQL Usage -----
        if raw_sql_usage:
            lines += [
                "## Raw SQL Usage\n",
                "| Location | Query Type | Purpose |",
                "|----------|-----------|---------|",
            ]
            for rs in raw_sql_usage[:50]:
                lines.append(
                    f"| {rs['location']} | {rs['query_type']} | {rs['purpose']} |"
                )
            lines.append("")

        # ----- Connection Management -----
        if session_patterns:
            lines += [
                "## Connection Management\n",
                "| Module | File | Pattern |",
                "|--------|------|---------|",
            ]
            seen_sp = set()
            for sp in session_patterns:
                key = sp["name"]
                if key in seen_sp:
                    continue
                seen_sp.add(key)
                lines.append(
                    f"| `{sp['name']}` | `{sp['file']}` | {sp['pattern']} |"
                )
            lines.append("")

        # Summary
        lines += [
            "## Summary\n",
            f"- **ORM Access Points**: {len(orm_usage)}",
            f"- **Raw SQL Queries**: {len(raw_sql_usage)}",
            f"- **Session Management Points**: {len(session_patterns)}",
            "",
        ]

    except Exception as e:
        logger.error("Python chapter 6 generation failed: %s", e, exc_info=True)
        lines.append(f"\n*Generation error: {e}*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chapter 7 Override: Call Graphs & Data Flow
# ---------------------------------------------------------------------------


def generate_python_call_graphs(
    db: DatabaseManager, project_id: str, pipeline: Any = None,
) -> str:
    """Show class-method call hierarchy, request flows, pipeline flows,
    and agentic loop patterns."""
    pid = _pid(project_id)
    lines = [f"# 7. Call Graphs & Data Flow\n"]

    try:
        calls = _fetch_calls(db, pid)
        imports = _fetch_imports(db, pid)
        analyses = _fetch_deep_analyses(db, pid)

        # Build call graph indexes
        callers_of: Dict[str, List[str]] = defaultdict(list)
        callees_of: Dict[str, List[str]] = defaultdict(list)
        for c in calls:
            caller = c.caller_qn or c.caller
            callee = c.callee_qn or c.callee
            callers_of[callee].append(caller)
            callees_of[caller].append(callee)

        # ----- Top-level call graph summary -----
        lines += [
            "## Call Graph Overview\n",
            f"- **Total call edges**: {len(calls)}",
            f"- **Total import edges**: {len(imports)}",
            f"- **Unique callers**: {len(callees_of)}",
            f"- **Unique callees**: {len(callers_of)}",
            "",
        ]

        # ----- Fan-in analysis (most-called functions) -----
        fan_in = sorted(callers_of.items(), key=lambda x: len(x[1]), reverse=True)
        if fan_in:
            lines += [
                "## Most-Called Functions (Fan-In)\n",
                "| Function | Called By (count) | Callers |",
                "|----------|-------------------|---------|",
            ]
            for callee, caller_list in fan_in[:15]:
                callers_str = ", ".join(sorted(set(caller_list))[:5])
                if len(set(caller_list)) > 5:
                    callers_str += f" (+{len(set(caller_list)) - 5} more)"
                lines.append(
                    f"| `{callee}` | {len(set(caller_list))} | {callers_str} |"
                )
            lines.append("")

        # ----- Fan-out analysis (functions calling the most others) -----
        fan_out = sorted(callees_of.items(), key=lambda x: len(x[1]), reverse=True)
        if fan_out:
            lines += [
                "## Highest Fan-Out (Orchestrator Functions)\n",
                "| Function | Calls (count) | Callees |",
                "|----------|---------------|---------|",
            ]
            for caller, callee_list in fan_out[:15]:
                callees_str = ", ".join(sorted(set(callee_list))[:5])
                if len(set(callee_list)) > 5:
                    callees_str += f" (+{len(set(callee_list)) - 5} more)"
                lines.append(
                    f"| `{caller}` | {len(set(callee_list))} | {callees_str} |"
                )
            lines.append("")

        # ----- Key Data Flows from deep analyses -----
        if analyses:
            lines.append("## Key Data Flows\n")
            for a in analyses[:10]:
                rj = a.result_json if isinstance(a.result_json, dict) else {}
                if isinstance(rj, str):
                    try:
                        rj = json.loads(rj)
                    except Exception:
                        rj = {}

                call_chain = rj.get("call_chain", [])
                if call_chain:
                    lines.append(f"### {a.entry_name}\n")
                    chain_str = " -> ".join(
                        str(c.get("name", c) if isinstance(c, dict) else c)
                        for c in call_chain[:10]
                    )
                    lines.append(f"```\n{chain_str}\n```\n")

        # ----- Import dependency graph -----
        if imports:
            import_by_source: Dict[str, List[str]] = defaultdict(list)
            for imp in imports:
                import_by_source[imp.from_file].append(imp.imported)

            lines += [
                "## Import Dependency Graph\n",
                "| Source Module | Imports |",
                "|--------------|---------|",
            ]
            for src_file in sorted(import_by_source.keys())[:30]:
                imported = ", ".join(sorted(set(import_by_source[src_file]))[:8])
                lines.append(f"| `{src_file}` | {imported} |")
            lines.append("")

    except Exception as e:
        logger.error("Python chapter 7 generation failed: %s", e, exc_info=True)
        lines.append(f"\n*Generation error: {e}*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chapter 8 Override: Dependencies & Integration Architecture
# ---------------------------------------------------------------------------


def generate_python_dependencies(
    db: DatabaseManager, project_id: str, pipeline: Any = None,
) -> str:
    """Catalog pip packages, external service dependencies, environment variables,
    and integration points."""
    pid = _pid(project_id)
    lines = [f"# 8. Dependencies & Integration Architecture\n"]

    try:
        with db.get_session() as session:
            # Fetch all source for pattern scanning
            all_units = session.execute(
                text("""
                    SELECT u.name, u.source, u.metadata, f.file_path
                    FROM code_units u
                    JOIN code_files f ON u.file_id = f.file_id
                    WHERE u.project_id = :pid
                      AND u.unit_type IN ('function', 'method', 'class', 'module')
                    ORDER BY f.file_path
                """),
                {"pid": pid},
            ).fetchall()

            # Fetch import edges for dependency tracking
            import_units = session.execute(
                text("""
                    SELECT DISTINCT tu.name AS imported_name
                    FROM code_edges e
                    JOIN code_units tu ON e.target_unit_id = tu.unit_id
                    WHERE e.project_id = :pid AND e.edge_type = 'imports'
                    ORDER BY tu.name
                """),
                {"pid": pid},
            ).fetchall()

        # Scan all source for external service patterns
        all_source = "\n".join(_safe_source(u) for u in all_units)

        # ----- External Service Dependencies -----
        services: List[Dict[str, str]] = []

        service_patterns = [
            (r"postgresql|psycopg|asyncpg", "PostgreSQL", "Primary data store"),
            (r"pgvector|PGVectorStore", "pgvector", "Vector similarity search"),
            (r"redis|Redis\(", "Redis", "Caching / message broker"),
            (r"openai|OpenAI", "OpenAI API", "LLM / Embeddings"),
            (r"anthropic|Anthropic", "Anthropic API", "LLM provider"),
            (r"requests\.(?:get|post|put|delete)", "HTTP Client", "External API calls"),
            (r"httpx|aiohttp", "Async HTTP Client", "Async external API calls"),
            (r"boto3|aws|s3", "AWS", "Cloud services"),
            (r"celery|Celery", "Celery", "Task queue"),
            (r"rabbitmq|pika", "RabbitMQ", "Message broker"),
            (r"kafka|confluent", "Kafka", "Event streaming"),
            (r"plantuml|PlantUML", "PlantUML", "Diagram rendering"),
            (r"smtp|email\.mime", "SMTP", "Email service"),
            (r"subprocess|Popen", "Subprocess", "External process execution"),
        ]

        for pattern, service_name, purpose in service_patterns:
            if re.search(pattern, all_source, re.IGNORECASE):
                # Find which files reference it
                files = set()
                for u in all_units:
                    src = _safe_source(u)
                    if re.search(pattern, src, re.IGNORECASE):
                        files.add(u.file_path)

                services.append({
                    "service": service_name,
                    "files": ", ".join(sorted(files)[:3]),
                    "purpose": purpose,
                })

        if services:
            lines += [
                "## External Service Dependencies\n",
                "| Service | Used In | Purpose |",
                "|---------|---------|---------|",
            ]
            for s in services:
                lines.append(
                    f"| {s['service']} | `{s['files']}` | {s['purpose']} |"
                )
            lines.append("")

        # ----- Python Package Dependencies (from imports) -----
        # Identify third-party packages from import edges
        stdlib_prefixes = {
            "os", "sys", "re", "json", "logging", "typing", "datetime",
            "collections", "pathlib", "uuid", "hashlib", "io", "abc",
            "functools", "itertools", "contextlib", "copy", "math",
            "dataclasses", "enum", "threading", "asyncio", "subprocess",
            "tempfile", "shutil", "time", "traceback", "unittest",
        }
        third_party: Set[str] = set()
        for imp in import_units:
            name = imp.imported_name or ""
            top_pkg = name.split(".")[0]
            if top_pkg and top_pkg not in stdlib_prefixes:
                third_party.add(top_pkg)

        if third_party:
            lines += [
                "## Python Package Dependencies\n",
                "| Package | Category |",
                "|---------|----------|",
            ]
            pkg_categories = {
                "fastapi": "Web Framework",
                "flask": "Web Framework",
                "django": "Web Framework",
                "sqlalchemy": "ORM",
                "alembic": "Migrations",
                "pydantic": "Validation",
                "llama_index": "RAG Framework",
                "tree_sitter": "AST Parsing",
                "uvicorn": "ASGI Server",
                "gunicorn": "WSGI Server",
                "celery": "Task Queue",
                "redis": "Cache/Broker",
                "boto3": "AWS SDK",
                "pytest": "Testing",
                "httpx": "HTTP Client",
                "numpy": "Numerical Computing",
                "pandas": "Data Processing",
                "torch": "Machine Learning",
                "transformers": "NLP Models",
            }
            for pkg in sorted(third_party):
                category = pkg_categories.get(pkg, "Library")
                lines.append(f"| `{pkg}` | {category} |")
            lines.append("")

        # ----- Environment Variables -----
        all_env_vars: Dict[str, Set[str]] = defaultdict(set)
        for u in all_units:
            source = _safe_source(u)
            for ev in _extract_env_vars(source):
                all_env_vars[ev].add(u.file_path or "")

        if all_env_vars:
            lines += [
                "## Environment Variables\n",
                "| Variable | Referenced In | Required |",
                "|----------|--------------|----------|",
            ]
            for var in sorted(all_env_vars.keys()):
                files = ", ".join(sorted(all_env_vars[var])[:3])
                # Heuristic: if getenv has a default, it's optional
                lines.append(
                    f"| `{var}` | `{files}` | -- |"
                )
            lines.append("")

        # ----- Integration Points from deep analyses -----
        analyses = _fetch_deep_analyses(db, pid)
        integrations: List[Dict[str, str]] = []
        for a in analyses:
            rj = a.result_json if isinstance(a.result_json, dict) else {}
            if isinstance(rj, str):
                try:
                    rj = json.loads(rj)
                except Exception:
                    rj = {}
            for integ in rj.get("integrations", []):
                if isinstance(integ, dict):
                    integrations.append({
                        "name": integ.get("name", "?"),
                        "type": integ.get("type", "?"),
                        "description": integ.get("description", "")[:80],
                        "source": a.entry_name,
                    })

        if integrations:
            lines += [
                "## Integration Points (from Deep Analysis)\n",
                "| Integration | Type | Description | Detected In |",
                "|-------------|------|-------------|-------------|",
            ]
            seen_integ = set()
            for integ in integrations:
                key = integ["name"]
                if key in seen_integ:
                    continue
                seen_integ.add(key)
                lines.append(
                    f"| {integ['name']} | {integ['type']} "
                    f"| {integ['description']} | `{integ['source']}` |"
                )
            lines.append("")

        # Summary
        lines += [
            "## Summary\n",
            f"- **External Services**: {len(services)}",
            f"- **Third-Party Packages**: {len(third_party)}",
            f"- **Environment Variables**: {len(all_env_vars)}",
            f"- **Integration Points**: {len(integrations)}",
            "",
        ]

    except Exception as e:
        logger.error("Python chapter 8 generation failed: %s", e, exc_info=True)
        lines.append(f"\n*Generation error: {e}*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chapter 9 Override: Error Handling & Resilience
# ---------------------------------------------------------------------------


def generate_python_error_handling(
    db: DatabaseManager, project_id: str, pipeline: Any = None,
) -> str:
    """Catalog exception hierarchy, try/except patterns, retry patterns,
    and graceful degradation strategies."""
    pid = _pid(project_id)
    lines = [f"# 9. Error Handling & Resilience\n"]

    try:
        with db.get_session() as session:
            # Fetch all units with source
            all_units = session.execute(
                text("""
                    SELECT u.name, u.qualified_name, u.source, u.unit_type,
                           u.signature, u.metadata, f.file_path
                    FROM code_units u
                    JOIN code_files f ON u.file_id = f.file_id
                    WHERE u.project_id = :pid
                      AND u.unit_type IN ('function', 'method', 'class')
                    ORDER BY f.file_path, u.start_line
                """),
                {"pid": pid},
            ).fetchall()

        # ----- Custom Exception Classes -----
        custom_exceptions: List[Dict[str, str]] = []
        for u in all_units:
            if u.unit_type != "class":
                continue
            sig = u.signature or ""
            name = u.name or ""
            if any(base in sig for base in ("Exception", "Error", "BaseException")):
                custom_exceptions.append({
                    "name": name,
                    "base": sig,
                    "file": u.file_path or "",
                    "docstring": (u.source or "")[:100] if not hasattr(u, "docstring") else "",
                })

        if custom_exceptions:
            lines += [
                "## Custom Exception Hierarchy\n",
                "| Exception | Base | File |",
                "|-----------|------|------|",
            ]
            for ce in custom_exceptions:
                lines.append(
                    f"| `{ce['name']}` | `{ce['base'][:60]}` | `{ce['file']}` |"
                )
            lines.append("")

        # ----- Exception Handler patterns -----
        handler_patterns: List[Dict[str, str]] = []
        for u in all_units:
            source = _safe_source(u)
            if not source:
                continue

            # Count try/except blocks
            try_count = len(re.findall(r"^\s*try\s*:", source, re.MULTILINE))
            except_types = re.findall(r"^\s*except\s+(\w[\w.]*)", source, re.MULTILINE)

            if try_count > 0:
                handler_patterns.append({
                    "name": u.qualified_name or u.name or "",
                    "file": u.file_path or "",
                    "try_blocks": str(try_count),
                    "exceptions_caught": ", ".join(sorted(set(except_types))[:5]),
                })

        if handler_patterns:
            lines += [
                "## Exception Handling Patterns\n",
                "| Function | Try Blocks | Exceptions Caught | File |",
                "|----------|-----------|-------------------|------|",
            ]
            # Show top handlers by try block count
            handler_patterns.sort(key=lambda x: int(x["try_blocks"]), reverse=True)
            for hp in handler_patterns[:25]:
                lines.append(
                    f"| `{hp['name']}` | {hp['try_blocks']} "
                    f"| {hp['exceptions_caught'] or 'bare except'} | `{hp['file']}` |"
                )
            lines.append("")

        # ----- Retry Patterns -----
        retry_patterns: List[Dict[str, str]] = []
        for u in all_units:
            source = _safe_source(u)
            if not source:
                continue

            has_retry = any(kw in source for kw in (
                "retry", "backoff", "max_retries", "tenacity", "exponential",
            ))
            if has_retry:
                # Extract retry config hints
                max_retries_match = re.search(r"max_retries\s*=\s*(\d+)", source)
                backoff_match = re.search(r"(?:base_delay|backoff)\s*=\s*([\d.]+)", source)
                retry_patterns.append({
                    "name": u.qualified_name or u.name or "",
                    "file": u.file_path or "",
                    "max_retries": max_retries_match.group(1) if max_retries_match else "?",
                    "backoff": backoff_match.group(1) if backoff_match else "?",
                })

        if retry_patterns:
            lines += [
                "## Retry Patterns\n",
                "| Component | Max Retries | Backoff | File |",
                "|-----------|-------------|---------|------|",
            ]
            for rp in retry_patterns:
                lines.append(
                    f"| `{rp['name']}` | {rp['max_retries']} "
                    f"| {rp['backoff']} | `{rp['file']}` |"
                )
            lines.append("")

        # ----- Graceful Degradation -----
        degradation_patterns: List[Dict[str, str]] = []
        for u in all_units:
            source = _safe_source(u)
            if not source:
                continue

            # Look for fallback patterns
            has_fallback = any(kw in source.lower() for kw in (
                "fallback", "graceful", "degrad", "best-effort", "best_effort",
                "continue", "skip", "disabled",
            ))
            if has_fallback and "except" in source:
                # Extract the fallback description from comments
                fallback_comments = re.findall(
                    r"#\s*(.*?(?:fallback|graceful|skip|degrad|best.effort).*)",
                    source, re.IGNORECASE,
                )
                desc = fallback_comments[0][:80] if fallback_comments else "Graceful degradation"
                degradation_patterns.append({
                    "name": u.qualified_name or u.name or "",
                    "file": u.file_path or "",
                    "description": desc,
                })

        if degradation_patterns:
            lines += [
                "## Graceful Degradation\n",
                "| Component | Fallback | File |",
                "|-----------|----------|------|",
            ]
            seen_deg = set()
            for dp in degradation_patterns:
                key = dp["name"]
                if key in seen_deg:
                    continue
                seen_deg.add(key)
                lines.append(
                    f"| `{dp['name']}` | {dp['description']} | `{dp['file']}` |"
                )
            lines.append("")

        # Summary
        lines += [
            "## Summary\n",
            f"- **Custom Exceptions**: {len(custom_exceptions)}",
            f"- **Functions with Try/Except**: {len(handler_patterns)}",
            f"- **Retry Patterns**: {len(retry_patterns)}",
            f"- **Graceful Degradation Points**: {len(degradation_patterns)}",
            "",
        ]

    except Exception as e:
        logger.error("Python chapter 9 generation failed: %s", e, exc_info=True)
        lines.append(f"\n*Generation error: {e}*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Override dispatch table
# ---------------------------------------------------------------------------

PYTHON_CHAPTER_OVERRIDES = {
    1: generate_module_inventory,
    3: generate_api_endpoints,
    4: generate_per_module_specs,
    5: generate_python_data_models,
    6: generate_database_patterns,
    7: generate_python_call_graphs,
    8: generate_python_dependencies,
    9: generate_python_error_handling,
}

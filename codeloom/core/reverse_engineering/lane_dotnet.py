"""C#/.NET-specific chapter overrides for reverse engineering documentation.

When the project is predominantly C# (.NET / ASP.NET Core), these generators
replace the generic chapters with deep, .NET-aware functional specifications:
  - Solution/project inventory with assembly and namespace mapping
  - Controller route mapping with attribute routing and auth info
  - Per-controller/service functional specs with DI analysis
  - EF Core entity models and DbContext catalogs
  - Database access pattern classification (EF Core vs raw SQL vs Dapper)
  - DI registrations + call graphs showing request pipeline flow
  - NuGet dependency inventories and framework references
  - Error handling middleware and exception filter patterns

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
    """Fetch import edges (using statements)."""
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


def _fetch_inherits(db: DatabaseManager, pid: UUID) -> list:
    """Fetch inheritance edges."""
    with db.get_session() as session:
        return session.execute(
            text("""
                SELECT su.name AS child, tu.name AS parent,
                       su.qualified_name AS child_qn, tu.qualified_name AS parent_qn,
                       sf.file_path AS child_file
                FROM code_edges e
                JOIN code_units su ON e.source_unit_id = su.unit_id
                JOIN code_units tu ON e.target_unit_id = tu.unit_id
                JOIN code_files sf ON su.file_id = sf.file_id
                WHERE e.project_id = :pid AND e.edge_type = 'inherits'
                ORDER BY su.name
            """),
            {"pid": pid},
        ).fetchall()


def _fetch_implements(db: DatabaseManager, pid: UUID) -> list:
    """Fetch implements edges (class implements interface)."""
    with db.get_session() as session:
        return session.execute(
            text("""
                SELECT su.name AS implementor, tu.name AS interface,
                       su.qualified_name AS impl_qn, tu.qualified_name AS iface_qn,
                       sf.file_path AS impl_file
                FROM code_edges e
                JOIN code_units su ON e.source_unit_id = su.unit_id
                JOIN code_units tu ON e.target_unit_id = tu.unit_id
                JOIN code_files sf ON su.file_id = sf.file_id
                WHERE e.project_id = :pid AND e.edge_type = 'implements'
                ORDER BY su.name
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


def _extract_attributes(source: str) -> List[str]:
    """Extract C# attribute decorations from source."""
    return re.findall(r"^\s*(\[[\w.]+(?:\([^]]*\))?\])", source, re.MULTILINE)


def _extract_route_info(source: str) -> List[Dict[str, str]]:
    """Extract ASP.NET Core route attributes with method and path."""
    routes: List[Dict[str, str]] = []
    # [HttpGet("path")], [HttpPost("path")], [Route("path")]
    for m in re.finditer(
        r"\[Http(Get|Post|Put|Delete|Patch|Options|Head)"
        r"(?:\s*\(\s*\"([^\"]*)\"\s*\))?\s*\]",
        source, re.IGNORECASE,
    ):
        method = m.group(1).upper()
        path = m.group(2) or ""
        routes.append({"path": path, "method": method})

    # [Route("...")] at class level
    for m in re.finditer(
        r"\[Route\s*\(\s*\"([^\"]+)\"\s*\)\]", source, re.IGNORECASE,
    ):
        if not any(r["path"] == m.group(1) for r in routes):
            routes.append({"path": m.group(1), "method": "ROUTE"})

    return routes


def _extract_di_registrations(source: str) -> List[Dict[str, str]]:
    """Extract DI registrations from Startup/Program.cs source."""
    registrations: List[Dict[str, str]] = []
    # services.AddTransient<IFoo, Foo>()
    for m in re.finditer(
        r"\.Add(Transient|Scoped|Singleton)\s*<\s*(\w+)\s*(?:,\s*(\w+))?\s*>\s*\(",
        source,
    ):
        lifetime = m.group(1)
        interface = m.group(2)
        impl = m.group(3) or interface
        registrations.append({
            "lifetime": lifetime,
            "interface": interface,
            "implementation": impl,
        })
    # services.AddDbContext<AppDbContext>(...)
    for m in re.finditer(
        r"\.AddDbContext\s*<\s*(\w+)\s*>", source,
    ):
        registrations.append({
            "lifetime": "Scoped",
            "interface": m.group(1),
            "implementation": m.group(1),
        })
    return registrations


def _extract_config_keys(source: str) -> List[str]:
    """Extract configuration key references from C# source."""
    keys: Set[str] = set()
    # Configuration["Key"], Configuration.GetSection("Key")
    for m in re.finditer(
        r'Configuration\s*\[\s*"([^"]+)"\s*\]', source,
    ):
        keys.add(m.group(1))
    for m in re.finditer(
        r'Configuration\.GetSection\s*\(\s*"([^"]+)"\s*\)', source,
    ):
        keys.add(m.group(1))
    # IOptions<T> pattern
    for m in re.finditer(
        r'IOptions(?:Monitor|Snapshot)?\s*<\s*(\w+)\s*>', source,
    ):
        keys.add(m.group(1))
    return sorted(keys)


def _extract_nuget_packages(source: str) -> List[Dict[str, str]]:
    """Extract NuGet package references from .csproj XML source."""
    packages: List[Dict[str, str]] = []
    for m in re.finditer(
        r'<PackageReference\s+Include="([^"]+)"\s+Version="([^"]*)"',
        source, re.IGNORECASE,
    ):
        packages.append({"name": m.group(1), "version": m.group(2)})
    return packages


def _is_controller(name: str, source: str, meta: dict) -> bool:
    """Detect if a class is an ASP.NET controller."""
    if name.endswith("Controller"):
        return True
    if ": Controller" in source or ": ControllerBase" in source:
        return True
    if "[ApiController]" in source:
        return True
    return False


def _is_dbcontext(name: str, source: str) -> bool:
    """Detect if a class is an EF Core DbContext."""
    if name.endswith("DbContext") or name.endswith("Context"):
        if ": DbContext" in source or "DbSet<" in source:
            return True
    return False


def _is_entity(source: str, meta: dict) -> bool:
    """Detect if a class is an EF Core entity model."""
    if "[Table(" in source or "[Key]" in source:
        return True
    if "DbSet<" in source:
        return False  # This is a DbContext, not an entity
    modifiers = meta.get("modifiers", [])
    for mod in modifiers:
        if isinstance(mod, str) and "Table" in mod:
            return True
    return False


def _namespace_from_path(file_path: str) -> str:
    """Infer namespace from file path."""
    path = file_path.replace("\\", "/")
    if path.endswith(".cs"):
        path = path[:-3]
    return path.replace("/", ".")


# ---------------------------------------------------------------------------
# Chapter 1 Override: Solution/Project Inventory
# ---------------------------------------------------------------------------


def generate_solution_inventory(
    db: DatabaseManager, project_id: str, pipeline: Any = None,
) -> str:
    """Solution/project catalog with assembly inventory, namespace mapping,
    language breakdown, and project structure tree."""
    pid = _pid(project_id)
    lines = ["# 1. Solution & Project Inventory\n"]

    try:
        with db.get_session() as session:
            # Project info
            proj = session.execute(
                text("SELECT name, primary_language, languages, file_count, total_lines "
                     "FROM projects WHERE project_id = :pid"),
                {"pid": pid},
            ).fetchone()

            if not proj:
                return "# 1. Solution & Project Inventory\n\nProject not found."

            # Language breakdown
            lang_rows = session.execute(
                text("SELECT language, COUNT(*) AS cnt FROM code_files "
                     "WHERE project_id = :pid GROUP BY language ORDER BY cnt DESC"),
                {"pid": pid},
            ).fetchall()

            # All C# files
            cs_files = session.execute(
                text("SELECT file_path FROM code_files "
                     "WHERE project_id = :pid AND language = 'csharp' "
                     "ORDER BY file_path"),
                {"pid": pid},
            ).fetchall()

            # .csproj files for project structure
            csproj_files = session.execute(
                text("SELECT file_path, source FROM code_files "
                     "WHERE project_id = :pid AND file_path LIKE '%%.csproj' "
                     "ORDER BY file_path"),
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

        # Application type detection
        app_type = "Library"
        all_paths = [f.file_path or "" for f in cs_files]
        if any("Controller" in p for p in all_paths):
            app_type = "ASP.NET Core Web API"
        elif any("Startup.cs" in p or "Program.cs" in p for p in all_paths):
            app_type = "ASP.NET Core Application"
        elif any(".razor" in p.lower() for p in all_paths):
            app_type = "Blazor Application"

        lines += [
            "## Application Type\n",
            f"- **Framework**: .NET / C#",
            f"- **Application Type**: {app_type}",
            f"- **Primary Language**: {proj.primary_language or 'C#'}",
            f"- **Total Files**: {proj.file_count or 0}",
            f"- **Total Lines**: {proj.total_lines or 0}",
            "",
        ]

        # .csproj project inventory
        if csproj_files:
            lines += [
                "## Project Files (.csproj)\n",
                "| Project | Path | Target Framework | Packages |",
                "|---------|------|-----------------|----------|",
            ]
            for cp in csproj_files:
                src = cp.source or ""
                # Extract target framework
                tf_match = re.search(
                    r"<TargetFramework>(.*?)</TargetFramework>", src,
                )
                tf = tf_match.group(1) if tf_match else "unknown"
                pkg_count = len(_extract_nuget_packages(src))
                proj_name = (cp.file_path or "").split("/")[-1]
                lines.append(
                    f"| {proj_name} | `{cp.file_path}` | {tf} | {pkg_count} |"
                )
            lines.append("")

        # Namespace catalog
        namespaces: Dict[str, Dict[str, int]] = defaultdict(lambda: {
            "classes": 0, "interfaces": 0, "methods": 0,
        })
        for u in unit_rows:
            fp = u.file_path or ""
            ut = u.unit_type or ""
            name = u.name or ""
            ns = _namespace_from_path(fp)
            if ut == "class":
                if name.startswith("I") and name[1:2].isupper():
                    namespaces[ns]["interfaces"] += 1
                else:
                    namespaces[ns]["classes"] += 1
            elif ut == "method":
                namespaces[ns]["methods"] += 1

        if namespaces:
            lines += [
                "## Namespace Catalog\n",
                "| Namespace | Classes | Interfaces | Methods |",
                "|-----------|---------|------------|---------|",
            ]
            for ns in sorted(namespaces.keys()):
                m = namespaces[ns]
                lines.append(
                    f"| `{ns}` | {m['classes']} | {m['interfaces']} | {m['methods']} |"
                )
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

        # Project structure tree
        dirs: Set[str] = set()
        for fp in sorted(f.file_path or "" for f in cs_files):
            parts = fp.replace("\\", "/").split("/")
            for i in range(1, len(parts)):
                dirs.add("/".join(parts[:i]))

        if dirs:
            lines.append("## Project Structure\n")
            lines.append("```")
            for d in sorted(dirs):
                depth = d.count("/")
                name = d.split("/")[-1]
                indent = "    " * depth
                prefix = "|- " if depth > 0 else ""
                lines.append(f"{indent}{prefix}{name}/")
            lines.append("```")
            lines.append("")

    except Exception as e:
        logger.error(".NET chapter 1 generation failed: %s", e, exc_info=True)
        lines.append(f"\n*Generation error: {e}*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chapter 3 Override: Controller Routes & Entry Points
# ---------------------------------------------------------------------------


def generate_controller_routes(
    db: DatabaseManager, project_id: str, pipeline: Any = None,
) -> str:
    """Map all ASP.NET Core controller routes with attribute routing,
    auth requirements, and middleware pipeline info."""
    pid = _pid(project_id)
    lines = ["# 3. Controller Routes & Entry Points\n"]

    try:
        classes = _fetch_classes(db, pid)
        methods = _fetch_methods(db, pid)
        entry_points = _fetch_entry_points(db, pid)

        # Index methods by class
        methods_by_class: Dict[str, list] = defaultdict(list)
        for m in methods:
            methods_by_class[m.class_name].append(m)

        # Identify controllers
        controllers = []
        for cls in classes:
            source = _safe_source(cls)
            meta = _safe_meta(cls)
            if _is_controller(cls.name, source, meta):
                controllers.append(cls)

        # Controller route table
        all_routes: List[Dict[str, str]] = []
        for ctrl in controllers:
            ctrl_source = _safe_source(ctrl)
            ctrl_name = ctrl.name

            # Extract class-level [Route("api/[controller]")]
            base_route = ""
            route_match = re.search(
                r'\[Route\s*\(\s*"([^"]+)"\s*\)\]', ctrl_source,
            )
            if route_match:
                base_route = route_match.group(1)
                base_route = base_route.replace(
                    "[controller]",
                    ctrl_name.replace("Controller", "").lower(),
                )

            # Extract per-action routes from methods
            ctrl_methods = methods_by_class.get(ctrl_name, [])
            for meth in ctrl_methods:
                m_source = _safe_source(meth)
                m_meta = _safe_meta(meth)
                routes = _extract_route_info(m_source)
                if not routes:
                    continue

                # Detect auth
                auth = "Unknown"
                if "[Authorize" in m_source or "[Authorize" in ctrl_source:
                    auth = "Required"
                if "[AllowAnonymous]" in m_source:
                    auth = "Public"

                for route in routes:
                    full_path = f"{base_route}/{route['path']}".strip("/")
                    all_routes.append({
                        "method": route["method"],
                        "path": f"/{full_path}" if full_path else "/",
                        "controller": ctrl_name,
                        "action": meth.name or "",
                        "file": ctrl.file_path or "",
                        "auth": auth,
                        "description": (meth.docstring or "")[:80]
                            if hasattr(meth, "docstring") else "",
                    })

        if all_routes:
            lines += [
                "## API Routes\n",
                "| Method | Path | Controller | Action | Auth | Description |",
                "|--------|------|------------|--------|------|-------------|",
            ]
            for r in sorted(all_routes, key=lambda x: (x["path"], x["method"])):
                lines.append(
                    f"| {r['method']} | `{r['path']}` | `{r['controller']}` "
                    f"| `{r['action']}()` | {r['auth']} | {r['description']} |"
                )
            lines.append("")

        # Controller inventory
        if controllers:
            lines += [
                "## Controller Inventory\n",
                "| Controller | File | Actions | Base Route |",
                "|------------|------|---------|------------|",
            ]
            for ctrl in controllers:
                ctrl_source = _safe_source(ctrl)
                route_match = re.search(
                    r'\[Route\s*\(\s*"([^"]+)"\s*\)\]', ctrl_source,
                )
                base = route_match.group(1) if route_match else "--"
                action_count = len(methods_by_class.get(ctrl.name, []))
                lines.append(
                    f"| `{ctrl.name}` | `{ctrl.file_path}` "
                    f"| {action_count} | `{base}` |"
                )
            lines.append("")

        # Entry Points from understanding engine
        if entry_points:
            ep_by_type: Dict[str, list] = defaultdict(list)
            for ep in entry_points:
                ep_by_type[ep.entry_type or "unknown"].append(ep)

            for ep_type, eps in sorted(ep_by_type.items()):
                if ep_type == "http_endpoint" and all_routes:
                    continue
                lines += [
                    f"## {ep_type.replace('_', ' ').title()} Entry Points\n",
                    "| Name | Qualified Name | File | Confidence |",
                    "|------|---------------|------|------------|",
                ]
                for ep in eps:
                    lines.append(
                        f"| `{ep.name}` | `{ep.qualified_name or ''}` "
                        f"| `{ep.file_path}` | {ep.confidence or 0:.0%} |"
                    )
                lines.append("")

        # Summary
        lines += [
            "## Summary\n",
            f"- **Controllers**: {len(controllers)}",
            f"- **API Routes**: {len(all_routes)}",
            f"- **Entry Points**: {len(entry_points)}",
            "",
        ]

    except Exception as e:
        logger.error(".NET chapter 3 generation failed: %s", e, exc_info=True)
        lines.append(f"\n*Generation error: {e}*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chapter 4 Override: Per-Controller/Service Functional Specs
# ---------------------------------------------------------------------------


def generate_per_service_specs(
    db: DatabaseManager, project_id: str, pipeline: Any = None,
) -> str:
    """Per-controller and per-service functional specifications with method
    signatures, DI dependencies, business logic, and configuration."""
    pid = _pid(project_id)
    lines = ["# 4. Functional Specifications\n"]

    try:
        classes = _fetch_classes(db, pid)
        methods = _fetch_methods(db, pid)
        analyses = _fetch_deep_analyses(db, pid)
        calls = _fetch_calls(db, pid)
        implements = _fetch_implements(db, pid)

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

        # Index methods by class
        methods_by_class: Dict[str, list] = defaultdict(list)
        for m in methods:
            methods_by_class[m.class_name].append(m)

        # Build call graph
        callee_by_caller: Dict[str, Set[str]] = defaultdict(set)
        for c in calls:
            callee_by_caller[c.caller_qn or c.caller].add(c.callee_qn or c.callee)

        # Build implements map
        ifaces_by_class: Dict[str, List[str]] = defaultdict(list)
        for imp in implements:
            ifaces_by_class[imp.implementor].append(imp.interface)

        # Sort classes: controllers first, then services, then others
        def _sort_key(cls):
            name = cls.name or ""
            if name.endswith("Controller"):
                return (0, name)
            if name.endswith("Service"):
                return (1, name)
            return (2, name)

        for cls in sorted(classes, key=_sort_key):
            source = _safe_source(cls)
            meta = _safe_meta(cls)
            cls_name = cls.name
            qn = cls.qualified_name or cls_name

            lines.append(f"## {qn}\n")

            # Classification
            cls_type = "Class"
            if _is_controller(cls_name, source, meta):
                cls_type = "Controller"
            elif cls_name.endswith("Service"):
                cls_type = "Service"
            elif _is_dbcontext(cls_name, source):
                cls_type = "DbContext"
            elif cls_name.startswith("I") and cls_name[1:2].isupper():
                cls_type = "Interface"

            # Interfaces implemented
            ifaces = ifaces_by_class.get(cls_name, [])

            lines.append(f"- **Type**: {cls_type}")
            if ifaces:
                lines.append(f"- **Implements**: {', '.join(f'`{i}`' for i in ifaces)}")
            lines.append(f"- **File**: `{cls.file_path}`")
            lines.append("")

            # Purpose from docstring or deep analysis
            docstring = cls.docstring or ""
            ainfo = analysis_by_name.get(cls_name, {})
            narrative = ainfo.get("narrative", "")

            purpose = docstring.split("\n")[0] if docstring else ""
            if not purpose and narrative:
                purpose = narrative[:200]
            if purpose:
                lines += ["### Purpose\n", purpose, ""]

            # Constructor DI dependencies (from source)
            ctor_deps: List[str] = []
            ctor_match = re.search(
                rf"{re.escape(cls_name)}\s*\(([^)]*)\)",
                source, re.DOTALL,
            )
            if ctor_match:
                params = ctor_match.group(1)
                for pm in re.finditer(r"(\w+(?:<\w+>)?)\s+\w+", params):
                    ctor_deps.append(pm.group(1))

            if ctor_deps:
                lines += [
                    "### Constructor Dependencies (DI)\n",
                    "| Type | Injected Via |",
                    "|------|-------------|",
                ]
                for dep in ctor_deps:
                    lines.append(f"| `{dep}` | Constructor injection |")
                lines.append("")

            # Method table
            cls_methods = methods_by_class.get(cls_name, [])
            if cls_methods:
                lines += [
                    "### Methods\n",
                    "| Method | Signature | Async | Return Type | Description |",
                    "|--------|-----------|-------|-------------|-------------|",
                ]
                for meth in cls_methods:
                    m_meta = _safe_meta(meth)
                    is_async = "Yes" if m_meta.get("is_async") else "No"
                    sig = (meth.signature or "")[:60]
                    ret = m_meta.get("return_type", "--")
                    doc = (meth.docstring or "")[:50] if hasattr(meth, "docstring") else ""
                    lines.append(
                        f"| `{meth.name}` | `{sig}` | {is_async} | `{ret}` | {doc} |"
                    )
                lines.append("")

            # Business logic from deep analysis
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
            cls_deps_set: Set[str] = set()
            for meth in cls_methods:
                meth_qn = meth.qualified_name or meth.name
                cls_deps_set.update(callee_by_caller.get(meth_qn, set()))
            cls_deps_set.update(callee_by_caller.get(qn, set()))

            if cls_deps_set:
                lines += [
                    "### Outgoing Calls\n",
                    "| Target | Relationship |",
                    "|--------|-------------|",
                ]
                for dep in sorted(cls_deps_set)[:15]:
                    lines.append(f"| `{dep}` | Calls |")
                lines.append("")

            # Configuration keys
            config_keys = _extract_config_keys(source)
            if config_keys:
                lines += [
                    "### Configuration\n",
                    "| Key | Source |",
                    "|-----|--------|",
                ]
                for ck in config_keys[:10]:
                    lines.append(f"| `{ck}` | `IConfiguration` / `IOptions<T>` |")
                lines.append("")

            lines.append("---\n")

    except Exception as e:
        logger.error(".NET chapter 4 generation failed: %s", e, exc_info=True)
        lines.append(f"\n*Generation error: {e}*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chapter 5 Override: EF Core Entity Models
# ---------------------------------------------------------------------------


def generate_ef_core_models(
    db: DatabaseManager, project_id: str, pipeline: Any = None,
) -> str:
    """Catalog EF Core DbContext classes, entity models, navigation properties,
    and Fluent API configuration."""
    pid = _pid(project_id)
    lines = ["# 5. EF Core Entity Models & Data Schema\n"]

    try:
        classes = _fetch_classes(db, pid)
        methods = _fetch_methods(db, pid)
        inherits = _fetch_inherits(db, pid)

        # Index methods by class
        methods_by_class: Dict[str, list] = defaultdict(list)
        for m in methods:
            methods_by_class[m.class_name].append(m)

        # Inheritance map
        parent_by_child: Dict[str, str] = {}
        for inh in inherits:
            parent_by_child[inh.child] = inh.parent

        # Separate DbContext classes and entity classes
        dbcontexts: list = []
        entities: list = []
        other_models: list = []

        for cls in classes:
            source = _safe_source(cls)
            meta = _safe_meta(cls)
            if _is_dbcontext(cls.name, source):
                dbcontexts.append(cls)
            elif _is_entity(source, meta):
                entities.append(cls)
            elif parent_by_child.get(cls.name) in ("DbContext", "IdentityDbContext"):
                dbcontexts.append(cls)

        # ----- DbContext Classes -----
        if dbcontexts:
            lines.append("## DbContext Classes\n")
            for ctx in dbcontexts:
                source = _safe_source(ctx)
                lines.append(f"### {ctx.name}\n")
                lines.append(f"- **File**: `{ctx.file_path}`")

                parent = parent_by_child.get(ctx.name, "DbContext")
                lines.append(f"- **Inherits**: `{parent}`")

                # Extract DbSet<T> properties
                dbsets: List[Tuple[str, str]] = []
                for m in re.finditer(
                    r"DbSet<(\w+)>\s+(\w+)", source,
                ):
                    dbsets.append((m.group(1), m.group(2)))

                if dbsets:
                    lines += [
                        "",
                        "| Entity Type | DbSet Property |",
                        "|-------------|---------------|",
                    ]
                    for entity_type, prop_name in dbsets:
                        lines.append(f"| `{entity_type}` | `{prop_name}` |")
                    lines.append("")

                # OnModelCreating info
                ctx_methods = methods_by_class.get(ctx.name, [])
                for meth in ctx_methods:
                    if meth.name == "OnModelCreating":
                        m_source = _safe_source(meth)
                        # Count Fluent API configurations
                        has_fluent = re.findall(
                            r"\.HasKey|\.HasIndex|\.HasOne|\.HasMany|\.ToTable|\.Property",
                            m_source,
                        )
                        if has_fluent:
                            lines.append(
                                f"- **Fluent API configurations**: "
                                f"{len(has_fluent)} property/relationship mappings"
                            )
                lines.append("")

        # ----- Entity Models -----
        if entities:
            lines += [
                "## Entity Models\n",
                "| Entity | File | Base Class | Key Properties |",
                "|--------|------|-----------|----------------|",
            ]
            for ent in entities:
                source = _safe_source(ent)
                meta = _safe_meta(ent)
                parent = parent_by_child.get(ent.name, "--")

                # Extract [Key] properties or Id convention
                key_props: List[str] = []
                fields = meta.get("fields", [])
                for f in fields:
                    fname = f.get("name", "") if isinstance(f, dict) else str(f)
                    if fname.lower() in ("id", ent.name.lower() + "id"):
                        key_props.append(fname)

                # Regex fallback for [Key] attribute
                for m in re.finditer(
                    r"\[Key\]\s*(?:\[.*?\]\s*)*public\s+\w+\s+(\w+)",
                    source, re.DOTALL,
                ):
                    if m.group(1) not in key_props:
                        key_props.append(m.group(1))

                keys_str = ", ".join(f"`{k}`" for k in key_props) if key_props else "--"
                lines.append(
                    f"| `{ent.name}` | `{ent.file_path}` "
                    f"| `{parent}` | {keys_str} |"
                )
            lines.append("")

            # Detailed per-entity breakdown
            for ent in entities[:20]:  # Cap at 20 to avoid huge output
                source = _safe_source(ent)
                meta = _safe_meta(ent)
                fields = meta.get("fields", [])

                if not fields:
                    continue

                lines.append(f"### {ent.name}\n")
                lines += [
                    "| Property | Type | Attributes |",
                    "|----------|------|-----------|",
                ]
                for f in fields[:30]:
                    if isinstance(f, dict):
                        fname = f.get("name", "?")
                        ftype = f.get("type", "?")
                        fattrs = ", ".join(f.get("modifiers", [])[:3]) if f.get("modifiers") else "--"
                    else:
                        fname = str(f)
                        ftype = "--"
                        fattrs = "--"
                    lines.append(f"| `{fname}` | `{ftype}` | {fattrs} |")
                lines.append("")

        # Summary
        lines += [
            "## Summary\n",
            f"- **DbContext classes**: {len(dbcontexts)}",
            f"- **Entity models**: {len(entities)}",
            "",
        ]

    except Exception as e:
        logger.error(".NET chapter 5 generation failed: %s", e, exc_info=True)
        lines.append(f"\n*Generation error: {e}*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chapter 6 Override: Database Access Patterns
# ---------------------------------------------------------------------------


def generate_database_patterns(
    db: DatabaseManager, project_id: str, pipeline: Any = None,
) -> str:
    """Classify database access patterns: EF Core LINQ, raw SQL, Dapper,
    stored procedure calls, and connection management."""
    pid = _pid(project_id)
    lines = ["# 6. Database Access Patterns\n"]

    try:
        with db.get_session() as session:
            all_units = session.execute(
                text("""
                    SELECT u.name, u.qualified_name, u.source, u.unit_type,
                           u.signature, u.metadata, f.file_path
                    FROM code_units u
                    JOIN code_files f ON u.file_id = f.file_id
                    WHERE u.project_id = :pid
                      AND u.unit_type IN ('method', 'class')
                    ORDER BY f.file_path, u.start_line
                """),
                {"pid": pid},
            ).fetchall()

        # Pattern detection
        ef_linq: List[Dict[str, str]] = []  # .Where().ToListAsync() etc
        raw_sql: List[Dict[str, str]] = []  # FromSqlRaw, ExecuteSqlRaw
        dapper: List[Dict[str, str]] = []  # QueryAsync, ExecuteAsync (Dapper)
        sp_calls: List[Dict[str, str]] = []  # Stored procedure invocations
        ado_net: List[Dict[str, str]] = []  # SqlCommand, SqlConnection

        for u in all_units:
            source = _safe_source(u)
            if not source:
                continue
            name = u.name or ""
            fp = u.file_path or ""

            # EF Core LINQ
            if re.search(r"\.\s*(?:Where|Select|Include|ThenInclude|FirstOrDefault"
                         r"|ToList|ToListAsync|FindAsync|AddAsync|SaveChanges)", source):
                ef_linq.append({"name": name, "file": fp, "pattern": "EF Core LINQ"})

            # Raw SQL via EF
            if re.search(r"(?:FromSqlRaw|ExecuteSqlRaw|SqlQueryRaw)", source):
                raw_sql.append({"name": name, "file": fp, "pattern": "EF Core Raw SQL"})

            # Dapper
            if re.search(r"\.(?:QueryAsync|ExecuteAsync|QueryFirstAsync|QuerySingle)", source):
                dapper.append({"name": name, "file": fp, "pattern": "Dapper"})

            # Stored procedure calls
            if re.search(r"CommandType\.StoredProcedure|EXEC\s+\w+|sp_\w+", source):
                sp_calls.append({"name": name, "file": fp, "pattern": "Stored Procedure"})

            # ADO.NET
            if re.search(r"SqlCommand|SqlConnection|SqlDataReader|SqlDataAdapter", source):
                ado_net.append({"name": name, "file": fp, "pattern": "ADO.NET"})

        # Pattern summary
        lines += [
            "## Access Pattern Distribution\n",
            "| Pattern | Occurrences | Description |",
            "|---------|-------------|-------------|",
            f"| EF Core LINQ | {len(ef_linq)} | Strongly-typed LINQ queries via DbContext |",
            f"| EF Core Raw SQL | {len(raw_sql)} | FromSqlRaw / ExecuteSqlRawAsync |",
            f"| Dapper | {len(dapper)} | Micro-ORM raw SQL with object mapping |",
            f"| Stored Procedures | {len(sp_calls)} | SP invocations via ADO.NET or EF |",
            f"| ADO.NET Direct | {len(ado_net)} | SqlCommand / SqlConnection direct |",
            "",
        ]

        # Detailed tables for each pattern
        for label, items in [
            ("EF Core LINQ Usage", ef_linq),
            ("Raw SQL Usage", raw_sql),
            ("Dapper Usage", dapper),
            ("Stored Procedure Calls", sp_calls),
            ("ADO.NET Direct Access", ado_net),
        ]:
            if items:
                seen = set()
                lines += [
                    f"## {label}\n",
                    "| Method/Class | File |",
                    "|-------------|------|",
                ]
                for item in items:
                    key = (item["name"], item["file"])
                    if key in seen:
                        continue
                    seen.add(key)
                    lines.append(f"| `{item['name']}` | `{item['file']}` |")
                lines.append("")

        # Migration risk assessment
        legacy_count = len(sp_calls) + len(ado_net)
        total_db = len(ef_linq) + len(raw_sql) + len(dapper) + legacy_count
        if total_db > 0:
            legacy_pct = (legacy_count / total_db) * 100
            lines += [
                "## Migration Risk\n",
                f"- **Total DB access points**: {total_db}",
                f"- **Legacy (SP + ADO.NET)**: {legacy_count} ({legacy_pct:.0f}%)",
                f"- **Modern (EF Core + Dapper)**: {total_db - legacy_count} ({100 - legacy_pct:.0f}%)",
                "",
            ]

    except Exception as e:
        logger.error(".NET chapter 6 generation failed: %s", e, exc_info=True)
        lines.append(f"\n*Generation error: {e}*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chapter 7 Override: DI Registrations & Call Graphs
# ---------------------------------------------------------------------------


def generate_di_and_call_graphs(
    db: DatabaseManager, project_id: str, pipeline: Any = None,
) -> str:
    """Map dependency injection registrations, service lifetimes, and
    call graphs showing request pipeline flow."""
    pid = _pid(project_id)
    lines = ["# 7. Dependency Injection & Call Graphs\n"]

    try:
        classes = _fetch_classes(db, pid)
        calls = _fetch_calls(db, pid)
        implements = _fetch_implements(db, pid)

        # Scan for DI registrations in Startup/Program files
        with db.get_session() as session:
            startup_units = session.execute(
                text("""
                    SELECT u.name, u.source, u.qualified_name, f.file_path
                    FROM code_units u
                    JOIN code_files f ON u.file_id = f.file_id
                    WHERE u.project_id = :pid
                      AND (f.file_path LIKE '%%Startup.cs'
                           OR f.file_path LIKE '%%Program.cs'
                           OR u.name LIKE '%%ServiceCollectionExtensions%%')
                    ORDER BY f.file_path
                """),
                {"pid": pid},
            ).fetchall()

        # Collect all DI registrations
        all_registrations: List[Dict[str, str]] = []
        for su in startup_units:
            source = _safe_source(su)
            regs = _extract_di_registrations(source)
            for r in regs:
                r["source_file"] = su.file_path or ""
            all_registrations.extend(regs)

        # DI Registration Table
        if all_registrations:
            lines += [
                "## Service Registrations\n",
                "| Interface | Implementation | Lifetime | Registered In |",
                "|-----------|----------------|----------|--------------|",
            ]
            for reg in sorted(all_registrations, key=lambda x: x["interface"]):
                lines.append(
                    f"| `{reg['interface']}` | `{reg['implementation']}` "
                    f"| {reg['lifetime']} | `{reg['source_file']}` |"
                )
            lines.append("")

            # Lifetime distribution
            lifetime_counts: Dict[str, int] = defaultdict(int)
            for reg in all_registrations:
                lifetime_counts[reg["lifetime"]] += 1
            lines += [
                "## Lifetime Distribution\n",
                "| Lifetime | Count |",
                "|----------|-------|",
            ]
            for lt in sorted(lifetime_counts.keys()):
                lines.append(f"| {lt} | {lifetime_counts[lt]} |")
            lines.append("")

        # Interface-Implementation map
        if implements:
            lines += [
                "## Interface Implementations\n",
                "| Interface | Implementation | File |",
                "|-----------|----------------|------|",
            ]
            seen = set()
            for imp in implements:
                key = (imp.interface, imp.implementor)
                if key in seen:
                    continue
                seen.add(key)
                lines.append(
                    f"| `{imp.interface}` | `{imp.implementor}` "
                    f"| `{imp.impl_file}` |"
                )
            lines.append("")

        # Call graph
        if calls:
            # Build adjacency for controllers -> services -> repositories
            caller_callee: Dict[str, Set[str]] = defaultdict(set)
            for c in calls:
                caller_callee[c.caller].add(c.callee)

            # Controller call chains
            controllers = [
                cls for cls in classes
                if _is_controller(cls.name, _safe_source(cls), _safe_meta(cls))
            ]
            if controllers:
                lines.append("## Controller Call Chains\n")
                for ctrl in controllers[:10]:
                    direct_calls = sorted(caller_callee.get(ctrl.name, set()))
                    if direct_calls:
                        lines.append(f"### {ctrl.name}\n")
                        lines.append("```")
                        for callee in direct_calls[:15]:
                            second_level = sorted(caller_callee.get(callee, set()))
                            lines.append(f"  {ctrl.name} -> {callee}")
                            for sl in second_level[:5]:
                                lines.append(f"    {callee} -> {sl}")
                        lines.append("```")
                        lines.append("")

        # Summary
        lines += [
            "## Summary\n",
            f"- **DI Registrations**: {len(all_registrations)}",
            f"- **Interface Implementations**: {len(implements)}",
            f"- **Call Edges**: {len(calls)}",
            "",
        ]

    except Exception as e:
        logger.error(".NET chapter 7 generation failed: %s", e, exc_info=True)
        lines.append(f"\n*Generation error: {e}*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chapter 8 Override: NuGet Dependencies
# ---------------------------------------------------------------------------


def generate_nuget_dependencies(
    db: DatabaseManager, project_id: str, pipeline: Any = None,
) -> str:
    """Catalog NuGet packages, framework references, and external service
    integrations from .csproj and source analysis."""
    pid = _pid(project_id)
    lines = ["# 8. NuGet Dependencies & External Services\n"]

    try:
        with db.get_session() as session:
            # .csproj files for package references
            csproj_files = session.execute(
                text("SELECT file_path, source FROM code_files "
                     "WHERE project_id = :pid AND file_path LIKE '%%.csproj' "
                     "ORDER BY file_path"),
                {"pid": pid},
            ).fetchall()

            # All units with source for service detection
            all_units = session.execute(
                text("""
                    SELECT u.name, u.qualified_name, u.source, u.unit_type,
                           u.metadata, f.file_path
                    FROM code_units u
                    JOIN code_files f ON u.file_id = f.file_id
                    WHERE u.project_id = :pid
                      AND u.unit_type IN ('class', 'method')
                    ORDER BY f.file_path, u.start_line
                """),
                {"pid": pid},
            ).fetchall()

        # Extract NuGet packages from all .csproj files
        all_packages: Dict[str, Dict[str, str]] = {}  # name -> {version, project}
        for cp in csproj_files:
            source = cp.source or ""
            packages = _extract_nuget_packages(source)
            proj_name = (cp.file_path or "").split("/")[-1]
            for pkg in packages:
                if pkg["name"] not in all_packages:
                    all_packages[pkg["name"]] = {
                        "version": pkg["version"],
                        "project": proj_name,
                    }

        # NuGet package categories
        pkg_categories = {
            "Microsoft.AspNetCore": "Web Framework",
            "Microsoft.EntityFrameworkCore": "ORM",
            "Microsoft.Extensions": "Framework Extensions",
            "Swashbuckle": "API Documentation",
            "Serilog": "Logging",
            "AutoMapper": "Object Mapping",
            "MediatR": "Mediator Pattern",
            "FluentValidation": "Validation",
            "Polly": "Resilience",
            "Hangfire": "Background Jobs",
            "Dapper": "Micro-ORM",
            "Newtonsoft.Json": "Serialization",
            "xunit": "Testing",
            "Moq": "Mocking",
            "NUnit": "Testing",
            "StackExchange.Redis": "Caching",
            "MassTransit": "Message Bus",
            "RabbitMQ": "Message Queue",
            "Azure": "Cloud (Azure)",
            "AWS": "Cloud (AWS)",
            "Npgsql": "PostgreSQL Driver",
            "Microsoft.Data.SqlClient": "SQL Server Driver",
        }

        if all_packages:
            lines += [
                "## NuGet Packages\n",
                "| Package | Version | Category | Project |",
                "|---------|---------|----------|---------|",
            ]
            for name in sorted(all_packages.keys()):
                info = all_packages[name]
                # Categorize
                category = "Library"
                for prefix, cat in pkg_categories.items():
                    if name.startswith(prefix):
                        category = cat
                        break
                lines.append(
                    f"| `{name}` | {info['version']} | {category} | {info['project']} |"
                )
            lines.append("")

        # Detect external service integrations from source
        services: Dict[str, Set[str]] = defaultdict(set)
        service_patterns = {
            "HttpClient": r"HttpClient|IHttpClientFactory",
            "Redis": r"IDistributedCache|StackExchange\.Redis|ConnectionMultiplexer",
            "Message Queue": r"RabbitMQ|MassTransit|IPublishEndpoint|IBus",
            "Azure Blob": r"BlobServiceClient|BlobContainerClient",
            "Azure Service Bus": r"ServiceBusClient|ServiceBusSender",
            "AWS S3": r"AmazonS3Client|IAmazonS3",
            "gRPC": r"GrpcChannel|\.Protos\.",
            "SignalR": r"HubConnection|IHubContext",
            "Email": r"SmtpClient|IEmailSender|SendGrid",
        }
        for u in all_units:
            source = _safe_source(u)
            for svc_name, pattern in service_patterns.items():
                if re.search(pattern, source):
                    services[svc_name].add(u.file_path or "")

        if services:
            lines += [
                "## External Service Integrations\n",
                "| Service | Files Using |",
                "|---------|------------|",
            ]
            for svc in sorted(services.keys()):
                files = ", ".join(sorted(services[svc])[:3])
                lines.append(f"| {svc} | `{files}` |")
            lines.append("")

        # Configuration keys across all source
        all_config: Dict[str, Set[str]] = defaultdict(set)
        for u in all_units:
            source = _safe_source(u)
            for ck in _extract_config_keys(source):
                all_config[ck].add(u.file_path or "")

        if all_config:
            lines += [
                "## Configuration Keys\n",
                "| Key | Referenced In |",
                "|-----|--------------|",
            ]
            for key in sorted(all_config.keys()):
                files = ", ".join(sorted(all_config[key])[:3])
                lines.append(f"| `{key}` | `{files}` |")
            lines.append("")

        # Deep analysis integrations
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
            seen_integ: Set[str] = set()
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
            f"- **NuGet Packages**: {len(all_packages)}",
            f"- **External Services**: {len(services)}",
            f"- **Configuration Keys**: {len(all_config)}",
            f"- **Integration Points**: {len(integrations)}",
            "",
        ]

    except Exception as e:
        logger.error(".NET chapter 8 generation failed: %s", e, exc_info=True)
        lines.append(f"\n*Generation error: {e}*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chapter 9 Override: Error Handling Middleware
# ---------------------------------------------------------------------------


def generate_dotnet_error_handling(
    db: DatabaseManager, project_id: str, pipeline: Any = None,
) -> str:
    """Catalog exception handling middleware, exception filters, try/catch
    patterns, and resilience strategies (Polly, circuit breakers)."""
    pid = _pid(project_id)
    lines = ["# 9. Error Handling & Middleware\n"]

    try:
        with db.get_session() as session:
            all_units = session.execute(
                text("""
                    SELECT u.name, u.qualified_name, u.source, u.unit_type,
                           u.signature, u.metadata, f.file_path
                    FROM code_units u
                    JOIN code_files f ON u.file_id = f.file_id
                    WHERE u.project_id = :pid
                      AND u.unit_type IN ('class', 'method')
                    ORDER BY f.file_path, u.start_line
                """),
                {"pid": pid},
            ).fetchall()

        # ----- Custom Exception Classes -----
        custom_exceptions: List[Dict[str, str]] = []
        for u in all_units:
            source = _safe_source(u)
            name = u.name or ""
            if u.unit_type == "class" and (
                name.endswith("Exception") or ": Exception" in source
                or ": ApplicationException" in source
            ):
                # Detect base class
                base = "Exception"
                base_match = re.search(r":\s*(\w+Exception)", source)
                if base_match:
                    base = base_match.group(1)
                custom_exceptions.append({
                    "name": name,
                    "base": base,
                    "file": u.file_path or "",
                })

        if custom_exceptions:
            lines += [
                "## Custom Exception Classes\n",
                "| Exception | Base Class | File |",
                "|-----------|-----------|------|",
            ]
            for exc in custom_exceptions:
                lines.append(
                    f"| `{exc['name']}` | `{exc['base']}` | `{exc['file']}` |"
                )
            lines.append("")

        # ----- Exception Filters & Middleware -----
        middleware: List[Dict[str, str]] = []
        exception_filters: List[Dict[str, str]] = []
        for u in all_units:
            source = _safe_source(u)
            name = u.name or ""
            if u.unit_type != "class":
                continue

            # Middleware (IMiddleware or has Invoke/InvokeAsync)
            if "IMiddleware" in source or re.search(
                r"(?:Invoke|InvokeAsync)\s*\(\s*HttpContext", source,
            ):
                middleware.append({
                    "name": name,
                    "file": u.file_path or "",
                    "type": "Middleware",
                })

            # Exception filters
            if (": IExceptionFilter" in source
                    or ": ExceptionFilterAttribute" in source
                    or "IExceptionHandler" in source):
                exception_filters.append({
                    "name": name,
                    "file": u.file_path or "",
                    "type": "Exception Filter",
                })

        if middleware:
            lines += [
                "## Middleware Pipeline\n",
                "| Middleware | File | Type |",
                "|-----------|------|------|",
            ]
            for mw in middleware:
                lines.append(
                    f"| `{mw['name']}` | `{mw['file']}` | {mw['type']} |"
                )
            lines.append("")

        if exception_filters:
            lines += [
                "## Exception Filters\n",
                "| Filter | File |",
                "|--------|------|",
            ]
            for ef in exception_filters:
                lines.append(f"| `{ef['name']}` | `{ef['file']}` |")
            lines.append("")

        # ----- Try/Catch Pattern Analysis -----
        catch_patterns: Dict[str, int] = defaultdict(int)
        methods_with_try = 0
        total_methods = 0
        for u in all_units:
            if u.unit_type != "method":
                continue
            total_methods += 1
            source = _safe_source(u)
            if "try" not in source:
                continue
            methods_with_try += 1

            # Extract caught exception types
            for m in re.finditer(r"catch\s*\(\s*(\w+(?:\.\w+)?)", source):
                catch_patterns[m.group(1)] += 1

        if catch_patterns:
            lines += [
                "## Caught Exception Types\n",
                "| Exception Type | Catch Count |",
                "|---------------|-------------|",
            ]
            for exc_type, count in sorted(
                catch_patterns.items(), key=lambda x: -x[1],
            ):
                lines.append(f"| `{exc_type}` | {count} |")
            lines.append("")

        # ----- Resilience Patterns -----
        resilience: List[Dict[str, str]] = []
        for u in all_units:
            source = _safe_source(u)
            name = u.name or ""
            if not source:
                continue

            if re.search(r"Policy\.|\.WaitAndRetry|\.CircuitBreaker|\.Bulkhead", source):
                resilience.append({
                    "name": name,
                    "file": u.file_path or "",
                    "pattern": "Polly Resilience Policy",
                })
            elif re.search(r"\.AddPolicyHandler|\.AddTransientHttpErrorPolicy", source):
                resilience.append({
                    "name": name,
                    "file": u.file_path or "",
                    "pattern": "HttpClient Resilience",
                })

        if resilience:
            lines += [
                "## Resilience Patterns\n",
                "| Component | File | Pattern |",
                "|-----------|------|---------|",
            ]
            for r in resilience:
                lines.append(
                    f"| `{r['name']}` | `{r['file']}` | {r['pattern']} |"
                )
            lines.append("")

        # Summary
        lines += [
            "## Summary\n",
            f"- **Custom Exceptions**: {len(custom_exceptions)}",
            f"- **Middleware Components**: {len(middleware)}",
            f"- **Exception Filters**: {len(exception_filters)}",
            f"- **Methods with try/catch**: {methods_with_try}/{total_methods}",
            f"- **Resilience Policies**: {len(resilience)}",
            "",
        ]

    except Exception as e:
        logger.error(".NET chapter 9 generation failed: %s", e, exc_info=True)
        lines.append(f"\n*Generation error: {e}*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Override dispatch table
# ---------------------------------------------------------------------------

DOTNET_CHAPTER_OVERRIDES = {
    1: generate_solution_inventory,
    3: generate_controller_routes,
    4: generate_per_service_specs,
    5: generate_ef_core_models,
    6: generate_database_patterns,
    7: generate_di_and_call_graphs,
    8: generate_nuget_dependencies,
    9: generate_dotnet_error_handling,
}

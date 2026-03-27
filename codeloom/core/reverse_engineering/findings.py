"""Deterministic Findings Engine for reverse engineering documentation.

Cross-references inventory data (ASG, code_units, source text) to find
architectural issues, configuration conflicts, inconsistencies, and hidden
problems. Every finding is provable from source evidence -- zero LLM.

Runs AFTER the 14-chapter doc is generated and produces a list of Finding
objects, each with file:line:snippet evidence.
"""

import logging
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID

from sqlalchemy import text as sa_text

from ..db import DatabaseManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Finding data structure
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    """A single deterministic finding with source evidence."""

    id: str  # "F-001"
    category: str  # config_conflict, dead_code_path, circular_dep, etc.
    severity: str  # critical, high, medium, low, info
    title: str  # Human-readable title
    description: str  # What's wrong and why it matters
    evidence: list = field(default_factory=list)  # [{"file", "line", "snippet"}]
    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Severity ordering (for sorting)
# ---------------------------------------------------------------------------

_SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}

# ---------------------------------------------------------------------------
# Regex patterns (compiled once)
# ---------------------------------------------------------------------------

# Python env var access patterns
_RE_GETENV = re.compile(
    r"""os\.getenv\s*\(\s*["']([A-Z_][A-Z0-9_]*)["']"""
    r"""(?:\s*,\s*["']([^"']*)["'])?\s*\)""",
    re.MULTILINE,
)
_RE_ENVIRON_GET = re.compile(
    r"""os\.environ\.get\s*\(\s*["']([A-Z_][A-Z0-9_]*)["']"""
    r"""(?:\s*,\s*["']([^"']*)["'])?\s*\)""",
    re.MULTILINE,
)
_RE_ENVIRON_BRACKET = re.compile(
    r"""os\.environ\s*\[\s*["']([A-Z_][A-Z0-9_]*)["']\s*\]""",
    re.MULTILINE,
)

# JS/TS env var access
_RE_PROCESS_ENV = re.compile(
    r"""process\.env\.([A-Z_][A-Z0-9_]*)""",
    re.MULTILINE,
)

# Java @Value annotation
_RE_JAVA_VALUE = re.compile(
    r"""@Value\s*\(\s*["']\$\{([^}]+)\}["']\s*\)""",
    re.MULTILINE,
)

# .env file key=value
_RE_DOTENV_LINE = re.compile(
    r"""^([A-Z_][A-Z0-9_]*)\s*=\s*(.*)$""",
    re.MULTILINE,
)

# YAML key: value (top-level only, simple heuristic)
_RE_YAML_KEY = re.compile(
    r"""^([a-z_][a-z0-9_]*)\s*:\s*(.+)$""",
    re.MULTILINE | re.IGNORECASE,
)

# Function name prefix pattern for error handling grouping
_RE_FUNC_PREFIX = re.compile(
    r"^(get_|set_|create_|update_|delete_|handle_|process_|fetch_|save_|load_|parse_|validate_|check_|find_|list_|remove_|add_)"
)

# Error handling keywords (language-agnostic)
_RE_ERROR_HANDLING = re.compile(
    r"\b(try\b|catch\b|except\b|rescue\b|on\s+error\b|error\s+handling)",
    re.IGNORECASE,
)

# Hardcoded value patterns
_RE_URL = re.compile(r"""https?://[^\s"'`)\]]+""")
_RE_IP_ADDR = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")
_RE_HARDCODED_SECRET = re.compile(
    r"""(?:password|passwd|secret|api_key|apikey|token|private_key)\s*=\s*["'][^"']{3,}["']""",
    re.IGNORECASE,
)
_RE_HARDCODED_PATH = re.compile(
    r"""["'](/usr/|/var/|/tmp/|/opt/|/etc/|C:\\\\|D:\\\\)[^"']*["']""",
    re.IGNORECASE,
)

# Test file detection
_RE_TEST_FILE = re.compile(
    r"(?:test_|_test\.|\.test\.|\.spec\.|tests/|__tests__/|spec/)",
    re.IGNORECASE,
)

# Config file detection
_RE_CONFIG_FILE = re.compile(
    r"(?:\.env|\.yaml|\.yml|\.json|\.properties|\.ini|\.toml|\.cfg)$",
    re.IGNORECASE,
)

# Constants file detection (exclude from hardcoded checks)
_RE_CONSTANTS_FILE = re.compile(
    r"(?:constants|config|settings|defaults|\.env)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class FindingsEngine:
    """Deterministic cross-reference analysis engine.

    Each check method returns a list of Finding objects. All checks are
    independent -- one failing check never blocks others.
    """

    # Ordered list of check methods (called in sequence)
    CHECKS = [
        "check_config_consistency",
        "check_dead_config",
        "check_circular_deps",
        "check_orphaned_code",
        "check_error_handling_consistency",
        "check_hidden_coupling",
        "check_api_contracts",
        "check_duplicate_logic",
        "check_execution_coverage",
        "check_complex_logic",
        "check_functional_inconsistencies",
        "check_hardcoded_values",
    ]

    def run(
        self, db: DatabaseManager, project_id: str
    ) -> List[Finding]:
        """Execute all checks and return sorted findings."""
        pid = UUID(project_id) if isinstance(project_id, str) else project_id
        all_findings: List[Finding] = []
        counter = 1

        for check_name in self.CHECKS:
            method = getattr(self, check_name, None)
            if not method:
                logger.warning("Check method not found: %s", check_name)
                continue
            try:
                findings = method(db, str(pid))
                # Assign sequential IDs
                for f in findings:
                    f.id = f"F-{counter:03d}"
                    counter += 1
                all_findings.extend(findings)
                logger.info(
                    "Check %s produced %d findings", check_name, len(findings)
                )
            except Exception as e:
                logger.error(
                    "Check %s failed: %s", check_name, e, exc_info=True
                )

        # Sort by severity
        all_findings.sort(key=lambda f: _SEVERITY_ORDER.get(f.severity, 99))
        return all_findings

    # ------------------------------------------------------------------
    # 1. Configuration Consistency
    # ------------------------------------------------------------------

    def check_config_consistency(
        self, db: DatabaseManager, project_id: str
    ) -> List[Finding]:
        """Detect config keys read with different defaults across call sites."""
        findings: List[Finding] = []

        with db.get_session() as session:
            rows = session.execute(
                sa_text("""
                    SELECT u.name, u.qualified_name, f.file_path,
                           u.source, u.start_line
                    FROM code_units u
                    JOIN code_files f ON u.file_id = f.file_id
                    WHERE u.project_id = :pid
                      AND u.source IS NOT NULL
                      AND (u.source ILIKE '%os.getenv%'
                           OR u.source ILIKE '%os.environ%'
                           OR u.source ILIKE '%process.env%'
                           OR u.source ILIKE '%@Value%')
                """),
                {"pid": project_id},
            ).fetchall()

        # key -> [(file_path, line, default_value, snippet)]
        env_refs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for row in rows:
            source = row.source or ""
            file_path = row.file_path
            base_line = row.start_line or 0

            for pattern in (_RE_GETENV, _RE_ENVIRON_GET):
                for m in pattern.finditer(source):
                    key = m.group(1)
                    default = m.group(2) if m.lastindex >= 2 else None
                    line_offset = source[: m.start()].count("\n")
                    env_refs[key].append({
                        "file": file_path,
                        "line": base_line + line_offset,
                        "default": default,
                        "snippet": m.group(0)[:120],
                    })

            for m in _RE_ENVIRON_BRACKET.finditer(source):
                key = m.group(1)
                line_offset = source[: m.start()].count("\n")
                env_refs[key].append({
                    "file": file_path,
                    "line": base_line + line_offset,
                    "default": None,
                    "snippet": m.group(0)[:120],
                })

            for m in _RE_PROCESS_ENV.finditer(source):
                key = m.group(1)
                line_offset = source[: m.start()].count("\n")
                env_refs[key].append({
                    "file": file_path,
                    "line": base_line + line_offset,
                    "default": None,
                    "snippet": m.group(0)[:120],
                })

            for m in _RE_JAVA_VALUE.finditer(source):
                key = m.group(1)
                line_offset = source[: m.start()].count("\n")
                env_refs[key].append({
                    "file": file_path,
                    "line": base_line + line_offset,
                    "default": None,
                    "snippet": m.group(0)[:120],
                })

        # Check for conflicting defaults
        for key, refs in env_refs.items():
            defaults = {r["default"] for r in refs if r["default"] is not None}
            if len(defaults) > 1:
                evidence = [
                    {
                        "file": r["file"],
                        "line": r["line"],
                        "snippet": f'{key} default="{r["default"]}" -- {r["snippet"]}',
                    }
                    for r in refs
                    if r["default"] is not None
                ]
                findings.append(Finding(
                    id="",
                    category="config_conflict",
                    severity="high",
                    title=f"Conflicting defaults for {key}",
                    description=(
                        f"Environment variable `{key}` is read in "
                        f"{len(refs)} locations with {len(defaults)} "
                        f"different default values: {defaults}. "
                        f"This can cause inconsistent behavior depending "
                        f"on which code path executes first."
                    ),
                    evidence=evidence,
                    recommendation=(
                        f"Centralize the default for `{key}` in a single "
                        f"settings/config module and import from there."
                    ),
                ))

        return findings

    # ------------------------------------------------------------------
    # 2. Dead Configuration
    # ------------------------------------------------------------------

    def check_dead_config(
        self, db: DatabaseManager, project_id: str
    ) -> List[Finding]:
        """Find config keys defined but never referenced in code."""
        findings: List[Finding] = []

        with db.get_session() as session:
            # Get config file contents
            config_rows = session.execute(
                sa_text("""
                    SELECT f.file_path, u.source, u.start_line
                    FROM code_units u
                    JOIN code_files f ON u.file_id = f.file_id
                    WHERE u.project_id = :pid
                      AND u.source IS NOT NULL
                      AND (f.file_path ILIKE '%%.env%%'
                           OR f.file_path ILIKE '%%.yaml'
                           OR f.file_path ILIKE '%%.yml'
                           OR f.file_path ILIKE '%%.properties'
                           OR f.file_path ILIKE '%%.json')
                """),
                {"pid": project_id},
            ).fetchall()

            # Get all non-config source code
            code_rows = session.execute(
                sa_text("""
                    SELECT u.source
                    FROM code_units u
                    JOIN code_files f ON u.file_id = f.file_id
                    WHERE u.project_id = :pid
                      AND u.source IS NOT NULL
                      AND f.file_path NOT ILIKE '%%.env%%'
                      AND f.file_path NOT ILIKE '%%.yaml'
                      AND f.file_path NOT ILIKE '%%.yml'
                      AND f.file_path NOT ILIKE '%%.properties'
                      AND f.file_path NOT ILIKE '%%.json'
                    LIMIT 5000
                """),
                {"pid": project_id},
            ).fetchall()

        # Build combined source text for searching (limit to avoid memory blow)
        all_source = "\n".join(r.source or "" for r in code_rows)

        # Extract defined config keys
        defined_keys: Dict[str, Dict[str, Any]] = {}
        for row in config_rows:
            source = row.source or ""
            file_path = row.file_path
            base_line = row.start_line or 0

            for m in _RE_DOTENV_LINE.finditer(source):
                key = m.group(1)
                line_offset = source[: m.start()].count("\n")
                defined_keys[key] = {
                    "file": file_path,
                    "line": base_line + line_offset,
                    "snippet": m.group(0)[:120],
                }

        # Check which keys are never referenced in code
        for key, info in defined_keys.items():
            # Search for the key name in source code (case-sensitive)
            if key not in all_source:
                findings.append(Finding(
                    id="",
                    category="dead_config",
                    severity="low",
                    title=f"Unused config key: {key}",
                    description=(
                        f"Configuration key `{key}` is defined in "
                        f"`{info['file']}` but never referenced in "
                        f"application source code."
                    ),
                    evidence=[{
                        "file": info["file"],
                        "line": info["line"],
                        "snippet": info["snippet"],
                    }],
                    recommendation=(
                        f"Remove `{key}` from the config file if it is "
                        f"no longer needed, or document its purpose."
                    ),
                ))

        return findings

    # ------------------------------------------------------------------
    # 3. Circular Dependencies
    # ------------------------------------------------------------------

    def check_circular_deps(
        self, db: DatabaseManager, project_id: str
    ) -> List[Finding]:
        """Detect import cycles: A -> B -> C -> A."""
        findings: List[Finding] = []

        with db.get_session() as session:
            # Get all import edges at file level
            edges = session.execute(
                sa_text("""
                    SELECT DISTINCT sf.file_path AS source_file,
                                    tf.file_path AS target_file
                    FROM code_edges e
                    JOIN code_units su ON e.source_unit_id = su.unit_id
                    JOIN code_units tu ON e.target_unit_id = tu.unit_id
                    JOIN code_files sf ON su.file_id = sf.file_id
                    JOIN code_files tf ON tu.file_id = tf.file_id
                    WHERE e.project_id = :pid
                      AND e.edge_type = 'imports'
                      AND sf.file_path != tf.file_path
                """),
                {"pid": project_id},
            ).fetchall()

        # Build adjacency list
        graph: Dict[str, Set[str]] = defaultdict(set)
        for row in edges:
            graph[row.source_file].add(row.target_file)

        # DFS cycle detection
        cycles = self._find_cycles(graph)

        for i, cycle in enumerate(cycles[:20]):  # Cap at 20 cycles
            evidence = [
                {"file": f, "line": "", "snippet": f"Part of import cycle"}
                for f in cycle
            ]
            cycle_str = " -> ".join(cycle) + " -> " + cycle[0]
            length = len(cycle)
            severity = "high" if length <= 3 else "medium"

            findings.append(Finding(
                id="",
                category="circular_dep",
                severity=severity,
                title=f"Circular import: {length} files",
                description=(
                    f"Import cycle detected: `{cycle_str}`. "
                    f"Circular dependencies make refactoring difficult, "
                    f"can cause import errors at runtime, and indicate "
                    f"that module boundaries need restructuring."
                ),
                evidence=evidence,
                recommendation=(
                    f"Break the cycle by extracting shared types/interfaces "
                    f"into a separate module, or use dependency inversion "
                    f"(depend on abstractions, not concretions)."
                ),
            ))

        return findings

    @staticmethod
    def _find_cycles(graph: Dict[str, Set[str]]) -> List[List[str]]:
        """Find all simple cycles using DFS with back-edge detection."""
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        path: List[str] = []
        cycles: List[List[str]] = []
        seen_cycle_sets: Set[frozenset] = set()

        def dfs(node: str) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in rec_stack:
                    # Found a cycle -- extract it
                    try:
                        idx = path.index(neighbor)
                        cycle = path[idx:]
                        cycle_set = frozenset(cycle)
                        if cycle_set not in seen_cycle_sets:
                            seen_cycle_sets.add(cycle_set)
                            cycles.append(list(cycle))
                    except ValueError:
                        pass

            path.pop()
            rec_stack.discard(node)

        for node in graph:
            if node not in visited:
                dfs(node)

        return cycles

    # ------------------------------------------------------------------
    # 4. Orphaned Code
    # ------------------------------------------------------------------

    def check_orphaned_code(
        self, db: DatabaseManager, project_id: str
    ) -> List[Finding]:
        """Find classes and functions with zero incoming references."""
        findings: List[Finding] = []

        with db.get_session() as session:
            # Classes with no incoming calls, type_dep, or inherits edges
            orphan_classes = session.execute(
                sa_text("""
                    SELECT u.name, u.qualified_name, f.file_path, u.start_line
                    FROM code_units u
                    JOIN code_files f ON u.file_id = f.file_id
                    LEFT JOIN code_edges e
                        ON e.target_unit_id = u.unit_id
                        AND e.edge_type IN ('calls', 'type_dep', 'inherits', 'implements')
                    WHERE u.project_id = :pid
                      AND u.unit_type = 'class'
                      AND e.id IS NULL
                      AND f.file_path NOT SIMILAR TO '%%(test_|_test|tests/|__tests__|spec/)%%'
                    LIMIT 100
                """),
                {"pid": project_id},
            ).fetchall()

            # Functions/methods with no incoming calls edges
            orphan_functions = session.execute(
                sa_text("""
                    SELECT u.name, u.qualified_name, f.file_path, u.start_line
                    FROM code_units u
                    JOIN code_files f ON u.file_id = f.file_id
                    LEFT JOIN code_edges e
                        ON e.target_unit_id = u.unit_id
                        AND e.edge_type = 'calls'
                    WHERE u.project_id = :pid
                      AND u.unit_type IN ('function', 'method')
                      AND e.id IS NULL
                      AND u.name NOT LIKE '\\_%'
                      AND u.name NOT IN ('__init__', '__str__', '__repr__',
                                         'main', 'setUp', 'tearDown',
                                         'setup', 'teardown')
                      AND f.file_path NOT SIMILAR TO '%%(test_|_test|tests/|__tests__|spec/)%%'
                    LIMIT 200
                """),
                {"pid": project_id},
            ).fetchall()

        for row in orphan_classes:
            findings.append(Finding(
                id="",
                category="orphaned_code",
                severity="medium",
                title=f"Unreferenced class: {row.name}",
                description=(
                    f"Class `{row.qualified_name or row.name}` has no "
                    f"incoming references (no instantiation, inheritance, "
                    f"or type dependency from other units). It may be "
                    f"dead code or missing integration."
                ),
                evidence=[{
                    "file": row.file_path,
                    "line": row.start_line or "",
                    "snippet": f"class {row.name} -- zero incoming edges",
                }],
                recommendation=(
                    f"Verify whether `{row.name}` is used via dynamic "
                    f"dispatch, reflection, or configuration. If unused, "
                    f"remove it to reduce maintenance burden."
                ),
            ))

        for row in orphan_functions:
            findings.append(Finding(
                id="",
                category="orphaned_code",
                severity="low",
                title=f"Uncalled function: {row.name}",
                description=(
                    f"Function `{row.qualified_name or row.name}` has "
                    f"no incoming call edges. It may be dead code, or "
                    f"called via dynamic dispatch not captured by the ASG."
                ),
                evidence=[{
                    "file": row.file_path,
                    "line": row.start_line or "",
                    "snippet": f"def {row.name}() -- zero call edges",
                }],
                recommendation=(
                    f"Check if `{row.name}` is used via callbacks, "
                    f"decorators, or framework conventions. Remove if "
                    f"genuinely unused."
                ),
            ))

        return findings

    # ------------------------------------------------------------------
    # 5. Inconsistent Error Handling
    # ------------------------------------------------------------------

    def check_error_handling_consistency(
        self, db: DatabaseManager, project_id: str
    ) -> List[Finding]:
        """Find function groups where some have error handling and some don't."""
        findings: List[Finding] = []

        with db.get_session() as session:
            rows = session.execute(
                sa_text("""
                    SELECT u.unit_id, u.name, u.qualified_name,
                           f.file_path, u.start_line, u.source
                    FROM code_units u
                    JOIN code_files f ON u.file_id = f.file_id
                    WHERE u.project_id = :pid
                      AND u.unit_type IN ('function', 'method')
                      AND u.source IS NOT NULL
                      AND f.file_path NOT SIMILAR TO '%%(test_|_test|tests/|__tests__|spec/)%%'
                    LIMIT 3000
                """),
                {"pid": project_id},
            ).fetchall()

        # Classify each function
        has_eh: Set[str] = set()
        func_info: Dict[str, Any] = {}

        for row in rows:
            uid = str(row.unit_id)
            source = row.source or ""
            func_info[uid] = row
            if _RE_ERROR_HANDLING.search(source):
                has_eh.add(uid)

        # Group by naming prefix
        groups: Dict[str, List[str]] = defaultdict(list)
        for uid, row in func_info.items():
            match = _RE_FUNC_PREFIX.match(row.name)
            if match:
                groups[match.group(1)].append(uid)

        # Check consistency within each group
        for prefix, uids in groups.items():
            if len(uids) < 3:
                continue  # Too small a group to be meaningful

            with_eh = [u for u in uids if u in has_eh]
            without_eh = [u for u in uids if u not in has_eh]

            # Flag only when majority has it but some don't
            if len(with_eh) > 0 and len(without_eh) > 0 and len(with_eh) > len(without_eh):
                for uid in without_eh:
                    row = func_info[uid]
                    findings.append(Finding(
                        id="",
                        category="inconsistent_error_handling",
                        severity="medium",
                        title=f"Missing error handling: {row.name}",
                        description=(
                            f"Function `{row.name}` in the `{prefix}*` "
                            f"group lacks error handling, while "
                            f"{len(with_eh)}/{len(uids)} siblings in the "
                            f"same group have try/except/catch blocks. "
                            f"Inconsistent error handling can lead to "
                            f"unhandled exceptions in production."
                        ),
                        evidence=[{
                            "file": row.file_path,
                            "line": row.start_line or "",
                            "snippet": f"{row.name} -- no try/except/catch",
                        }],
                        recommendation=(
                            f"Add error handling to `{row.name}` consistent "
                            f"with the pattern used by its `{prefix}*` "
                            f"siblings."
                        ),
                    ))

        return findings

    # ------------------------------------------------------------------
    # 6. Hidden Coupling
    # ------------------------------------------------------------------

    def check_hidden_coupling(
        self, db: DatabaseManager, project_id: str
    ) -> List[Finding]:
        """Find modules sharing state without explicit dependency edges."""
        findings: List[Finding] = []

        with db.get_session() as session:
            # Find DB table names referenced in source (via SQL or ORM)
            rows = session.execute(
                sa_text("""
                    SELECT u.name, u.qualified_name, f.file_path,
                           u.start_line, u.source
                    FROM code_units u
                    JOIN code_files f ON u.file_id = f.file_id
                    WHERE u.project_id = :pid
                      AND u.source IS NOT NULL
                      AND u.unit_type IN ('function', 'method', 'class')
                      AND (u.source ILIKE '%%SELECT %%FROM %%'
                           OR u.source ILIKE '%%INSERT INTO %%'
                           OR u.source ILIKE '%%UPDATE %%SET %%'
                           OR u.source ILIKE '%%DELETE FROM %%'
                           OR u.source ILIKE '%%__tablename__%%')
                    LIMIT 2000
                """),
                {"pid": project_id},
            ).fetchall()

        # table_name -> set of files that access it
        table_accessors: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        re_sql_table = re.compile(
            r"\b(?:FROM|INTO|UPDATE|JOIN)\s+([a-z_][a-z0-9_]*)\b",
            re.IGNORECASE,
        )
        re_tablename = re.compile(
            r"""__tablename__\s*=\s*["']([a-z_][a-z0-9_]*)["']"""
        )

        for row in rows:
            source = row.source or ""
            file_path = row.file_path

            tables: Set[str] = set()
            for m in re_sql_table.finditer(source):
                tbl = m.group(1).lower()
                # Skip SQL keywords that look like table names
                if tbl not in ("set", "where", "and", "or", "not", "null",
                               "select", "insert", "update", "delete",
                               "values", "into", "from"):
                    tables.add(tbl)
            for m in re_tablename.finditer(source):
                tables.add(m.group(1).lower())

            for tbl in tables:
                table_accessors[tbl].append({
                    "file": file_path,
                    "line": row.start_line or "",
                    "unit": row.qualified_name or row.name,
                })

        # Flag tables accessed from multiple distinct files
        for tbl, accessors in table_accessors.items():
            unique_files = {a["file"] for a in accessors}
            if len(unique_files) >= 3:
                evidence = [
                    {
                        "file": a["file"],
                        "line": a["line"],
                        "snippet": f"{a['unit']} accesses table `{tbl}`",
                    }
                    for a in accessors[:10]
                ]
                findings.append(Finding(
                    id="",
                    category="hidden_coupling",
                    severity="medium",
                    title=f"Shared DB table: {tbl} ({len(unique_files)} files)",
                    description=(
                        f"Table `{tbl}` is accessed from {len(unique_files)} "
                        f"different files without a centralized repository "
                        f"pattern. Direct table access from multiple modules "
                        f"creates hidden coupling and makes schema changes "
                        f"risky."
                    ),
                    evidence=evidence,
                    recommendation=(
                        f"Centralize all access to `{tbl}` through a "
                        f"single repository/DAO class to encapsulate "
                        f"the data access pattern."
                    ),
                ))

        return findings

    # ------------------------------------------------------------------
    # 7. API Contract Gaps
    # ------------------------------------------------------------------

    def check_api_contracts(
        self, db: DatabaseManager, project_id: str
    ) -> List[Finding]:
        """Find mismatches between backend response models and frontend types."""
        findings: List[Finding] = []

        with db.get_session() as session:
            # Find backend response models (Pydantic BaseModel, TypedDict, dataclass)
            backend_rows = session.execute(
                sa_text("""
                    SELECT u.name, u.qualified_name, f.file_path,
                           u.start_line, u.source
                    FROM code_units u
                    JOIN code_files f ON u.file_id = f.file_id
                    WHERE u.project_id = :pid
                      AND u.unit_type = 'class'
                      AND u.source IS NOT NULL
                      AND (u.source ILIKE '%%BaseModel%%'
                           OR u.source ILIKE '%%TypedDict%%'
                           OR u.source ILIKE '%%@dataclass%%'
                           OR u.source ILIKE '%%dataclass%%')
                      AND (u.name ILIKE '%%Response%%'
                           OR u.name ILIKE '%%Schema%%'
                           OR u.name ILIKE '%%DTO%%')
                    LIMIT 200
                """),
                {"pid": project_id},
            ).fetchall()

            # Find frontend TypeScript interfaces/types
            frontend_rows = session.execute(
                sa_text("""
                    SELECT u.name, u.qualified_name, f.file_path,
                           u.start_line, u.source
                    FROM code_units u
                    JOIN code_files f ON u.file_id = f.file_id
                    WHERE u.project_id = :pid
                      AND u.source IS NOT NULL
                      AND u.unit_type IN ('interface', 'class', 'type')
                      AND f.file_path ILIKE '%%frontend%%'
                    LIMIT 200
                """),
                {"pid": project_id},
            ).fetchall()

        if not backend_rows or not frontend_rows:
            return findings

        # Extract field names from backend models
        re_py_field = re.compile(r"^\s+(\w+)\s*[:=]", re.MULTILINE)

        backend_models: Dict[str, Dict[str, Any]] = {}
        for row in backend_rows:
            fields = set(re_py_field.findall(row.source or ""))
            # Filter out dunder and private
            fields = {f for f in fields if not f.startswith("_")}
            if fields:
                backend_models[row.name] = {
                    "fields": fields,
                    "file": row.file_path,
                    "line": row.start_line,
                }

        # Extract field names from frontend types
        re_ts_field = re.compile(r"^\s+(\w+)\s*[?:]", re.MULTILINE)

        frontend_types: Dict[str, Dict[str, Any]] = {}
        for row in frontend_rows:
            fields = set(re_ts_field.findall(row.source or ""))
            fields = {f for f in fields if not f.startswith("_")}
            if fields:
                frontend_types[row.name] = {
                    "fields": fields,
                    "file": row.file_path,
                    "line": row.start_line,
                }

        # Cross-reference: find types with similar names
        for be_name, be_info in backend_models.items():
            # Normalize name for matching (strip Response/Schema/DTO suffix)
            base_name = re.sub(
                r"(Response|Schema|DTO|Model)$", "", be_name, flags=re.IGNORECASE
            )
            if not base_name:
                continue

            for fe_name, fe_info in frontend_types.items():
                fe_base = re.sub(
                    r"(Props|Type|Interface|Data)$", "", fe_name, flags=re.IGNORECASE
                )
                if not fe_base:
                    continue

                # Check if names are similar enough to be a contract pair
                if base_name.lower() == fe_base.lower() or base_name.lower() in fe_name.lower():
                    be_fields = be_info["fields"]
                    fe_fields = fe_info["fields"]

                    # Fields in backend but missing in frontend
                    missing_in_fe = be_fields - fe_fields
                    # Fields in frontend but missing in backend
                    missing_in_be = fe_fields - be_fields

                    if missing_in_fe or missing_in_be:
                        evidence = [
                            {
                                "file": be_info["file"],
                                "line": be_info["line"] or "",
                                "snippet": f"Backend {be_name}: {sorted(be_fields)[:10]}",
                            },
                            {
                                "file": fe_info["file"],
                                "line": fe_info["line"] or "",
                                "snippet": f"Frontend {fe_name}: {sorted(fe_fields)[:10]}",
                            },
                        ]
                        desc_parts = []
                        if missing_in_fe:
                            desc_parts.append(
                                f"Fields in backend but not frontend: "
                                f"`{sorted(missing_in_fe)[:10]}`"
                            )
                        if missing_in_be:
                            desc_parts.append(
                                f"Fields in frontend but not backend: "
                                f"`{sorted(missing_in_be)[:10]}`"
                            )

                        findings.append(Finding(
                            id="",
                            category="api_contract_gap",
                            severity="medium",
                            title=f"API contract mismatch: {be_name} vs {fe_name}",
                            description=(
                                f"Backend model `{be_name}` and frontend type "
                                f"`{fe_name}` appear to represent the same "
                                f"API contract but have field mismatches. "
                                + " ".join(desc_parts)
                            ),
                            evidence=evidence,
                            recommendation=(
                                f"Synchronize field definitions between "
                                f"`{be_name}` and `{fe_name}`, or generate "
                                f"TypeScript types from the backend schema."
                            ),
                        ))

        return findings

    # ------------------------------------------------------------------
    # 8. Duplicate Logic
    # ------------------------------------------------------------------

    def check_duplicate_logic(
        self, db: DatabaseManager, project_id: str
    ) -> List[Finding]:
        """Find functions with suspiciously similar source code."""
        findings: List[Finding] = []

        with db.get_session() as session:
            rows = session.execute(
                sa_text("""
                    SELECT u.unit_id, u.name, u.qualified_name,
                           f.file_path, u.start_line, u.source
                    FROM code_units u
                    JOIN code_files f ON u.file_id = f.file_id
                    WHERE u.project_id = :pid
                      AND u.unit_type IN ('function', 'method')
                      AND u.source IS NOT NULL
                      AND LENGTH(u.source) > 100
                      AND f.file_path NOT SIMILAR TO '%%(test_|_test|tests/|__tests__|spec/)%%'
                    LIMIT 1000
                """),
                {"pid": project_id},
            ).fetchall()

        # Tokenize each function
        units: List[Tuple[Any, Counter]] = []
        for row in rows:
            tokens = self._tokenize(row.source or "")
            if sum(tokens.values()) >= 10:  # Skip trivially small
                units.append((row, tokens))

        # Compare pairs (O(n^2) but capped at 1000)
        seen_pairs: Set[frozenset] = set()

        for i in range(len(units)):
            for j in range(i + 1, len(units)):
                row_a, tokens_a = units[i]
                row_b, tokens_b = units[j]

                # Quick check: size ratio must be within 2x
                size_a = sum(tokens_a.values())
                size_b = sum(tokens_b.values())
                if max(size_a, size_b) > 2 * min(size_a, size_b):
                    continue

                similarity = self._jaccard(tokens_a, tokens_b)
                if similarity >= 0.80:
                    pair_key = frozenset([str(row_a.unit_id), str(row_b.unit_id)])
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    findings.append(Finding(
                        id="",
                        category="duplicate_logic",
                        severity="medium" if similarity >= 0.90 else "low",
                        title=(
                            f"Duplicate logic: {row_a.name} ~ {row_b.name} "
                            f"({similarity:.0%})"
                        ),
                        description=(
                            f"Functions `{row_a.qualified_name or row_a.name}` "
                            f"and `{row_b.qualified_name or row_b.name}` have "
                            f"{similarity:.0%} token similarity, indicating "
                            f"potential code duplication (DRY violation)."
                        ),
                        evidence=[
                            {
                                "file": row_a.file_path,
                                "line": row_a.start_line or "",
                                "snippet": f"{row_a.name} ({size_a} tokens)",
                            },
                            {
                                "file": row_b.file_path,
                                "line": row_b.start_line or "",
                                "snippet": f"{row_b.name} ({size_b} tokens)",
                            },
                        ],
                        recommendation=(
                            f"Extract common logic into a shared function "
                            f"and call it from both `{row_a.name}` and "
                            f"`{row_b.name}`."
                        ),
                    ))

        return findings[:50]  # Cap total duplicate findings

    @staticmethod
    def _tokenize(source: str) -> Counter:
        """Tokenize source into a bag of words (lowercased)."""
        tokens = re.findall(r"\b\w+\b", source.lower())
        return Counter(tokens)

    @staticmethod
    def _jaccard(a: Counter, b: Counter) -> float:
        """Jaccard similarity on token multisets."""
        intersection = sum((a & b).values())
        union = sum((a | b).values())
        return intersection / max(union, 1)

    # ------------------------------------------------------------------
    # 9. Missing Test Coverage Indicators
    # ------------------------------------------------------------------

    def check_execution_coverage(
        self, db: DatabaseManager, project_id: str
    ) -> List[Finding]:
        """Find functions unreachable from any entry point (dead execution paths)."""
        findings: List[Finding] = []

        with db.get_session() as session:
            # Get all entry point unit IDs
            entry_ids = session.execute(
                sa_text("""
                    SELECT DISTINCT entry_unit_id FROM deep_analyses
                    WHERE project_id = :pid
                """),
                {"pid": project_id},
            ).fetchall()
            entry_set = {str(r.entry_unit_id) for r in entry_ids}

            if not entry_set:
                return findings  # No entry points detected yet

            # Get all reachable units via transitive call edges from entry points
            reachable = session.execute(
                sa_text("""
                    WITH RECURSIVE reachable AS (
                        SELECT target_unit_id AS unit_id, 1 AS depth
                        FROM code_edges
                        WHERE source_unit_id = ANY(:eids::uuid[])
                          AND edge_type = 'calls' AND project_id = :pid
                        UNION
                        SELECT e.target_unit_id, r.depth + 1
                        FROM code_edges e
                        JOIN reachable r ON e.source_unit_id = r.unit_id
                        WHERE e.edge_type = 'calls' AND e.project_id = :pid
                          AND r.depth < 10
                    )
                    SELECT DISTINCT unit_id FROM reachable
                """),
                {"pid": project_id, "eids": list(entry_set)},
            ).fetchall()
            reachable_set = {str(r.unit_id) for r in reachable} | entry_set

            # Get all non-test functions/methods
            all_funcs = session.execute(
                sa_text("""
                    SELECT u.unit_id, u.name, u.qualified_name, f.file_path, u.start_line
                    FROM code_units u
                    JOIN code_files f ON u.file_id = f.file_id
                    WHERE u.project_id = :pid
                      AND u.unit_type IN ('function', 'method')
                      AND f.file_path NOT SIMILAR TO '%%(test_|_test|tests/)%%'
                      AND u.name NOT LIKE '\\_%\\_%'
                    ORDER BY f.file_path, u.name
                """),
                {"pid": project_id},
            ).fetchall()

        unreachable = [f for f in all_funcs if str(f.unit_id) not in reachable_set]
        total = len(all_funcs)
        reachable_count = total - len(unreachable)
        coverage_pct = reachable_count / max(total, 1) * 100

        if unreachable and coverage_pct < 90:
            findings.append(Finding(
                id="",
                category="execution_coverage",
                severity="high" if coverage_pct < 50 else "medium",
                title=f"Execution coverage: {coverage_pct:.0f}% ({reachable_count}/{total} functions reachable from entry points)",
                description=(
                    f"{len(unreachable)} functions are not reachable from any detected entry point. "
                    f"These may be dead code, dynamically invoked, or entry points not yet detected."
                ),
                evidence=[{
                    "file": f.file_path,
                    "line": f.start_line or "",
                    "snippet": f"{f.qualified_name or f.name}",
                } for f in unreachable[:20]],
                recommendation=(
                    "Review unreachable functions — they may be dead code to remove, "
                    "or dynamically invoked code that needs explicit entry point registration."
                ),
            ))

        return findings

    # ------------------------------------------------------------------
    # 10. Complex Logic Documentation
    # ------------------------------------------------------------------

    def check_complex_logic(
        self, db: DatabaseManager, project_id: str
    ) -> List[Finding]:
        """Document functions with complex branching — what each path does."""
        findings: List[Finding] = []

        with db.get_session() as session:
            # Get high-complexity functions with source
            complex_funcs = session.execute(
                sa_text("""
                    SELECT u.name, u.qualified_name, f.file_path,
                           u.start_line, u.end_line, u.source,
                           (u.metadata->>'cyclomatic_complexity')::int AS complexity
                    FROM code_units u
                    JOIN code_files f ON u.file_id = f.file_id
                    WHERE u.project_id = :pid
                      AND u.unit_type IN ('function', 'method')
                      AND u.metadata->>'cyclomatic_complexity' IS NOT NULL
                      AND (u.metadata->>'cyclomatic_complexity')::int >= 10
                    ORDER BY (u.metadata->>'cyclomatic_complexity')::int DESC
                    LIMIT 30
                """),
                {"pid": project_id},
            ).fetchall()

        for func in complex_funcs:
            source = func.source or ""
            # Count and describe branching paths
            if_count = len(re.findall(r'\bif\b|\bIF\b', source))
            elif_count = len(re.findall(r'\belif\b|\bELIF\b|\bELSE\s+IF\b|\bELSEIF\b', source))
            else_count = len(re.findall(r'\belse\b|\bELSE\b', source))
            switch_count = len(re.findall(r'\bswitch\b|\bEVALUATE\b|\bmatch\b', source))
            loop_count = len(re.findall(r'\bfor\b|\bwhile\b|\bFOR\b|\bWHILE\b|\bPERFORM\b', source))
            try_count = len(re.findall(r'\btry\b|\bTRY\b|\bBEGIN\b', source))
            return_count = len(re.findall(r'\breturn\b|\bRETURN\b|\bGOBACK\b', source))

            # Extract key conditions (first line of each if/elif)
            conditions = []
            for m in re.finditer(
                r'(?:if|elif|IF|EVALUATE|WHEN|case)\s+(.+?)(?:\s*[:{]|\s*THEN|\s*$)',
                source, re.MULTILINE
            ):
                cond = m.group(1).strip()[:80]
                if cond and cond not in conditions:
                    conditions.append(cond)
                if len(conditions) >= 8:
                    break

            path_desc = (
                f"{if_count} if-branches, {elif_count} elif, {else_count} else, "
                f"{switch_count} switch/evaluate, {loop_count} loops, "
                f"{try_count} try blocks, {return_count} exit points"
            )

            findings.append(Finding(
                id="",
                category="complex_logic",
                severity="high" if func.complexity >= 20 else "medium",
                title=f"Complex logic: {func.name} (complexity {func.complexity})",
                description=(
                    f"`{func.qualified_name or func.name}` has cyclomatic complexity "
                    f"{func.complexity} with {path_desc}. "
                    f"Key conditions: {'; '.join(conditions[:5]) if conditions else 'N/A'}"
                ),
                evidence=[{
                    "file": func.file_path,
                    "line": func.start_line or "",
                    "snippet": f"Complexity={func.complexity}, lines={func.start_line}-{func.end_line}",
                }],
                recommendation=(
                    f"Consider decomposing `{func.name}` into smaller functions. "
                    f"Each major branch ({if_count} branches) could be a separate method."
                ),
            ))

        return findings

    # ------------------------------------------------------------------
    # 11. Functional Inconsistencies
    # ------------------------------------------------------------------

    def check_functional_inconsistencies(
        self, db: DatabaseManager, project_id: str
    ) -> List[Finding]:
        """Find functions with similar purposes but different implementations."""
        findings: List[Finding] = []

        with db.get_session() as session:
            # Get all functions with source
            funcs = session.execute(
                sa_text("""
                    SELECT u.unit_id, u.name, u.qualified_name, f.file_path,
                           u.start_line, u.source, u.metadata
                    FROM code_units u
                    JOIN code_files f ON u.file_id = f.file_id
                    WHERE u.project_id = :pid
                      AND u.unit_type IN ('function', 'method')
                      AND u.source IS NOT NULL
                      AND LENGTH(u.source) > 50
                      AND f.file_path NOT SIMILAR TO '%%(test_|_test|tests/)%%'
                    ORDER BY u.name
                """),
                {"pid": project_id},
            ).fetchall()

        # Group by semantic purpose (extracted from name)
        purpose_groups: Dict[str, list] = defaultdict(list)
        purpose_patterns = {
            "validate": r"^(validate|check|verify|is_valid|assert)",
            "create": r"^(create|make|build|generate|new|add|insert)",
            "update": r"^(update|modify|change|edit|set|patch)",
            "delete": r"^(delete|remove|destroy|drop|clear|purge)",
            "get": r"^(get|fetch|find|load|read|retrieve|lookup|query)",
            "format": r"^(format|render|display|show|present|to_string|to_dict)",
            "parse": r"^(parse|extract|decode|deserialize|from_)",
            "authenticate": r"^(auth|login|verify_token|check_perm|is_auth)",
            "send": r"^(send|notify|emit|publish|dispatch|post|push)",
            "save": r"^(save|store|persist|write|dump|serialize|export)",
        }

        for func in funcs:
            name_lower = func.name.lower()
            for purpose, pattern in purpose_patterns.items():
                if re.match(pattern, name_lower):
                    purpose_groups[purpose].append(func)
                    break

        # For each group with 2+ members, compare implementations
        for purpose, group in purpose_groups.items():
            if len(group) < 2:
                continue

            # Compare each pair for inconsistencies
            checked = set()
            for i, a in enumerate(group):
                for b in group[i + 1:]:
                    key = (str(a.unit_id), str(b.unit_id))
                    if key in checked:
                        continue
                    checked.add(key)

                    # Skip if in same file (likely overloads, not inconsistencies)
                    if a.file_path == b.file_path:
                        continue

                    src_a = (a.source or "").lower()
                    src_b = (b.source or "").lower()

                    # Check for structural differences in similar-purpose functions
                    a_has_try = bool(re.search(r'\btry\b|\bexcept\b|\bcatch\b', src_a))
                    b_has_try = bool(re.search(r'\btry\b|\bexcept\b|\bcatch\b', src_b))
                    a_has_validation = bool(re.search(r'\bif\b.*\bnot\b|\bif\b.*\bnone\b|\bif\b.*\bnull\b|\braise\b|\bthrow\b', src_a))
                    b_has_validation = bool(re.search(r'\bif\b.*\bnot\b|\bif\b.*\bnone\b|\bif\b.*\bnull\b|\braise\b|\bthrow\b', src_b))
                    a_has_logging = bool(re.search(r'\blogger\b|\blogging\b|\bprint\b|\bconsole\b', src_a))
                    b_has_logging = bool(re.search(r'\blogger\b|\blogging\b|\bprint\b|\bconsole\b', src_b))

                    inconsistencies = []
                    if a_has_try != b_has_try:
                        inconsistencies.append("error handling (one has try/except, other doesn't)")
                    if a_has_validation != b_has_validation:
                        inconsistencies.append("input validation (one validates, other doesn't)")
                    if a_has_logging != b_has_logging:
                        inconsistencies.append("logging (one logs, other doesn't)")

                    if inconsistencies:
                        findings.append(Finding(
                            id="",
                            category="functional_inconsistency",
                            severity="medium",
                            title=f"Inconsistent `{purpose}` implementations: {a.name} vs {b.name}",
                            description=(
                                f"Two functions with similar purpose (`{purpose}`) have different "
                                f"implementation patterns: {', '.join(inconsistencies)}. "
                                f"This may indicate missing safeguards in one implementation."
                            ),
                            evidence=[
                                {"file": a.file_path, "line": a.start_line or "", "snippet": f"{a.qualified_name or a.name}"},
                                {"file": b.file_path, "line": b.start_line or "", "snippet": f"{b.qualified_name or b.name}"},
                            ],
                            recommendation=(
                                f"Review both `{purpose}` implementations and align their patterns. "
                                f"Consider extracting shared logic into a common helper."
                            ),
                        ))

            if len(findings) >= 30:
                break

        return findings

    # ------------------------------------------------------------------
    # 10. Hardcoded Values
    # ------------------------------------------------------------------

    def check_hardcoded_values(
        self, db: DatabaseManager, project_id: str
    ) -> List[Finding]:
        """Find magic numbers, hardcoded URLs, embedded credentials, paths."""
        findings: List[Finding] = []

        with db.get_session() as session:
            rows = session.execute(
                sa_text("""
                    SELECT u.name, u.qualified_name, f.file_path,
                           u.start_line, u.source
                    FROM code_units u
                    JOIN code_files f ON u.file_id = f.file_id
                    WHERE u.project_id = :pid
                      AND u.source IS NOT NULL
                      AND u.unit_type IN ('function', 'method', 'class', 'module')
                      AND f.file_path NOT SIMILAR TO '%%(test_|_test|tests/|__tests__|spec/)%%'
                    LIMIT 3000
                """),
                {"pid": project_id},
            ).fetchall()

        for row in rows:
            source = row.source or ""
            file_path = row.file_path
            base_line = row.start_line or 0

            # Skip constants/config definition files
            if _RE_CONSTANTS_FILE.search(file_path):
                continue

            # Hardcoded secrets (CRITICAL)
            for m in _RE_HARDCODED_SECRET.finditer(source):
                line_offset = source[: m.start()].count("\n")
                # Skip if it looks like a placeholder
                value_part = m.group(0)
                if any(p in value_part.lower() for p in
                       ("changeme", "placeholder", "xxx", "your_", "example",
                        "todo", "fixme", "<", "{")):
                    continue
                findings.append(Finding(
                    id="",
                    category="hardcoded_value",
                    severity="critical",
                    title=f"Hardcoded credential in {row.name}",
                    description=(
                        f"Potential hardcoded credential found in "
                        f"`{row.qualified_name or row.name}`. Secrets "
                        f"should never be stored in source code."
                    ),
                    evidence=[{
                        "file": file_path,
                        "line": base_line + line_offset,
                        "snippet": m.group(0)[:80] + "...",
                    }],
                    recommendation=(
                        "Move this credential to environment variables "
                        "or a secrets manager. Never commit secrets to "
                        "version control."
                    ),
                ))

            # Hardcoded IP addresses (HIGH)
            for m in _RE_IP_ADDR.finditer(source):
                ip = m.group(0)
                # Skip common non-issue IPs
                if ip in ("0.0.0.0", "127.0.0.1", "255.255.255.255",
                          "255.255.255.0"):
                    continue
                line_offset = source[: m.start()].count("\n")
                findings.append(Finding(
                    id="",
                    category="hardcoded_value",
                    severity="high",
                    title=f"Hardcoded IP address: {ip}",
                    description=(
                        f"Hardcoded IP address `{ip}` found in "
                        f"`{row.qualified_name or row.name}`. This "
                        f"reduces portability across environments."
                    ),
                    evidence=[{
                        "file": file_path,
                        "line": base_line + line_offset,
                        "snippet": source[max(0, m.start() - 20):m.end() + 20][:100],
                    }],
                    recommendation=(
                        f"Replace hardcoded IP `{ip}` with a "
                        f"configuration variable."
                    ),
                ))

            # Hardcoded file paths (MEDIUM)
            for m in _RE_HARDCODED_PATH.finditer(source):
                line_offset = source[: m.start()].count("\n")
                findings.append(Finding(
                    id="",
                    category="hardcoded_value",
                    severity="medium",
                    title=f"Hardcoded path in {row.name}",
                    description=(
                        f"Hardcoded file system path found in "
                        f"`{row.qualified_name or row.name}`. Platform-"
                        f"specific paths reduce portability."
                    ),
                    evidence=[{
                        "file": file_path,
                        "line": base_line + line_offset,
                        "snippet": m.group(0)[:100],
                    }],
                    recommendation=(
                        "Use configuration variables or platform-"
                        "independent path construction."
                    ),
                ))

        # Cap to avoid overwhelming output
        return findings[:80]

    # ------------------------------------------------------------------
    # Markdown formatter
    # ------------------------------------------------------------------

    def format_as_markdown(self, findings: List[Finding]) -> str:
        """Format findings as a markdown chapter."""
        lines = [
            "# 15. Cross-Reference Findings & Architectural Issues",
            "",
            "This chapter contains deterministic, evidence-based findings "
            "produced by cross-referencing the codebase inventory (ASG, code "
            "units, source text). Every finding below is provable from source "
            "evidence -- no LLM was used.",
            "",
        ]

        if not findings:
            lines.append("No findings detected.")
            return "\n".join(lines)

        # Summary table
        by_severity: Dict[str, List[Finding]] = defaultdict(list)
        by_category: Dict[str, List[Finding]] = defaultdict(list)
        for f in findings:
            by_severity[f.severity].append(f)
            by_category[f.category].append(f)

        lines += [
            f"**Total Findings**: {len(findings)}",
            "",
            "| Severity | Count |",
            "|----------|-------|",
        ]
        for sev in ["critical", "high", "medium", "low", "info"]:
            if sev in by_severity:
                lines.append(f"| {sev.upper()} | {len(by_severity[sev])} |")
        lines += [""]

        lines += [
            "| Category | Count |",
            "|----------|-------|",
        ]
        for cat in sorted(by_category.keys()):
            lines.append(f"| {cat} | {len(by_category[cat])} |")
        lines += [""]

        # Each finding
        for f in findings:
            lines += [
                f"## {f.id}: {f.title}",
                "",
                f"**Severity**: {f.severity.upper()} | **Category**: {f.category}",
                "",
                f"{f.description}",
                "",
                "**Evidence**:",
                "",
            ]
            for ev in f.evidence:
                loc = f"`{ev.get('file', '?')}:{ev.get('line', '')}`"
                snippet = (ev.get("snippet", "") or "")[:120]
                lines.append(f"- {loc} -- {snippet}")
            lines += [
                "",
                f"**Recommendation**: {f.recommendation}",
                "",
                "---",
                "",
            ]

        return "\n".join(lines)

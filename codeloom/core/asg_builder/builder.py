"""ASG Builder — extracts semantic edges from parsed code units.

Runs as a post-processing step after AST parsing. Reads all code units
for a project from the database, detects relationships (contains, imports,
calls, inherits, implements, overrides, type_dep), and stores them as
CodeEdge records.
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from ..db import DatabaseManager
from ..db.models import CodeEdge, CodeUnit

logger = logging.getLogger(__name__)

# Regex to extract base classes from class signatures
_PYTHON_BASES_RE = re.compile(r"class\s+\w+\s*\(([^)]+)\)\s*:")
_JS_EXTENDS_RE = re.compile(r"class\s+\w+\s+extends\s+(\w+)")

# Regex to extract call targets from source
_CALL_RE = re.compile(r"(?<!\w)(\w+)\s*\(")
# Qualified calls: obj.method() or Package.Class.method()
_QUALIFIED_CALL_RE = re.compile(r"(?:\w+\.)+(\w+)\s*\(")

# ── SP call detection patterns ──────────────────────────────────────
# Java: CallableStatement / prepareCall("{ call usp_Name(...) }")
_JAVA_SP_CALL_RE = re.compile(
    r"""(?:prepareCall|callproc)\s*\(\s*["']\{?\s*call\s+(\w+)""",
    re.IGNORECASE,
)
# Java: @Procedure(name = "usp_Name") or @Procedure("usp_Name")
_JAVA_PROC_ANNOTATION_RE = re.compile(
    r"""@Procedure\s*\(\s*(?:name\s*=\s*)?["'](\w+)["']""",
    re.IGNORECASE,
)
# C#: SqlCommand(..., "usp_Name") + CommandType.StoredProcedure
_CSHARP_SP_RE = re.compile(
    r"""CommandText\s*=\s*["'](\w+)["']|new\s+SqlCommand\s*\(\s*["'](\w+)["']""",
    re.IGNORECASE,
)
# Python: cursor.callproc("usp_Name")
_PYTHON_CALLPROC_RE = re.compile(
    r"""callproc\s*\(\s*["'](\w+)["']""",
    re.IGNORECASE,
)
# Generic: EXEC[UTE] usp_Name in string literals
_EXEC_IN_STRING_RE = re.compile(
    r"""["'].*?\bEXEC(?:UTE)?\s+(\w+).*?["']""",
    re.IGNORECASE,
)


class ASGBuilder:
    """Build the Abstract Semantic Graph for a project.

    Edge types produced:
    - contains:   class -> method/property/constructor (from parent_name field)
    - imports:    file unit -> imported unit (resolved within project)
    - calls:      function/method -> function/method (identifier matching)
    - inherits:   class -> base class (from signature parsing or metadata)
    - implements: class/struct/record -> interface (from metadata["implements"])
    - overrides:  method -> parent method (from @Override / override modifier)
    - calls_sp:   app code unit -> stored procedure (SP invocation patterns)
    - type_dep:   consumer -> referenced type (field types, param types, return types)
    """

    def __init__(self, db_manager: DatabaseManager):
        self._db = db_manager

    def build_edges(self, project_id: str) -> int:
        """Build all ASG edges for a project.

        Loads all code units, detects relationships, and bulk-inserts
        CodeEdge records. Uses ON CONFLICT DO NOTHING to be idempotent.

        Args:
            project_id: UUID string of the project

        Returns:
            Number of edges created
        """
        pid = UUID(project_id) if isinstance(project_id, str) else project_id

        with self._db.get_session() as session:
            # Load all code units for this project
            units: List[CodeUnit] = (
                session.query(CodeUnit)
                .filter(CodeUnit.project_id == pid)
                .all()
            )

            if not units:
                logger.info(f"No code units found for project {project_id}")
                return 0

            logger.info(f"Building ASG for {len(units)} code units")

            # Build lookup structures
            unit_by_id: Dict[UUID, CodeUnit] = {u.unit_id: u for u in units}
            unit_by_name: Dict[str, CodeUnit] = {}
            unit_by_qualified: Dict[str, CodeUnit] = {}
            units_by_file: Dict[str, List[CodeUnit]] = {}

            for u in units:
                unit_by_name.setdefault(u.name, u)
                if u.qualified_name:
                    unit_by_qualified[u.qualified_name] = u
                file_key = str(u.file_id)
                units_by_file.setdefault(file_key, []).append(u)

            # Collect all edges
            edges: List[dict] = []

            # 1. Contains edges (class -> method/property/constructor)
            edges.extend(self._detect_contains(units, unit_by_name, unit_by_qualified, units_by_file, pid))

            # 2. Inherits edges (class -> base class)
            edges.extend(self._detect_inherits(units, unit_by_name, unit_by_qualified, pid))

            # 3. Implements edges (class/struct/record -> interface)
            edges.extend(self._detect_implements(units, unit_by_name, unit_by_qualified, pid))

            # 4. Calls edges (function -> function via identifier matching)
            all_names: Set[str] = {u.name for u in units}
            edges.extend(self._detect_calls(units, unit_by_name, all_names, pid))

            # 5. Imports edges (units importing other units)
            edges.extend(self._detect_imports(units, unit_by_name, unit_by_qualified, units_by_file, pid))

            # 6. Overrides edges (method -> parent method)
            edges.extend(self._detect_overrides(units, unit_by_name, unit_by_qualified, units_by_file, pid))

            # 7. SP call edges (app code -> stored procedure)
            sp_units = [u for u in units if u.unit_type in ("stored_procedure", "sql_function")]
            if sp_units:
                edges.extend(self._detect_sp_calls(units, sp_units, pid))

            # 8. Type dependency edges (field types, param types, return types)
            edges.extend(self._detect_type_deps(units, unit_by_name, unit_by_qualified, pid))

            # Deduplicate
            seen: Set[Tuple] = set()
            unique_edges = []
            for e in edges:
                key = (e["source_unit_id"], e["target_unit_id"], e["edge_type"])
                if key not in seen:
                    seen.add(key)
                    unique_edges.append(e)

            # Bulk insert with ON CONFLICT DO NOTHING
            if unique_edges:
                stmt = pg_insert(CodeEdge).values(unique_edges)
                stmt = stmt.on_conflict_do_nothing(
                    constraint="uq_code_edge"
                )
                session.execute(stmt)

            logger.info(f"ASG built: {len(unique_edges)} edges for project {project_id}")
            return len(unique_edges)

    # ── Edge detectors ──────────────────────────────────────────────────

    def _detect_contains(
        self,
        units: List[CodeUnit],
        unit_by_name: Dict[str, CodeUnit],
        unit_by_qualified: Dict[str, CodeUnit],
        units_by_file: Dict[str, List[CodeUnit]],
        project_id: UUID,
    ) -> List[dict]:
        """Detect contains edges: class -> method/property/constructor.

        Uses three strategies in order:
        1. unit_metadata["parent_name"] (set during ingestion)
        2. qualified_name parsing (e.g., "Module.ClassName.method" -> parent is "ClassName")
        3. Line range containment (method start_line within class start/end)
        """
        edges = []
        member_types = ("method", "constructor", "property")
        parent_types = ("class", "interface", "struct", "record")

        for u in units:
            if u.unit_type not in member_types:
                continue

            # Strategy 1: metadata parent_name
            meta = u.unit_metadata or {}
            parent_name = meta.get("parent_name")

            # Strategy 2: parse qualified_name
            if not parent_name and u.qualified_name:
                parts = u.qualified_name.rsplit(".", 2)
                if len(parts) >= 2:
                    parent_name = parts[-2]

            if not parent_name:
                continue

            # Find the parent class/interface/struct unit in the same file
            file_units = units_by_file.get(str(u.file_id), [])
            parent = None
            for fu in file_units:
                if fu.unit_type in parent_types and fu.name == parent_name:
                    parent = fu
                    break

            if parent:
                edges.append({
                    "project_id": project_id,
                    "source_unit_id": parent.unit_id,
                    "target_unit_id": u.unit_id,
                    "edge_type": "contains",
                    "edge_metadata": {},
                })
        return edges

    def _detect_inherits(
        self,
        units: List[CodeUnit],
        unit_by_name: Dict[str, CodeUnit],
        unit_by_qualified: Dict[str, CodeUnit],
        project_id: UUID,
    ) -> List[dict]:
        """Detect inheritance edges from metadata or class signatures.

        Prefers unit_metadata["extends"] (Java/C#) when available,
        falls back to regex parsing of signature (Python/JS/TS).
        """
        edges = []
        inheritable_types = ("class", "struct", "record")

        for u in units:
            if u.unit_type not in inheritable_types:
                continue

            meta = u.unit_metadata or {}
            base_names = []

            # Prefer metadata (Java/C# parsers set this explicitly)
            extends_meta = meta.get("extends")
            if extends_meta:
                if isinstance(extends_meta, str):
                    base_names.append(extends_meta)
                elif isinstance(extends_meta, list):
                    base_names.extend(extends_meta)
            elif u.signature:
                # Fallback: regex on signature (Python/JS/TS)
                base_names = self._extract_base_classes(u.signature, u.language)

            for base_name in base_names:
                target = self._resolve_unit(
                    base_name, unit_by_name, unit_by_qualified,
                    preferred_types=("class", "struct", "record"),
                )
                if target and target.unit_id != u.unit_id:
                    edges.append({
                        "project_id": project_id,
                        "source_unit_id": u.unit_id,
                        "target_unit_id": target.unit_id,
                        "edge_type": "inherits",
                        "edge_metadata": {"base_class": base_name},
                    })
        return edges

    def _detect_implements(
        self,
        units: List[CodeUnit],
        unit_by_name: Dict[str, CodeUnit],
        unit_by_qualified: Dict[str, CodeUnit],
        project_id: UUID,
    ) -> List[dict]:
        """Detect implements edges from unit_metadata["implements"].

        Source: class/struct/record unit with metadata["implements"] list.
        Target: interface unit resolved by name.
        """
        edges = []
        for u in units:
            if u.unit_type not in ("class", "struct", "record"):
                continue

            meta = u.unit_metadata or {}
            implements_list = meta.get("implements")
            if not implements_list:
                continue

            if isinstance(implements_list, str):
                implements_list = [implements_list]

            for iface_name in implements_list:
                target = self._resolve_unit(
                    iface_name, unit_by_name, unit_by_qualified,
                    preferred_types=("interface",),
                )
                if target and target.unit_id != u.unit_id:
                    edges.append({
                        "project_id": project_id,
                        "source_unit_id": u.unit_id,
                        "target_unit_id": target.unit_id,
                        "edge_type": "implements",
                        "edge_metadata": {"interface_name": iface_name},
                    })
        return edges

    def _detect_overrides(
        self,
        units: List[CodeUnit],
        unit_by_name: Dict[str, CodeUnit],
        unit_by_qualified: Dict[str, CodeUnit],
        units_by_file: Dict[str, List[CodeUnit]],
        project_id: UUID,
    ) -> List[dict]:
        """Detect override edges for methods annotated with @Override or override modifier.

        Walks the inheritance chain to find the parent method being overridden.
        """
        edges = []
        for u in units:
            if u.unit_type != "method":
                continue

            meta = u.unit_metadata or {}

            # Check for override indicator
            is_override = meta.get("is_override", False)
            if not is_override:
                annotations = meta.get("annotations", [])
                if "@Override" not in annotations:
                    modifiers = meta.get("modifiers", [])
                    if "override" not in modifiers:
                        continue

            # Find the parent class
            parent_class_name = meta.get("parent_name")
            if not parent_class_name and u.qualified_name:
                parts = u.qualified_name.rsplit(".", 2)
                if len(parts) >= 2:
                    parent_class_name = parts[-2]

            if not parent_class_name:
                continue

            # Find the parent class unit
            parent_class = self._resolve_unit(
                parent_class_name, unit_by_name, unit_by_qualified,
                preferred_types=("class", "struct", "record"),
            )
            if not parent_class:
                continue

            # Walk the inheritance chain to find the overridden method
            parent_class_meta = parent_class.unit_metadata or {}
            base_name = parent_class_meta.get("extends")
            if isinstance(base_name, list):
                base_name = base_name[0] if base_name else None

            # Also check implements for interface default methods
            search_bases = []
            if base_name:
                search_bases.append(base_name)
            impl_list = parent_class_meta.get("implements", [])
            if isinstance(impl_list, str):
                impl_list = [impl_list]
            search_bases.extend(impl_list)

            for base in search_bases:
                base_unit = self._resolve_unit(
                    base, unit_by_name, unit_by_qualified,
                    preferred_types=("class", "interface", "struct", "record"),
                )
                if not base_unit:
                    continue

                # Look for a method with the same name in the base
                target_qn = f"{base_unit.qualified_name}.{u.name}" if base_unit.qualified_name else f"{base_unit.name}.{u.name}"
                target = unit_by_qualified.get(target_qn)

                if not target:
                    # Broader search: any method with same name that belongs to base
                    for qu_qn, qu in unit_by_qualified.items():
                        if qu.name == u.name and qu.unit_type == "method":
                            qu_meta = qu.unit_metadata or {}
                            qu_parent = qu_meta.get("parent_name", "")
                            if qu_parent == base_unit.name:
                                target = qu
                                break

                if target and target.unit_id != u.unit_id:
                    edges.append({
                        "project_id": project_id,
                        "source_unit_id": u.unit_id,
                        "target_unit_id": target.unit_id,
                        "edge_type": "overrides",
                        "edge_metadata": {
                            "overriding_class": parent_class_name,
                            "parent_class": base_unit.name,
                        },
                    })
                    break  # Found the overridden method, stop searching

        return edges

    def _detect_sp_calls(
        self,
        units: List[CodeUnit],
        sp_units: List[CodeUnit],
        project_id: UUID,
    ) -> List[dict]:
        """Detect stored procedure invocations in application code.

        Scans Java, C#, and Python source for patterns that call stored procedures.
        Matches against known SP units parsed from .sql files.

        Patterns detected:
        - Java: prepareCall("{ call usp_Name }"), @Procedure("usp_Name")
        - C#:   CommandText = "usp_Name" (with StoredProcedure), SqlCommand("usp_Name")
        - Python: cursor.callproc("usp_Name"), cursor.execute("EXEC usp_Name")
        - Generic: EXEC usp_Name in string literals
        """
        edges = []
        sp_by_name: Dict[str, CodeUnit] = {u.name.lower(): u for u in sp_units}
        app_units = [u for u in units if u.language != "sql" and u.source]

        for u in app_units:
            if u.unit_type not in ("function", "method", "class"):
                continue

            sp_refs = self._extract_sp_references(u.source, u.language)

            for sp_name, call_pattern in sp_refs:
                target = sp_by_name.get(sp_name.lower())
                if target and target.unit_id != u.unit_id:
                    edges.append({
                        "project_id": project_id,
                        "source_unit_id": u.unit_id,
                        "target_unit_id": target.unit_id,
                        "edge_type": "calls_sp",
                        "edge_metadata": {
                            "call_pattern": call_pattern,
                            "sp_name": sp_name,
                        },
                    })

        return edges

    @staticmethod
    def _extract_sp_references(source: str, language: Optional[str]) -> List[Tuple[str, str]]:
        """Extract SP names referenced in application code.

        Returns list of (sp_name, call_pattern) tuples.
        """
        refs: List[Tuple[str, str]] = []
        seen: Set[str] = set()

        def _add(name: str, pattern: str):
            key = name.lower()
            if key not in seen:
                seen.add(key)
                refs.append((name, pattern))

        if language == "java":
            for m in _JAVA_SP_CALL_RE.finditer(source):
                _add(m.group(1), "CallableStatement")
            for m in _JAVA_PROC_ANNOTATION_RE.finditer(source):
                _add(m.group(1), "@Procedure")

        elif language == "csharp":
            # Check for StoredProcedure command type hint
            has_sp_type = "StoredProcedure" in source or "CommandType.StoredProcedure" in source
            if has_sp_type:
                for m in _CSHARP_SP_RE.finditer(source):
                    name = m.group(1) or m.group(2)
                    if name:
                        _add(name, "SqlCommand.StoredProcedure")

        elif language == "python":
            for m in _PYTHON_CALLPROC_RE.finditer(source):
                _add(m.group(1), "cursor.callproc")

        # Generic EXEC pattern in string literals (any language)
        for m in _EXEC_IN_STRING_RE.finditer(source):
            _add(m.group(1), "EXEC_in_string")

        return refs

    def _detect_calls(
        self,
        units: List[CodeUnit],
        unit_by_name: Dict[str, CodeUnit],
        all_names: Set[str],
        project_id: UUID,
    ) -> List[dict]:
        """Detect call edges by scanning function bodies for known identifiers.

        Uses both simple call patterns (func()) and qualified call patterns
        (obj.method(), Package.Class.method()).
        """
        edges = []
        for u in units:
            if u.unit_type not in ("function", "method") or not u.source:
                continue

            # Find all function-call-like patterns in the source
            called_names = set(_CALL_RE.findall(u.source))
            # Also capture qualified calls: obj.method() -> "method"
            called_names.update(_QUALIFIED_CALL_RE.findall(u.source))

            # Intersect with known unit names in the project
            matched = called_names & all_names
            # Remove self-calls and common builtins
            matched.discard(u.name)
            matched -= _BUILTINS

            for callee_name in matched:
                target = unit_by_name.get(callee_name)
                if target and target.unit_id != u.unit_id:
                    confidence = "high" if target.qualified_name and u.qualified_name else "medium"
                    edges.append({
                        "project_id": project_id,
                        "source_unit_id": u.unit_id,
                        "target_unit_id": target.unit_id,
                        "edge_type": "calls",
                        "edge_metadata": {"confidence": confidence},
                    })
        return edges

    def _detect_imports(
        self,
        units: List[CodeUnit],
        unit_by_name: Dict[str, CodeUnit],
        unit_by_qualified: Dict[str, CodeUnit],
        units_by_file: Dict[str, List[CodeUnit]],
        project_id: UUID,
    ) -> List[dict]:
        """Detect import edges by matching import statements to project units.

        Two strategies per file:
        1. metadata["file_imports"] — stamped during ingestion (Java/C#/Python/JS/TS)
        2. Fallback: scan unit source for import patterns (Python/JS/TS only —
           Java/C# imports live at file level, outside any unit's source)
        """
        edges = []
        processed_files: Set[str] = set()

        for u in units:
            file_key = str(u.file_id)
            if file_key in processed_files:
                continue
            processed_files.add(file_key)

            file_units = units_by_file.get(file_key, [])
            if not file_units:
                continue

            # Strategy 1: file_imports from metadata (set during ingestion)
            imported_names: Set[str] = set()
            source_unit = file_units[0]
            for fu in file_units:
                meta = fu.unit_metadata or {}
                file_imports = meta.get("file_imports")
                if file_imports:
                    imported_names.update(
                        self._names_from_import_statements(file_imports)
                    )
                    source_unit = fu
                    break

            # Strategy 2: fallback — scan unit source for import patterns
            if not imported_names:
                for fu in file_units:
                    if not fu.source:
                        continue
                    imported_names = self._extract_imported_names(fu)
                    if imported_names:
                        source_unit = fu
                        break

            # Resolve each imported name to a project unit
            for name in imported_names:
                target = unit_by_name.get(name)
                if not target:
                    for qn, qu in unit_by_qualified.items():
                        if qn.endswith(f".{name}"):
                            target = qu
                            break

                if target and target.file_id != source_unit.file_id:
                    edges.append({
                        "project_id": project_id,
                        "source_unit_id": source_unit.unit_id,
                        "target_unit_id": target.unit_id,
                        "edge_type": "imports",
                        "edge_metadata": {"import_name": name},
                    })

        return edges

    @staticmethod
    def _names_from_import_statements(statements: List[str]) -> Set[str]:
        """Extract type/class names from raw import statements.

        Handles:
          Java:   "import com.example.HazelcastService;"  -> "HazelcastService"
          Java:   "import static com.example.Utils.foo;"   -> "Utils" (class, not method)
          C#:     "using Namespace.ClassName;"             -> "ClassName"
          Python: "from foo import Bar"                    -> "Bar"
          JS/TS:  "import { Foo } from './bar'"            -> "Foo"
        """
        names: Set[str] = set()
        for stmt in statements:
            stmt = stmt.strip().rstrip(";")

            # Java / C# / VB.NET: dotted path — take last capitalized segment
            # "import com.example.service.HazelcastService" -> "HazelcastService"
            # "import static com.example.Utils.method" -> "Utils"
            # "Imports System.Collections.Generic" -> "Generic"
            if stmt.startswith(("import ", "using ", "Imports ")):
                # Remove keywords
                path = (stmt.replace("import static ", "")
                        .replace("import ", "")
                        .replace("using ", "")
                        .replace("Imports ", "")
                        .strip())
                if path.endswith(".*") or path.endswith("*"):
                    continue  # Wildcard
                segments = path.split(".")
                # Walk from the end to find the last capitalized segment (the type name)
                for seg in reversed(segments):
                    seg = seg.strip()
                    if seg and seg[0].isupper():
                        names.add(seg)
                        break

            # Python: "from foo.bar import Baz, Qux"
            match = re.search(r"from\s+\S+\s+import\s+(.+)", stmt)
            if match:
                for part in match.group(1).split(","):
                    clean = part.strip().split(" as ")[0].strip()
                    if clean and clean[0].isupper():
                        names.add(clean)

            # JS/TS: "import { Foo, Bar } from './baz'"
            match = re.search(r"import\s*\{([^}]+)\}\s*from", stmt)
            if match:
                for part in match.group(1).split(","):
                    clean = part.strip().split(" as ")[0].strip()
                    if clean:
                        names.add(clean)

        return names

    def _detect_type_deps(
        self,
        units: List[CodeUnit],
        unit_by_name: Dict[str, CodeUnit],
        unit_by_qualified: Dict[str, CodeUnit],
        project_id: UUID,
    ) -> List[dict]:
        """Detect type dependency edges from structured metadata.

        Reads pre-extracted metadata (no regex) to find type references:
        - Strategy A: metadata["parsed_params"][*].type → method/constructor depends on param type
        - Strategy B: metadata["return_type"] → method depends on return type
        - Strategy C: metadata["fields"][*].type → class depends on field type

        Direction: consumer --type_dep--> referenced_type
        """
        edges = []
        seen_pairs: Set[Tuple[UUID, UUID, str]] = set()

        for u in units:
            meta = u.unit_metadata or {}

            type_refs: List[Tuple[str, str]] = []  # (type_string, kind)

            # Strategy A: param types (methods/constructors)
            if u.unit_type in ("method", "constructor", "function"):
                parsed_params = meta.get("parsed_params", [])
                for param in parsed_params:
                    ptype = param.get("type")
                    if ptype:
                        type_refs.append((ptype, "param"))

            # Strategy B: return types (methods/functions)
            if u.unit_type in ("method", "function"):
                ret_type = meta.get("return_type")
                if ret_type:
                    type_refs.append((ret_type, "return"))

            # Strategy C: field types (classes/interfaces/structs)
            if u.unit_type in ("class", "interface", "struct", "record"):
                fields = meta.get("fields", [])
                for field in fields:
                    ftype = field.get("type")
                    if ftype:
                        type_refs.append((ftype, "field"))

            # Resolve each type reference to project units
            for type_str, kind in type_refs:
                identifiers = self._extract_type_identifiers(type_str)
                for ident in identifiers:
                    target = self._resolve_unit(
                        ident, unit_by_name, unit_by_qualified,
                        preferred_types=("class", "interface", "struct", "record", "enum"),
                    )
                    if not target or target.unit_id == u.unit_id:
                        continue

                    pair_key = (u.unit_id, target.unit_id, kind)
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    edges.append({
                        "project_id": project_id,
                        "source_unit_id": u.unit_id,
                        "target_unit_id": target.unit_id,
                        "edge_type": "type_dep",
                        "edge_metadata": {"kind": kind},
                    })

        logger.debug(f"type_dep detection: {len(edges)} edges")
        return edges

    @staticmethod
    def _extract_type_identifiers(type_str: str) -> List[str]:
        """Extract individual type identifiers from a type expression.

        Handles generics, arrays, nullables, and compound types:
          "Map<String, HazelcastService>" -> ["HazelcastService"]
          "List<User>"                    -> ["User"]
          "IMap<String, SittingMessage>"  -> ["SittingMessage"]
          "Optional[UserProfile]"         -> ["UserProfile"]
          "int"                           -> []
          "String[]"                      -> []

        Filters out primitives and common framework types that never
        correspond to user-defined code units.
        """
        # Find all capitalized identifiers (candidate user-defined types)
        candidates = re.findall(r"\b([A-Z]\w+)", type_str)
        return [c for c in candidates if c not in _PRIMITIVE_TYPES]

    # ── Re-enrichment for existing projects ──────────────────────────────

    @staticmethod
    def enrich_class_fields_from_db(db_manager: "DatabaseManager", project_id: str) -> int:
        """Re-enrich class units with field metadata for existing projects.

        For projects already ingested (class units in DB but no metadata["fields"]),
        this reads each class unit's stored source, parses it with tree-sitter,
        and updates the JSONB metadata with extracted fields.

        Called by the build-asg endpoint before edge building so that
        _detect_type_deps() has field data to work with.

        Returns:
            Number of class units updated with field metadata.
        """
        from ..ast_parser.enricher import SemanticEnricher

        pid = UUID(project_id) if isinstance(project_id, str) else project_id
        enricher = SemanticEnricher()
        updated = 0

        with db_manager.get_session() as session:
            class_units = (
                session.query(CodeUnit)
                .filter(
                    CodeUnit.project_id == pid,
                    CodeUnit.unit_type.in_(("class", "interface", "struct", "record")),
                )
                .all()
            )

            for cu in class_units:
                if not cu.source or not cu.language:
                    continue

                # Skip if fields already extracted
                meta = cu.unit_metadata or {}
                if meta.get("fields"):
                    continue

                fields = enricher.enrich_class_source(cu.source, cu.language)
                if fields:
                    new_meta = dict(meta)
                    new_meta["fields"] = fields
                    cu.unit_metadata = new_meta
                    updated += 1

        logger.info(f"Field re-enrichment: {updated}/{len(class_units)} class units updated")
        return updated

    # ── Resolution helpers ──────────────────────────────────────────────

    @staticmethod
    def _resolve_unit(
        name: str,
        unit_by_name: Dict[str, CodeUnit],
        unit_by_qualified: Dict[str, CodeUnit],
        preferred_types: tuple = (),
    ) -> Optional[CodeUnit]:
        """Resolve a unit by name with optional type preference.

        Resolution order:
        1. Exact name match (filtered by preferred_types if provided)
        2. Qualified name suffix match
        3. Fallback: exact name match without type filter
        """
        # Strip generics for lookup: List<User> -> List, IRepository<T> -> IRepository
        bare_name = name.split("<")[0].strip()
        # Strip namespace prefix for lookup: System.IDisposable -> IDisposable
        short_name = bare_name.rsplit(".", 1)[-1]

        # 1. Exact name match with type preference
        candidate = unit_by_name.get(short_name)
        if candidate:
            if not preferred_types or candidate.unit_type in preferred_types:
                return candidate

        # 2. Qualified name suffix match
        suffix = f".{short_name}"
        for qn, qu in unit_by_qualified.items():
            if qn.endswith(suffix):
                if not preferred_types or qu.unit_type in preferred_types:
                    return qu

        # 3. Fallback: exact name match without type filter
        if preferred_types and candidate:
            return candidate

        return None

    @staticmethod
    def _extract_base_classes(signature: str, language: Optional[str]) -> List[str]:
        """Extract base class names from a class signature (Python/JS/TS fallback)."""
        bases = []
        if language == "python":
            match = _PYTHON_BASES_RE.search(signature)
            if match:
                raw = match.group(1)
                for part in raw.split(","):
                    name = part.strip().split("(")[0].split("[")[0].strip()
                    if name and name not in ("object", "ABC", "metaclass"):
                        # Take the last component of dotted names
                        bases.append(name.rsplit(".", 1)[-1])
        else:
            # JS/TS: class Foo extends Bar
            match = _JS_EXTENDS_RE.search(signature)
            if match:
                bases.append(match.group(1))
        return bases

    @staticmethod
    def _extract_imported_names(unit: CodeUnit) -> Set[str]:
        """Extract specific imported names from a unit's context.

        Parses import statements to extract the individual names being
        imported. Supports Python, JS/TS, Java, and C#.
        """
        names: Set[str] = set()
        if not hasattr(unit, "source") or not unit.source:
            return names

        # Python: from foo.bar import Baz, Qux
        for match in re.finditer(r"from\s+\S+\s+import\s+([^;\n]+)", unit.source):
            for name in match.group(1).split(","):
                clean = name.strip().split(" as ")[0].strip()
                if clean and clean[0].isupper():
                    names.add(clean)

        # JS/TS: import { Foo, Bar } from './baz'
        for match in re.finditer(r"import\s*\{([^}]+)\}\s*from", unit.source):
            for name in match.group(1).split(","):
                clean = name.strip().split(" as ")[0].strip()
                if clean:
                    names.add(clean)

        # Java: import com.example.HazelcastService;
        # Also handles static imports: import static com.example.Utils.method;
        # Skips wildcard imports (import com.example.*)
        for match in re.finditer(
            r"import\s+(?:static\s+)?([\w.]+)\s*;", unit.source
        ):
            fqn = match.group(1)
            if fqn.endswith(".*"):
                continue  # Wildcard — can't resolve to specific unit
            last_segment = fqn.rsplit(".", 1)[-1]
            if last_segment and last_segment[0].isupper():
                names.add(last_segment)

        # C#: using Namespace.ClassName;
        # Skips using directives that are aliased or static
        for match in re.finditer(
            r"using\s+(?!static\b)([\w.]+)\s*;", unit.source
        ):
            fqn = match.group(1)
            last_segment = fqn.rsplit(".", 1)[-1]
            if last_segment and last_segment[0].isupper():
                names.add(last_segment)

        # VB.NET: Imports Namespace.TypeName
        for match in re.finditer(
            r"Imports\s+([\w.]+)", unit.source
        ):
            fqn = match.group(1)
            last_segment = fqn.rsplit(".", 1)[-1]
            if last_segment and last_segment[0].isupper():
                names.add(last_segment)

        return names


# Common builtins/keywords to exclude from call detection
_BUILTINS = frozenset({
    # Python builtins
    "print", "len", "range", "int", "str", "float", "bool", "list", "dict",
    "set", "tuple", "type", "isinstance", "issubclass", "getattr", "setattr",
    "hasattr", "super", "property", "classmethod", "staticmethod", "enumerate",
    "zip", "map", "filter", "sorted", "reversed", "any", "all", "min", "max",
    "sum", "abs", "round", "open", "format", "repr", "hash", "id", "input",
    "vars", "dir", "callable", "iter", "next", "slice",
    # JS/TS builtins
    "console", "log", "error", "warn", "setTimeout", "setInterval",
    "clearTimeout", "clearInterval", "parseInt", "parseFloat",
    "Array", "Object", "String", "Number", "Boolean", "Date", "Math",
    "JSON", "Promise", "Error", "RegExp", "Map", "Set", "WeakMap", "WeakSet",
    "Symbol", "Proxy", "Reflect", "fetch", "require",
    # C# builtins / common framework types
    "Console", "WriteLine", "Write", "ReadLine", "ToString", "GetType",
    "Equals", "GetHashCode", "ReferenceEquals",
    "Task", "Func", "Action", "Predicate", "Delegate",
    "List", "Dictionary", "HashSet", "Queue", "Stack",
    "var", "nameof", "typeof", "sizeof", "default",
    "Dispose", "ConfigureAwait", "GetAwaiter", "GetResult",
    # Java builtins / common framework types
    "System", "out", "println", "equals", "hashCode", "getClass",
    "toString", "valueOf", "compareTo", "iterator",
    # Common patterns that look like calls but aren't meaningful edges
    "self", "this", "cls", "return", "raise", "throw", "new", "delete",
    "if", "for", "while", "switch", "catch", "try", "finally",
    # VB.NET builtins
    "MsgBox", "InputBox", "CStr", "CInt", "CLng", "CDbl", "CSng", "CBool",
    "CByte", "CChar", "CDate", "CDec", "CObj", "CShort", "CType",
    "DirectCast", "TryCast", "IsNothing", "IsNumeric",
    "MyBase", "MyClass", "Me",
    "Len", "Mid", "Left", "Right", "Trim", "UCase", "LCase",
    "Val", "Asc", "Chr",
})

# Common framework / primitive types to exclude from type_dep resolution.
# These never correspond to user-defined code units in a project.
_PRIMITIVE_TYPES = frozenset({
    # Java primitives + wrappers
    "String", "Integer", "Long", "Double", "Float", "Boolean", "Byte",
    "Short", "Character", "Number", "Object", "Class", "Void",
    # Java collections / standard lib
    "List", "Set", "Map", "Collection", "Queue", "Deque",
    "ArrayList", "LinkedList", "HashMap", "TreeMap", "LinkedHashMap",
    "HashSet", "TreeSet", "LinkedHashSet", "ConcurrentHashMap",
    "Iterator", "Iterable", "Comparable", "Serializable", "Cloneable",
    "Optional", "Stream", "Collectors",
    "CompletableFuture", "Future", "Callable", "Runnable",
    "Supplier", "Consumer", "Function", "Predicate", "BiFunction",
    "BiConsumer", "BiPredicate", "UnaryOperator", "BinaryOperator",
    # C# / .NET primitives + standard lib
    "IList", "ISet", "IMap", "IDictionary", "IEnumerable", "IEnumerator",
    "ICollection", "IReadOnlyList", "IReadOnlyDictionary", "IReadOnlyCollection",
    "IDisposable", "IComparable", "IEquatable", "IFormattable", "ICloneable",
    "Task", "ValueTask", "Action", "Func", "Predicate",
    "CancellationToken", "CancellationTokenSource",
    "StringBuilder", "StringComparer",
    "DateTime", "DateTimeOffset", "TimeSpan", "Guid",
    "Nullable", "Lazy", "Tuple", "KeyValuePair",
    "Exception", "ArgumentException", "InvalidOperationException",
    "NotImplementedException", "NullReferenceException",
    "ILogger", "IConfiguration", "IServiceProvider", "IOptions",
    # Python standard types
    "Any", "Dict", "Tuple", "Type", "Callable", "Generator",
    "AsyncGenerator", "Awaitable", "Coroutine", "Protocol",
    "ClassVar", "Final", "Literal", "TypeVar", "Generic",
    "Union", "Sequence", "Mapping", "MutableMapping", "Iterable",
    "AbstractSet", "MutableSet", "MutableSequence",
    # TypeScript/JS standard types
    "Array", "Record", "Partial", "Required", "Readonly", "Pick",
    "Omit", "Exclude", "Extract", "NonNullable", "ReturnType",
    "InstanceType", "Parameters", "ConstructorParameters",
    "Promise", "Date", "RegExp", "Error", "TypeError",
    "Uint8Array", "Int32Array", "Float64Array", "ArrayBuffer",
    "ReadonlyArray", "PropertyKey", "Symbol",
})

"""Structural edge detectors — contains, imports, calls.

All functions take an EdgeContext and return List[dict] of edge records
ready for bulk insertion.
"""

import re
import logging
from typing import Dict, List, Set

from ..db.models import CodeUnit

from .constants import (
    BUILTINS, CALL_RE, QUALIFIED_CALL_RE,
    COBOL_PERFORM_RE, COBOL_PERFORM_THRU_RE, COBOL_GOTO_RE, COBOL_CALL_RE,
    PL1_CALL_RE, PL1_GOTO_RE,
)
from .context import EdgeContext

logger = logging.getLogger(__name__)


# ── Contains ─────────────────────────────────────────────────────────


def detect_contains(ctx: EdgeContext) -> List[dict]:
    """Detect contains edges: class -> method/property/constructor.

    Uses three strategies in order:
    1. unit_metadata["parent_name"] (set during ingestion)
    2. qualified_name parsing (e.g., "Module.ClassName.method" -> parent is "ClassName")
    3. Line range containment (method start_line within class start/end)
    """
    edges = []
    member_types = ("method", "constructor", "property")
    parent_types = ("class", "interface", "struct", "record")

    for u in ctx.units:
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
        file_units = ctx.units_by_file.get(str(u.file_id), [])
        parent = None
        for fu in file_units:
            if fu.unit_type in parent_types and fu.name == parent_name:
                parent = fu
                break

        if parent:
            edges.append({
                "project_id": ctx.project_id,
                "source_unit_id": parent.unit_id,
                "target_unit_id": u.unit_id,
                "edge_type": "contains",
                "edge_metadata": {},
            })

    # COBOL/PL1: program → paragraph/section containment
    # Programs contain their paragraphs — needed for call tree traversal
    # in the Understanding Engine (CTE follows 'contains' edges).
    mainframe_parent_types = ("program",)
    mainframe_child_types = ("paragraph", "section", "procedure")

    for u in ctx.units:
        if u.unit_type not in mainframe_child_types:
            continue
        if u.language not in ("cobol", "pl1"):
            continue

        file_units = ctx.units_by_file.get(str(u.file_id), [])
        for fu in file_units:
            if fu.unit_type not in mainframe_parent_types:
                continue
            if fu.unit_id == u.unit_id:
                continue
            # Paragraph must be within program's line range
            if (fu.start_line is not None and u.start_line is not None
                    and fu.start_line <= u.start_line
                    and (fu.end_line is None or u.end_line is None
                         or fu.end_line >= u.end_line)):
                edges.append({
                    "project_id": ctx.project_id,
                    "source_unit_id": fu.unit_id,
                    "target_unit_id": u.unit_id,
                    "edge_type": "contains",
                    "edge_metadata": {},
                })
                break  # A paragraph belongs to exactly one program

    return edges


# ── Imports ──────────────────────────────────────────────────────────


def detect_imports(ctx: EdgeContext) -> List[dict]:
    """Detect import edges by matching import statements to project units.

    Two strategies per file:
    1. metadata["file_imports"] -- stamped during ingestion (Java/C#/Python/JS/TS)
    2. Fallback: scan unit source for import patterns (Python/JS/TS only --
       Java/C# imports live at file level, outside any unit's source)
    """
    edges = []
    processed_files: Set[str] = set()

    for u in ctx.units:
        file_key = str(u.file_id)
        if file_key in processed_files:
            continue
        processed_files.add(file_key)

        file_units = ctx.units_by_file.get(file_key, [])
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
                    _names_from_import_statements(file_imports)
                )
                source_unit = fu
                break

        # Strategy 2: fallback -- scan unit source for import patterns
        if not imported_names:
            for fu in file_units:
                if not fu.source:
                    continue
                imported_names = _extract_imported_names(fu)
                if imported_names:
                    source_unit = fu
                    break

        # Resolve each imported name to a project unit
        for name in imported_names:
            target = ctx.unit_by_name.get(name)
            if not target:
                for qn, qu in ctx.unit_by_qualified.items():
                    if qn.endswith(f".{name}"):
                        target = qu
                        break

            if target and target.file_id != source_unit.file_id:
                edges.append({
                    "project_id": ctx.project_id,
                    "source_unit_id": source_unit.unit_id,
                    "target_unit_id": target.unit_id,
                    "edge_type": "imports",
                    "edge_metadata": {"import_name": name},
                })

    return edges


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

        # Java / C# / VB.NET: dotted path -- take last capitalized segment
        if stmt.startswith(("import ", "using ", "Imports ")):
            path = (stmt.replace("import static ", "")
                    .replace("import ", "")
                    .replace("using ", "")
                    .replace("Imports ", "")
                    .strip())
            if path.endswith(".*") or path.endswith("*"):
                continue  # Wildcard
            segments = path.split(".")
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

        # COBOL: "COPY CUSTMSTR" or "COPY 'ACCTFILE' IN COPYLIB"
        if re.match(r"COPY\s+", stmt, re.IGNORECASE):
            m = re.match(r"COPY\s+['\"]?([\w-]+)['\"]?", stmt, re.IGNORECASE)
            if m:
                names.add(m.group(1))

        # PL/1: "%INCLUDE filename" or "%INCLUDE 'filename.inc'"
        if re.match(r"%INCLUDE\s+", stmt, re.IGNORECASE):
            m = re.match(r"%INCLUDE\s+['\"]?([\w.]+)['\"]?", stmt, re.IGNORECASE)
            if m:
                names.add(m.group(1).rsplit(".", 1)[0])  # strip extension

    return names


def _extract_imported_names(unit: CodeUnit) -> Set[str]:
    """Extract specific imported names from a unit's context.

    Parses import statements to extract the individual names being
    imported. Supports Python, JS/TS, Java, C#, and VB.NET.
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
    for match in re.finditer(
        r"import\s+(?:static\s+)?([\w.]+)\s*;", unit.source
    ):
        fqn = match.group(1)
        if fqn.endswith(".*"):
            continue
        last_segment = fqn.rsplit(".", 1)[-1]
        if last_segment and last_segment[0].isupper():
            names.add(last_segment)

    # C#: using Namespace.ClassName;
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


# ── Calls ────────────────────────────────────────────────────────────


def detect_calls(ctx: EdgeContext) -> List[dict]:
    """Detect call edges by scanning function bodies for known identifiers.

    Uses both simple call patterns (func()) and qualified call patterns
    (obj.method(), Package.Class.method()).
    """
    edges = []
    for u in ctx.units:
        if u.unit_type not in ("function", "method", "paragraph", "procedure", "entry", "step") or not u.source:
            continue

        # Find all function-call-like patterns in the source
        called_names = set(CALL_RE.findall(u.source))
        # Also capture qualified calls: obj.method() -> "method"
        called_names.update(QUALIFIED_CALL_RE.findall(u.source))

        # Language-specific call detection
        if u.language == "cobol":
            # COBOL scoping rule:
            #   PERFORM / GO TO → intra-program (same file); use same-file lookup
            #   CALL 'PROGNAME' → cross-program (external); use global lookup
            # Mixing both into unit_by_name (first-wins) causes wrong edges when
            # multiple programs share a paragraph name like 0000-MAIN-PARA.
            perform_names: set = set(COBOL_PERFORM_RE.findall(u.source))
            perform_names.update(COBOL_PERFORM_THRU_RE.findall(u.source))
            perform_names.update(COBOL_GOTO_RE.findall(u.source))
            call_names: set = set(COBOL_CALL_RE.findall(u.source))

            # Build a same-file name → unit map (paragraphs/sections of THIS program)
            file_units = ctx.units_by_file.get(str(u.file_id), [])
            same_prog_by_name = {fu.name: fu for fu in file_units}

            # Intra-program PERFORM/GOTO edges — prefer same-file target
            for name in perform_names:
                if not name or name == u.name or name in BUILTINS:
                    continue
                target = same_prog_by_name.get(name) or ctx.unit_by_name.get(name)
                if target and target.unit_id != u.unit_id:
                    edges.append({
                        "project_id": ctx.project_id,
                        "source_unit_id": u.unit_id,
                        "target_unit_id": target.unit_id,
                        "edge_type": "calls",
                        "edge_metadata": {"confidence": "high"},
                    })

            # Cross-program CALL edges — global lookup (program units)
            for name in call_names:
                if not name or name in BUILTINS:
                    continue
                target = ctx.unit_by_name.get(name)
                if not target:
                    for qn, qu in ctx.unit_by_qualified.items():
                        if qn.endswith(f".{name}"):
                            target = qu
                            break
                if target and target.unit_id != u.unit_id:
                    edges.append({
                        "project_id": ctx.project_id,
                        "source_unit_id": u.unit_id,
                        "target_unit_id": target.unit_id,
                        "edge_type": "calls",
                        "edge_metadata": {"confidence": "high"},
                    })
            continue  # COBOL handled above; skip generic resolution

        elif u.language == "pl1":
            called_names.update(PL1_CALL_RE.findall(u.source))
            called_names.update(PL1_GOTO_RE.findall(u.source))
        elif u.language == "jcl" and u.unit_type == "step":
            # JCL steps invoke programs/procs via metadata — not inline source calls.
            meta = u.unit_metadata or {}
            if meta.get("pgm"):
                called_names.add(meta["pgm"])      # e.g. "CUSTUPD" → COBOL program
            if meta.get("proc_name"):
                called_names.add(meta["proc_name"])  # e.g. "PAYRPT" → JCL proc

        # Intersect with known unit names in the project
        matched = called_names & ctx.all_names
        # Remove self-calls and common builtins
        matched.discard(u.name)
        matched -= BUILTINS

        for callee_name in matched:
            target = ctx.unit_by_name.get(callee_name)
            if target and target.unit_id != u.unit_id:
                confidence = "high" if target.qualified_name and u.qualified_name else "medium"
                edges.append({
                    "project_id": ctx.project_id,
                    "source_unit_id": u.unit_id,
                    "target_unit_id": target.unit_id,
                    "edge_type": "calls",
                    "edge_metadata": {"confidence": confidence},
                })
    return edges

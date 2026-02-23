"""Stored procedure call detector — calls_sp edges.

Detects invocations of stored procedures from application code (Java,
C#, Python) by pattern-matching against known SP regex patterns.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

from ..db.models import CodeUnit

from .constants import (
    JAVA_SP_CALL_RE,
    JAVA_PROC_ANNOTATION_RE,
    CSHARP_SP_RE,
    PYTHON_CALLPROC_RE,
    EXEC_IN_STRING_RE,
)
from .context import EdgeContext

logger = logging.getLogger(__name__)


def detect_sp_calls(ctx: EdgeContext) -> List[dict]:
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
    sp_units = [u for u in ctx.units if u.unit_type in ("stored_procedure", "sql_function")]
    sp_by_name: Dict[str, CodeUnit] = {u.name.lower(): u for u in sp_units}
    app_units = [u for u in ctx.units if u.language != "sql" and u.source]

    for u in app_units:
        if u.unit_type not in ("function", "method", "class"):
            continue

        sp_refs = _extract_sp_references(u.source, u.language)

        for sp_name, call_pattern in sp_refs:
            target = sp_by_name.get(sp_name.lower())
            if target and target.unit_id != u.unit_id:
                edges.append({
                    "project_id": ctx.project_id,
                    "source_unit_id": u.unit_id,
                    "target_unit_id": target.unit_id,
                    "edge_type": "calls_sp",
                    "edge_metadata": {
                        "call_pattern": call_pattern,
                        "sp_name": sp_name,
                    },
                })

    return edges


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
        for m in JAVA_SP_CALL_RE.finditer(source):
            _add(m.group(1), "CallableStatement")
        for m in JAVA_PROC_ANNOTATION_RE.finditer(source):
            _add(m.group(1), "@Procedure")

    elif language == "csharp":
        has_sp_type = "StoredProcedure" in source or "CommandType.StoredProcedure" in source
        if has_sp_type:
            for m in CSHARP_SP_RE.finditer(source):
                name = m.group(1) or m.group(2)
                if name:
                    _add(name, "SqlCommand.StoredProcedure")

    elif language == "python":
        for m in PYTHON_CALLPROC_RE.finditer(source):
            _add(m.group(1), "cursor.callproc")

    # Generic EXEC pattern in string literals (any language)
    for m in EXEC_IN_STRING_RE.finditer(source):
        _add(m.group(1), "EXEC_in_string")

    return refs

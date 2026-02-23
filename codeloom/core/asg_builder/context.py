"""EdgeContext — shared lookup structures for ASG edge detection.

Built once per build_edges() call and passed to every domain-specific
detector module. This avoids passing 5+ positional dicts through every
function signature.
"""

import re
from typing import Dict, List, Optional, Set
from uuid import UUID

from ..db.models import CodeUnit

from .constants import PYTHON_BASES_RE, JS_EXTENDS_RE, PRIMITIVE_TYPES


class EdgeContext:
    """Shared lookup structures built once, passed to all edge detectors.

    Attributes:
        project_id: UUID of the project being analyzed.
        units: All CodeUnit records for the project.
        unit_by_name: First unit with a given short name.
        unit_by_qualified: Units indexed by qualified_name.
        units_by_file: Units grouped by file_id string.
        all_names: Set of all unit short names (for fast intersection).
    """

    __slots__ = (
        "project_id", "units", "unit_by_name", "unit_by_qualified",
        "units_by_file", "all_names",
    )

    def __init__(
        self,
        project_id: UUID,
        units: List[CodeUnit],
        unit_by_name: Dict[str, CodeUnit],
        unit_by_qualified: Dict[str, CodeUnit],
        units_by_file: Dict[str, List[CodeUnit]],
        all_names: Set[str],
    ):
        self.project_id = project_id
        self.units = units
        self.unit_by_name = unit_by_name
        self.unit_by_qualified = unit_by_qualified
        self.units_by_file = units_by_file
        self.all_names = all_names

    @classmethod
    def from_units(cls, units: List[CodeUnit], project_id: UUID) -> "EdgeContext":
        """Build all lookup indexes from a list of CodeUnits."""
        unit_by_name: Dict[str, CodeUnit] = {}
        unit_by_qualified: Dict[str, CodeUnit] = {}
        units_by_file: Dict[str, List[CodeUnit]] = {}

        for u in units:
            unit_by_name.setdefault(u.name, u)
            if u.qualified_name:
                unit_by_qualified[u.qualified_name] = u
            file_key = str(u.file_id)
            units_by_file.setdefault(file_key, []).append(u)

        all_names = {u.name for u in units}

        return cls(
            project_id=project_id,
            units=units,
            unit_by_name=unit_by_name,
            unit_by_qualified=unit_by_qualified,
            units_by_file=units_by_file,
            all_names=all_names,
        )


# ── Resolution helpers ──────────────────────────────────────────────


def resolve_unit(
    name: str,
    ctx: EdgeContext,
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
    candidate = ctx.unit_by_name.get(short_name)
    if candidate:
        if not preferred_types or candidate.unit_type in preferred_types:
            return candidate

    # 2. Qualified name suffix match
    suffix = f".{short_name}"
    for qn, qu in ctx.unit_by_qualified.items():
        if qn.endswith(suffix):
            if not preferred_types or qu.unit_type in preferred_types:
                return qu

    # 3. Fallback: exact name match without type filter
    if preferred_types and candidate:
        return candidate

    return None


def extract_type_identifiers(type_str: str) -> List[str]:
    """Extract individual type identifiers from a type expression.

    Handles generics, arrays, nullables, and compound types:
      "Map<String, HazelcastService>" -> ["HazelcastService"]
      "List<User>"                    -> ["User"]
      "Optional[UserProfile]"         -> ["UserProfile"]
      "int"                           -> []

    Filters out primitives and common framework types that never
    correspond to user-defined code units.
    """
    candidates = re.findall(r"\b([A-Z]\w+)", type_str)
    return [c for c in candidates if c not in PRIMITIVE_TYPES]


def extract_base_classes(signature: str, language: Optional[str]) -> List[str]:
    """Extract base class names from a class signature (Python/JS/TS fallback)."""
    bases = []
    if language == "python":
        match = PYTHON_BASES_RE.search(signature)
        if match:
            raw = match.group(1)
            for part in raw.split(","):
                name = part.strip().split("(")[0].split("[")[0].strip()
                if name and name not in ("object", "ABC", "metaclass"):
                    bases.append(name.rsplit(".", 1)[-1])
    else:
        # JS/TS: class Foo extends Bar
        match = JS_EXTENDS_RE.search(signature)
        if match:
            bases.append(match.group(1))
    return bases

"""PL/1 parser — regex-based.

No tree-sitter grammar is available for PL/1. This parser follows the same
standalone pattern as VbNetParser and SqlParser: implements parse_file /
parse_source directly without subclassing BaseLanguageParser.

Grammar reference: IBM OS PL/I V2R3 formal grammar (Lämmel & Verhoef, 1999)
  https://www.cs.vu.nl/grammarware/browsable/os-pli-v2r3/

Extracts:
  LABEL: PROCEDURE [(params)] [RETURNS(...)] [OPTIONS(...)];  → unit_type="procedure"
  LABEL: ENTRY [(params)] [RETURNS(...)];                     → unit_type="entry"
  LABEL: PACKAGE;                                             → unit_type="package"

Multiple labels on one declaration are captured; additional labels stored as aliases.

%INCLUDE directives → imports list (drives ASG imports edges).
CALL and GO TO names preserved in unit source for ASG call edge detection.
ON condition handlers scanned and tagged in procedure metadata.

PL/1 syntax notes (from OS PL/I V2R3 grammar):
  - One or more labels before PROCEDURE/ENTRY:  { entry-constant ":" }+
  - PROC is an accepted abbreviation for PROCEDURE
  - Names: alphanumeric + underscore (no hyphens unlike COBOL)
  - Comments: /* ... */ (multi-line supported)
  - END [label]; closes the innermost open block
  - PACKAGE groups procedures (package.procedure qualified naming)
  - PROCEDURE may have RETURNS(...) and RECURSIVE clauses
  - ENTRY marks alternate entry points inside a PROCEDURE (no matching END)
  - ON condition handlers: ON ERROR BEGIN; ... END;  / ON ZERODIVIDE SYSTEM;
  - Case-insensitive keywords
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .models import CodeUnit, ParseResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns — aligned with OS PL/I V2R3 formal grammar
# ---------------------------------------------------------------------------

# Strip block comments before line processing (greedy, handles multi-line)
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)

# One or more "label:" prefixes: captures the full prefix group as group 1
# Grammar: { entry-constant ":" }+
_LABELS_PREFIX = r"((?:\w+\s*:\s*)+)"

# PROCEDURE / PROC declaration
# Grammar: { entry-constant ":" }+ "PROCEDURE" [params] [RETURNS] [OPTIONS]
#          [RECURSIVE] [ORDER|REORDER] [CHARGRAPHIC|NOCHARGRAPHIC] ";"
# Matches:
#   "MYPROC: PROCEDURE(arg1, arg2) RETURNS(FIXED) OPTIONS(MAIN);"
#   "MYPROC: ALTNAME: PROC RECURSIVE;"
#   "PROCEDURE OPTIONS(MAIN);"   (no label — unnamed block, rare)
_PROC_RE = re.compile(
    r"^(?:" + _LABELS_PREFIX + r")?"
    r"(?P<kw>PROCEDURE|PROC)\b"
    r"(?:\s*\((?P<params>[^)]*)\))?"         # optional parameter list
    r"(?:\s*RETURNS\s*\([^)]*\))?"           # optional RETURNS clause
    r"(?:\s*OPTIONS\s*\([^)]*\))?"           # optional OPTIONS clause
    r"(?:\s*RECURSIVE)?"                     # optional RECURSIVE keyword
    r"(?:\s*(?:ORDER|REORDER))?"             # optional ORDER/REORDER
    r"(?:\s*(?:CHARGRAPHIC|NOCHARGRAPHIC))?" # optional CHARGRAPHIC
    r"\s*;",
    re.IGNORECASE,
)

# ENTRY statement — alternate entry point inside a PROCEDURE block
# Grammar: { entry-constant ":" }+ "ENTRY" [params] [RETURNS] [OPTIONS] ";"
# Unlike PROCEDURE, ENTRY has no matching END — it's a single-line marker.
# Matches:
#   "CALCPAY: ENTRY(BASE_SAL, HOURS);"
#   "CALCPAY: ALTCALC: ENTRY RETURNS(FIXED BINARY);"
_ENTRY_RE = re.compile(
    _LABELS_PREFIX
    + r"ENTRY\b"
    + r"(?:\s*\((?P<params>[^)]*)\))?"
    + r"(?:\s*RETURNS\s*\([^)]*\))?"
    + r"(?:\s*OPTIONS\s*\([^)]*\))?"
    + r"\s*;",
    re.IGNORECASE,
)

# PACKAGE declaration
# Matches: "MYPKG: PACKAGE;"  "MYPKG: PACKAGE EXPORTS(...);"
_PACKAGE_RE = re.compile(
    r"^(?:" + _LABELS_PREFIX + r")?PACKAGE\b[^;]*;",
    re.IGNORECASE,
)

# END statement — closes innermost open block
# Matches: "END MYPROC;"  "END;"
_END_RE = re.compile(
    r"^\s*END\b\s*(?P<label>\w+)?\s*;",
    re.IGNORECASE,
)

# BEGIN statement — starts a compound on-unit (not a named block)
# Grammar: "BEGIN" ";" — appears in "ON condition BEGIN;" or standalone "BEGIN;"
# We track depth to avoid confusing BEGIN-END with PROCEDURE-END.
_BEGIN_RE = re.compile(r"\bBEGIN\s*;", re.IGNORECASE)

# %INCLUDE / %INSCAN directive
# Grammar covers both forms; %INSCAN is IBM-specific macro inclusion
# Matches: "%INCLUDE filename;"  "%INCLUDE 'filename.inc';"
_INCLUDE_RE = re.compile(
    r"%(?:INCLUDE|INSCAN)\s+['\"]?(?P<name>[\w.]+)['\"]?\s*;",
    re.IGNORECASE,
)

# ON condition handler detection (for metadata tagging)
# Grammar: "ON" condition ["SNAP"] ("SYSTEM" ";" | on-unit)
# 24 predefined conditions from OS PL/I V2R3 grammar
_ON_CONDITION_RE = re.compile(
    r"\bON\s+("
    r"AREA|ATTENTION|CHECK|CONDITION|CONVERSION|"
    r"ENDFILE|ENDPAGE|ERROR|FINISH|FIXEDOVERFLOW|"
    r"KEY|NAME|OVERFLOW|PENDING|RECORD|SIZE|"
    r"STRINGRANGE|STRINGSIZE|SUBSCRIPTRANGE|"
    r"TRANSMIT|UNDEFINEDFILE|UNDERFLOW|ZERODIVIDE"
    r")\b",
    re.IGNORECASE,
)


def _parse_labels(labels_str: str) -> List[str]:
    """Extract label names from a '{ label ":" }+' prefix string.

    E.g. 'MYPROC: ALTNAME: ' → ['MYPROC', 'ALTNAME']
    """
    parts = [p.strip() for p in labels_str.split(":")]
    return [p for p in parts if p]


# ---------------------------------------------------------------------------
# Block tracking
# ---------------------------------------------------------------------------

@dataclass
class _Block:
    keyword: str               # "procedure", "package"
    name: str                  # primary label
    aliases: List[str]         # additional labels (multiple-label syntax)
    start_line: int            # 1-indexed
    params: str                # raw param string (for procedures)
    source_lines: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class Pl1Parser:
    """Regex-based PL/1 parser aligned with IBM OS PL/I V2R3 grammar.

    Standalone class (not a BaseLanguageParser subclass) — same interface
    as VbNetParser: implements parse_file() and parse_source().

    Unit types emitted:
      "procedure"  — LABEL: PROCEDURE [params] [RETURNS] ... ; ... END LABEL;
      "entry"      — LABEL: ENTRY [params]; (alternate entry point, no END)
      "package"    — LABEL: PACKAGE; ... END LABEL;
    """

    def parse_file(self, file_path: str, project_root: str = "") -> ParseResult:
        """Parse a PL/1 source file."""
        if project_root and file_path.startswith(project_root):
            rel_path = file_path[len(project_root):].lstrip("/")
        else:
            rel_path = file_path

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                source_text = f.read()
        except OSError as e:
            logger.error(f"Cannot read {file_path}: {e}")
            return ParseResult(
                file_path=rel_path,
                language="pl1",
                units=[],
                imports=[],
                line_count=0,
            )

        return self.parse_source(source_text, rel_path)

    def parse_source(self, source_text: str, file_path: str) -> ParseResult:
        """Parse PL/1 source text into a ParseResult."""
        lines = source_text.split("\n")
        line_count = len(lines)

        # Strip block comments for pattern matching; keep original for source.
        clean_text = _BLOCK_COMMENT_RE.sub(
            lambda m: " " * (m.end() - m.start()), source_text
        )
        clean_lines = clean_text.split("\n")

        imports: List[str] = []
        units: List[CodeUnit] = []
        block_stack: List[_Block] = []   # open blocks (innermost last)
        name_stack: List[str] = []       # for qualified naming
        begin_depth: int = 0             # depth of open BEGIN...END compound blocks

        for lineno, (raw_line, clean_line) in enumerate(
            zip(lines, clean_lines), start=1
        ):
            stripped = clean_line.strip()

            # ── %INCLUDE / %INSCAN (any nesting level) ────────────────────
            include_m = _INCLUDE_RE.search(clean_line)
            if include_m:
                name = include_m.group("name").rsplit(".", 1)[0]
                imports.append(f"%INCLUDE {name}")

            # ── PACKAGE declaration ───────────────────────────────────────
            pkg_m = _PACKAGE_RE.match(stripped)
            if pkg_m:
                labels_str = pkg_m.group(1) or ""
                labels = _parse_labels(labels_str)
                primary = labels[0] if labels else f"_pkg_{lineno}"
                block = _Block(
                    keyword="package",
                    name=primary,
                    aliases=labels[1:],
                    start_line=lineno,
                    params="",
                )
                block_stack.append(block)
                name_stack.append(primary)
                continue

            # ── PROCEDURE / PROC declaration ──────────────────────────────
            proc_m = _PROC_RE.match(stripped)
            if proc_m:
                labels_str = proc_m.group(1) or ""
                labels = _parse_labels(labels_str)
                primary = labels[0] if labels else f"_proc_{lineno}"
                params = (proc_m.group("params") or "").strip()
                block = _Block(
                    keyword="procedure",
                    name=primary,
                    aliases=labels[1:],
                    start_line=lineno,
                    params=params,
                )
                block_stack.append(block)
                name_stack.append(primary)
                continue

            # ── ENTRY statement (alternate entry point, no matching END) ──
            # Only meaningful when inside a procedure/package block.
            entry_m = _ENTRY_RE.match(stripped)
            if entry_m and block_stack:
                labels_str = entry_m.group(1) or ""
                labels = _parse_labels(labels_str)
                if not labels:
                    # Unlabelled ENTRY — skip (no callable name)
                    for block in block_stack:
                        block.source_lines.append(raw_line)
                    continue

                primary = labels[0]
                params = (entry_m.group("params") or "").strip()

                # Build qualified name from enclosing blocks
                enclosing = list(name_stack)
                if enclosing:
                    qualified_name = ".".join(enclosing) + "." + primary
                    parent_name = enclosing[-1]
                else:
                    qualified_name = primary
                    parent_name = None

                sig = f"{primary}: ENTRY"
                if params:
                    sig += f"({params})"
                sig += ";"

                unit = CodeUnit(
                    unit_type="entry",
                    name=primary,
                    qualified_name=qualified_name,
                    language="pl1",
                    start_line=lineno,
                    end_line=lineno,
                    source=raw_line,
                    file_path=file_path,
                    signature=sig,
                    parent_name=parent_name,
                    metadata={
                        "params": params,
                        "aliases": labels[1:],
                    },
                )
                units.append(unit)
                # Also track in open blocks (entry line is part of their source)
                for block in block_stack:
                    block.source_lines.append(raw_line)
                continue

            # ── BEGIN compound block (on-unit body) ───────────────────────
            # "ON condition BEGIN;" starts a compound block that ends with
            # "END;" — track depth so unlabelled END; doesn't close a PROC.
            if _BEGIN_RE.search(stripped):
                begin_depth += 1
                for block in block_stack:
                    block.source_lines.append(raw_line)
                continue

            # ── END statement — closes innermost matching block ────────────
            end_m = _END_RE.match(stripped)
            if end_m and block_stack:
                end_label = end_m.group("label") or ""
                # Unlabelled END closes the innermost BEGIN block first.
                if begin_depth > 0 and not end_label:
                    begin_depth -= 1
                    for block in block_stack:
                        block.source_lines.append(raw_line)
                    continue
                close_idx = len(block_stack) - 1
                if end_label:
                    for i in range(len(block_stack) - 1, -1, -1):
                        if block_stack[i].name.upper() == end_label.upper():
                            close_idx = i
                            break

                closed = block_stack.pop(close_idx)
                if close_idx < len(name_stack):
                    name_stack.pop(close_idx)

                block_source_lines = lines[closed.start_line - 1 : lineno]
                block_source = "\n".join(block_source_lines)

                enclosing = name_stack[:close_idx]
                if enclosing:
                    qualified_name = ".".join(enclosing) + "." + closed.name
                    parent_name = enclosing[-1]
                else:
                    qualified_name = closed.name
                    parent_name = None

                if closed.keyword == "procedure":
                    sig = f"{closed.name}: PROCEDURE"
                    if closed.params:
                        sig += f"({closed.params})"
                    sig += ";"
                else:
                    sig = f"{closed.name}: PACKAGE;"

                # Scan for ON condition handlers in procedure source
                on_conditions = _extract_on_conditions(block_source)

                unit = CodeUnit(
                    unit_type=closed.keyword,
                    name=closed.name,
                    qualified_name=qualified_name,
                    language="pl1",
                    start_line=closed.start_line,
                    end_line=lineno,
                    source=block_source,
                    file_path=file_path,
                    signature=sig,
                    parent_name=parent_name,
                    metadata={
                        "params": closed.params,
                        "aliases": closed.aliases,
                        "on_conditions": on_conditions,
                    },
                )
                units.append(unit)
                continue

            # ── Accumulate source lines for all open blocks ───────────────
            for block in block_stack:
                block.source_lines.append(raw_line)

        # Handle unclosed blocks (malformed source — emit what we have)
        for block in block_stack:
            logger.warning(
                f"{file_path}: unclosed {block.keyword} block '{block.name}' "
                f"starting at line {block.start_line}"
            )
            block_source = "\n".join(block.source_lines)
            qualified_name = ".".join(
                [b.name for b in block_stack[: block_stack.index(block) + 1]]
            )
            on_conditions = _extract_on_conditions(block_source)
            unit = CodeUnit(
                unit_type=block.keyword,
                name=block.name,
                qualified_name=qualified_name,
                language="pl1",
                start_line=block.start_line,
                end_line=line_count,
                source=block_source,
                file_path=file_path,
                signature=f"{block.name}: {block.keyword.upper()};",
                metadata={
                    "params": block.params,
                    "aliases": block.aliases,
                    "on_conditions": on_conditions,
                    "unclosed": True,
                },
            )
            units.append(unit)

        # Stamp imports on every unit (consistent with other parsers)
        for unit in units:
            unit.imports = imports

        return ParseResult(
            file_path=file_path,
            language="pl1",
            units=units,
            imports=imports,
            line_count=line_count,
        )


def _extract_on_conditions(source: str) -> List[str]:
    """Return sorted list of ON condition names found in PL/1 source.

    E.g. 'ON ERROR BEGIN' + 'ON ZERODIVIDE SYSTEM' → ['ERROR', 'ZERODIVIDE']
    Used to tag procedure metadata for migration (→ try/catch blocks).
    """
    found: Set[str] = set()
    for m in _ON_CONDITION_RE.finditer(source):
        found.add(m.group(1).upper())
    return sorted(found)

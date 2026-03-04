"""PL/1 parser — regex-based.

No tree-sitter grammar is available for PL/1. This parser follows the same
standalone pattern as VbNetParser and SqlParser: implements parse_file /
parse_source directly without subclassing BaseLanguageParser.

Extracts:
  LABEL: PROCEDURE [(params)] [OPTIONS(...)];  → unit_type="procedure"
  LABEL: PACKAGE;                              → unit_type="package"

%INCLUDE directives → imports list (drives ASG imports edges).
CALL statements are preserved in unit source for ASG call edge detection.

PL/1 syntax notes:
  - Labels are on the same statement as PROCEDURE: "MYPROC: PROCEDURE;"
  - PROC is an accepted abbreviation for PROCEDURE
  - Names are alphanumeric + underscore (no hyphens unlike COBOL)
  - Comments: /* ... */ (multi-line supported)
  - END [label]; closes the innermost open block
  - PACKAGE groups procedures (package.procedure qualified naming)
  - Case-insensitive keywords
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .models import CodeUnit, ParseResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Strip block comments before line processing (greedy, handles multi-line)
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)

# PROCEDURE / PROC declaration (with optional label, params, OPTIONS)
# Matches: "MYPROC: PROCEDURE(arg1, arg2) OPTIONS(MAIN);"
#          "MYPROC: PROC;"
#          "PROCEDURE OPTIONS(MAIN);"   (no label — unnamed block, rare)
_PROC_RE = re.compile(
    r"^(?:(?P<label>\w+)\s*:\s*)?"
    r"(?P<kw>PROCEDURE|PROC)\b"
    r"(?:\s*\((?P<params>[^)]*)\))?"
    r"(?:\s*OPTIONS\s*\([^)]*\))?"
    r"\s*;",
    re.IGNORECASE,
)

# PACKAGE declaration
# Matches: "MYPKG: PACKAGE;"  "MYPKG: PACKAGE EXPORTS(...);"
_PACKAGE_RE = re.compile(
    r"^(?:(?P<label>\w+)\s*:\s*)?PACKAGE\b[^;]*;",
    re.IGNORECASE,
)

# END statement — closes innermost open block
# Matches: "END MYPROC;"  "END;"
_END_RE = re.compile(
    r"^\s*END\b\s*(?P<label>\w+)?\s*;",
    re.IGNORECASE,
)

# %INCLUDE directive
# Matches: "%INCLUDE filename;"  "%INCLUDE 'filename.inc';"
_INCLUDE_RE = re.compile(
    r"%INCLUDE\s+['\"]?(?P<name>[\w.]+)['\"]?\s*;",
    re.IGNORECASE,
)

# Inline comment stripping (-- or // style not standard in PL/1, skip)
# PL/1 uses only /* ... */ block comments, handled above


# ---------------------------------------------------------------------------
# Block tracking
# ---------------------------------------------------------------------------

@dataclass
class _Block:
    keyword: str          # "procedure" or "package"
    name: str             # block label / name
    start_line: int       # 1-indexed
    params: str           # raw param string (for procedures)
    source_lines: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class Pl1Parser:
    """Regex-based PL/1 parser.

    Standalone class (not a BaseLanguageParser subclass) — same interface
    as VbNetParser: implements parse_file() and parse_source().
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

        # Strip block comments from the full source for pattern matching,
        # but keep the original lines for source extraction.
        clean_text = _BLOCK_COMMENT_RE.sub(
            lambda m: " " * (m.end() - m.start()), source_text
        )
        clean_lines = clean_text.split("\n")

        imports: List[str] = []
        units: List[CodeUnit] = []
        block_stack: List[_Block] = []   # open blocks (innermost last)
        name_stack: List[str] = []       # for qualified naming

        for lineno, (raw_line, clean_line) in enumerate(
            zip(lines, clean_lines), start=1
        ):
            stripped = clean_line.strip()

            # Collect %INCLUDE at any nesting level
            include_m = _INCLUDE_RE.search(clean_line)
            if include_m:
                name = include_m.group("name").rsplit(".", 1)[0]  # strip extension
                imports.append(f"%INCLUDE {name}")

            # PACKAGE declaration
            pkg_m = _PACKAGE_RE.match(stripped)
            if pkg_m:
                label = pkg_m.group("label") or f"_pkg_{lineno}"
                block = _Block(
                    keyword="package",
                    name=label,
                    start_line=lineno,
                    params="",
                )
                block_stack.append(block)
                name_stack.append(label)
                continue

            # PROCEDURE / PROC declaration
            proc_m = _PROC_RE.match(stripped)
            if proc_m:
                label = proc_m.group("label")
                if not label:
                    # Unnamed block — use synthetic name
                    label = f"_proc_{lineno}"
                params = (proc_m.group("params") or "").strip()
                block = _Block(
                    keyword="procedure",
                    name=label,
                    start_line=lineno,
                    params=params,
                )
                block_stack.append(block)
                name_stack.append(label)
                continue

            # END statement — closes innermost block
            end_m = _END_RE.match(stripped)
            if end_m and block_stack:
                end_label = end_m.group("label") or ""
                # Find matching block (END label matches block name, or innermost)
                close_idx = len(block_stack) - 1
                if end_label:
                    for i in range(len(block_stack) - 1, -1, -1):
                        if block_stack[i].name.upper() == end_label.upper():
                            close_idx = i
                            break

                closed = block_stack.pop(close_idx)
                if close_idx < len(name_stack):
                    name_stack.pop(close_idx)

                # Accumulate source (original lines, not cleaned)
                block_source_lines = lines[closed.start_line - 1: lineno]
                block_source = "\n".join(block_source_lines)

                # Build qualified name from enclosing blocks
                enclosing = name_stack[:close_idx]
                if enclosing:
                    qualified_name = ".".join(enclosing) + "." + closed.name
                    parent_name = enclosing[-1]
                else:
                    qualified_name = closed.name
                    parent_name = None

                # Build signature
                if closed.keyword == "procedure":
                    sig = f"{closed.name}: PROCEDURE"
                    if closed.params:
                        sig += f"({closed.params})"
                    sig += ";"
                else:
                    sig = f"{closed.name}: PACKAGE;"

                unit = CodeUnit(
                    unit_type=closed.keyword,  # "procedure" or "package"
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
                    },
                )
                units.append(unit)
                continue

            # Track source lines for all open blocks
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
                [b.name for b in block_stack[:block_stack.index(block) + 1]]
            )
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
                metadata={"params": block.params, "unclosed": True},
            )
            units.append(unit)

        # Stamp imports on the first unit (consistent with other parsers)
        for unit in units:
            unit.imports = imports

        return ParseResult(
            file_path=file_path,
            language="pl1",
            units=units,
            imports=imports,
            line_count=line_count,
        )

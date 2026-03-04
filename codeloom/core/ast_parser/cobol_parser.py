"""COBOL parser using tree-sitter via tree-sitter-language-pack.

Grammar: nolanlwin/tree-sitter-cobol (COBOL85, MIT).
Loaded from tree-sitter-language-pack — no separate pip package required.

Extracts:
  PROGRAM-ID paragraph  → unit_type="program"   (1 per file, top-level container)
  Paragraph labels      → unit_type="paragraph"  (primary callable unit in PROCEDURE DIVISION)
  Section labels        → unit_type="section"    (groups paragraphs; optional nesting)

COPY statements → imports list (drives ASG imports edges).
PERFORM, CALL, and GO TO names are preserved in unit source for ASG call edge detection.

Node types confirmed against grammar.js:
  program_definition, identification_division, program_name
  procedure_division, section_header, paragraph_header
  copy_statement, perform_statement_call_proc, call_statement, goto_statement

EXEC SQL/CICS handling:
  The COBOL85 tree-sitter grammar does not parse EXEC SQL/CICS preprocessor blocks.
  These blocks are stripped (lines blanked, line count preserved) before parsing so
  tree-sitter can extract the structural skeleton without errors. Original source text
  is restored in units during post-processing. Paragraphs/sections are tagged with
  has_exec_sql and has_exec_cics metadata flags using the original source.
"""

import logging
import re
from typing import Dict, List, Optional

import tree_sitter
from tree_sitter_language_pack import get_language as _ts_get_language

from .base import BaseLanguageParser
from .models import CodeUnit, ParseResult
from ..asg_builder.constants import COBOL_EXEC_SQL_RE, COBOL_EXEC_CICS_RE

logger = logging.getLogger(__name__)

# Loaded once at import time — thread-safe (Language objects are immutable)
_COBOL_LANGUAGE = _ts_get_language("cobol")

# Matches the start of an EXEC SQL or EXEC CICS block (line-level check)
_EXEC_START_RE = re.compile(r"\bEXEC\s+(SQL|CICS)\b", re.IGNORECASE)

# SELECT...ASSIGN TO: maps internal COBOL file name to JCL DDNAME
# Common forms:
#   SELECT file-name ASSIGN TO INFILE
#   SELECT file-name ASSIGN TO UT-S-INFILE  (device class prefix)
#   SELECT file-name ASSIGN TO DISK-INFILE
# The DDNAME is the last hyphen-delimited segment (or the whole word if no prefix).
_SELECT_ASSIGN_RE = re.compile(
    r"\bSELECT\s+([\w-]+)\s+ASSIGN\s+(?:TO\s+)?(?:[\w]+-[A-Z]-)?([A-Z0-9@#$][\w-]*)",
    re.IGNORECASE,
)
# Optional ORGANIZATION IS clause following SELECT
_ORGANIZATION_RE = re.compile(r"\bORGANIZATION\s+IS\s+(\w+)", re.IGNORECASE)


class CobolParser(BaseLanguageParser):
    """tree-sitter based COBOL parser (COBOL85 dialect).

    Maps COBOL structural concepts onto CodeLoom's CodeUnit model:
    - PROGRAM-ID        → unit_type="program"
    - SECTION label     → unit_type="section"
    - Paragraph label   → unit_type="paragraph"

    Paragraphs are COBOL's primary callable units (PERFORMed by name).
    They don't have a wrapping node in the grammar — boundaries are
    calculated from paragraph_header to the next header or end of division.
    """

    def get_language(self) -> str:
        return "cobol"

    def get_tree_sitter_language(self) -> tree_sitter.Language:
        return _COBOL_LANGUAGE

    # =========================================================================
    # parse_source override — EXEC SQL/CICS pre-stripping
    # =========================================================================

    def parse_source(self, source_text: str, file_path: str) -> ParseResult:
        """Parse COBOL source, stripping EXEC SQL/CICS blocks before tree-sitter.

        The COBOL85 grammar treats EXEC SQL/CICS as syntax errors, causing
        ERROR nodes that hide all paragraphs below the first EXEC block.
        We blank those lines (preserving line numbers), parse the skeleton,
        then restore original source text and tag metadata in post-processing.
        """
        original_lines = source_text.splitlines()
        stripped_text = self._strip_exec_blocks(source_text)

        # Parse with tree-sitter using stripped source
        result = super().parse_source(stripped_text, file_path)

        # Post-process: restore original source and tag EXEC SQL/CICS metadata
        for unit in result.units:
            if unit.unit_type in ("paragraph", "section", "program") and unit.start_line:
                end = unit.end_line if unit.end_line else unit.start_line
                orig_src = "\n".join(original_lines[unit.start_line - 1 : end])
                unit.source = orig_src
                if unit.unit_type in ("paragraph", "section"):
                    unit.metadata = unit.metadata or {}
                    unit.metadata["has_exec_sql"] = bool(COBOL_EXEC_SQL_RE.search(orig_src))
                    unit.metadata["has_exec_cics"] = bool(COBOL_EXEC_CICS_RE.search(orig_src))

        return result

    def _strip_exec_blocks(self, source_text: str) -> str:
        """Replace EXEC SQL/CICS ... END-EXEC blocks with blank lines.

        Each content line in the block is replaced with spaces of the same
        length so that byte offsets for all lines *after* the block remain
        identical, keeping tree-sitter's start_point / end_point accurate.
        """
        lines = source_text.splitlines(keepends=True)
        in_exec = False
        for i, line in enumerate(lines):
            upper = line.upper()
            if not in_exec and _EXEC_START_RE.search(upper):
                in_exec = True
            if in_exec:
                nl = "\n" if line.endswith("\n") else ""
                lines[i] = " " * (len(line) - len(nl)) + nl
            if in_exec and "END-EXEC" in upper:
                in_exec = False
        return "".join(lines)

    # =========================================================================
    # Required abstract implementations
    # =========================================================================

    def extract_imports(self, tree: tree_sitter.Tree, source: bytes) -> List[str]:
        """Extract COPY statements — COBOL's include/import mechanism.

        Returns raw statement text for each COPY directive, e.g.:
          "COPY CUSTMSTR"
          "COPY 'ACCTFILE' IN COPYLIB"
        """
        imports: List[str] = []
        self._collect_copy_statements(tree.root_node, source, imports)
        return imports

    def extract_units(
        self, tree: tree_sitter.Tree, source: bytes, file_path: str
    ) -> List[CodeUnit]:
        """Extract program, section, and paragraph units from the COBOL AST."""
        units: List[CodeUnit] = []
        root = tree.root_node

        for child in root.children:
            if child.type == "program_definition":
                units.extend(
                    self._extract_program(child, source, file_path)
                )

        return units

    # =========================================================================
    # Program-level extraction
    # =========================================================================

    def _extract_program(
        self,
        program_node: tree_sitter.Node,
        source: bytes,
        file_path: str,
    ) -> List[CodeUnit]:
        """Extract a program_definition and all its paragraphs/sections."""
        units: List[CodeUnit] = []

        # Extract PROGRAM-ID name
        program_id = self._extract_program_name(program_node, source)
        if not program_id:
            program_id = "UNKNOWN"

        # Build program-level CodeUnit (the whole file)
        # Source will be restored to original in parse_source post-processing
        program_source = source[program_node.start_byte:program_node.end_byte].decode(
            "utf-8", errors="replace"
        )
        file_assignments = self._extract_file_assignments(program_source)
        program_unit = CodeUnit(
            unit_type="program",
            name=program_id,
            qualified_name=program_id,
            language="cobol",
            start_line=program_node.start_point[0] + 1,
            end_line=program_node.end_point[0] + 1,
            source=program_source,
            file_path=file_path,
            signature=f"PROGRAM-ID. {program_id}.",
            metadata={
                "cobol_dialect": "cobol85",
                "file_assignments": file_assignments,
            },
        )
        units.append(program_unit)

        # Walk procedure_division for sections and paragraphs
        for child in program_node.children:
            if child.type == "procedure_division":
                proc_units = self._extract_procedure_division(
                    child, source, file_path, program_id
                )
                units.extend(proc_units)

        return units

    def _extract_program_name(
        self, program_node: tree_sitter.Node, source: bytes
    ) -> Optional[str]:
        """Extract PROGRAM-ID name from identification_division → program_name."""
        for child in program_node.children:
            if child.type == "identification_division":
                for sub in child.children:
                    if sub.type == "program_name":
                        return source[sub.start_byte:sub.end_byte].decode(
                            "utf-8", errors="replace"
                        ).strip()
        return None

    # =========================================================================
    # Procedure Division: sections + paragraphs
    # =========================================================================

    def _extract_procedure_division(
        self,
        proc_node: tree_sitter.Node,
        source: bytes,
        file_path: str,
        program_id: str,
    ) -> List[CodeUnit]:
        """Extract section and paragraph units from the procedure_division.

        Paragraphs have no wrapping node — boundaries are header-to-next-header.
        We walk the direct children and accumulate statement bytes between headers.
        """
        units: List[CodeUnit] = []
        children = proc_node.children

        current_section: Optional[str] = None
        # Track open paragraph: (name, start_byte, start_line, section_name)
        open_para: Optional[tuple] = None

        for i, child in enumerate(children):
            if child.type == "section_header":
                # Close any open paragraph
                if open_para:
                    units.append(
                        self._close_paragraph(
                            open_para, child.start_byte, child.start_point[0],
                            source, file_path, program_id,
                        )
                    )
                    open_para = None

                section_name = self._header_name(child, source)
                current_section = section_name

                # Emit section unit (source restored in post-processing)
                section_end_byte, section_end_line = self._next_section_start(
                    children, i, source
                )
                section_source = source[child.start_byte:section_end_byte].decode(
                    "utf-8", errors="replace"
                )
                units.append(CodeUnit(
                    unit_type="section",
                    name=section_name,
                    qualified_name=f"{program_id}.{section_name}",
                    language="cobol",
                    start_line=child.start_point[0] + 1,
                    end_line=section_end_line,
                    source=section_source,
                    file_path=file_path,
                    signature=f"{section_name} SECTION.",
                    parent_name=program_id,
                    metadata={"program_id": program_id},
                ))

            elif child.type == "paragraph_header":
                # Close any previously open paragraph
                if open_para:
                    units.append(
                        self._close_paragraph(
                            open_para, child.start_byte, child.start_point[0],
                            source, file_path, program_id,
                        )
                    )

                para_name = self._header_name(child, source)
                open_para = (
                    para_name,
                    child.start_byte,
                    child.start_point[0],
                    current_section,
                )

        # Close last open paragraph at end of procedure_division
        if open_para:
            units.append(
                self._close_paragraph(
                    open_para, proc_node.end_byte, proc_node.end_point[0],
                    source, file_path, program_id,
                )
            )

        return units

    def _close_paragraph(
        self,
        open_para: tuple,
        end_byte: int,
        end_row: int,
        source: bytes,
        file_path: str,
        program_id: str,
    ) -> CodeUnit:
        """Build a CodeUnit for the just-closed paragraph.

        Source text and EXEC SQL/CICS metadata are set from the stripped source
        here; parse_source post-processing replaces them with original content.
        """
        para_name, start_byte, start_row, section_name = open_para
        para_source = source[start_byte:end_byte].decode("utf-8", errors="replace")

        if section_name:
            qualified_name = f"{program_id}.{section_name}.{para_name}"
            parent_name = section_name
        else:
            qualified_name = f"{program_id}.{para_name}"
            parent_name = program_id

        return CodeUnit(
            unit_type="paragraph",
            name=para_name,
            qualified_name=qualified_name,
            language="cobol",
            start_line=start_row + 1,
            end_line=end_row,
            source=para_source,
            file_path=file_path,
            signature=f"{para_name}.",
            parent_name=parent_name,
            metadata={
                "program_id": program_id,
                "section": section_name or "",
                # Flags set to False here; parse_source restores original source
                # and re-evaluates these using the un-stripped text.
                "has_exec_sql": False,
                "has_exec_cics": False,
            },
        )

    # =========================================================================
    # ENVIRONMENT DIVISION: SELECT...ASSIGN extraction
    # =========================================================================

    def _extract_file_assignments(self, program_source: str) -> List[Dict]:
        """Extract SELECT...ASSIGN TO clauses from COBOL ENVIRONMENT DIVISION.

        Returns a list of file assignment records:
          [{"cobol_name": "EMPLOYEE-FILE", "ddname": "INFILE", "org": "SEQUENTIAL"}, ...]

        Used to correlate COBOL internal file names with JCL DDNAME allocations.
        The DDNAME is the right-most hyphen-segment that matches a JCL name pattern;
        device-class prefixes like UT-S-, DISK-, or TAPE- are stripped automatically.
        """
        assignments: List[Dict] = []
        seen: set = set()

        for m in _SELECT_ASSIGN_RE.finditer(program_source):
            cobol_name = m.group(1).strip().upper()
            ddname = m.group(2).strip().upper()
            if cobol_name in seen:
                continue
            seen.add(cobol_name)

            # Try to find ORGANIZATION IS clause near this SELECT (within 200 chars)
            org = "SEQUENTIAL"
            org_m = _ORGANIZATION_RE.search(program_source, m.start(), m.start() + 300)
            if org_m:
                org = org_m.group(1).upper()

            assignments.append({
                "cobol_name": cobol_name,
                "ddname": ddname,
                "org": org,
            })

        return assignments

    # =========================================================================
    # COPY statement collection
    # =========================================================================

    def _collect_copy_statements(
        self,
        node: tree_sitter.Node,
        source: bytes,
        result: List[str],
    ) -> None:
        """Recursively collect all copy_statement nodes."""
        if node.type == "copy_statement":
            # Extract copybook name: first WORD child of copy_statement
            copybook = self._first_word(node, source)
            if copybook:
                result.append(f"COPY {copybook}")
            else:
                # Fallback: use raw statement text
                raw = source[node.start_byte:node.end_byte].decode(
                    "utf-8", errors="replace"
                ).strip().rstrip(".")
                result.append(raw)
            return  # Don't recurse into copy_statement children

        for child in node.children:
            self._collect_copy_statements(child, source, result)

    # =========================================================================
    # Helpers
    # =========================================================================

    def _header_name(self, node: tree_sitter.Node, source: bytes) -> str:
        """Extract the label name from a paragraph_header or section_header.

        The text of the node is "1000-INIT." or "MAIN SECTION." —
        strip the trailing dot and SECTION keyword if present.
        """
        raw = source[node.start_byte:node.end_byte].decode(
            "utf-8", errors="replace"
        ).strip()
        # Remove trailing period
        raw = raw.rstrip(".")
        # For section headers: "MAIN SECTION" → "MAIN"
        if raw.upper().endswith(" SECTION"):
            raw = raw[:-8].strip()
        return raw.strip()

    def _first_word(self, node: tree_sitter.Node, source: bytes) -> Optional[str]:
        """Get text of the first WORD-type child (for COPY copybook name)."""
        for child in node.children:
            if child.type == "WORD":
                return source[child.start_byte:child.end_byte].decode(
                    "utf-8", errors="replace"
                ).strip()
        return None

    def _next_section_start(
        self,
        children: List[tree_sitter.Node],
        current_idx: int,
        source: bytes,
    ) -> tuple:
        """Find start of the next section_header after current_idx.

        Returns (end_byte, end_line) for the current section.
        """
        for i in range(current_idx + 1, len(children)):
            if children[i].type == "section_header":
                return children[i].start_byte, children[i].start_point[0]
        # No next section — use end of last child
        if children:
            last = children[-1]
            return last.end_byte, last.end_point[0] + 1
        return 0, 0

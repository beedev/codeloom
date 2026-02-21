"""SQL stored procedure parser — regex-based.

Extracts stored procedures, functions, triggers, and views from SQL files.
Starts with T-SQL, structured for PL/SQL and PL/pgSQL additions.

Does NOT use tree-sitter (SQL grammars are too dialect-specific).
Follows the FallbackParser pattern with parse_file/parse_source interface.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .models import CodeUnit, ParseResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# T-SQL Patterns
# ---------------------------------------------------------------------------

# Match CREATE [OR ALTER] PROCEDURE/PROC <name> ... AS/BEGIN
_TSQL_PROC_RE = re.compile(
    r"^\s*CREATE\s+(?:OR\s+ALTER\s+)?PROC(?:EDURE)?\s+"
    r"(?:\[?(\w+)\]?\.)?"            # optional schema [dbo].
    r"\[?(\w+)\]?"                    # procedure name
    r"(.*?)(?:\bAS\b|\bBEGIN\b)",
    re.IGNORECASE | re.DOTALL | re.MULTILINE,
)

# Match CREATE [OR ALTER] FUNCTION <name>
_TSQL_FUNC_RE = re.compile(
    r"^\s*CREATE\s+(?:OR\s+ALTER\s+)?FUNCTION\s+"
    r"(?:\[?(\w+)\]?\.)?"
    r"\[?(\w+)\]?"
    r"(.*?)(?:\bRETURNS\b)",
    re.IGNORECASE | re.DOTALL | re.MULTILINE,
)

# Match CREATE [OR ALTER] TRIGGER <name>
_TSQL_TRIGGER_RE = re.compile(
    r"^\s*CREATE\s+(?:OR\s+ALTER\s+)?TRIGGER\s+"
    r"(?:\[?(\w+)\]?\.)?"
    r"\[?(\w+)\]?"
    r"\s+ON\s+(?:\[?(\w+)\]?\.)?(\[?\w+\]?)",
    re.IGNORECASE | re.DOTALL | re.MULTILINE,
)

# Match CREATE [OR ALTER] VIEW <name>
_TSQL_VIEW_RE = re.compile(
    r"^\s*CREATE\s+(?:OR\s+ALTER\s+)?VIEW\s+"
    r"(?:\[?(\w+)\]?\.)?"
    r"\[?(\w+)\]?",
    re.IGNORECASE | re.DOTALL | re.MULTILINE,
)

# Parameter extraction: @Name TYPE [(size)] [= default] [OUTPUT|OUT]
_TSQL_PARAM_RE = re.compile(
    r"(@\w+)\s+"
    r"(\w+(?:\(\s*\w+(?:\s*,\s*\w+)?\s*\))?)"
    r"(?:\s*=\s*\S+)?"
    r"(\s+(?:OUT(?:PUT)?))?",
    re.IGNORECASE,
)

# Table references: FROM/JOIN/INSERT INTO/UPDATE/DELETE FROM <table>
_TABLE_REF_RE = re.compile(
    r"(?:FROM|JOIN|INTO|UPDATE|DELETE\s+FROM)\s+"
    r"(?:\[?(\w+)\]?\.)?"
    r"\[?(\w+)\]?",
    re.IGNORECASE,
)

# SP call references: EXEC[UTE] <name> or CALL <name>
_SP_CALL_RE = re.compile(
    r"(?:EXEC(?:UTE)?|CALL)\s+"
    r"(?:\[?(\w+)\]?\.)?"
    r"\[?(\w+)\]?",
    re.IGNORECASE,
)

# Return type for functions: RETURNS TABLE / RETURNS <scalar>
_RETURNS_RE = re.compile(
    r"\bRETURNS\s+(TABLE\b|\w+(?:\(\s*\w+(?:\s*,\s*\w+)?\s*\))?)",
    re.IGNORECASE,
)

# GO statement — batch separator in T-SQL
_GO_RE = re.compile(r"^\s*GO\s*$", re.IGNORECASE | re.MULTILINE)


class SqlParser:
    """Regex-based SQL parser for stored procedures, functions, triggers, views.

    Produces CodeUnit objects compatible with the ingestion pipeline.
    Does not subclass BaseLanguageParser (no tree-sitter dependency).
    """

    def __init__(self, dialect: str = "tsql"):
        self._dialect = dialect

    def get_language(self) -> str:
        return "sql"

    def parse_file(self, file_path: str, project_root: str = "") -> ParseResult:
        """Parse a SQL file into structured CodeUnit objects."""
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
                language="sql",
                units=[],
                imports=[],
                line_count=0,
            )

        return self.parse_source(source_text, rel_path)

    def parse_source(self, source_text: str, file_path: str) -> ParseResult:
        """Parse SQL source text into CodeUnit objects."""
        line_count = source_text.count("\n") + (
            1 if source_text and not source_text.endswith("\n") else 0
        )

        # Split on GO batches (T-SQL) to isolate individual objects
        batches = _GO_RE.split(source_text) if self._dialect == "tsql" else [source_text]

        units: List[CodeUnit] = []
        offset = 0  # cumulative line offset

        for batch in batches:
            if not batch.strip():
                offset += batch.count("\n")
                continue

            batch_units = self._extract_from_batch(batch, file_path, offset)
            units.extend(batch_units)
            offset += batch.count("\n")

        # If no structured units found, fall back to block splitting
        if not units:
            units = self._fallback_blocks(source_text, file_path)

        return ParseResult(
            file_path=file_path,
            language="sql",
            units=units,
            imports=[],  # SQL has no imports
            line_count=line_count,
        )

    def _extract_from_batch(
        self, batch: str, file_path: str, line_offset: int
    ) -> List[CodeUnit]:
        """Try each construct pattern against a single batch."""
        units: List[CodeUnit] = []

        # Try procedures
        for m in _TSQL_PROC_RE.finditer(batch):
            schema = m.group(1) or "dbo"
            name = m.group(2)
            param_text = m.group(3)
            start_line = line_offset + batch[: m.start()].count("\n") + 1
            end_line = line_offset + batch.count("\n") + 1
            source = batch.strip()

            units.append(self._make_unit(
                unit_type="stored_procedure",
                name=name,
                schema=schema,
                source=source,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                param_text=param_text,
            ))

        # Try functions
        for m in _TSQL_FUNC_RE.finditer(batch):
            schema = m.group(1) or "dbo"
            name = m.group(2)
            param_text = m.group(3)
            start_line = line_offset + batch[: m.start()].count("\n") + 1
            end_line = line_offset + batch.count("\n") + 1
            source = batch.strip()

            ret = _RETURNS_RE.search(batch)
            return_type = ret.group(1).upper() if ret else "UNKNOWN"

            units.append(self._make_unit(
                unit_type="sql_function",
                name=name,
                schema=schema,
                source=source,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                param_text=param_text,
                return_type=return_type,
            ))

        # Try triggers
        for m in _TSQL_TRIGGER_RE.finditer(batch):
            schema = m.group(1) or "dbo"
            name = m.group(2)
            table_name = m.group(4)
            start_line = line_offset + batch[: m.start()].count("\n") + 1
            end_line = line_offset + batch.count("\n") + 1
            source = batch.strip()

            units.append(self._make_unit(
                unit_type="trigger",
                name=name,
                schema=schema,
                source=source,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                trigger_table=table_name,
            ))

        # Try views
        for m in _TSQL_VIEW_RE.finditer(batch):
            schema = m.group(1) or "dbo"
            name = m.group(2)
            start_line = line_offset + batch[: m.start()].count("\n") + 1
            end_line = line_offset + batch.count("\n") + 1
            source = batch.strip()

            units.append(self._make_unit(
                unit_type="view",
                name=name,
                schema=schema,
                source=source,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
            ))

        return units

    def _make_unit(
        self,
        unit_type: str,
        name: str,
        schema: str,
        source: str,
        file_path: str,
        start_line: int,
        end_line: int,
        param_text: str = "",
        return_type: str = "VOID",
        trigger_table: str = "",
    ) -> CodeUnit:
        """Build a CodeUnit with SQL-specific metadata."""
        qualified = f"{schema}.{name}"

        # Extract parameters
        parameters = self._extract_parameters(param_text) if param_text else []

        # Extract table references from body
        tables = self._extract_table_refs(source)

        # Extract SP calls from body
        sp_calls = self._extract_sp_calls(source, name)

        # Detect deprecation hints
        is_deprecated = bool(
            re.search(r"--\s*deprecated|/\*.*deprecated.*\*/", source, re.IGNORECASE)
        )

        # Build signature
        param_sig = ", ".join(
            f"{p['name']} {p['type']}" + (" OUTPUT" if p["direction"] == "OUTPUT" else "")
            for p in parameters
        )
        signature = f"CREATE {unit_type.upper().replace('_', ' ')} {qualified}({param_sig})"

        # Build metadata
        metadata: Dict[str, Any] = {
            "dialect": self._dialect,
            "schema": schema,
            "parameters": parameters,
            "tables_referenced": tables,
            "sp_calls": sp_calls,
            "is_deprecated": is_deprecated,
        }

        if unit_type == "sql_function":
            metadata["return_type"] = return_type
        if unit_type == "trigger":
            metadata["trigger_table"] = trigger_table

        return CodeUnit(
            unit_type=unit_type,
            name=name,
            qualified_name=qualified,
            language="sql",
            start_line=start_line,
            end_line=end_line,
            source=source,
            file_path=file_path,
            signature=signature,
            metadata=metadata,
        )

    def _extract_parameters(self, param_text: str) -> List[Dict[str, str]]:
        """Extract parameter definitions from the text between name and AS/BEGIN."""
        params: List[Dict[str, str]] = []
        for m in _TSQL_PARAM_RE.finditer(param_text):
            direction = "OUTPUT" if m.group(3) and m.group(3).strip() else "IN"
            params.append({
                "name": m.group(1),
                "type": m.group(2).upper(),
                "direction": direction,
            })
        return params

    def _extract_table_refs(self, source: str) -> List[str]:
        """Extract unique table names referenced in FROM/JOIN/INSERT/UPDATE/DELETE."""
        tables: set[str] = set()
        # Skip references to well-known non-table targets
        skip = {"inserted", "deleted", "sys", "information_schema"}

        for m in _TABLE_REF_RE.finditer(source):
            table = m.group(2)
            if table.lower() not in skip and not table.startswith("#"):
                tables.add(table)
        return sorted(tables)

    def _extract_sp_calls(self, source: str, self_name: str) -> List[str]:
        """Extract names of stored procedures called via EXEC/CALL."""
        calls: set[str] = set()
        for m in _SP_CALL_RE.finditer(source):
            sp_name = m.group(2)
            # Skip self-reference and system SPs
            if sp_name.lower() != self_name.lower() and not sp_name.startswith("sp_"):
                calls.add(sp_name)
        return sorted(calls)

    def _fallback_blocks(self, source_text: str, file_path: str) -> List[CodeUnit]:
        """Split unrecognized SQL into generic blocks (same as FallbackParser)."""
        lines = source_text.split("\n")
        block_text = source_text.strip()
        if not block_text:
            return []
        return [
            CodeUnit(
                unit_type="block",
                name="sql_block_1",
                qualified_name=f"{file_path}::sql_block_1",
                language="sql",
                start_line=1,
                end_line=len(lines),
                source=block_text,
                file_path=file_path,
                metadata={"dialect": self._dialect},
            )
        ]

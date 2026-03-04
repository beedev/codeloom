"""JCL (Job Control Language) parser — regex-based.

JCL orchestrates z/OS batch jobs: it names a JOB, defines execution STEPS
(each running a COBOL or PL/1 program), and allocates datasets via DD statements.
No tree-sitter grammar exists for JCL. Standalone class, same interface as
Pl1Parser and VbNetParser (parse_file / parse_source).

IBM z/OS JCL fixed-format column layout (72-column record):
  1-2:  // (statement indicator)
  3-10: Name field (0-8 chars: A-Z, 0-9, @, #, $; case-insensitive in practice)
  11:   Required space (separator)
  12-71: Operation and operands
  72:   Non-blank = continuation follows on next line (detected but not joined)
  73-80: Sequence numbers (ignored — columns 73+ stripped)

Extracts:
  JOB statement      → unit_type="job"     (one per job card; top-level container)
  EXEC PGM=PROGNAME  → unit_type="step"    (program execution step)
  EXEC PROC=PROCNAME → unit_type="step"    (cataloged procedure invocation step)
  EXEC PROCNAME      → unit_type="step"    (shorthand cataloged proc invocation)
  Inline PROC...PEND → unit_type="proc"    (inline cataloged procedure definition)

ASG edges produced:
  step → program/procedure (calls edge via metadata["pgm"] or metadata["proc_name"])
    e.g. EXEC PGM=CUSTUPD → calls edge to COBOL PROGRAM-ID. CUSTUPD

DSNAME= / DSN= references → imports list (dataset dependency tracking).
Continuation lines (// + spaces in name field): operand spanning is not joined;
the key EXEC PGM=/PROC= info on the first line is always captured.
"""

import logging
import re
from typing import Dict, List, Optional

from .models import CodeUnit, ParseResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Full JCL statement: //[name] operation [operands...]
# Name field: 0-8 chars from [A-Z0-9@#$]. Operation: alphabetic keyword.
_JCL_STMT_RE = re.compile(
    r"^//(?P<name>[A-Z0-9@#$]{0,8})\s+(?P<op>[A-Z]{1,8})(?:\s+(?P<operands>.*))?$",
    re.IGNORECASE,
)

# Comment line: //*
_JCL_COMMENT_RE = re.compile(r"^//\*")

# Continuation line: name field (cols 3-10) is blank → "//   continuation data"
# Detected when _JCL_STMT_RE matches with empty name AND operation looks like
# a keyword-value (e.g. "//             PARM=..."). We skip these for the parser
# since PGM= / PROC= are always on the first EXEC line in practice.
_JCL_CONTINUATION_RE = re.compile(r"^//\s{2,}\S")

# EXEC operand: PGM=program-name (load module to execute)
_PGM_RE = re.compile(r"\bPGM=([A-Z0-9@#$]{1,8})", re.IGNORECASE)

# EXEC operand: PROC=proc-name (explicit cataloged proc keyword)
_PROC_KW_RE = re.compile(r"\bPROC=([A-Z0-9@#$]{1,8})", re.IGNORECASE)

# DD operand: DSNAME= or DSN= (dataset name, up to 44 chars, dot-separated qualifiers)
# Leading && indicates a temporary dataset — strip && prefix but keep for data_flow.
_DSN_RE = re.compile(
    r"\b(?:DSNAME|DSN)=(&&?[A-Z0-9@#$][A-Z0-9@#$.]{0,42}|[A-Z0-9@#$][A-Z0-9@#$.]{0,43})",
    re.IGNORECASE,
)

# DD operand: DISP= (dataset disposition — determines producer/consumer role)
# Forms: DISP=SHR, DISP=OLD, DISP=(NEW,CATLG,DELETE), DISP=(OLD,PASS)
_DISP_RE = re.compile(r"\bDISP=(\([^)]+\)|\w+)", re.IGNORECASE)

# EXEC operand: PARM= (runtime parameter string passed to COBOL LINKAGE SECTION)
# Forms: PARM='value', PARM=value, PARM=(val1,val2)
_PARM_RE = re.compile(r"\bPARM=(?:'([^']*)'|(\([^)]*\))|([^,\s/]+))", re.IGNORECASE)

# Valid JCL name: starts with letter/@/#/$, up to 8 chars (for shorthand EXEC)
_JCL_NAME_RE = re.compile(r"^[A-Z@#$][A-Z0-9@#$]{0,7}$", re.IGNORECASE)

# JCL operations that are NOT proc shorthand (to avoid treating keywords as proc names)
_JCL_OPERATIONS = frozenset({
    "JOB", "EXEC", "DD", "PROC", "PEND", "IF", "ELSE", "ENDIF", "INCLUDE",
    "JCLLIB", "NOTIFY", "OUTPUT", "SCHEDULE", "SET", "XMIT",
})


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class JclParser:
    """Regex-based JCL parser.

    Standalone class (not a BaseLanguageParser subclass) — same interface as
    Pl1Parser: implements parse_file() and parse_source().

    Unit types emitted:
      "job"   — JOB statement (top-level job container)
      "step"  — EXEC PGM= or EXEC [PROC=]name (program/proc invocation)
      "proc"  — Inline PROC...PEND definition (cataloged procedure template)

    Key metadata on step units:
      "pgm": str              — program name when EXEC PGM= is used
      "proc_name": str        — proc name when EXEC PROC= or EXEC name is used
      "parm": str | None      — PARM= value (runtime parameters to program)
      "dd_statements": list   — DD allocations for this step:
          [{"ddname": "INFILE", "dsn": "PROD.CUST.MASTER", "disp": "SHR"}, ...]
    These drive calls edges (pgm/proc_name) and data_flow edges (dd_statements)
    in the ASG builder.
    """

    def parse_file(self, file_path: str, project_root: str = "") -> ParseResult:
        """Parse a JCL source file."""
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
                language="jcl",
                units=[],
                imports=[],
                line_count=0,
            )

        return self.parse_source(source_text, rel_path)

    def parse_source(self, source_text: str, file_path: str) -> ParseResult:
        """Parse JCL source text into a ParseResult."""
        # Truncate each line to column 72 (cols 73-80 are sequence numbers)
        raw_lines = source_text.split("\n")
        lines = [ln[:72].rstrip() for ln in raw_lines]
        line_count = len(lines)

        units: List[CodeUnit] = []
        imports: List[str] = []
        seen_dsns: set = set()

        # Tracking state
        current_job: Optional[Dict] = None    # {name, start_line}
        current_proc: Optional[Dict] = None   # {name, start_line} — inline PROC
        open_step: Optional[Dict] = None      # open EXEC step accumulating DD lines

        def _parent_name() -> str:
            if current_proc:
                return current_proc["name"]
            if current_job:
                return current_job["name"]
            return ""

        def _close_step(end_line: int) -> None:
            nonlocal open_step
            if open_step is None:
                return
            s = open_step
            step_src = "\n".join(raw_lines[s["start_line"] - 1 : end_line])
            parent = s["parent_name"]
            qname = f"{parent}.{s['name']}" if parent else s["name"]

            meta: Dict = {}
            if s["pgm"]:
                meta["pgm"] = s["pgm"]
                sig = f"{s['name']} EXEC PGM={s['pgm']}"
            elif s["proc_name"]:
                meta["proc_name"] = s["proc_name"]
                sig = f"{s['name']} EXEC {s['proc_name']}"
            else:
                sig = f"{s['name']} EXEC"
            if parent:
                meta["parent_name"] = parent
            if s.get("parm"):
                meta["parm"] = s["parm"]
            if s.get("dd_statements"):
                meta["dd_statements"] = s["dd_statements"]

            units.append(CodeUnit(
                unit_type="step",
                name=s["name"],
                qualified_name=qname,
                language="jcl",
                start_line=s["start_line"],
                end_line=end_line,
                source=step_src,
                file_path=file_path,
                signature=sig,
                parent_name=parent or None,
                metadata=meta,
            ))
            open_step = None

        def _close_job(end_line: int) -> None:
            nonlocal current_job
            if current_job is None:
                return
            job_src = "\n".join(raw_lines[current_job["start_line"] - 1 : end_line])
            units.insert(0, CodeUnit(
                unit_type="job",
                name=current_job["name"],
                qualified_name=current_job["name"],
                language="jcl",
                start_line=current_job["start_line"],
                end_line=end_line,
                source=job_src,
                file_path=file_path,
                signature=f"{current_job['name']} JOB",
                metadata={},
            ))
            current_job = None

        for lineno, line in enumerate(lines, start=1):
            if not line.startswith("//"):
                continue

            if _JCL_COMMENT_RE.match(line):
                continue

            # Continuation lines — skip as standalone (operand info is on first line)
            if _JCL_CONTINUATION_RE.match(line):
                continue

            m = _JCL_STMT_RE.match(line)
            if not m:
                continue

            name = m.group("name").strip().upper()
            op = m.group("op").upper()
            operands = (m.group("operands") or "").strip()
            # Remove inline comments (operands end at first non-quoted space for
            # simple cases; this heuristic handles the majority of real JCL)
            operands = _strip_jcl_comment(operands)

            if op == "JOB":
                _close_step(lineno - 1)
                _close_job(lineno - 1)
                current_job = {"name": name or f"_job_{lineno}", "start_line": lineno}

            elif op == "EXEC":
                _close_step(lineno - 1)
                step_name = name if name else f"_step_{lineno}"
                pgm = None
                proc_name = None

                pgm_m = _PGM_RE.search(operands)
                if pgm_m:
                    pgm = pgm_m.group(1).upper()
                else:
                    proc_m = _PROC_KW_RE.search(operands)
                    if proc_m:
                        proc_name = proc_m.group(1).upper()
                    elif operands:
                        # Positional shorthand: "EXEC PROCNAME[,...]"
                        first_tok = operands.split(",")[0].strip()
                        if (
                            _JCL_NAME_RE.match(first_tok)
                            and first_tok.upper() not in _JCL_OPERATIONS
                        ):
                            proc_name = first_tok.upper()

                # Extract PARM= from EXEC operands
                parm_val: Optional[str] = None
                parm_m = _PARM_RE.search(operands)
                if parm_m:
                    parm_val = (
                        parm_m.group(1) or parm_m.group(2) or parm_m.group(3) or ""
                    ).strip()

                open_step = {
                    "name": step_name,
                    "start_line": lineno,
                    "pgm": pgm,
                    "proc_name": proc_name,
                    "parent_name": _parent_name(),
                    "parm": parm_val,
                    "dd_statements": [],
                }

            elif op == "PROC":
                _close_step(lineno - 1)
                current_proc = {
                    "name": name if name else f"_proc_{lineno}",
                    "start_line": lineno,
                }

            elif op == "PEND":
                _close_step(lineno - 1)
                if current_proc:
                    proc_src = "\n".join(
                        raw_lines[current_proc["start_line"] - 1 : lineno]
                    )
                    parent = current_job["name"] if current_job else ""
                    qname = (
                        f"{parent}.{current_proc['name']}" if parent
                        else current_proc["name"]
                    )
                    units.append(CodeUnit(
                        unit_type="proc",
                        name=current_proc["name"],
                        qualified_name=qname,
                        language="jcl",
                        start_line=current_proc["start_line"],
                        end_line=lineno,
                        source=proc_src,
                        file_path=file_path,
                        signature=f"{current_proc['name']} PROC",
                        parent_name=parent or None,
                        metadata={"inline": True},
                    ))
                    current_proc = None

            elif op == "DD":
                # Per-step DD allocation capture (ddname, dsn, disp)
                ddname = name  # Name field of DD statement IS the DDNAME
                dsn_val: Optional[str] = None
                disp_val: Optional[str] = None

                dsn_m = _DSN_RE.search(operands)
                if dsn_m:
                    raw_dsn = dsn_m.group(1).rstrip(".")
                    dsn_val = raw_dsn  # Keep && prefix for temp dataset detection
                    # Strip && for imports deduplication
                    clean_dsn = raw_dsn.lstrip("&")
                    if clean_dsn not in seen_dsns:
                        seen_dsns.add(clean_dsn)
                        imports.append(f"DSNAME={clean_dsn}")

                disp_m = _DISP_RE.search(operands)
                if disp_m:
                    disp_val = disp_m.group(1).strip("()").strip()

                # Append to open step's dd_statements list
                if open_step is not None and ddname:
                    open_step["dd_statements"].append({
                        "ddname": ddname,
                        "dsn": dsn_val,
                        "disp": disp_val,
                    })

        # Close any open step and job at EOF
        _close_step(line_count)
        _close_job(line_count)

        # Stamp imports on all units (consistent with other parsers)
        for unit in units:
            unit.imports = imports

        return ParseResult(
            file_path=file_path,
            language="jcl",
            units=units,
            imports=imports,
            line_count=line_count,
        )


def _strip_jcl_comment(operands: str) -> str:
    """Strip trailing inline comment from JCL operand string.

    JCL operands end at the first unquoted blank (simple heuristic: split on
    two or more consecutive spaces, which conventionally separate operands from
    comments in mainframe JCL).  This is good enough for extracting PGM= and DSN=.
    """
    # Two or more spaces signal the start of a JCL comment field
    idx = operands.find("  ")
    if idx > 0:
        return operands[:idx].strip()
    return operands

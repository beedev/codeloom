"""AST Parser utilities.

Language detection, parser registry, and helper functions.
"""

import os
import re
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseLanguageParser

# Extension → language mapping
SUPPORTED_EXTENSIONS: Dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".cs": "csharp",
    ".sql": "sql",
    ".vb": "vbnet",
    ".xml": "xml_config",
    ".jsp": "jsp",
    ".jspf": "jsp",
    ".properties": "properties",
    ".aspx": "aspx",
    ".ascx": "aspx",
    ".master": "aspx",
    # COBOL (tree-sitter via tree-sitter-language-pack)
    ".cbl": "cobol",
    ".cob": "cobol",
    ".cobol": "cobol",
    ".cpy": "cobol",  # Copybook (COPY member) — shared data layouts
    # PL/1 (regex-based parser)
    ".pl1": "pl1",
    ".pli": "pl1",
    ".plx": "pl1",
    # JCL (regex-based parser)
    ".jcl": "jcl",
    ".jcllib": "jcl",
    ".proc": "jcl",
}

# Directories to skip during file walking
SKIP_DIRECTORIES = frozenset({
    "__pycache__",
    ".git",
    "venv",
    "env",
    ".venv",
    ".env",
    "node_modules",
    ".tox",
    "__pypackages__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".eggs",
    "*.egg-info",
    # C# / .NET
    "bin",
    "obj",
    "packages",
})

# Parser registry — lazy-loaded to avoid import overhead
_parser_registry: Dict[str, "BaseLanguageParser"] = {}


def detect_language(file_path: str) -> Optional[str]:
    """Detect programming language from file extension, with content fallback.

    For files with a known extension (e.g. .py, .java, .cbl) the extension
    table is used — no file I/O needed.  When the file has no extension (common
    with mainframe datasets exported to disk: MAINPGM, COBRUN, SORT1, …) the
    first 20 lines are read to apply COBOL / JCL / PL/1 heuristics.

    Args:
        file_path: Absolute or relative path to the source file.

    Returns:
        Language identifier string or None if unsupported / unrecognised.
    """
    _, ext = os.path.splitext(file_path)
    if ext:
        return SUPPORTED_EXTENSIONS.get(ext.lower())

    # No extension — try content-based detection (mainframe files, etc.)
    return _detect_language_from_content(file_path)


# ---------------------------------------------------------------------------
# Content-based language heuristics for extensionless files
# ---------------------------------------------------------------------------

# COBOL: fixed-format divisions / PROGRAM-ID paragraph
_COBOL_KEYWORDS_RE = re.compile(
    r"\b(IDENTIFICATION\s+DIVISION|PROGRAM-ID|PROCEDURE\s+DIVISION|DATA\s+DIVISION"
    r"|ENVIRONMENT\s+DIVISION|WORKING-STORAGE\s+SECTION)\b",
    re.IGNORECASE,
)

# PL/1: labelled PROCEDURE or standalone PROCEDURE/PACKAGE keyword
_PL1_KEYWORDS_RE = re.compile(
    r"\bPROCEDURE\b|\bPROC\b|\bPACKAGE\b|\bDECLARE\b|\bDCL\b|\bPUT\s+SKIP\b",
    re.IGNORECASE,
)


def _detect_language_from_content(file_path: str) -> Optional[str]:
    """Read the first 20 lines of a file and apply COBOL/JCL/PL/1 heuristics.

    Returns a language identifier or None if the content is unrecognised.
    Only called for extensionless files to avoid unnecessary file I/O.
    Silently returns None on any I/O error.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
            head = [fh.readline() for _ in range(20)]
    except OSError:
        return None

    # Count lines that start with // (JCL statement indicator)
    jcl_lines = sum(1 for ln in head if ln.startswith("//"))
    if jcl_lines >= 2:
        return "jcl"

    text = "".join(head)

    # COBOL: any canonical division or PROGRAM-ID keyword present
    if _COBOL_KEYWORDS_RE.search(text):
        return "cobol"

    # PL/1: PROCEDURE/PROC/DECLARE keywords without COBOL divisions
    # Only trigger when multiple PL/1-specific tokens are present to avoid
    # false positives from SQL or other languages.
    pl1_hits = len(_PL1_KEYWORDS_RE.findall(text))
    if pl1_hits >= 2:
        return "pl1"

    return None


def get_parser(language: str) -> "BaseLanguageParser":
    """Get a parser instance for the given language.

    Uses a lazy-initialized registry to avoid importing
    all parser modules at startup.

    Args:
        language: Language identifier (e.g., "python")

    Returns:
        Parser instance

    Raises:
        ValueError: If language is not supported
    """
    if language not in _parser_registry:
        if language == "python":
            from .python_parser import PythonParser
            _parser_registry["python"] = PythonParser()
        elif language == "javascript":
            from .javascript_parser import JavaScriptParser
            _parser_registry["javascript"] = JavaScriptParser()
        elif language == "typescript":
            from .typescript_parser import TypeScriptParser
            _parser_registry["typescript"] = TypeScriptParser()
        elif language == "java":
            from .java_parser import JavaParser
            _parser_registry["java"] = JavaParser()
        elif language == "csharp":
            from .csharp_parser import CSharpParser
            _parser_registry["csharp"] = CSharpParser()
        elif language == "sql":
            from .sql_parser import SqlParser
            _parser_registry["sql"] = SqlParser()
        elif language == "vbnet":
            from .vbnet_parser import VbNetParser
            _parser_registry["vbnet"] = VbNetParser()
        elif language == "xml_config":
            from .xml_config_parser import XmlConfigParser
            _parser_registry["xml_config"] = XmlConfigParser()
        elif language == "jsp":
            from .jsp_parser import JspParser
            _parser_registry["jsp"] = JspParser()
        elif language == "properties":
            from .properties_parser import PropertiesParser
            _parser_registry["properties"] = PropertiesParser()
        elif language == "aspx":
            from .asp_parser import AspParser
            _parser_registry["aspx"] = AspParser()
        elif language == "cobol":
            from .cobol_parser import CobolParser
            _parser_registry["cobol"] = CobolParser()
        elif language == "pl1":
            from .pl1_parser import Pl1Parser
            _parser_registry["pl1"] = Pl1Parser()
        elif language == "jcl":
            from .jcl_parser import JclParser
            _parser_registry["jcl"] = JclParser()
        else:
            raise ValueError(
                f"Unsupported language: {language}. "
                f"Supported: {list(SUPPORTED_EXTENSIONS.values())}"
            )

    return _parser_registry[language]


def should_skip_directory(dir_name: str) -> bool:
    """Check if a directory should be skipped during file walking.

    Args:
        dir_name: Directory name (not full path)

    Returns:
        True if directory should be skipped
    """
    return dir_name in SKIP_DIRECTORIES or dir_name.startswith(".")


def is_supported_file(file_path: str) -> bool:
    """Check if a file has a supported language extension.

    Args:
        file_path: Path to the file

    Returns:
        True if file type is supported for AST parsing
    """
    return detect_language(file_path) is not None

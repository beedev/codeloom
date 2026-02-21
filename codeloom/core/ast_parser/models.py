"""AST Parser data models.

Defines the core data structures for parsed code representation.
These are pure data containers — no parsing logic.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CodeUnit:
    """A single parsed code entity (function, class, method, module block).

    Represents one logical code unit extracted by tree-sitter AST parsing.
    Each CodeUnit maps 1:1 to a code chunk for embedding.
    """

    unit_type: str  # "function" | "class" | "method" | "module"
    name: str  # "parse_file"
    qualified_name: str  # "codeloom.core.ast_parser.parser.parse_file"
    language: str  # "python"
    start_line: int
    end_line: int
    source: str  # Raw source code
    file_path: str  # Relative path within project
    signature: Optional[str] = None  # "def parse_file(path: str) -> ParseResult:"
    docstring: Optional[str] = None
    parent_name: Optional[str] = None  # For methods: class name
    decorators: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)  # Module-level imports
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParseError:
    """An error encountered during parsing."""

    file_path: str
    line: int
    message: str
    severity: str = "warning"  # "warning" | "error"


@dataclass
class ParseResult:
    """Complete parse output for a single file.

    Contains all extracted code units, imports, and any parse errors.
    """

    file_path: str
    language: str
    units: List[CodeUnit]
    imports: List[str]  # All import statements in file
    module_docstring: Optional[str] = None
    line_count: int = 0
    errors: List[ParseError] = field(default_factory=list)

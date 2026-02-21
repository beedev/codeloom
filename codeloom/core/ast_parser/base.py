"""Base interface for language-specific AST parsers.

Defines the Strategy pattern base class that all language parsers implement.
Shared parsing logic lives here; language-specific extraction is delegated.
"""

import logging
from abc import ABC, abstractmethod
from typing import List

import tree_sitter

from .models import CodeUnit, ParseError, ParseResult

logger = logging.getLogger(__name__)


class BaseLanguageParser(ABC):
    """Abstract base for language-specific tree-sitter parsers.

    Subclasses implement:
    - get_language(): returns language name string
    - get_tree_sitter_language(): returns tree-sitter Language object
    - extract_units(): walks AST tree and extracts CodeUnit objects
    - extract_imports(): extracts import statements from AST
    """

    @abstractmethod
    def get_language(self) -> str:
        """Return the language identifier (e.g., 'python', 'javascript')."""
        ...

    @abstractmethod
    def get_tree_sitter_language(self) -> tree_sitter.Language:
        """Return the tree-sitter Language object for this language."""
        ...

    @abstractmethod
    def extract_units(
        self, tree: tree_sitter.Tree, source: bytes, file_path: str
    ) -> List[CodeUnit]:
        """Extract code units from a parsed tree-sitter AST.

        Args:
            tree: Parsed tree-sitter tree
            source: Raw source bytes
            file_path: Relative file path within project

        Returns:
            List of CodeUnit objects
        """
        ...

    @abstractmethod
    def extract_imports(self, tree: tree_sitter.Tree, source: bytes) -> List[str]:
        """Extract import statements from the AST.

        Args:
            tree: Parsed tree-sitter tree
            source: Raw source bytes

        Returns:
            List of import statement strings
        """
        ...

    def parse_file(self, file_path: str, project_root: str = "") -> ParseResult:
        """Parse a source file into a ParseResult.

        Shared logic: reads file, creates tree-sitter parser,
        delegates to subclass extract methods.

        Args:
            file_path: Absolute path to the source file
            project_root: Project root for computing relative paths

        Returns:
            ParseResult with extracted units and metadata
        """
        errors: List[ParseError] = []

        # Compute relative path
        if project_root and file_path.startswith(project_root):
            rel_path = file_path[len(project_root):].lstrip("/")
        else:
            rel_path = file_path

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                source_text = f.read()
        except OSError as e:
            return ParseResult(
                file_path=rel_path,
                language=self.get_language(),
                units=[],
                imports=[],
                line_count=0,
                errors=[ParseError(file_path=rel_path, line=0, message=str(e), severity="error")],
            )

        return self.parse_source(source_text, rel_path)

    def parse_source(self, source_text: str, file_path: str) -> ParseResult:
        """Parse source code string into a ParseResult.

        Args:
            source_text: Source code as string
            file_path: Relative file path (for metadata)

        Returns:
            ParseResult with extracted units and metadata
        """
        errors: List[ParseError] = []
        source_bytes = source_text.encode("utf-8")
        line_count = source_text.count("\n") + (1 if source_text and not source_text.endswith("\n") else 0)

        # Create parser and parse
        parser = tree_sitter.Parser(self.get_tree_sitter_language())
        tree = parser.parse(source_bytes)

        # Check for parse errors
        if tree.root_node.has_error:
            errors.append(
                ParseError(
                    file_path=file_path,
                    line=0,
                    message="Tree-sitter reported parse errors in file",
                    severity="warning",
                )
            )

        # Extract imports
        try:
            imports = self.extract_imports(tree, source_bytes)
        except Exception as e:
            logger.warning(f"Failed to extract imports from {file_path}: {e}")
            imports = []
            errors.append(ParseError(file_path=file_path, line=0, message=f"Import extraction failed: {e}"))

        # Extract code units
        try:
            units = self.extract_units(tree, source_bytes, file_path)
        except Exception as e:
            logger.error(f"Failed to extract units from {file_path}: {e}")
            units = []
            errors.append(ParseError(file_path=file_path, line=0, message=f"Unit extraction failed: {e}", severity="error"))

        # Inject imports into each unit for preamble building
        for unit in units:
            unit.imports = imports

        # Extract module docstring (first string literal at module level)
        module_docstring = self._extract_module_docstring(tree, source_bytes)

        return ParseResult(
            file_path=file_path,
            language=self.get_language(),
            units=units,
            imports=imports,
            module_docstring=module_docstring,
            line_count=line_count,
            errors=errors,
        )

    def _extract_module_docstring(self, tree: tree_sitter.Tree, source: bytes) -> str | None:
        """Extract module-level docstring if present.

        Default implementation checks for first expression_statement
        containing a string at the module level. Subclasses can override.
        """
        root = tree.root_node
        for child in root.children:
            if child.type == "expression_statement":
                for sub in child.children:
                    if sub.type == "string":
                        raw = source[sub.start_byte:sub.end_byte].decode("utf-8", errors="replace")
                        return raw.strip("\"'").strip()
            # Stop at first non-comment, non-docstring node
            if child.type not in ("expression_statement", "comment"):
                break
        return None

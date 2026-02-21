"""Python-specific AST parser using tree-sitter.

Walks the tree-sitter AST to extract functions, classes, methods,
and module-level code with signatures, docstrings, and decorators.
"""

import logging
from typing import List, Optional

import tree_sitter
import tree_sitter_python

from .base import BaseLanguageParser
from .models import CodeUnit

logger = logging.getLogger(__name__)

# Create the Language object once (wraps the PyCapsule)
_PYTHON_LANGUAGE = tree_sitter.Language(tree_sitter_python.language())


class PythonParser(BaseLanguageParser):
    """tree-sitter based Python parser.

    Extracts:
    - Module-level functions → unit_type="function"
    - Classes → unit_type="class"
    - Methods inside classes → unit_type="method"
    - Module-level code blocks → unit_type="module"
    """

    def get_language(self) -> str:
        return "python"

    def get_tree_sitter_language(self) -> tree_sitter.Language:
        return _PYTHON_LANGUAGE

    def extract_imports(self, tree: tree_sitter.Tree, source: bytes) -> List[str]:
        """Extract all import statements from the AST."""
        imports = []
        for child in tree.root_node.children:
            if child.type in ("import_statement", "import_from_statement"):
                text = source[child.start_byte:child.end_byte].decode("utf-8", errors="replace").strip()
                imports.append(text)
        return imports

    def extract_units(
        self, tree: tree_sitter.Tree, source: bytes, file_path: str
    ) -> List[CodeUnit]:
        """Extract code units from the Python AST.

        Walks top-level children of the module, extracting:
        - function_definition → CodeUnit(unit_type="function")
        - class_definition → CodeUnit(unit_type="class") + methods
        - Remaining module-level code → CodeUnit(unit_type="module")
        """
        units: List[CodeUnit] = []
        root = tree.root_node

        # Compute module qualified name prefix from file path
        module_prefix = self._file_path_to_module(file_path)

        for child in root.children:
            if child.type == "function_definition":
                unit = self._extract_function(child, source, file_path, module_prefix, parent_name=None)
                if unit:
                    units.append(unit)

            elif child.type == "decorated_definition":
                # Decorated function or class
                inner = self._get_decorated_inner(child)
                if inner and inner.type == "function_definition":
                    unit = self._extract_function(inner, source, file_path, module_prefix, parent_name=None, decorator_node=child)
                    if unit:
                        units.append(unit)
                elif inner and inner.type == "class_definition":
                    class_units = self._extract_class(inner, source, file_path, module_prefix, decorator_node=child)
                    units.extend(class_units)

            elif child.type == "class_definition":
                class_units = self._extract_class(child, source, file_path, module_prefix)
                units.extend(class_units)

        return units

    def _extract_function(
        self,
        node: tree_sitter.Node,
        source: bytes,
        file_path: str,
        module_prefix: str,
        parent_name: Optional[str] = None,
        decorator_node: Optional[tree_sitter.Node] = None,
    ) -> Optional[CodeUnit]:
        """Extract a single function/method definition."""
        name = self._get_child_text(node, "name", source)
        if not name:
            return None

        # Use the decorator node for full source range if decorated
        source_node = decorator_node if decorator_node else node
        unit_source = source[source_node.start_byte:source_node.end_byte].decode("utf-8", errors="replace")

        # Build qualified name
        if parent_name:
            qualified_name = f"{module_prefix}.{parent_name}.{name}" if module_prefix else f"{parent_name}.{name}"
            unit_type = "method"
        else:
            qualified_name = f"{module_prefix}.{name}" if module_prefix else name
            unit_type = "function"

        # Extract signature (first line up to colon)
        signature = self._extract_signature(node, source)

        # Extract docstring
        docstring = self._extract_docstring(node, source)

        # Extract decorators
        decorators = self._extract_decorators(decorator_node or node, source)

        return CodeUnit(
            unit_type=unit_type,
            name=name,
            qualified_name=qualified_name,
            language="python",
            start_line=source_node.start_point.row + 1,
            end_line=source_node.end_point.row + 1,
            source=unit_source,
            file_path=file_path,
            signature=signature,
            docstring=docstring,
            parent_name=parent_name,
            decorators=decorators,
        )

    def _extract_class(
        self,
        node: tree_sitter.Node,
        source: bytes,
        file_path: str,
        module_prefix: str,
        decorator_node: Optional[tree_sitter.Node] = None,
    ) -> List[CodeUnit]:
        """Extract a class definition and its methods."""
        units: List[CodeUnit] = []

        class_name = self._get_child_text(node, "name", source)
        if not class_name:
            return units

        # Class-level CodeUnit (includes full class source)
        source_node = decorator_node if decorator_node else node
        class_source = source[source_node.start_byte:source_node.end_byte].decode("utf-8", errors="replace")
        qualified_name = f"{module_prefix}.{class_name}" if module_prefix else class_name

        # Extract class signature (e.g., "class Foo(BaseClass):")
        signature = self._extract_class_signature(node, source)
        docstring = self._extract_docstring(node, source)
        decorators = self._extract_decorators(decorator_node or node, source)

        class_unit = CodeUnit(
            unit_type="class",
            name=class_name,
            qualified_name=qualified_name,
            language="python",
            start_line=source_node.start_point.row + 1,
            end_line=source_node.end_point.row + 1,
            source=class_source,
            file_path=file_path,
            signature=signature,
            docstring=docstring,
            decorators=decorators,
        )
        units.append(class_unit)

        # Extract methods within the class body
        body = self._get_child_by_type(node, "block")
        if body:
            for child in body.children:
                if child.type == "function_definition":
                    method = self._extract_function(
                        child, source, file_path, module_prefix,
                        parent_name=class_name,
                    )
                    if method:
                        units.append(method)
                elif child.type == "decorated_definition":
                    inner = self._get_decorated_inner(child)
                    if inner and inner.type == "function_definition":
                        method = self._extract_function(
                            inner, source, file_path, module_prefix,
                            parent_name=class_name,
                            decorator_node=child,
                        )
                        if method:
                            units.append(method)

        return units

    # =========================================================================
    # Helper methods
    # =========================================================================

    @staticmethod
    def _get_child_text(node: tree_sitter.Node, field_name: str, source: bytes) -> Optional[str]:
        """Get text content of a named child field."""
        child = node.child_by_field_name(field_name)
        if child:
            return source[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
        return None

    @staticmethod
    def _get_child_by_type(node: tree_sitter.Node, type_name: str) -> Optional[tree_sitter.Node]:
        """Find first child of a given type."""
        for child in node.children:
            if child.type == type_name:
                return child
        return None

    @staticmethod
    def _get_decorated_inner(node: tree_sitter.Node) -> Optional[tree_sitter.Node]:
        """Get the inner definition from a decorated_definition node."""
        for child in node.children:
            if child.type in ("function_definition", "class_definition"):
                return child
        return None

    @staticmethod
    def _extract_signature(node: tree_sitter.Node, source: bytes) -> str:
        """Extract function signature (def line up to colon)."""
        # Find the parameters node to build a clean signature
        name_node = node.child_by_field_name("name")
        params_node = node.child_by_field_name("parameters")
        return_type = node.child_by_field_name("return_type")

        if not name_node:
            # Fallback: first line of source
            text = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
            first_line = text.split("\n")[0].rstrip()
            return first_line

        parts = ["def ", source[name_node.start_byte:name_node.end_byte].decode("utf-8")]
        if params_node:
            parts.append(source[params_node.start_byte:params_node.end_byte].decode("utf-8"))
        if return_type:
            parts.append(" -> ")
            parts.append(source[return_type.start_byte:return_type.end_byte].decode("utf-8"))
        parts.append(":")

        return "".join(parts)

    @staticmethod
    def _extract_class_signature(node: tree_sitter.Node, source: bytes) -> str:
        """Extract class signature (class line up to colon)."""
        # Get text from class keyword to the colon
        text = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        first_line = text.split("\n")[0].rstrip()
        return first_line

    @staticmethod
    def _extract_docstring(node: tree_sitter.Node, source: bytes) -> Optional[str]:
        """Extract docstring from a function or class body."""
        body = None
        for child in node.children:
            if child.type == "block":
                body = child
                break

        if not body or not body.children:
            return None

        # First statement in body
        first_stmt = body.children[0]
        if first_stmt.type == "expression_statement":
            for sub in first_stmt.children:
                if sub.type == "string":
                    raw = source[sub.start_byte:sub.end_byte].decode("utf-8", errors="replace")
                    # Strip triple quotes
                    for quote in ('"""', "'''"):
                        if raw.startswith(quote) and raw.endswith(quote):
                            return raw[3:-3].strip()
                    # Strip single/double quotes
                    return raw.strip("\"'").strip()

        return None

    @staticmethod
    def _extract_decorators(node: tree_sitter.Node, source: bytes) -> List[str]:
        """Extract decorator strings from a node."""
        decorators = []
        for child in node.children:
            if child.type == "decorator":
                text = source[child.start_byte:child.end_byte].decode("utf-8", errors="replace").strip()
                decorators.append(text)
        return decorators

    @staticmethod
    def _file_path_to_module(file_path: str) -> str:
        """Convert file path to Python module dotted name.

        e.g., "codeloom/core/ast_parser/parser.py" → "codeloom.core.ast_parser.parser"
        """
        # Remove extension
        path = file_path
        if path.endswith(".py"):
            path = path[:-3]
        # Replace separators with dots
        path = path.replace("/", ".").replace("\\", ".")
        # Remove trailing .__init__
        if path.endswith(".__init__"):
            path = path[:-9]
        return path

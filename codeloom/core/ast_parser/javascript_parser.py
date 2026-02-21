"""JavaScript AST parser using tree-sitter.

Walks the tree-sitter AST to extract functions, classes, methods,
and import statements from JavaScript source files.
"""

import logging
from typing import List, Optional

import tree_sitter
import tree_sitter_javascript

from .base import BaseLanguageParser
from .models import CodeUnit

logger = logging.getLogger(__name__)

_JS_LANGUAGE = tree_sitter.Language(tree_sitter_javascript.language())


class JavaScriptParser(BaseLanguageParser):
    """tree-sitter based JavaScript parser.

    Extracts:
    - Top-level function declarations -> unit_type="function"
    - Arrow functions assigned to const/let/var -> unit_type="function"
    - Class declarations -> unit_type="class"
    - Methods inside classes -> unit_type="method"
    - Exported variants of all the above
    """

    def get_language(self) -> str:
        return "javascript"

    def get_tree_sitter_language(self) -> tree_sitter.Language:
        return _JS_LANGUAGE

    def extract_imports(self, tree: tree_sitter.Tree, source: bytes) -> List[str]:
        """Extract import statements from the AST."""
        imports = []
        for child in tree.root_node.children:
            if child.type == "import_statement":
                text = source[child.start_byte:child.end_byte].decode("utf-8", errors="replace").strip()
                imports.append(text)
            elif child.type == "export_statement":
                # Re-exports: export { foo } from './bar'
                for sub in child.children:
                    if sub.type == "import_statement":
                        text = source[sub.start_byte:sub.end_byte].decode("utf-8", errors="replace").strip()
                        imports.append(text)
        return imports

    def extract_units(
        self, tree: tree_sitter.Tree, source: bytes, file_path: str
    ) -> List[CodeUnit]:
        """Extract code units from the JavaScript AST."""
        units: List[CodeUnit] = []
        root = tree.root_node
        module_prefix = self._file_path_to_module(file_path)

        for child in root.children:
            if child.type == "function_declaration":
                unit = self._extract_function(child, source, file_path, module_prefix)
                if unit:
                    units.append(unit)

            elif child.type == "class_declaration":
                class_units = self._extract_class(child, source, file_path, module_prefix)
                units.extend(class_units)

            elif child.type in ("lexical_declaration", "variable_declaration"):
                # const foo = () => {} or const foo = function() {}
                arrow_units = self._extract_assigned_functions(child, source, file_path, module_prefix)
                units.extend(arrow_units)

            elif child.type == "export_statement":
                export_units = self._extract_export(child, source, file_path, module_prefix)
                units.extend(export_units)

        return units

    def _extract_export(
        self,
        node: tree_sitter.Node,
        source: bytes,
        file_path: str,
        module_prefix: str,
    ) -> List[CodeUnit]:
        """Extract units from an export statement (default or named)."""
        units: List[CodeUnit] = []
        for child in node.children:
            if child.type == "function_declaration":
                unit = self._extract_function(child, source, file_path, module_prefix, export_node=node)
                if unit:
                    units.append(unit)
            elif child.type == "class_declaration":
                class_units = self._extract_class(child, source, file_path, module_prefix, export_node=node)
                units.extend(class_units)
            elif child.type in ("lexical_declaration", "variable_declaration"):
                arrow_units = self._extract_assigned_functions(child, source, file_path, module_prefix, export_node=node)
                units.extend(arrow_units)
        return units

    def _extract_function(
        self,
        node: tree_sitter.Node,
        source: bytes,
        file_path: str,
        module_prefix: str,
        parent_name: Optional[str] = None,
        export_node: Optional[tree_sitter.Node] = None,
    ) -> Optional[CodeUnit]:
        """Extract a function declaration."""
        name = self._get_child_text(node, "name", source)
        if not name:
            return None

        source_node = export_node if export_node else node
        unit_source = source[source_node.start_byte:source_node.end_byte].decode("utf-8", errors="replace")

        if parent_name:
            qualified_name = f"{module_prefix}.{parent_name}.{name}" if module_prefix else f"{parent_name}.{name}"
            unit_type = "method"
        else:
            qualified_name = f"{module_prefix}.{name}" if module_prefix else name
            unit_type = "function"

        signature = self._extract_function_signature(node, source)
        docstring = self._extract_jsdoc(node, source)

        return CodeUnit(
            unit_type=unit_type,
            name=name,
            qualified_name=qualified_name,
            language="javascript",
            start_line=source_node.start_point.row + 1,
            end_line=source_node.end_point.row + 1,
            source=unit_source,
            file_path=file_path,
            signature=signature,
            docstring=docstring,
            parent_name=parent_name,
        )

    def _extract_assigned_functions(
        self,
        node: tree_sitter.Node,
        source: bytes,
        file_path: str,
        module_prefix: str,
        export_node: Optional[tree_sitter.Node] = None,
    ) -> List[CodeUnit]:
        """Extract arrow functions or function expressions assigned to variables.

        Handles: const foo = () => {} and const foo = function() {}
        """
        units: List[CodeUnit] = []
        for child in node.children:
            if child.type != "variable_declarator":
                continue

            name_node = child.child_by_field_name("name")
            value_node = child.child_by_field_name("value")
            if not name_node or not value_node:
                continue
            if value_node.type not in ("arrow_function", "function_expression"):
                continue

            name = source[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace")
            source_node = export_node if export_node else node
            unit_source = source[source_node.start_byte:source_node.end_byte].decode("utf-8", errors="replace")
            qualified_name = f"{module_prefix}.{name}" if module_prefix else name

            # Build signature from the variable assignment
            params_node = value_node.child_by_field_name("parameters")
            if params_node:
                params = source[params_node.start_byte:params_node.end_byte].decode("utf-8", errors="replace")
                signature = f"const {name} = {params} =>"
            else:
                # function_expression
                signature = f"const {name} = function()"

            docstring = self._extract_jsdoc(source_node, source)

            units.append(CodeUnit(
                unit_type="function",
                name=name,
                qualified_name=qualified_name,
                language="javascript",
                start_line=source_node.start_point.row + 1,
                end_line=source_node.end_point.row + 1,
                source=unit_source,
                file_path=file_path,
                signature=signature,
                docstring=docstring,
            ))
        return units

    def _extract_class(
        self,
        node: tree_sitter.Node,
        source: bytes,
        file_path: str,
        module_prefix: str,
        export_node: Optional[tree_sitter.Node] = None,
    ) -> List[CodeUnit]:
        """Extract a class and its methods."""
        units: List[CodeUnit] = []

        class_name = self._get_child_text(node, "name", source)
        if not class_name:
            return units

        source_node = export_node if export_node else node
        class_source = source[source_node.start_byte:source_node.end_byte].decode("utf-8", errors="replace")
        qualified_name = f"{module_prefix}.{class_name}" if module_prefix else class_name

        signature = self._extract_class_signature(node, source)
        docstring = self._extract_jsdoc(source_node, source)

        class_unit = CodeUnit(
            unit_type="class",
            name=class_name,
            qualified_name=qualified_name,
            language="javascript",
            start_line=source_node.start_point.row + 1,
            end_line=source_node.end_point.row + 1,
            source=class_source,
            file_path=file_path,
            signature=signature,
            docstring=docstring,
        )
        units.append(class_unit)

        # Extract methods from class body
        body = self._get_child_by_type(node, "class_body")
        if body:
            for child in body.children:
                if child.type == "method_definition":
                    method = self._extract_method(child, source, file_path, module_prefix, class_name)
                    if method:
                        units.append(method)

        return units

    def _extract_method(
        self,
        node: tree_sitter.Node,
        source: bytes,
        file_path: str,
        module_prefix: str,
        class_name: str,
    ) -> Optional[CodeUnit]:
        """Extract a method definition from a class body."""
        name = self._get_child_text(node, "name", source)
        if not name:
            return None

        unit_source = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        qualified_name = f"{module_prefix}.{class_name}.{name}" if module_prefix else f"{class_name}.{name}"

        # Build signature
        params_node = node.child_by_field_name("parameters")
        params = source[params_node.start_byte:params_node.end_byte].decode("utf-8", errors="replace") if params_node else "()"
        signature = f"{name}{params}"

        docstring = self._extract_jsdoc(node, source)

        return CodeUnit(
            unit_type="method",
            name=name,
            qualified_name=qualified_name,
            language="javascript",
            start_line=node.start_point.row + 1,
            end_line=node.end_point.row + 1,
            source=unit_source,
            file_path=file_path,
            signature=signature,
            docstring=docstring,
            parent_name=class_name,
        )

    # =========================================================================
    # Helpers
    # =========================================================================

    @staticmethod
    def _get_child_text(node: tree_sitter.Node, field_name: str, source: bytes) -> Optional[str]:
        child = node.child_by_field_name(field_name)
        if child:
            return source[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
        return None

    @staticmethod
    def _get_child_by_type(node: tree_sitter.Node, type_name: str) -> Optional[tree_sitter.Node]:
        for child in node.children:
            if child.type == type_name:
                return child
        return None

    @staticmethod
    def _extract_function_signature(node: tree_sitter.Node, source: bytes) -> str:
        """Extract function signature: function name(params)."""
        name_node = node.child_by_field_name("name")
        params_node = node.child_by_field_name("parameters")

        if not name_node:
            text = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
            return text.split("\n")[0].rstrip()

        name = source[name_node.start_byte:name_node.end_byte].decode("utf-8")
        params = source[params_node.start_byte:params_node.end_byte].decode("utf-8") if params_node else "()"
        return f"function {name}{params}"

    @staticmethod
    def _extract_class_signature(node: tree_sitter.Node, source: bytes) -> str:
        """Extract class signature line."""
        text = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        first_line = text.split("\n")[0].rstrip()
        # Trim the opening brace for cleaner signature
        return first_line.rstrip(" {")

    @staticmethod
    def _extract_jsdoc(node: tree_sitter.Node, source: bytes) -> Optional[str]:
        """Extract JSDoc comment preceding a node.

        Looks at the previous sibling for a comment node starting with /**.
        """
        prev = node.prev_named_sibling
        if prev and prev.type == "comment":
            text = source[prev.start_byte:prev.end_byte].decode("utf-8", errors="replace").strip()
            if text.startswith("/**"):
                # Strip /** and */
                content = text[3:]
                if content.endswith("*/"):
                    content = content[:-2]
                # Clean up leading * on each line
                lines = []
                for line in content.split("\n"):
                    stripped = line.strip()
                    if stripped.startswith("* "):
                        lines.append(stripped[2:])
                    elif stripped.startswith("*"):
                        lines.append(stripped[1:].strip())
                    else:
                        lines.append(stripped)
                return "\n".join(lines).strip()
        return None

    @staticmethod
    def _file_path_to_module(file_path: str) -> str:
        """Convert file path to a dotted module name.

        e.g., "src/utils/helpers.js" -> "src.utils.helpers"
        """
        path = file_path
        for ext in (".js", ".jsx", ".mjs", ".cjs"):
            if path.endswith(ext):
                path = path[: -len(ext)]
                break
        path = path.replace("/", ".").replace("\\", ".")
        if path.endswith(".index"):
            path = path[:-6]
        return path

    def _extract_module_docstring(self, tree: tree_sitter.Tree, source: bytes) -> Optional[str]:
        """Extract module-level JSDoc if present."""
        root = tree.root_node
        for child in root.children:
            if child.type == "comment":
                text = source[child.start_byte:child.end_byte].decode("utf-8", errors="replace").strip()
                if text.startswith("/**"):
                    content = text[3:]
                    if content.endswith("*/"):
                        content = content[:-2]
                    return content.strip()
            if child.type not in ("comment",):
                break
        return None

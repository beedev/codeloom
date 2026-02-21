"""Java AST parser using tree-sitter.

Walks the tree-sitter AST to extract classes, interfaces, methods,
constructors, enums, and import/package statements from Java source files.
"""

import logging
from typing import List, Optional

import tree_sitter
import tree_sitter_java

from .base import BaseLanguageParser
from .models import CodeUnit

logger = logging.getLogger(__name__)

_JAVA_LANGUAGE = tree_sitter.Language(tree_sitter_java.language())


class JavaParser(BaseLanguageParser):
    """tree-sitter based Java parser.

    Extracts:
    - Class declarations -> unit_type="class"
    - Interface declarations -> unit_type="interface"
    - Enum declarations -> unit_type="enum"
    - Method declarations inside classes -> unit_type="method"
    - Constructor declarations -> unit_type="constructor"
    - Package and import declarations -> metadata
    """

    def get_language(self) -> str:
        return "java"

    def get_tree_sitter_language(self) -> tree_sitter.Language:
        return _JAVA_LANGUAGE

    def extract_imports(self, tree: tree_sitter.Tree, source: bytes) -> List[str]:
        """Extract import and package declarations from the AST."""
        imports = []
        for child in tree.root_node.children:
            if child.type in ("import_declaration", "package_declaration"):
                text = source[child.start_byte:child.end_byte].decode("utf-8", errors="replace").strip()
                imports.append(text)
        return imports

    def extract_units(
        self, tree: tree_sitter.Tree, source: bytes, file_path: str
    ) -> List[CodeUnit]:
        """Extract code units from the Java AST."""
        units: List[CodeUnit] = []
        root = tree.root_node

        # Extract package name for qualified name prefix
        package_name = self._extract_package(root, source)

        for child in root.children:
            if child.type == "class_declaration":
                class_units = self._extract_class(child, source, file_path, package_name)
                units.extend(class_units)

            elif child.type == "interface_declaration":
                iface_units = self._extract_interface(child, source, file_path, package_name)
                units.extend(iface_units)

            elif child.type == "enum_declaration":
                unit = self._extract_enum(child, source, file_path, package_name)
                if unit:
                    units.append(unit)

        return units

    # =========================================================================
    # Top-level extractors
    # =========================================================================

    def _extract_class(
        self,
        node: tree_sitter.Node,
        source: bytes,
        file_path: str,
        package_name: str,
        parent_class: Optional[str] = None,
    ) -> List[CodeUnit]:
        """Extract a class declaration and its members."""
        units: List[CodeUnit] = []

        class_name = self._get_child_text(node, "name", source)
        if not class_name:
            return units

        unit_source = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

        if parent_class:
            qualified_name = f"{package_name}.{parent_class}.{class_name}" if package_name else f"{parent_class}.{class_name}"
        else:
            qualified_name = f"{package_name}.{class_name}" if package_name else class_name

        signature = self._extract_class_signature(node, source)
        docstring = self._extract_javadoc(node, source)

        # Extract inheritance info for ASG
        extends, implements = self._extract_inheritance(node, source)
        metadata = {}
        if extends:
            metadata["extends"] = extends
        if implements:
            metadata["implements"] = implements

        # Extract annotations
        annotations = self._extract_annotations(node, source)
        if annotations:
            metadata["annotations"] = annotations

        class_unit = CodeUnit(
            unit_type="class",
            name=class_name,
            qualified_name=qualified_name,
            language="java",
            start_line=node.start_point.row + 1,
            end_line=node.end_point.row + 1,
            source=unit_source,
            file_path=file_path,
            signature=signature,
            docstring=docstring,
            metadata=metadata,
        )
        units.append(class_unit)

        # Extract members from class body
        body = self._get_child_by_type(node, "class_body")
        if body:
            for child in body.children:
                if child.type == "method_declaration":
                    method = self._extract_method(child, source, file_path, package_name, class_name)
                    if method:
                        units.append(method)

                elif child.type == "constructor_declaration":
                    ctor = self._extract_constructor(child, source, file_path, package_name, class_name)
                    if ctor:
                        units.append(ctor)

                elif child.type == "class_declaration":
                    # Nested class
                    nested = self._extract_class(child, source, file_path, package_name, parent_class=class_name)
                    units.extend(nested)

                elif child.type == "interface_declaration":
                    nested = self._extract_interface(child, source, file_path, package_name, parent_class=class_name)
                    units.extend(nested)

                elif child.type == "enum_declaration":
                    enum_unit = self._extract_enum(child, source, file_path, package_name, parent_class=class_name)
                    if enum_unit:
                        units.append(enum_unit)

        return units

    def _extract_interface(
        self,
        node: tree_sitter.Node,
        source: bytes,
        file_path: str,
        package_name: str,
        parent_class: Optional[str] = None,
    ) -> List[CodeUnit]:
        """Extract an interface declaration and its method signatures."""
        units: List[CodeUnit] = []

        iface_name = self._get_child_text(node, "name", source)
        if not iface_name:
            return units

        unit_source = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

        if parent_class:
            qualified_name = f"{package_name}.{parent_class}.{iface_name}" if package_name else f"{parent_class}.{iface_name}"
        else:
            qualified_name = f"{package_name}.{iface_name}" if package_name else iface_name

        signature = self._extract_class_signature(node, source)
        docstring = self._extract_javadoc(node, source)

        # Interface extends
        extends_list = self._extract_interface_extends(node, source)
        metadata = {}
        if extends_list:
            metadata["extends"] = extends_list

        annotations = self._extract_annotations(node, source)
        if annotations:
            metadata["annotations"] = annotations

        iface_unit = CodeUnit(
            unit_type="interface",
            name=iface_name,
            qualified_name=qualified_name,
            language="java",
            start_line=node.start_point.row + 1,
            end_line=node.end_point.row + 1,
            source=unit_source,
            file_path=file_path,
            signature=signature,
            docstring=docstring,
            metadata=metadata,
        )
        units.append(iface_unit)

        # Extract method declarations in interface body
        body = self._get_child_by_type(node, "interface_body")
        if body:
            for child in body.children:
                if child.type == "method_declaration":
                    method = self._extract_method(child, source, file_path, package_name, iface_name)
                    if method:
                        units.append(method)

        return units

    def _extract_enum(
        self,
        node: tree_sitter.Node,
        source: bytes,
        file_path: str,
        package_name: str,
        parent_class: Optional[str] = None,
    ) -> Optional[CodeUnit]:
        """Extract an enum declaration."""
        enum_name = self._get_child_text(node, "name", source)
        if not enum_name:
            return None

        unit_source = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

        if parent_class:
            qualified_name = f"{package_name}.{parent_class}.{enum_name}" if package_name else f"{parent_class}.{enum_name}"
        else:
            qualified_name = f"{package_name}.{enum_name}" if package_name else enum_name

        signature = self._extract_class_signature(node, source)
        docstring = self._extract_javadoc(node, source)

        metadata = {}
        annotations = self._extract_annotations(node, source)
        if annotations:
            metadata["annotations"] = annotations

        return CodeUnit(
            unit_type="enum",
            name=enum_name,
            qualified_name=qualified_name,
            language="java",
            start_line=node.start_point.row + 1,
            end_line=node.end_point.row + 1,
            source=unit_source,
            file_path=file_path,
            signature=signature,
            docstring=docstring,
            metadata=metadata,
        )

    def _extract_method(
        self,
        node: tree_sitter.Node,
        source: bytes,
        file_path: str,
        package_name: str,
        class_name: str,
    ) -> Optional[CodeUnit]:
        """Extract a method declaration from a class or interface body."""
        name = self._get_child_text(node, "name", source)
        if not name:
            return None

        unit_source = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        qualified_name = f"{package_name}.{class_name}.{name}" if package_name else f"{class_name}.{name}"

        signature = self._extract_method_signature(node, source)
        docstring = self._extract_javadoc(node, source)

        metadata = {}
        annotations = self._extract_annotations(node, source)
        if annotations:
            metadata["annotations"] = annotations

        return CodeUnit(
            unit_type="method",
            name=name,
            qualified_name=qualified_name,
            language="java",
            start_line=node.start_point.row + 1,
            end_line=node.end_point.row + 1,
            source=unit_source,
            file_path=file_path,
            signature=signature,
            docstring=docstring,
            parent_name=class_name,
            metadata=metadata,
        )

    def _extract_constructor(
        self,
        node: tree_sitter.Node,
        source: bytes,
        file_path: str,
        package_name: str,
        class_name: str,
    ) -> Optional[CodeUnit]:
        """Extract a constructor declaration."""
        name = self._get_child_text(node, "name", source)
        if not name:
            name = class_name

        unit_source = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        qualified_name = f"{package_name}.{class_name}.{name}" if package_name else f"{class_name}.{name}"

        signature = self._extract_method_signature(node, source)
        docstring = self._extract_javadoc(node, source)

        metadata = {}
        annotations = self._extract_annotations(node, source)
        if annotations:
            metadata["annotations"] = annotations

        return CodeUnit(
            unit_type="constructor",
            name=name,
            qualified_name=qualified_name,
            language="java",
            start_line=node.start_point.row + 1,
            end_line=node.end_point.row + 1,
            source=unit_source,
            file_path=file_path,
            signature=signature,
            docstring=docstring,
            parent_name=class_name,
            metadata=metadata,
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
    def _extract_package(root: tree_sitter.Node, source: bytes) -> str:
        """Extract package name from the compilation unit."""
        for child in root.children:
            if child.type == "package_declaration":
                # package com.example.foo;
                text = source[child.start_byte:child.end_byte].decode("utf-8", errors="replace").strip()
                # Remove 'package ' prefix and trailing ';'
                pkg = text.replace("package ", "").rstrip(";").strip()
                return pkg
        return ""

    @staticmethod
    def _extract_class_signature(node: tree_sitter.Node, source: bytes) -> str:
        """Extract class/interface/enum signature (first line up to opening brace)."""
        text = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        first_line = text.split("\n")[0].rstrip()
        return first_line.rstrip(" {").rstrip()

    @staticmethod
    def _extract_method_signature(node: tree_sitter.Node, source: bytes) -> str:
        """Extract method/constructor signature including modifiers, return type, params.

        Builds signature from the declaration up to the opening brace or semicolon.
        """
        text = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        # Take everything before the method body
        for i, char in enumerate(text):
            if char == '{':
                return text[:i].strip()
            if char == ';':
                return text[:i].strip()
        # Fallback: first line
        return text.split("\n")[0].rstrip()

    @staticmethod
    def _extract_javadoc(node: tree_sitter.Node, source: bytes) -> Optional[str]:
        """Extract Javadoc comment preceding a node.

        Looks at the previous sibling for a block_comment starting with /**.
        """
        prev = node.prev_named_sibling
        if prev and prev.type == "block_comment":
            text = source[prev.start_byte:prev.end_byte].decode("utf-8", errors="replace").strip()
            if text.startswith("/**"):
                content = text[3:]
                if content.endswith("*/"):
                    content = content[:-2]
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
    def _extract_inheritance(node: tree_sitter.Node, source: bytes) -> tuple:
        """Extract extends and implements clauses from a class declaration.

        Returns:
            (extends: str | None, implements: list[str])
        """
        extends = None
        implements = []

        for child in node.children:
            if child.type == "superclass":
                # superclass node contains the type identifier
                for sub in child.children:
                    if sub.type == "type_identifier" or sub.is_named:
                        text = source[sub.start_byte:sub.end_byte].decode("utf-8", errors="replace")
                        if text not in ("extends",):
                            extends = text
                            break

            elif child.type == "super_interfaces":
                # super_interfaces contains a type_list
                for sub in child.children:
                    if sub.type == "type_list":
                        for type_node in sub.children:
                            if type_node.is_named:
                                text = source[type_node.start_byte:type_node.end_byte].decode("utf-8", errors="replace")
                                implements.append(text)

        return extends, implements

    @staticmethod
    def _extract_interface_extends(node: tree_sitter.Node, source: bytes) -> List[str]:
        """Extract extends list from an interface declaration."""
        extends = []
        for child in node.children:
            if child.type == "extends_interfaces":
                for sub in child.children:
                    if sub.type == "type_list":
                        for type_node in sub.children:
                            if type_node.is_named:
                                text = source[type_node.start_byte:type_node.end_byte].decode("utf-8", errors="replace")
                                extends.append(text)
        return extends

    @staticmethod
    def _extract_annotations(node: tree_sitter.Node, source: bytes) -> List[str]:
        """Extract annotation strings preceding a declaration.

        Handles both standalone annotations and marker annotations.
        """
        annotations = []
        prev = node.prev_named_sibling
        while prev and prev.type in ("marker_annotation", "annotation"):
            text = source[prev.start_byte:prev.end_byte].decode("utf-8", errors="replace").strip()
            annotations.insert(0, text)
            prev = prev.prev_named_sibling

        # Also check direct children (modifiers can contain annotations)
        for child in node.children:
            if child.type == "modifiers":
                for mod_child in child.children:
                    if mod_child.type in ("marker_annotation", "annotation"):
                        text = source[mod_child.start_byte:mod_child.end_byte].decode("utf-8", errors="replace").strip()
                        if text not in annotations:
                            annotations.append(text)

        return annotations

    def _extract_module_docstring(self, tree: tree_sitter.Tree, source: bytes) -> Optional[str]:
        """Extract file-level Javadoc if present (first block comment)."""
        root = tree.root_node
        for child in root.children:
            if child.type == "block_comment":
                text = source[child.start_byte:child.end_byte].decode("utf-8", errors="replace").strip()
                if text.startswith("/**"):
                    content = text[3:]
                    if content.endswith("*/"):
                        content = content[:-2]
                    return content.strip()
            if child.type not in ("block_comment", "line_comment", "package_declaration", "import_declaration"):
                break
        return None

    @staticmethod
    def _file_path_to_module(file_path: str) -> str:
        """Convert file path to Java-style dotted name.

        e.g., "com/example/service/UserService.java" -> "com.example.service.UserService"
        """
        path = file_path
        if path.endswith(".java"):
            path = path[:-5]
        return path.replace("/", ".").replace("\\", ".")

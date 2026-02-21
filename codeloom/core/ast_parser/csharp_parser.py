"""C# AST parser using tree-sitter.

Walks the tree-sitter AST to extract classes, interfaces, structs, enums,
records, methods, constructors, and properties from C# source files.
"""

import logging
from typing import List, Optional

import tree_sitter
import tree_sitter_c_sharp

from .base import BaseLanguageParser
from .models import CodeUnit

logger = logging.getLogger(__name__)

_CSHARP_LANGUAGE = tree_sitter.Language(tree_sitter_c_sharp.language())


class CSharpParser(BaseLanguageParser):
    """tree-sitter based C# parser.

    Extracts:
    - Class declarations -> unit_type="class"
    - Interface declarations -> unit_type="interface"
    - Struct declarations -> unit_type="struct"
    - Enum declarations -> unit_type="enum"
    - Record declarations -> unit_type="record"
    - Method declarations -> unit_type="method"
    - Constructor declarations -> unit_type="constructor"
    - Property declarations -> unit_type="property"
    - Namespace and using declarations -> metadata
    """

    def get_language(self) -> str:
        return "csharp"

    def get_tree_sitter_language(self) -> tree_sitter.Language:
        return _CSHARP_LANGUAGE

    def extract_imports(self, tree: tree_sitter.Tree, source: bytes) -> List[str]:
        """Extract using directives from the AST."""
        imports = []
        for child in tree.root_node.children:
            if child.type == "using_directive":
                text = source[child.start_byte:child.end_byte].decode("utf-8", errors="replace").strip()
                imports.append(text)
        return imports

    def extract_units(
        self, tree: tree_sitter.Tree, source: bytes, file_path: str
    ) -> List[CodeUnit]:
        """Extract code units from the C# AST."""
        units: List[CodeUnit] = []
        root = tree.root_node
        self._walk_members(root, source, file_path, namespace="", parent_class=None, units=units)
        return units

    # =========================================================================
    # Recursive member walker
    # =========================================================================

    def _walk_members(
        self,
        node: tree_sitter.Node,
        source: bytes,
        file_path: str,
        namespace: str,
        parent_class: Optional[str],
        units: List[CodeUnit],
    ) -> None:
        """Recursively walk AST nodes to extract type and member declarations."""
        for child in node.children:
            if child.type == "namespace_declaration":
                ns_name = self._extract_namespace_name(child, source)
                full_ns = f"{namespace}.{ns_name}" if namespace else ns_name
                self._walk_members(child, source, file_path, full_ns, parent_class, units)

            elif child.type == "file_scoped_namespace_declaration":
                ns_name = self._extract_namespace_name(child, source)
                full_ns = f"{namespace}.{ns_name}" if namespace else ns_name
                self._walk_members(child, source, file_path, full_ns, parent_class, units)

            elif child.type == "declaration_list":
                self._walk_members(child, source, file_path, namespace, parent_class, units)

            elif child.type == "class_declaration":
                self._extract_type(child, source, file_path, namespace, parent_class, "class", units)

            elif child.type == "interface_declaration":
                self._extract_type(child, source, file_path, namespace, parent_class, "interface", units)

            elif child.type == "struct_declaration":
                self._extract_type(child, source, file_path, namespace, parent_class, "struct", units)

            elif child.type == "enum_declaration":
                self._extract_enum(child, source, file_path, namespace, parent_class, units)

            elif child.type == "record_declaration":
                self._extract_type(child, source, file_path, namespace, parent_class, "record", units)

    # =========================================================================
    # Type-level extractors
    # =========================================================================

    def _extract_type(
        self,
        node: tree_sitter.Node,
        source: bytes,
        file_path: str,
        namespace: str,
        parent_class: Optional[str],
        unit_type: str,
        units: List[CodeUnit],
    ) -> None:
        """Extract a class/interface/struct/record declaration and its members."""
        name = self._get_child_text(node, "name", source)
        if not name:
            return

        unit_source = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        qualified_name = self._build_qualified_name(namespace, parent_class, name)
        signature = self._extract_type_signature(node, source)
        docstring = self._extract_xml_doc(node, source)

        metadata = {}

        # Inheritance: class Foo : Bar, IDisposable
        extends, implements = self._extract_inheritance(node, source)
        if extends:
            metadata["extends"] = extends
        if implements:
            metadata["implements"] = implements

        # Attributes: [Serializable], [HttpGet("route")]
        attributes = self._extract_attributes(node, source)
        if attributes:
            metadata["annotations"] = attributes

        # Modifiers: public, abstract, sealed, static, partial
        modifiers = self._extract_modifiers(node, source)
        if modifiers:
            metadata["modifiers"] = modifiers

        # Generic type parameters: class Repo<T> where T : IEntity
        generics = self._extract_generic_params(node, source)
        if generics:
            metadata["generic_params"] = generics

        type_unit = CodeUnit(
            unit_type=unit_type,
            name=name,
            qualified_name=qualified_name,
            language="csharp",
            start_line=node.start_point.row + 1,
            end_line=node.end_point.row + 1,
            source=unit_source,
            file_path=file_path,
            signature=signature,
            docstring=docstring,
            metadata=metadata,
        )
        units.append(type_unit)

        # Walk the declaration body for members
        body = self._get_child_by_type(node, "declaration_list")
        if body:
            for child in body.children:
                if child.type == "method_declaration":
                    method = self._extract_method(child, source, file_path, namespace, name)
                    if method:
                        units.append(method)

                elif child.type == "constructor_declaration":
                    ctor = self._extract_constructor(child, source, file_path, namespace, name)
                    if ctor:
                        units.append(ctor)

                elif child.type == "property_declaration":
                    prop = self._extract_property(child, source, file_path, namespace, name)
                    if prop:
                        units.append(prop)

                elif child.type == "class_declaration":
                    self._extract_type(child, source, file_path, namespace, name, "class", units)
                elif child.type == "interface_declaration":
                    self._extract_type(child, source, file_path, namespace, name, "interface", units)
                elif child.type == "struct_declaration":
                    self._extract_type(child, source, file_path, namespace, name, "struct", units)
                elif child.type == "enum_declaration":
                    self._extract_enum(child, source, file_path, namespace, name, units)
                elif child.type == "record_declaration":
                    self._extract_type(child, source, file_path, namespace, name, "record", units)

    def _extract_enum(
        self,
        node: tree_sitter.Node,
        source: bytes,
        file_path: str,
        namespace: str,
        parent_class: Optional[str],
        units: List[CodeUnit],
    ) -> None:
        """Extract an enum declaration."""
        name = self._get_child_text(node, "name", source)
        if not name:
            return

        unit_source = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        qualified_name = self._build_qualified_name(namespace, parent_class, name)
        signature = self._extract_type_signature(node, source)
        docstring = self._extract_xml_doc(node, source)

        metadata = {}
        attributes = self._extract_attributes(node, source)
        if attributes:
            metadata["annotations"] = attributes
        modifiers = self._extract_modifiers(node, source)
        if modifiers:
            metadata["modifiers"] = modifiers

        units.append(CodeUnit(
            unit_type="enum",
            name=name,
            qualified_name=qualified_name,
            language="csharp",
            start_line=node.start_point.row + 1,
            end_line=node.end_point.row + 1,
            source=unit_source,
            file_path=file_path,
            signature=signature,
            docstring=docstring,
            metadata=metadata,
        ))

    # =========================================================================
    # Member extractors
    # =========================================================================

    def _extract_method(
        self,
        node: tree_sitter.Node,
        source: bytes,
        file_path: str,
        namespace: str,
        class_name: str,
    ) -> Optional[CodeUnit]:
        """Extract a method declaration."""
        name = self._get_child_text(node, "name", source)
        if not name:
            return None

        unit_source = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        qualified_name = self._build_qualified_name(namespace, class_name, name)
        signature = self._extract_method_signature(node, source)
        docstring = self._extract_xml_doc(node, source)

        metadata = {}
        attributes = self._extract_attributes(node, source)
        if attributes:
            metadata["annotations"] = attributes
        modifiers = self._extract_modifiers(node, source)
        if modifiers:
            metadata["modifiers"] = modifiers

        return CodeUnit(
            unit_type="method",
            name=name,
            qualified_name=qualified_name,
            language="csharp",
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
        namespace: str,
        class_name: str,
    ) -> Optional[CodeUnit]:
        """Extract a constructor declaration."""
        name = self._get_child_text(node, "name", source) or class_name

        unit_source = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        qualified_name = self._build_qualified_name(namespace, class_name, name)
        signature = self._extract_method_signature(node, source)
        docstring = self._extract_xml_doc(node, source)

        metadata = {}
        attributes = self._extract_attributes(node, source)
        if attributes:
            metadata["annotations"] = attributes
        modifiers = self._extract_modifiers(node, source)
        if modifiers:
            metadata["modifiers"] = modifiers

        return CodeUnit(
            unit_type="constructor",
            name=name,
            qualified_name=qualified_name,
            language="csharp",
            start_line=node.start_point.row + 1,
            end_line=node.end_point.row + 1,
            source=unit_source,
            file_path=file_path,
            signature=signature,
            docstring=docstring,
            parent_name=class_name,
            metadata=metadata,
        )

    def _extract_property(
        self,
        node: tree_sitter.Node,
        source: bytes,
        file_path: str,
        namespace: str,
        class_name: str,
    ) -> Optional[CodeUnit]:
        """Extract a property declaration."""
        name = self._get_child_text(node, "name", source)
        if not name:
            return None

        unit_source = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        qualified_name = self._build_qualified_name(namespace, class_name, name)
        signature = self._extract_property_signature(node, source)
        docstring = self._extract_xml_doc(node, source)

        metadata = {}
        attributes = self._extract_attributes(node, source)
        if attributes:
            metadata["annotations"] = attributes
        modifiers = self._extract_modifiers(node, source)
        if modifiers:
            metadata["modifiers"] = modifiers

        return CodeUnit(
            unit_type="property",
            name=name,
            qualified_name=qualified_name,
            language="csharp",
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
    def _build_qualified_name(namespace: str, parent: Optional[str], name: str) -> str:
        """Build dotted qualified name from namespace, parent, and name."""
        parts = []
        if namespace:
            parts.append(namespace)
        if parent:
            parts.append(parent)
        parts.append(name)
        return ".".join(parts)

    @staticmethod
    def _extract_namespace_name(node: tree_sitter.Node, source: bytes) -> str:
        """Extract namespace name from a namespace declaration."""
        name_node = node.child_by_field_name("name")
        if name_node:
            return source[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace")
        return ""

    @staticmethod
    def _extract_type_signature(node: tree_sitter.Node, source: bytes) -> str:
        """Extract class/interface/struct/enum/record signature (first line up to brace)."""
        text = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        first_line = text.split("\n")[0].rstrip()
        return first_line.rstrip(" {").rstrip()

    @staticmethod
    def _extract_method_signature(node: tree_sitter.Node, source: bytes) -> str:
        """Extract method/constructor signature up to opening brace or semicolon."""
        text = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        for i, char in enumerate(text):
            if char == '{':
                return text[:i].strip()
            if char == ';':
                return text[:i].strip()
        return text.split("\n")[0].rstrip()

    @staticmethod
    def _extract_property_signature(node: tree_sitter.Node, source: bytes) -> str:
        """Extract property signature (type + name + accessors hint)."""
        text = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        first_line = text.split("\n")[0].rstrip()
        return first_line.rstrip(" {").rstrip()

    @staticmethod
    def _extract_xml_doc(node: tree_sitter.Node, source: bytes) -> Optional[str]:
        """Extract XML doc comment (///) preceding a node."""
        lines = []
        prev = node.prev_named_sibling
        # tree-sitter-c-sharp may produce comment nodes as siblings
        while prev and prev.type == "comment":
            text = source[prev.start_byte:prev.end_byte].decode("utf-8", errors="replace").strip()
            if text.startswith("///"):
                content = text[3:].strip()
                # Strip XML tags for plain text summary
                content = content.replace("<summary>", "").replace("</summary>", "")
                content = content.replace("<param ", "").replace("</param>", "")
                content = content.replace("<returns>", "Returns: ").replace("</returns>", "")
                if content:
                    lines.insert(0, content)
                prev = prev.prev_named_sibling
            else:
                break

        return "\n".join(lines).strip() if lines else None

    @staticmethod
    def _extract_inheritance(node: tree_sitter.Node, source: bytes) -> tuple:
        """Extract base class and implemented interfaces from base_list.

        C# uses `class Foo : Bar, IDisposable` syntax.
        Heuristic: names starting with 'I' + uppercase are interfaces,
        the first non-interface is the base class.

        Returns:
            (extends: str | None, implements: list[str])
        """
        extends = None
        implements = []

        base_list = None
        for child in node.children:
            if child.type == "base_list":
                base_list = child
                break

        if not base_list:
            return extends, implements

        for child in base_list.children:
            if not child.is_named or child.type in (":",):
                continue
            text = source[child.start_byte:child.end_byte].decode("utf-8", errors="replace").strip()
            if not text:
                continue

            # Heuristic: I + uppercase = interface
            bare_name = text.split("<")[0].split(".")[~0]  # Get last part, strip generics
            if len(bare_name) >= 2 and bare_name[0] == "I" and bare_name[1].isupper():
                implements.append(text)
            elif extends is None:
                extends = text
            else:
                # Ambiguous — could be another interface without I prefix
                implements.append(text)

        return extends, implements

    @staticmethod
    def _extract_attributes(node: tree_sitter.Node, source: bytes) -> List[str]:
        """Extract C# attributes ([Serializable], [HttpGet("route")]) from a declaration."""
        attributes = []
        for child in node.children:
            if child.type == "attribute_list":
                text = source[child.start_byte:child.end_byte].decode("utf-8", errors="replace").strip()
                # Strip outer brackets
                if text.startswith("[") and text.endswith("]"):
                    text = text[1:-1]
                # Split multiple attributes in one list: [Attr1, Attr2]
                for attr in text.split(","):
                    attr = attr.strip()
                    if attr:
                        attributes.append(f"[{attr}]")
        return attributes

    @staticmethod
    def _extract_modifiers(node: tree_sitter.Node, source: bytes) -> List[str]:
        """Extract modifier keywords (public, static, override, virtual, etc.)."""
        modifiers = []
        for child in node.children:
            if child.type == "modifier":
                text = source[child.start_byte:child.end_byte].decode("utf-8", errors="replace").strip()
                modifiers.append(text)
        return modifiers

    @staticmethod
    def _extract_generic_params(node: tree_sitter.Node, source: bytes) -> List[str]:
        """Extract generic type parameters and constraints."""
        params = []
        for child in node.children:
            if child.type == "type_parameter_list":
                text = source[child.start_byte:child.end_byte].decode("utf-8", errors="replace").strip()
                # Strip angle brackets and split
                if text.startswith("<") and text.endswith(">"):
                    text = text[1:-1]
                for param in text.split(","):
                    param = param.strip()
                    if param:
                        params.append(param)
            elif child.type == "type_parameter_constraints_clause":
                text = source[child.start_byte:child.end_byte].decode("utf-8", errors="replace").strip()
                params.append(text)
        return params

    def _extract_module_docstring(self, tree: tree_sitter.Tree, source: bytes) -> Optional[str]:
        """Extract file-level XML doc comment if present."""
        root = tree.root_node
        for child in root.children:
            if child.type == "comment":
                text = source[child.start_byte:child.end_byte].decode("utf-8", errors="replace").strip()
                if text.startswith("///") or text.startswith("//"):
                    return text.lstrip("/").strip()
            if child.type not in ("comment", "using_directive"):
                break
        return None

    @staticmethod
    def _file_path_to_module(file_path: str) -> str:
        """Convert file path to C#-style dotted name."""
        path = file_path
        if path.endswith(".cs"):
            path = path[:-3]
        return path.replace("/", ".").replace("\\", ".")

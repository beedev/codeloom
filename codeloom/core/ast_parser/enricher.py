"""Semantic metadata enricher — tree-sitter second-pass.

Adds structured type info to CodeUnit.metadata without modifying existing
fields. Extracts parsed_params, return_type, modifiers, is_async,
is_override, is_abstract from tree-sitter AST nodes.

Also extracts field declarations from class/interface/struct bodies to
populate metadata["fields"] for type_dep edge detection in the ASG builder.

No external dependencies beyond tree-sitter itself.
"""

import logging
from typing import Any, Dict, List, Optional

import tree_sitter

from .models import CodeUnit

logger = logging.getLogger(__name__)


class SemanticEnricher:
    """Lightweight second-pass enricher that adds structured metadata
    to CodeUnit objects using tree-sitter reparsing.

    Supports: Java, C#, Python, TypeScript.
    Safe to call on any language — unsupported languages are silently skipped.
    """

    # Languages this enricher knows how to process
    _SUPPORTED = frozenset({"java", "csharp", "python", "typescript"})

    def enrich_units(
        self,
        units: List[CodeUnit],
        source_text: str,
        language: str,
    ) -> List[CodeUnit]:
        """Enrich code units with structured parameter/return type metadata.

        Modifies units in-place (adds to metadata dict). Returns the same
        list for chaining convenience.

        Args:
            units: CodeUnit objects from initial parsing
            source_text: Full source text of the file
            language: Language identifier

        Returns:
            The same list of units (modified in-place)
        """
        if language not in self._SUPPORTED:
            return units

        source_bytes = source_text.encode("utf-8")

        try:
            ts_lang = self._get_language(language)
        except Exception:
            logger.debug(f"Could not load tree-sitter language for enrichment: {language}")
            return units

        parser = tree_sitter.Parser(ts_lang)
        tree = parser.parse(source_bytes)

        for unit in units:
            if unit.unit_type in ("method", "function", "constructor"):
                try:
                    self._enrich_callable(unit, tree, source_bytes, language)
                except Exception as e:
                    logger.debug(f"Enrichment failed for {unit.qualified_name}: {e}")

        # Second pass: extract field declarations from class/interface/struct
        for unit in units:
            if unit.unit_type in ("class", "interface", "struct"):
                try:
                    self._enrich_class_fields(unit, tree, source_bytes, language)
                except Exception as e:
                    logger.debug(f"Field enrichment failed for {unit.qualified_name}: {e}")

        return units

    # =========================================================================
    # Per-unit enrichment
    # =========================================================================

    def _enrich_callable(
        self,
        unit: CodeUnit,
        tree: tree_sitter.Tree,
        source: bytes,
        language: str,
    ) -> None:
        """Enrich a method/function/constructor with structured metadata."""
        target_types = self._declaration_types(language)
        node = self._find_typed_node_at_line(tree.root_node, unit.start_line, target_types)
        if not node:
            return

        metadata = unit.metadata

        if language == "python":
            self._enrich_python(node, source, metadata)
        elif language == "java":
            self._enrich_java(node, source, metadata)
        elif language == "csharp":
            self._enrich_csharp(node, source, metadata)

    def _enrich_class_fields(
        self,
        unit: CodeUnit,
        tree: tree_sitter.Tree,
        source: bytes,
        language: str,
    ) -> None:
        """Extract field declarations from a class/interface/struct body."""
        target_types = self._class_declaration_types(language)
        node = self._find_typed_node_at_line(tree.root_node, unit.start_line, target_types)
        if not node:
            return

        body = self._get_class_body(node, language)
        if not body:
            return

        fields = self._dispatch_field_extraction(body, source, language)
        if fields:
            unit.metadata["fields"] = fields

    def enrich_class_source(
        self,
        source_text: str,
        language: str,
    ) -> List[Dict[str, Any]]:
        """Extract fields from a class unit's stored source string.

        Used by build-asg to enrich existing class units without re-ingestion.
        Parses the source fragment, finds the class node, and extracts fields.

        Returns:
            List of {"name": ..., "type": ...} dicts, or empty list.
        """
        if language not in self._SUPPORTED:
            return []

        source_bytes = source_text.encode("utf-8")
        try:
            ts_lang = self._get_language(language)
        except Exception:
            return []

        parser = tree_sitter.Parser(ts_lang)
        tree = parser.parse(source_bytes)

        # The class declaration should be near the root of the parsed fragment
        target_types = self._class_declaration_types(language)
        node = self._find_first_typed_node(tree.root_node, target_types)
        if not node:
            return []

        body = self._get_class_body(node, language)
        if not body:
            return []

        return self._dispatch_field_extraction(body, source_bytes, language)

    def _dispatch_field_extraction(
        self,
        body_node: tree_sitter.Node,
        source: bytes,
        language: str,
    ) -> List[Dict[str, Any]]:
        """Dispatch field extraction to the appropriate language handler."""
        if language == "java":
            return self._extract_java_fields(body_node, source)
        elif language == "csharp":
            return self._extract_csharp_fields(body_node, source)
        elif language == "python":
            return self._extract_python_fields(body_node, source)
        elif language == "typescript":
            return self._extract_typescript_fields(body_node, source)
        return []

    @staticmethod
    def _get_class_body(
        node: tree_sitter.Node, language: str
    ) -> Optional[tree_sitter.Node]:
        """Get the body container node of a class/interface/struct."""
        if language == "java":
            for child in node.children:
                if child.type in ("class_body", "interface_body", "enum_body"):
                    return child
        elif language == "csharp":
            for child in node.children:
                if child.type == "declaration_list":
                    return child
        elif language == "python":
            for child in node.children:
                if child.type == "block":
                    return child
        elif language == "typescript":
            for child in node.children:
                if child.type == "class_body":
                    return child
        return None

    # =========================================================================
    # Python enrichment
    # =========================================================================

    def _enrich_python(
        self, node: tree_sitter.Node, source: bytes, metadata: Dict[str, Any]
    ) -> None:
        """Extract params, return type, and async/decorator info from Python AST."""
        params_node = node.child_by_field_name("parameters")
        if params_node:
            metadata["parsed_params"] = self._parse_python_params(params_node, source)

        ret_node = node.child_by_field_name("return_type")
        if ret_node:
            metadata["return_type"] = self._node_text(ret_node, source)

        # Check async
        metadata["is_async"] = any(
            c.type == "async" for c in (node.parent.children if node.parent else node.children)
        ) or node.type == "function_definition" and any(
            self._node_text(c, source) == "async" for c in node.children
        )

        # Check decorators
        if node.parent and node.parent.type == "decorated_definition":
            for c in node.parent.children:
                if c.type == "decorator":
                    text = self._node_text(c, source)
                    if "abstractmethod" in text:
                        metadata["is_abstract"] = True
                    if "override" in text:
                        metadata["is_override"] = True

        metadata.setdefault("is_override", False)
        metadata.setdefault("is_abstract", False)

    def _parse_python_params(
        self, params_node: tree_sitter.Node, source: bytes
    ) -> List[Dict[str, Any]]:
        """Parse Python function parameters into structured list."""
        params = []
        for child in params_node.children:
            if child.type in ("identifier", "typed_parameter", "default_parameter",
                              "typed_default_parameter"):
                param = self._parse_single_python_param(child, source)
                if param and param["name"] not in ("self", "cls"):
                    params.append(param)
        return params

    def _parse_single_python_param(
        self, node: tree_sitter.Node, source: bytes
    ) -> Optional[Dict[str, Any]]:
        """Parse a single Python parameter node."""
        if node.type == "identifier":
            return {"name": self._node_text(node, source), "type": None, "default": None, "optional": False}

        if node.type == "typed_parameter":
            name_node = node.child_by_field_name("name") or node.children[0] if node.children else None
            type_node = node.child_by_field_name("type")
            name = self._node_text(name_node, source) if name_node else "?"
            ptype = self._node_text(type_node, source) if type_node else None
            return {"name": name, "type": ptype, "default": None, "optional": False}

        if node.type == "default_parameter":
            name_node = node.child_by_field_name("name") or node.children[0] if node.children else None
            value_node = node.child_by_field_name("value")
            name = self._node_text(name_node, source) if name_node else "?"
            default = self._node_text(value_node, source) if value_node else None
            return {"name": name, "type": None, "default": default, "optional": True}

        if node.type == "typed_default_parameter":
            name_node = node.child_by_field_name("name") or node.children[0] if node.children else None
            type_node = node.child_by_field_name("type")
            value_node = node.child_by_field_name("value")
            name = self._node_text(name_node, source) if name_node else "?"
            ptype = self._node_text(type_node, source) if type_node else None
            default = self._node_text(value_node, source) if value_node else None
            return {"name": name, "type": ptype, "default": default, "optional": True}

        return None

    # =========================================================================
    # Java enrichment
    # =========================================================================

    def _enrich_java(
        self, node: tree_sitter.Node, source: bytes, metadata: Dict[str, Any]
    ) -> None:
        """Extract params, return type, modifiers from Java AST."""
        # Parameters: formal_parameters
        params_node = node.child_by_field_name("parameters")
        if params_node:
            metadata["parsed_params"] = self._parse_java_params(params_node, source)

        # Return type
        type_node = node.child_by_field_name("type")
        if type_node:
            metadata["return_type"] = self._node_text(type_node, source)

        # Modifiers
        modifiers = self._collect_modifiers(node, source)
        if modifiers:
            metadata["modifiers"] = modifiers

        metadata["is_async"] = False  # Java doesn't have async keyword
        metadata["is_override"] = any(
            a == "@Override" for a in metadata.get("annotations", [])
        )
        metadata["is_abstract"] = "abstract" in modifiers

    def _parse_java_params(
        self, params_node: tree_sitter.Node, source: bytes
    ) -> List[Dict[str, Any]]:
        """Parse Java formal_parameters into structured list."""
        params = []
        for child in params_node.children:
            if child.type == "formal_parameter" or child.type == "spread_parameter":
                name_node = child.child_by_field_name("name")
                type_node = child.child_by_field_name("type")
                name = self._node_text(name_node, source) if name_node else "?"
                ptype = self._node_text(type_node, source) if type_node else None
                params.append({
                    "name": name,
                    "type": ptype,
                    "default": None,
                    "optional": False,
                })
        return params

    # =========================================================================
    # C# enrichment
    # =========================================================================

    def _enrich_csharp(
        self, node: tree_sitter.Node, source: bytes, metadata: Dict[str, Any]
    ) -> None:
        """Extract params, return type, modifiers from C# AST."""
        # Parameters: parameter_list
        params_node = node.child_by_field_name("parameters")
        if params_node:
            metadata["parsed_params"] = self._parse_csharp_params(params_node, source)

        # Return type
        type_node = node.child_by_field_name("type")
        if type_node:
            metadata["return_type"] = self._node_text(type_node, source)

        # Modifiers
        modifiers = self._collect_modifiers(node, source)
        if modifiers:
            metadata["modifiers"] = modifiers

        metadata["is_async"] = "async" in modifiers
        metadata["is_override"] = "override" in modifiers
        metadata["is_abstract"] = "abstract" in modifiers

    def _parse_csharp_params(
        self, params_node: tree_sitter.Node, source: bytes
    ) -> List[Dict[str, Any]]:
        """Parse C# parameter_list into structured list."""
        params = []
        for child in params_node.children:
            if child.type == "parameter":
                name_node = child.child_by_field_name("name")
                type_node = child.child_by_field_name("type")
                default_node = child.child_by_field_name("default_value")
                name = self._node_text(name_node, source) if name_node else "?"
                ptype = self._node_text(type_node, source) if type_node else None
                default = self._node_text(default_node, source) if default_node else None
                params.append({
                    "name": name,
                    "type": ptype,
                    "default": default,
                    "optional": default is not None,
                })
        return params

    # =========================================================================
    # Field extraction — Java
    # =========================================================================

    def _extract_java_fields(
        self, body_node: tree_sitter.Node, source: bytes
    ) -> List[Dict[str, Any]]:
        """Extract field declarations from a Java class/interface body."""
        fields: List[Dict[str, Any]] = []
        for child in body_node.children:
            if child.type == "field_declaration":
                type_node = child.child_by_field_name("type")
                if not type_node:
                    continue
                type_text = self._node_text(type_node, source)
                # A field_declaration can have multiple variable_declarators
                for sub in child.children:
                    if sub.type == "variable_declarator":
                        name_node = sub.child_by_field_name("name")
                        if name_node:
                            name = self._node_text(name_node, source)
                            if name and type_text:
                                fields.append({"name": name, "type": type_text})
            elif child.type == "constant_declaration":
                # Interface constants: Type NAME = value;
                type_node = child.child_by_field_name("type")
                if not type_node:
                    continue
                type_text = self._node_text(type_node, source)
                for sub in child.children:
                    if sub.type == "variable_declarator":
                        name_node = sub.child_by_field_name("name")
                        if name_node:
                            name = self._node_text(name_node, source)
                            if name and type_text:
                                fields.append({"name": name, "type": type_text})
        return fields

    # =========================================================================
    # Field extraction — C#
    # =========================================================================

    def _extract_csharp_fields(
        self, body_node: tree_sitter.Node, source: bytes
    ) -> List[Dict[str, Any]]:
        """Extract field and property declarations from a C# type body."""
        fields: List[Dict[str, Any]] = []
        for child in body_node.children:
            if child.type == "field_declaration":
                # C# field_declaration > variable_declaration > type + declarators
                var_decl = None
                for sub in child.children:
                    if sub.type == "variable_declaration":
                        var_decl = sub
                        break
                if not var_decl:
                    continue
                type_node = var_decl.child_by_field_name("type")
                if not type_node:
                    continue
                type_text = self._node_text(type_node, source)
                for sub in var_decl.children:
                    if sub.type == "variable_declarator":
                        # C# variable_declarator's first named child is the identifier
                        name_node = sub.children[0] if sub.children else None
                        if name_node and name_node.is_named:
                            name = self._node_text(name_node, source)
                            if name and type_text:
                                fields.append({"name": name, "type": type_text})

            elif child.type == "property_declaration":
                type_node = child.child_by_field_name("type")
                name_node = child.child_by_field_name("name")
                if type_node and name_node:
                    type_text = self._node_text(type_node, source)
                    name = self._node_text(name_node, source)
                    if name and type_text:
                        fields.append({"name": name, "type": type_text})
        return fields

    # =========================================================================
    # Field extraction — Python
    # =========================================================================

    def _extract_python_fields(
        self, body_node: tree_sitter.Node, source: bytes
    ) -> List[Dict[str, Any]]:
        """Extract type-annotated class variables from a Python class body.

        Handles:
          x: int = 5      → {"name": "x", "type": "int"}
          y: str           → {"name": "y", "type": "str"}
          z = "hello"      → skipped (no type annotation)
        """
        fields: List[Dict[str, Any]] = []
        for child in body_node.children:
            if child.type != "expression_statement":
                continue
            # Look for typed assignments or bare annotations
            inner = child.children[0] if child.children else None
            if not inner:
                continue

            if inner.type == "assignment":
                # x: Type = value — assignment with type annotation
                type_node = inner.child_by_field_name("type")
                if not type_node:
                    continue  # Untyped assignment — skip
                left_node = inner.child_by_field_name("left")
                if left_node and left_node.type == "identifier":
                    name = self._node_text(left_node, source)
                    ptype = self._node_text(type_node, source)
                    if name and ptype:
                        fields.append({"name": name, "type": ptype})

            elif inner.type == "type":
                # Bare annotation: y: str (no value)
                # tree-sitter-python parses this as type node
                # containing the identifier and type
                name_node = inner.children[0] if inner.children else None
                type_node = inner.children[1] if len(inner.children) > 1 else None
                if name_node and type_node:
                    name = self._node_text(name_node, source)
                    ptype = self._node_text(type_node, source)
                    if name and ptype:
                        fields.append({"name": name, "type": ptype})
        return fields

    # =========================================================================
    # Field extraction — TypeScript
    # =========================================================================

    def _extract_typescript_fields(
        self, body_node: tree_sitter.Node, source: bytes
    ) -> List[Dict[str, Any]]:
        """Extract typed class properties from a TypeScript class body.

        Handles:
          private service: HazelcastService;
          readonly name: string = "default";
        Skips arrow function properties (those are methods).
        """
        fields: List[Dict[str, Any]] = []
        for child in body_node.children:
            if child.type == "public_field_definition":
                # Skip arrow function properties — they're method-like
                if self._has_function_value(child):
                    continue
                name_node = child.child_by_field_name("name")
                type_ann = child.child_by_field_name("type")
                if name_node and type_ann:
                    name = self._node_text(name_node, source)
                    ptype = self._node_text(type_ann, source)
                    if name and ptype:
                        fields.append({"name": name, "type": ptype})
        return fields

    @staticmethod
    def _has_function_value(node: tree_sitter.Node) -> bool:
        """Check if a field definition has an arrow function or function value."""
        value = node.child_by_field_name("value")
        if value and value.type in ("arrow_function", "function"):
            return True
        return False

    # =========================================================================
    # Shared helpers
    # =========================================================================

    @staticmethod
    def _node_text(node: Optional[tree_sitter.Node], source: bytes) -> str:
        """Get text of a node, stripping leading colons/arrows for type annotations."""
        if not node:
            return ""
        text = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace").strip()
        # Python type annotations come after ": " or "-> "
        if text.startswith(":"):
            text = text[1:].strip()
        if text.startswith("->"):
            text = text[2:].strip()
        return text

    @staticmethod
    def _collect_modifiers(node: tree_sitter.Node, source: bytes) -> List[str]:
        """Collect modifier keywords from a declaration node."""
        modifiers = []
        for child in node.children:
            if child.type in ("modifiers", "modifier"):
                if child.type == "modifiers":
                    for sub in child.children:
                        if sub.is_named and sub.type not in ("marker_annotation", "annotation", "attribute_list"):
                            text = source[sub.start_byte:sub.end_byte].decode("utf-8", errors="replace").strip()
                            if text and text.isalpha():
                                modifiers.append(text)
                else:
                    text = source[child.start_byte:child.end_byte].decode("utf-8", errors="replace").strip()
                    if text and text.isalpha():
                        modifiers.append(text)
        return modifiers

    def _find_typed_node_at_line(
        self,
        root: tree_sitter.Node,
        target_line: int,
        target_types: frozenset,
    ) -> Optional[tree_sitter.Node]:
        """Find a node of specified types at the given line (1-indexed).

        Searches the tree for nodes whose type is in target_types and whose
        start line matches target_line.
        """
        target_row = target_line - 1  # tree-sitter uses 0-indexed rows

        def search(node: tree_sitter.Node) -> Optional[tree_sitter.Node]:
            if node.type in target_types and node.start_point.row == target_row:
                return node
            for child in node.children:
                if child.end_point.row < target_row:
                    continue
                if child.start_point.row > target_row:
                    break
                result = search(child)
                if result:
                    return result
            return None

        return search(root)

    @staticmethod
    def _find_first_typed_node(
        root: tree_sitter.Node, target_types: frozenset
    ) -> Optional[tree_sitter.Node]:
        """Find the first node of specified types via breadth-first search.

        Used for parsing source fragments where the target node is near the root.
        """
        queue = [root]
        while queue:
            node = queue.pop(0)
            if node.type in target_types:
                return node
            queue.extend(child for child in node.children if child.is_named)
        return None

    @staticmethod
    def _declaration_types(language: str) -> frozenset:
        """Return tree-sitter node types for callable declarations."""
        if language == "python":
            return frozenset({"function_definition"})
        elif language == "java":
            return frozenset({"method_declaration", "constructor_declaration"})
        elif language == "csharp":
            return frozenset({"method_declaration", "constructor_declaration"})
        elif language == "typescript":
            return frozenset({"method_definition", "function_declaration"})
        return frozenset()

    @staticmethod
    def _class_declaration_types(language: str) -> frozenset:
        """Return tree-sitter node types for class/interface/struct declarations."""
        if language == "java":
            return frozenset({"class_declaration", "interface_declaration", "enum_declaration"})
        elif language == "csharp":
            return frozenset({
                "class_declaration", "interface_declaration",
                "struct_declaration", "record_declaration",
            })
        elif language == "python":
            return frozenset({"class_definition"})
        elif language == "typescript":
            return frozenset({"class_declaration"})
        return frozenset()

    @staticmethod
    def _get_language(language: str) -> tree_sitter.Language:
        """Get tree-sitter Language object by language identifier."""
        if language == "python":
            import tree_sitter_python
            return tree_sitter.Language(tree_sitter_python.language())
        elif language == "java":
            import tree_sitter_java
            return tree_sitter.Language(tree_sitter_java.language())
        elif language == "csharp":
            import tree_sitter_c_sharp
            return tree_sitter.Language(tree_sitter_c_sharp.language())
        elif language == "typescript":
            import tree_sitter_typescript
            return tree_sitter.Language(tree_sitter_typescript.language_typescript())
        raise ValueError(f"Unsupported language for enrichment: {language}")

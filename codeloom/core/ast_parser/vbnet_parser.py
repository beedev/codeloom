"""VB.NET parser — regex-based.

Extracts classes, interfaces, structures, modules, enums, methods (Sub/Function),
properties, events, and constructors from VB.NET (.vb) files.

Does NOT use tree-sitter (no pip-installable VB.NET grammar exists).
Follows the SqlParser pattern with parse_file/parse_source interface.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .models import CodeUnit, ParseResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VB.NET Block Patterns (case-insensitive, line-oriented)
# ---------------------------------------------------------------------------

# Access modifiers and other keywords that precede declarations
_MODIFIER_WORDS = frozenset({
    "public", "private", "protected", "friend", "internal",
    "shared", "static", "overridable", "mustoverride", "notoverridable",
    "overrides", "overloads", "shadows", "readonly", "writeonly",
    "default", "withevents", "partial", "mustinherit", "notinheritable",
    "async", "iterator", "narrowing", "widening",
})

# Block-opening keywords and their matching End keyword
_BLOCK_TYPES = {
    "namespace": "namespace",
    "class": "class",
    "interface": "interface",
    "structure": "struct",
    "module": "module",
    "enum": "enum",
    "sub": "method",
    "function": "method",
    "property": "property",
}

# Regex: optional attributes, optional modifiers, then keyword + name
_DECLARATION_RE = re.compile(
    r"^(?P<attrs>(?:\s*<[^>]+>\s*)*)"              # <Attribute> lines (greedy)
    r"\s*(?P<mods>(?:(?:" + "|".join(_MODIFIER_WORDS) + r")\s+)*)"
    r"(?P<keyword>Namespace|Class|Interface|Structure|Module|Enum|Sub|Function|Property|Event)"
    r"\s+(?P<name>\w+)"
    r"(?:\s*\((?P<of>Of\s+[^)]+)\))?"              # (Of T) generics
    r"(?P<rest>.*)",
    re.IGNORECASE,
)

# End block: End Sub, End Class, etc.
_END_BLOCK_RE = re.compile(
    r"^\s*End\s+(?P<keyword>Namespace|Class|Interface|Structure|Module|Enum|Sub|Function|Property)\b",
    re.IGNORECASE,
)

# Inherits line inside a class/structure
_INHERITS_RE = re.compile(r"^\s*Inherits\s+(.+)", re.IGNORECASE)

# Implements line inside a class/structure
_IMPLEMENTS_RE = re.compile(r"^\s*Implements\s+(.+)", re.IGNORECASE)

# Import statement
_IMPORTS_RE = re.compile(r"^\s*Imports\s+(.+)", re.IGNORECASE)

# Parameter pattern: [ByVal|ByRef] [Optional] name As Type [= default]
_PARAM_RE = re.compile(
    r"(?:(?P<passing>ByVal|ByRef|ParamArray)\s+)?"
    r"(?:(?P<optional>Optional)\s+)?"
    r"(?P<name>\w+)"
    r"\s+As\s+(?P<type>[^,=)]+?)"
    r"(?:\s*=\s*(?P<default>[^,)]+))?"
    r"(?=\s*[,)]|$)",
    re.IGNORECASE,
)

# Return type: As <type> at end of Function/Property signature
_RETURN_TYPE_RE = re.compile(r"\)\s*As\s+(.+?)$", re.IGNORECASE)

# VB.NET doc comment: ''' or '''<summary>
_DOC_COMMENT_RE = re.compile(r"^\s*'''(.*)$")

# Attribute line: <AttributeName(...)>
_ATTRIBUTE_RE = re.compile(r"<(\w+)(?:\([^)]*\))?\s*>")

# Event declaration (single-line): [mods] Event Name As EventHandler
_EVENT_RE = re.compile(
    r"^\s*(?:(?:" + "|".join(_MODIFIER_WORDS) + r")\s+)*"
    r"Event\s+(\w+)",
    re.IGNORECASE,
)


class VbNetParser:
    """Regex-based VB.NET parser for classes, interfaces, modules, methods, etc.

    Produces CodeUnit objects compatible with the ingestion pipeline.
    Does not subclass BaseLanguageParser (no tree-sitter dependency).
    """

    def get_language(self) -> str:
        return "vbnet"

    def parse_file(self, file_path: str, project_root: str = "") -> ParseResult:
        """Parse a VB.NET file into structured CodeUnit objects."""
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
                language="vbnet",
                units=[],
                imports=[],
                line_count=0,
            )

        return self.parse_source(source_text, rel_path)

    def parse_source(self, source_text: str, file_path: str) -> ParseResult:
        """Parse VB.NET source text into CodeUnit objects."""
        lines = source_text.split("\n")
        line_count = len(lines)

        # Extract file-level imports
        file_imports = self._extract_imports(lines)

        # Parse declarations via line-by-line block tracking
        units = self._parse_blocks(lines, file_path, file_imports)

        return ParseResult(
            file_path=file_path,
            language="vbnet",
            units=units,
            imports=file_imports,
            line_count=line_count,
        )

    # ── Block parser ──────────────────────────────────────────────────

    def _parse_blocks(
        self, lines: List[str], file_path: str, file_imports: List[str]
    ) -> List[CodeUnit]:
        """Line-by-line block tracking with depth counter for End <keyword> matching."""
        units: List[CodeUnit] = []

        # Stack tracks open blocks: (keyword, name, start_line, modifiers, attrs, rest, doc_lines)
        block_stack: List[Dict[str, Any]] = []
        # Namespace stack for qualified name building
        namespace_stack: List[str] = []

        # Accumulate doc comments preceding a declaration
        pending_doc: List[str] = []
        # Track inherits/implements inside current type block
        current_type_meta: Dict[str, Any] = {}
        # Flag: file_imports already stamped on first unit
        imports_stamped = False

        for line_num, raw_line in enumerate(lines, start=1):
            line = raw_line.rstrip()
            stripped = line.strip()

            # Skip blank lines and single-line comments (but preserve doc comments)
            doc_match = _DOC_COMMENT_RE.match(stripped)
            if doc_match:
                pending_doc.append(doc_match.group(1).strip())
                continue

            if not stripped or stripped.startswith("'"):
                if not stripped:
                    pending_doc = []  # Reset doc on blank line
                continue

            # Check for End Block
            end_match = _END_BLOCK_RE.match(stripped)
            if end_match:
                end_kw = end_match.group("keyword").lower()

                # Pop matching block from stack
                if block_stack and block_stack[-1]["keyword"] == end_kw:
                    block = block_stack.pop()

                    if end_kw == "namespace":
                        if namespace_stack:
                            namespace_stack.pop()
                        continue

                    # Build the CodeUnit
                    source_block = "\n".join(lines[block["start_line"] - 1 : line_num])
                    unit = self._make_unit(
                        block=block,
                        end_line=line_num,
                        source=source_block,
                        file_path=file_path,
                        namespace_stack=namespace_stack,
                        block_stack=block_stack,
                        type_meta=block.get("type_meta", {}),
                        file_imports=file_imports if not imports_stamped else None,
                    )
                    if unit:
                        units.append(unit)
                        imports_stamped = True

                    # Reset type meta when closing a type block
                    if end_kw in ("class", "interface", "structure", "module"):
                        current_type_meta = {}

                continue

            # Check for Inherits / Implements lines (inside type blocks)
            if block_stack and block_stack[-1]["keyword"] in ("class", "structure"):
                inherits_match = _INHERITS_RE.match(stripped)
                if inherits_match:
                    base = inherits_match.group(1).strip()
                    current_type_meta["extends"] = base.split("(")[0].strip()
                    block_stack[-1].setdefault("type_meta", {})["extends"] = current_type_meta["extends"]
                    continue

                implements_match = _IMPLEMENTS_RE.match(stripped)
                if implements_match:
                    ifaces = [i.strip() for i in implements_match.group(1).split(",")]
                    existing = block_stack[-1].get("type_meta", {}).get("implements", [])
                    block_stack[-1].setdefault("type_meta", {})["implements"] = existing + ifaces
                    continue

            # Check for Event declaration (single-line, no End Event)
            event_match = _EVENT_RE.match(stripped)
            if event_match and not _DECLARATION_RE.match(stripped):
                # Events are single-line declarations
                event_name = event_match.group(1)
                parent_name = block_stack[-1]["name"] if block_stack else None
                qualified = self._build_qualified_name(
                    event_name, namespace_stack, block_stack
                )
                meta: Dict[str, Any] = {"modifiers": self._extract_modifiers(stripped)}
                if not imports_stamped and file_imports:
                    meta["file_imports"] = file_imports
                    imports_stamped = True

                units.append(CodeUnit(
                    unit_type="event",
                    name=event_name,
                    qualified_name=qualified,
                    language="vbnet",
                    start_line=line_num,
                    end_line=line_num,
                    source=stripped,
                    file_path=file_path,
                    signature=stripped,
                    docstring="\n".join(pending_doc) if pending_doc else None,
                    parent_name=parent_name,
                    metadata=meta,
                ))
                pending_doc = []
                continue

            # Check for block-opening declaration
            decl_match = _DECLARATION_RE.match(stripped)
            if decl_match:
                keyword = decl_match.group("keyword").lower()
                name = decl_match.group("name")
                mods_str = (decl_match.group("mods") or "").strip()
                attrs_str = (decl_match.group("attrs") or "").strip()
                rest = (decl_match.group("rest") or "").strip()

                if keyword == "event":
                    # Single-line event (handled above if regex missed it)
                    parent_name = block_stack[-1]["name"] if block_stack else None
                    qualified = self._build_qualified_name(
                        name, namespace_stack, block_stack
                    )
                    meta = {"modifiers": self._extract_modifiers(stripped)}
                    if not imports_stamped and file_imports:
                        meta["file_imports"] = file_imports
                        imports_stamped = True
                    units.append(CodeUnit(
                        unit_type="event",
                        name=name,
                        qualified_name=qualified,
                        language="vbnet",
                        start_line=line_num,
                        end_line=line_num,
                        source=stripped,
                        file_path=file_path,
                        signature=stripped,
                        docstring="\n".join(pending_doc) if pending_doc else None,
                        parent_name=parent_name,
                        metadata=meta,
                    ))
                    pending_doc = []
                    continue

                # Detect single-line declarations that have no End block:
                # 1. Auto-property: "Public Property Age As Integer" (As on same line)
                # 2. MustOverride/Declare methods: no body
                # 3. Interface members: Sub/Function inside Interface are declarations
                is_single_line = False
                mods_lower = mods_str.lower()

                if keyword == "property" and re.search(r"\bAs\b", rest, re.IGNORECASE):
                    # Auto-implemented property (no Get/Set block)
                    is_single_line = True
                elif keyword in ("sub", "function") and "mustoverride" in mods_lower:
                    is_single_line = True
                elif keyword in ("sub", "function") and "declare" in mods_lower:
                    is_single_line = True
                elif keyword in ("sub", "function") and block_stack:
                    # Inside an interface, methods are declarations only
                    if block_stack[-1]["keyword"] == "interface":
                        is_single_line = True

                if is_single_line:
                    # Emit as a single-line unit without pushing to stack
                    parent_name = block_stack[-1]["name"] if block_stack else None
                    qualified = self._build_qualified_name(
                        name, namespace_stack, block_stack
                    )
                    if keyword == "sub" and name.lower() == "new":
                        unit_type = "constructor"
                    elif keyword in _BLOCK_TYPES:
                        unit_type = _BLOCK_TYPES[keyword]
                    else:
                        unit_type = "block"

                    meta: Dict[str, Any] = {"modifiers": self._extract_modifiers(mods_str)}
                    if "mustoverride" in mods_lower:
                        meta["is_abstract"] = True
                    if "overrides" in mods_lower:
                        meta["is_override"] = True
                    if parent_name:
                        meta["parent_name"] = parent_name
                    # Parse params for methods
                    if keyword in ("sub", "function", "property"):
                        param_text = self._extract_param_text(rest)
                        if param_text:
                            meta["parsed_params"] = self._parse_params(param_text)
                    # Return type for functions/properties
                    if keyword in ("function", "property"):
                        rt_match = _RETURN_TYPE_RE.search(rest)
                        if rt_match:
                            meta["return_type"] = rt_match.group(1).strip()
                        elif keyword == "property" and re.search(r"\bAs\b", rest, re.IGNORECASE):
                            # Auto-property: "Property Age As Integer"
                            as_match = re.search(r"\bAs\s+(.+?)$", rest, re.IGNORECASE)
                            if as_match:
                                meta["return_type"] = as_match.group(1).strip()
                    # Annotations
                    attrs = _ATTRIBUTE_RE.findall(attrs_str)
                    if attrs:
                        meta["annotations"] = attrs
                    if not imports_stamped and file_imports:
                        meta["file_imports"] = file_imports
                        imports_stamped = True

                    units.append(CodeUnit(
                        unit_type=unit_type,
                        name=name,
                        qualified_name=qualified,
                        language="vbnet",
                        start_line=line_num,
                        end_line=line_num,
                        source=stripped,
                        file_path=file_path,
                        signature=stripped,
                        docstring="\n".join(pending_doc) if pending_doc else None,
                        parent_name=parent_name,
                        metadata=meta,
                    ))
                    pending_doc = []
                    continue

                block_stack.append({
                    "keyword": keyword,
                    "name": name,
                    "start_line": line_num,
                    "modifiers": mods_str,
                    "attrs": attrs_str,
                    "rest": rest,
                    "doc_lines": list(pending_doc),
                    "type_meta": {},
                })
                pending_doc = []

                if keyword == "namespace":
                    namespace_stack.append(name)

                continue

            # Reset doc comment accumulator on non-doc, non-declaration lines
            pending_doc = []

        # Handle unclosed blocks (best-effort)
        while block_stack:
            block = block_stack.pop()
            if block["keyword"] == "namespace":
                continue
            source_block = "\n".join(lines[block["start_line"] - 1:])
            unit = self._make_unit(
                block=block,
                end_line=len(lines),
                source=source_block,
                file_path=file_path,
                namespace_stack=namespace_stack,
                block_stack=block_stack,
                type_meta=block.get("type_meta", {}),
                file_imports=file_imports if not imports_stamped else None,
            )
            if unit:
                units.append(unit)
                imports_stamped = True

        return units

    # ── Unit construction ──────────────────────────────────────────────

    def _make_unit(
        self,
        block: Dict[str, Any],
        end_line: int,
        source: str,
        file_path: str,
        namespace_stack: List[str],
        block_stack: List[Dict[str, Any]],
        type_meta: Dict[str, Any],
        file_imports: Optional[List[str]],
    ) -> Optional[CodeUnit]:
        """Build a CodeUnit from a completed block."""
        keyword = block["keyword"]
        name = block["name"]
        start_line = block["start_line"]
        mods_str = block["modifiers"]
        attrs_str = block["attrs"]
        rest = block["rest"]
        doc_lines = block["doc_lines"]

        # Determine unit_type
        if keyword == "sub" and name.lower() == "new":
            unit_type = "constructor"
        elif keyword in _BLOCK_TYPES:
            unit_type = _BLOCK_TYPES[keyword]
        else:
            unit_type = "block"

        # Build qualified name
        qualified = self._build_qualified_name(name, namespace_stack, block_stack)

        # Parent name (for methods/properties/constructors inside types)
        parent_name = None
        if unit_type in ("method", "constructor", "property", "event"):
            if block_stack:
                parent_name = block_stack[-1]["name"]

        # Modifiers list
        modifiers = self._extract_modifiers(mods_str)

        # Annotations from <Attribute> syntax
        annotations = _ATTRIBUTE_RE.findall(attrs_str)

        # Parameters (for Sub, Function, Property)
        parsed_params: List[Dict[str, Any]] = []
        if keyword in ("sub", "function", "property"):
            param_text = self._extract_param_text(rest)
            if param_text:
                parsed_params = self._parse_params(param_text)

        # Return type (for Function, Property)
        return_type = None
        if keyword == "function":
            rt_match = _RETURN_TYPE_RE.search(rest)
            if rt_match:
                return_type = rt_match.group(1).strip()

        # Signature
        signature = source.split("\n")[0].strip() if source else ""

        # Docstring from accumulated ''' comments
        docstring = "\n".join(doc_lines) if doc_lines else None

        # Build metadata
        metadata: Dict[str, Any] = {
            "modifiers": modifiers,
        }

        if annotations:
            metadata["annotations"] = annotations

        if parsed_params:
            metadata["parsed_params"] = parsed_params

        if return_type:
            metadata["return_type"] = return_type

        # Inherits / Implements (from type_meta collected during block parsing)
        if type_meta.get("extends"):
            metadata["extends"] = type_meta["extends"]

        if type_meta.get("implements"):
            metadata["implements"] = type_meta["implements"]

        # Override detection
        mods_lower = {m.lower() for m in modifiers}
        if "overrides" in mods_lower:
            metadata["is_override"] = True
        if "mustoverride" in mods_lower:
            metadata["is_abstract"] = True

        # Parent name in metadata (for ASG builder)
        if parent_name:
            metadata["parent_name"] = parent_name

        # Stamp file imports on first unit
        if file_imports:
            metadata["file_imports"] = file_imports

        return CodeUnit(
            unit_type=unit_type,
            name=name,
            qualified_name=qualified,
            language="vbnet",
            start_line=start_line,
            end_line=end_line,
            source=source,
            file_path=file_path,
            signature=signature,
            docstring=docstring,
            parent_name=parent_name,
            metadata=metadata,
        )

    # ── Helper methods ──────────────────────────────────────────────────

    def _build_qualified_name(
        self,
        name: str,
        namespace_stack: List[str],
        block_stack: List[Dict[str, Any]],
    ) -> str:
        """Build fully qualified name from namespace + type hierarchy."""
        parts: List[str] = list(namespace_stack)
        for block in block_stack:
            if block["keyword"] != "namespace":
                parts.append(block["name"])
        parts.append(name)
        return ".".join(parts)

    @staticmethod
    def _extract_modifiers(text: str) -> List[str]:
        """Extract modifier keywords from a declaration line."""
        words = text.split()
        return [w for w in words if w.lower() in _MODIFIER_WORDS]

    @staticmethod
    def _extract_param_text(rest: str) -> str:
        """Extract the parameter text from between parentheses."""
        # rest is everything after the name in the declaration line
        paren_start = rest.find("(")
        if paren_start == -1:
            return ""

        depth = 0
        result: List[str] = []
        for ch in rest[paren_start:]:
            if ch == "(":
                depth += 1
                if depth == 1:
                    continue  # Skip opening paren
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    break
            if depth >= 1:
                result.append(ch)

        return "".join(result)

    @staticmethod
    def _parse_params(param_text: str) -> List[Dict[str, Any]]:
        """Parse VB.NET parameter list into structured dicts.

        Handles: ByVal, ByRef, Optional ... = default, ParamArray
        """
        params: List[Dict[str, Any]] = []
        if not param_text.strip():
            return params

        # Split on commas that aren't inside parentheses (Of T, ...)
        depth = 0
        segments: List[str] = []
        current: List[str] = []
        for ch in param_text:
            if ch in ("(", "<"):
                depth += 1
            elif ch in (")", ">"):
                depth -= 1
            elif ch == "," and depth == 0:
                segments.append("".join(current).strip())
                current = []
                continue
            current.append(ch)
        if current:
            segments.append("".join(current).strip())

        for seg in segments:
            if not seg:
                continue

            m = _PARAM_RE.search(seg)
            if m:
                passing = (m.group("passing") or "ByVal").strip()
                is_optional = m.group("optional") is not None
                pname = m.group("name")
                ptype = m.group("type").strip()
                default = m.group("default")
                if default:
                    default = default.strip()

                params.append({
                    "name": pname,
                    "type": ptype,
                    "default": default,
                    "optional": is_optional or passing.lower() == "paramarray",
                    "passing": passing,
                })
            else:
                # Best-effort: just grab the name
                words = seg.split()
                pname = words[-1] if words else seg
                params.append({
                    "name": pname,
                    "type": None,
                    "default": None,
                    "optional": False,
                })

        return params

    @staticmethod
    def _extract_imports(lines: List[str]) -> List[str]:
        """Extract Imports statements from source lines."""
        imports: List[str] = []
        for line in lines:
            m = _IMPORTS_RE.match(line)
            if m:
                imports.append(f"Imports {m.group(1).strip()}")
        return imports

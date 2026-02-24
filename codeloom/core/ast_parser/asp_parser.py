"""ASP.NET Web Forms parser — regex-based.

Extracts a single CodeUnit per .aspx/.ascx/.master file with rich metadata about:
- Page/Master/Control directives (CodeBehind, Inherits, MasterPageFile)
- Register directives (user controls and tag prefixes)
- Import directives (namespace imports)
- Server controls (<asp:TextBox>, <asp:GridView>, etc.)
- Validation controls (<asp:RequiredFieldValidator>, etc.)
- Event handlers (OnClick, OnLoad, OnSelectedIndexChanged, etc.)
- Data binding expressions (<%# Eval("...") %>, <%# Bind("...") %>)
- Inline code blocks (<% %>) and expressions (<%= %>)
- ViewState and Session state access patterns
- PostBack detection (IsPostBack / Page.IsPostBack)

Does NOT use tree-sitter. Follows the regex/fallback parser pattern
with parse_file/parse_source interface.
"""

import logging
import re
from typing import Any, Dict, List, Set

from .models import CodeUnit, ParseResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns — directives
# ---------------------------------------------------------------------------

# Page directive: <%@ Page Language="VB" CodeBehind="Login.aspx.vb" ... %>
_PAGE_DIRECTIVE_RE = re.compile(
    r"<%@\s*Page\s+.*?%>",
    re.IGNORECASE | re.DOTALL,
)

# Master directive: <%@ Master Language="C#" CodeBehind="Site.master.cs" ... %>
_MASTER_DIRECTIVE_RE = re.compile(
    r"<%@\s*Master\s+.*?%>",
    re.IGNORECASE | re.DOTALL,
)

# Control directive: <%@ Control Language="C#" CodeBehind="Header.ascx.cs" ... %>
_CONTROL_DIRECTIVE_RE = re.compile(
    r"<%@\s*Control\s+.*?%>",
    re.IGNORECASE | re.DOTALL,
)

# Attribute extractors (reusable across directive types)
_CODEBEHIND_RE = re.compile(r'CodeBehind\s*=\s*"([^"]*)"', re.IGNORECASE)
_INHERITS_RE = re.compile(r'Inherits\s*=\s*"([^"]*)"', re.IGNORECASE)
_MASTERPAGE_RE = re.compile(r'MasterPageFile\s*=\s*"([^"]*)"', re.IGNORECASE)
_LANGUAGE_RE = re.compile(r'Language\s*=\s*"([^"]*)"', re.IGNORECASE)

# ---------------------------------------------------------------------------
# Register and Import directives
# ---------------------------------------------------------------------------

# Register directive: <%@ Register TagPrefix="uc1" TagName="Header" Src="..." %>
_REGISTER_DIRECTIVE_RE = re.compile(
    r"<%@\s*Register\s+.*?%>",
    re.IGNORECASE | re.DOTALL,
)
_TAGPREFIX_RE = re.compile(r'TagPrefix\s*=\s*"([^"]*)"', re.IGNORECASE)
_TAGNAME_RE = re.compile(r'TagName\s*=\s*"([^"]*)"', re.IGNORECASE)
_SRC_RE = re.compile(r'Src\s*=\s*"([^"]*)"', re.IGNORECASE)

# Import directive: <%@ Import Namespace="System.Data" %>
_IMPORT_DIRECTIVE_RE = re.compile(
    r'<%@\s*Import\s+Namespace\s*=\s*"([^"]*)"[^%]*%>',
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Server controls and validation
# ---------------------------------------------------------------------------

# Server controls: <asp:TextBox ID="txtName" runat="server" ... />
# Captures the control type name after "asp:"
_SERVER_CONTROL_RE = re.compile(
    r"<asp:(\w+)\s([^>]*(?:>|/>))",
    re.IGNORECASE | re.DOTALL,
)

# Control attributes
_CONTROL_ID_RE = re.compile(r'ID\s*=\s*"([^"]*)"', re.IGNORECASE)
_CONTROL_RUNAT_RE = re.compile(r'runat\s*=\s*"server"', re.IGNORECASE)

# Generic attribute extractor for building the properties dict
_ATTRIBUTE_RE = re.compile(r'(\w+)\s*=\s*"([^"]*)"')

# Validation controls: <asp:RequiredFieldValidator ...>, <asp:RangeValidator ...>
_VALIDATION_CONTROL_RE = re.compile(
    r"<asp:(\w*Validator)\s([^>]*(?:>|/>))",
    re.IGNORECASE | re.DOTALL,
)
_CONTROL_TO_VALIDATE_RE = re.compile(
    r'ControlToValidate\s*=\s*"([^"]*)"', re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------

# On[Event]="handlerName" — captures event name and handler method name
_EVENT_HANDLER_RE = re.compile(
    r'On(\w+)\s*=\s*"([^"]*)"',
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Inline code blocks and expressions
# ---------------------------------------------------------------------------

# Code block: <% ... %> (but NOT directives <%@ or expressions <%= or bindings <%#)
_CODE_BLOCK_RE = re.compile(
    r"<%(?![=@#])(.*?)%>",
    re.DOTALL,
)

# Expression: <%= ... %>
_EXPRESSION_RE = re.compile(
    r"<%=(.*?)%>",
    re.DOTALL,
)

# Data binding: <%# ... %> (Eval, Bind, etc.)
_DATABIND_RE = re.compile(
    r"<%#(.*?)%>",
    re.DOTALL,
)

# ---------------------------------------------------------------------------
# State access patterns
# ---------------------------------------------------------------------------

# ViewState["key"] or ViewState("key") — VB uses parens, C# uses brackets
_VIEWSTATE_RE = re.compile(
    r'ViewState\s*[\[("]\s*"([^"]*)"',
    re.IGNORECASE,
)

# Session["key"] or Session("key")
_SESSION_RE = re.compile(
    r'Session\s*[\[("]\s*"([^"]*)"',
    re.IGNORECASE,
)

# PostBack detection: IsPostBack or Page.IsPostBack
_POSTBACK_RE = re.compile(
    r'(?:Page\.)?IsPostBack',
    re.IGNORECASE,
)


class AspParser:
    """Regex-based ASP.NET Web Forms parser extracting page-level metadata.

    Each .aspx/.ascx/.master file becomes one CodeUnit with metadata
    describing server controls, event handlers, validation, data bindings,
    state management, and code-behind references.
    """

    def get_language(self) -> str:
        return "aspx"

    def parse_file(self, file_path: str, project_root: str = "") -> ParseResult:
        """Parse an ASP.NET Web Forms file into a single CodeUnit."""
        if project_root and file_path.startswith(project_root):
            rel_path = file_path[len(project_root):].lstrip("/")
        else:
            rel_path = file_path

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                source_text = f.read()
        except OSError as e:
            logger.error("Cannot read %s: %s", file_path, e)
            return ParseResult(
                file_path=rel_path,
                language="aspx",
                units=[],
                imports=[],
                line_count=0,
            )

        return self.parse_source(source_text, rel_path)

    def parse_source(self, source_text: str, file_path: str) -> ParseResult:
        """Parse ASP.NET Web Forms source into a single CodeUnit with rich metadata."""
        line_count = source_text.count("\n") + (
            1 if source_text and not source_text.endswith("\n") else 0
        )

        # --- Determine page type from directive ---
        page_type = "page"
        directive_text = ""

        master_m = _MASTER_DIRECTIVE_RE.search(source_text)
        control_m = _CONTROL_DIRECTIVE_RE.search(source_text)
        page_m = _PAGE_DIRECTIVE_RE.search(source_text)

        if master_m:
            page_type = "master"
            directive_text = master_m.group(0)
        elif control_m:
            page_type = "control"
            directive_text = control_m.group(0)
        elif page_m:
            page_type = "page"
            directive_text = page_m.group(0)

        # --- Extract directive attributes ---
        code_behind = ""
        inherits = ""
        master_page = ""
        language = ""

        if directive_text:
            cb_m = _CODEBEHIND_RE.search(directive_text)
            if cb_m:
                code_behind = cb_m.group(1)
            inh_m = _INHERITS_RE.search(directive_text)
            if inh_m:
                inherits = inh_m.group(1)
            mp_m = _MASTERPAGE_RE.search(directive_text)
            if mp_m:
                master_page = mp_m.group(1)
            lang_m = _LANGUAGE_RE.search(directive_text)
            if lang_m:
                language = lang_m.group(1)

        # --- Import directives ---
        imports: List[str] = []
        for m in _IMPORT_DIRECTIVE_RE.finditer(source_text):
            ns = m.group(1).strip()
            if ns and ns not in imports:
                imports.append(ns)

        # --- Register directives ---
        registered_controls: List[Dict[str, str]] = []
        for m in _REGISTER_DIRECTIVE_RE.finditer(source_text):
            block = m.group(0)
            prefix_m = _TAGPREFIX_RE.search(block)
            tag_m = _TAGNAME_RE.search(block)
            src_m = _SRC_RE.search(block)
            registered_controls.append({
                "prefix": prefix_m.group(1) if prefix_m else "",
                "tag_name": tag_m.group(1) if tag_m else "",
                "src": src_m.group(1) if src_m else "",
            })

        # --- Server controls ---
        server_controls: List[Dict[str, Any]] = []
        for m in _SERVER_CONTROL_RE.finditer(source_text):
            ctrl_type = m.group(1)
            attr_block = m.group(2)

            # Skip validator controls — handled separately below
            if ctrl_type.lower().endswith("validator"):
                continue

            # Extract all attributes as properties
            props: Dict[str, str] = {}
            ctrl_id = ""
            for attr_m in _ATTRIBUTE_RE.finditer(attr_block):
                attr_name = attr_m.group(1)
                attr_val = attr_m.group(2)
                if attr_name.upper() == "ID":
                    ctrl_id = attr_val
                elif attr_name.lower() == "runat":
                    continue  # Skip runat="server", it's implicit
                else:
                    props[attr_name] = attr_val

            server_controls.append({
                "type": ctrl_type,
                "id": ctrl_id,
                "properties": props,
            })

        # --- Validation controls ---
        validation_controls: List[Dict[str, str]] = []
        for m in _VALIDATION_CONTROL_RE.finditer(source_text):
            val_type = m.group(1)
            attr_block = m.group(2)

            id_m = _CONTROL_ID_RE.search(attr_block)
            ctv_m = _CONTROL_TO_VALIDATE_RE.search(attr_block)

            validation_controls.append({
                "type": val_type,
                "id": id_m.group(1) if id_m else "",
                "control_to_validate": ctv_m.group(1) if ctv_m else "",
            })

        # --- Event handlers ---
        event_handlers: Set[str] = set()
        for m in _EVENT_HANDLER_RE.finditer(source_text):
            handler_name = m.group(2).strip()
            if handler_name:
                event_handlers.add(handler_name)

        # --- Data binding expressions ---
        data_bindings: Set[str] = set()
        for m in _DATABIND_RE.finditer(source_text):
            expr = m.group(1).strip()
            if expr:
                data_bindings.add(expr)

        # --- Code blocks (count only) ---
        code_block_count = len(_CODE_BLOCK_RE.findall(source_text))

        # --- Expressions ---
        expressions: Set[str] = set()
        for m in _EXPRESSION_RE.finditer(source_text):
            expr = m.group(1).strip()
            if expr:
                expressions.add(expr)

        # --- ViewState keys (search in both code blocks and attribute values) ---
        viewstate_keys: Set[str] = set()
        for m in _VIEWSTATE_RE.finditer(source_text):
            viewstate_keys.add(m.group(1))

        # --- Session keys ---
        session_keys: Set[str] = set()
        for m in _SESSION_RE.finditer(source_text):
            session_keys.add(m.group(1))

        # --- PostBack detection ---
        has_postback = bool(_POSTBACK_RE.search(source_text))
        has_viewstate = len(viewstate_keys) > 0

        # --- Assemble metadata ---
        metadata: Dict[str, Any] = {
            "page_type": page_type,
            "code_behind": code_behind,
            "inherits": inherits,
            "master_page": master_page,
            "language": language,
            "imports": imports,
            "registered_controls": registered_controls,
            "server_controls": server_controls,
            "validation_controls": validation_controls,
            "event_handlers": sorted(event_handlers),
            "data_bindings": sorted(data_bindings),
            "code_blocks": code_block_count,
            "expressions": sorted(expressions),
            "viewstate_keys": sorted(viewstate_keys),
            "session_keys": sorted(session_keys),
            "has_postback": has_postback,
            "has_viewstate": has_viewstate,
        }

        # --- Build unit type and signature ---
        unit_type_map = {
            "page": "aspx_page",
            "master": "aspx_master",
            "control": "aspx_control",
        }
        unit_type = unit_type_map.get(page_type, "aspx_page")

        ctrl_count = len(server_controls) + len(validation_controls)
        handler_count = len(event_handlers)
        signature = (
            f"aspx_{page_type} {file_path} "
            f"controls[{ctrl_count}] handlers[{handler_count}]"
        )

        unit = CodeUnit(
            unit_type=unit_type,
            name=file_path,
            qualified_name=f"{file_path}:page",
            language="aspx",
            start_line=1,
            end_line=line_count,
            source=source_text,
            file_path=file_path,
            signature=signature,
            metadata=metadata,
        )

        return ParseResult(
            file_path=file_path,
            language="aspx",
            units=[unit],
            imports=imports,
            line_count=line_count,
        )

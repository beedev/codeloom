"""VB.NET to .NET Core (C#) migration lane.

Deterministic transforms, prompt augmentation, quality gates, and asset
strategy overrides for migrating VB.NET applications to C# on .NET Core.

Covers three application archetypes:
  * **WebForms** -- code-behind → Controller + Service + Repository;
    .aspx pages → React components + API services + TypeScript types
  * **WinForms** -- Form classes → Blazor component stubs
  * **Library**  -- modules, classes, interfaces, structs, enums → C#

Key philosophy: **decompose by concern**, not 1-to-1 syntactic translation.
A WebForms code-behind with mixed business logic and data access splits into
a thin Controller, a Service with business rules, and a Repository for data
access.  An .aspx page with server controls becomes a React component, an
API service module, and a TypeScript types file.

VB.NET parser metadata consumed per unit (from ``vbnet_parser.py``):
    modifiers       -- [str, ...]
    annotations     -- [str, ...]   (from <Attribute> syntax)
    parsed_params   -- [{name, type, default, optional, passing}, ...]
    return_type     -- str          (Functions / Properties)
    extends         -- str          (Inherits line)
    implements      -- [str, ...]   (Implements line)
    is_override     -- bool
    is_abstract     -- bool
    parent_name     -- str          (methods inside types)
    file_imports    -- [str, ...]   (Imports statements, first unit only)

ASP parser metadata consumed per unit (from ``asp_parser.py``):
    page_type           -- "page" | "master" | "control"
    code_behind         -- str
    inherits            -- str
    server_controls     -- [{type, id, properties}, ...]
    validation_controls -- [{type, control_to_validate, id}, ...]
    event_handlers      -- [str, ...]
    data_bindings       -- [str, ...]
    viewstate_keys      -- [str, ...]
    session_keys        -- [str, ...]
    has_postback        -- bool
    has_viewstate       -- bool
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set

from .base import (
    GateCategory,
    GateDefinition,
    GateResult,
    MigrationLane,
    TransformResult,
    TransformRule,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Application archetype constants
# ═══════════════════════════════════════════════════════════════════════

_ARCH_WEBFORMS = "webforms"
_ARCH_WINFORMS = "winforms"
_ARCH_LIBRARY = "library"
_ARCH_UNKNOWN = "unknown"

# ── WebForms indicators ─────────────────────────────────────────────

_WEBFORMS_BASES = frozenset({
    "Page", "System.Web.UI.Page",
    "MasterPage", "System.Web.UI.MasterPage",
    "UserControl", "System.Web.UI.UserControl",
    "WebService", "System.Web.Services.WebService",
})

_WEBFORMS_IMPORTS = frozenset({
    "System.Web", "System.Web.UI", "System.Web.UI.WebControls",
    "System.Web.UI.HtmlControls", "System.Web.Services",
    "System.Web.SessionState", "System.Web.Security",
})

# ── WinForms indicators ─────────────────────────────────────────────

_WINFORMS_BASES = frozenset({
    "Form", "System.Windows.Forms.Form",
    "System.Windows.Forms.UserControl",
})

_WINFORMS_IMPORTS = frozenset({
    "System.Windows.Forms", "System.Drawing",
    "System.ComponentModel",
})

# ── VB.NET type → C# type mapping ───────────────────────────────────

_VB_TYPE_MAP: Dict[str, str] = {
    "Integer": "int",
    "Long": "long",
    "Short": "short",
    "Byte": "byte",
    "SByte": "sbyte",
    "UInteger": "uint",
    "ULong": "ulong",
    "UShort": "ushort",
    "Single": "float",
    "Double": "double",
    "Decimal": "decimal",
    "Boolean": "bool",
    "String": "string",
    "Char": "char",
    "Object": "object",
    "Date": "DateTime",
    "Nothing": "null",
}

# ── VB.NET modifier → C# modifier mapping ───────────────────────────

_VB_MODIFIER_MAP: Dict[str, str] = {
    "Shared": "static",
    "MustOverride": "abstract",
    "NotOverridable": "sealed",
    "Overridable": "virtual",
    "MustInherit": "abstract",
    "NotInheritable": "sealed",
    "Overrides": "override",
    "Overloads": "new",
    "Shadows": "new",
    "ReadOnly": "readonly",
    "WriteOnly": "",
    "WithEvents": "",
    "Friend": "internal",
    "Narrowing": "explicit",
    "Widening": "implicit",
}

# ── VB.NET access modifier priority ─────────────────────────────────

_VB_ACCESS_MAP: Dict[str, str] = {
    "Public": "public",
    "Private": "private",
    "Protected": "protected",
    "Friend": "internal",
    "Protected Friend": "protected internal",
}

# ── ASP.NET server control → React/HTML mapping ─────────────────────

_ASP_CONTROL_MAP: Dict[str, Dict[str, Any]] = {
    "TextBox": {"element": "input", "type": "text", "value_prop": "value", "change_event": "onChange"},
    "Button": {"element": "button", "type": "submit"},
    "LinkButton": {"element": "button", "type": "button"},
    "ImageButton": {"element": "button", "type": "submit"},
    "Label": {"element": "span"},
    "Literal": {"element": "span"},
    "HyperLink": {"element": "Link", "import": "react-router-dom"},
    "DropDownList": {"element": "select", "value_prop": "value", "change_event": "onChange"},
    "ListBox": {"element": "select", "multiple": True},
    "CheckBox": {"element": "input", "type": "checkbox", "value_prop": "checked", "change_event": "onChange"},
    "CheckBoxList": {"element": "div", "note": "Use .map() with checkboxes", "complex": True},
    "RadioButton": {"element": "input", "type": "radio"},
    "RadioButtonList": {"element": "div", "note": "Use .map() with radio buttons", "complex": True},
    "HiddenField": {"element": "input", "type": "hidden"},
    "FileUpload": {"element": "input", "type": "file"},
    "Image": {"element": "img"},
    "Panel": {"element": "div"},
    "PlaceHolder": {"element": "div"},
    "GridView": {"element": "table", "note": "Use .map() for rows", "complex": True},
    "Repeater": {"element": "div", "note": "Use .map() for items", "complex": True},
    "DataList": {"element": "div", "note": "Use .map() for items", "complex": True},
    "ListView": {"element": "div", "note": "Use .map() for items", "complex": True},
    "FormView": {"element": "form", "note": "Single-record form", "complex": True},
    "DetailsView": {"element": "div", "note": "Detail view with .map()", "complex": True},
    "Calendar": {"element": "input", "type": "date"},
    "Menu": {"element": "nav"},
    "TreeView": {"element": "ul", "note": "Recursive tree component", "complex": True},
    "Wizard": {"element": "div", "note": "Multi-step form component", "complex": True},
    "MultiView": {"element": "div", "note": "Conditional rendering"},
    "UpdatePanel": {"element": "div", "note": "Remove — React handles updates"},
    "ScriptManager": {"element": None, "note": "Remove — not needed in React"},
}

# ── Validation control → HTML5 / React validation ───────────────────

_VALIDATION_MAP: Dict[str, Dict[str, str]] = {
    "RequiredFieldValidator": {"attr": "required", "msg": "This field is required"},
    "RangeValidator": {"attr": "min/max", "msg": "Value must be in range"},
    "RegularExpressionValidator": {"attr": "pattern", "msg": "Invalid format"},
    "CompareValidator": {"attr": "custom", "msg": "Values must match"},
    "CustomValidator": {"attr": "custom", "msg": "Validation failed"},
}

# ── DB access patterns (for repository extraction) ──────────────────

_DB_ACCESS_RE = re.compile(
    r"(?:SqlConnection|SqlCommand|OleDbConnection|OleDbCommand"
    r"|SqlDataAdapter|OleDbDataAdapter|DataAdapter"
    r"|DataSet|DataTable|DataReader"
    r"|ExecuteReader|ExecuteNonQuery|ExecuteScalar"
    r"|ConnectionString|\.Open\(\)|\.Close\(\)"
    r"|EntityFramework|DbContext|ObjectContext"
    r"|LinqToSql|DataContext)",
    re.IGNORECASE,
)

# ── WebForms lifecycle handlers (to separate from business methods) ──

_WEBFORMS_LIFECYCLE_HANDLERS = frozenset({
    "Page_Load", "Page_Init", "Page_PreInit", "Page_PreRender",
    "Page_Unload", "Page_Error", "Page_LoadComplete",
    "Page_PreRenderComplete", "Page_InitComplete",
    "OnInit", "OnLoad", "OnPreRender", "OnUnload",
})

# ── VB.NET method signature regex ────────────────────────────────────

_VB_METHOD_SIG_RE = re.compile(
    r"(?:Public|Protected|Friend)\s+(?:(?:Shared|Overrides|Overridable|"
    r"MustOverride|NotOverridable)\s+)*"
    r"(?:Sub|Function)\s+(\w+)\s*\(([^)]*)\)",
    re.IGNORECASE,
)

# ── Of(...) generic pattern ──────────────────────────────────────────

_OF_GENERIC_RE = re.compile(r"\(Of\s+(.+)\)", re.IGNORECASE)

# ── Array type pattern ───────────────────────────────────────────────

_ARRAY_SUFFIX_RE = re.compile(r"^(.+?)\(\s*\)$")


# ═══════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════


def _pascal_case(name: str) -> str:
    """Convert a slash/dash/underscore/dot-delimited name to PascalCase."""
    parts = re.split(r"[/\-_.]+", name.strip("/"))
    return "".join(p.capitalize() for p in parts if p)


def _camel_case(name: str) -> str:
    """Convert to camelCase."""
    pascal = _pascal_case(name)
    return pascal[0].lower() + pascal[1:] if pascal else ""


def _vb_type_to_cs(vb_type: str) -> str:
    """Map a VB.NET type to its C# equivalent.

    Handles generics ``List(Of String)`` → ``List<string>``,
    array types ``Integer()`` → ``int[]``,
    and nullable ``Integer?`` → ``int?``.
    """
    if not vb_type:
        return "object"

    vb_type = vb_type.strip()

    # Nullable: Integer? → int?
    nullable = vb_type.endswith("?")
    if nullable:
        vb_type = vb_type[:-1].strip()

    # Array: Integer() → int[]
    arr_m = _ARRAY_SUFFIX_RE.match(vb_type)
    if arr_m:
        inner = _vb_type_to_cs(arr_m.group(1))
        return f"{inner}[]{'?' if nullable else ''}"

    # Generics: List(Of String) → List<string>
    of_m = _OF_GENERIC_RE.search(vb_type)
    if of_m:
        outer = vb_type[: of_m.start()].strip()
        inner_parts = [_vb_type_to_cs(p.strip()) for p in of_m.group(1).split(",")]
        cs_outer = _VB_TYPE_MAP.get(outer, outer)
        cs_inner = ", ".join(inner_parts)
        return f"{cs_outer}<{cs_inner}>{'?' if nullable else ''}"

    # Simple lookup
    cs = _VB_TYPE_MAP.get(vb_type, vb_type)
    return f"{cs}?" if nullable else cs


def _vb_param_to_cs(param: Dict[str, Any]) -> str:
    """Convert a VB.NET parameter dict to a C# parameter string.

    Handles ByRef → ``ref``, Optional with default → ``= default``,
    ParamArray → ``params``.
    """
    name = param.get("name", "arg")
    ptype = _vb_type_to_cs(param.get("type") or "object")
    passing = (param.get("passing") or "ByVal").strip()
    default = param.get("default")
    is_optional = param.get("optional", False)

    prefix = ""
    if passing.lower() == "byref":
        prefix = "ref "
    elif passing.lower() == "paramarray":
        prefix = "params "
        if not ptype.endswith("[]"):
            ptype += "[]"

    result = f"{prefix}{ptype} {name}"

    if is_optional and default is not None:
        cs_default = default.strip()
        # Convert VB Nothing → null
        if cs_default.lower() == "nothing":
            cs_default = "null"
        elif cs_default.lower() == "true":
            cs_default = "true"
        elif cs_default.lower() == "false":
            cs_default = "false"
        result += f" = {cs_default}"

    return result


def _vb_modifier_to_cs(modifier: str) -> str:
    """Convert a single VB.NET modifier to its C# equivalent."""
    return _VB_MODIFIER_MAP.get(modifier, modifier.lower())


def _vb_access_to_cs(modifiers: List[str]) -> str:
    """Extract the C# access modifier from a VB.NET modifier list.

    Returns the first matching access modifier, defaulting to ``public``.
    """
    for mod in modifiers:
        cs = _VB_ACCESS_MAP.get(mod)
        if cs:
            return cs
    return "public"


def _vb_nonaccess_mods_to_cs(modifiers: List[str]) -> List[str]:
    """Convert non-access VB.NET modifiers to C# equivalents."""
    result: List[str] = []
    for mod in modifiers:
        if mod in _VB_ACCESS_MAP:
            continue
        cs = _VB_MODIFIER_MAP.get(mod, "")
        if cs and cs not in result:
            result.append(cs)
    return result


def _classify_vb_unit(unit: Dict[str, Any]) -> str:
    """Classify a VB.NET unit by application archetype.

    Priority: file_imports > extends > file path heuristic.
    """
    if unit.get("unit_type") not in ("class", "module", "interface"):
        return _ARCH_LIBRARY
    if unit.get("language", "").lower() != "vbnet":
        return _ARCH_UNKNOWN

    meta = unit.get("metadata", {})
    extends = meta.get("extends", "")
    file_imports = set()
    for imp in meta.get("file_imports", []):
        # "Imports System.Web" → "System.Web"
        ns = imp.replace("Imports ", "").strip()
        file_imports.add(ns)

    # 1. Check imports
    if file_imports & _WEBFORMS_IMPORTS:
        return _ARCH_WEBFORMS
    if file_imports & _WINFORMS_IMPORTS:
        return _ARCH_WINFORMS

    # 2. Check extends
    if extends in _WEBFORMS_BASES:
        return _ARCH_WEBFORMS
    if extends in _WINFORMS_BASES:
        return _ARCH_WINFORMS

    # 3. File path heuristic
    fp = unit.get("file_path", "").lower()
    if ".aspx." in fp or "app_code" in fp or "web" in fp:
        return _ARCH_WEBFORMS

    return _ARCH_LIBRARY


def _asp_control_to_jsx(control: Dict[str, Any]) -> str:
    """Convert a single ASP.NET server control to a JSX string.

    Uses ``_ASP_CONTROL_MAP`` to find the HTML/React element equivalent.
    """
    ctrl_type = control.get("type", "")
    ctrl_id = control.get("id", "")
    props = control.get("properties", {})

    mapping = _ASP_CONTROL_MAP.get(ctrl_type, {"element": "div"})
    element = mapping.get("element")

    if element is None:
        return f"{{/* {ctrl_type} removed — not needed in React */}}"

    # Build props string
    jsx_props: List[str] = []
    if ctrl_id:
        safe_id = _camel_case(ctrl_id)
        jsx_props.append(f'id="{safe_id}"')

    html_type = mapping.get("type")
    if html_type:
        jsx_props.append(f'type="{html_type}"')

    value_prop = mapping.get("value_prop")
    change_event = mapping.get("change_event")
    if value_prop and ctrl_id:
        safe_name = _camel_case(ctrl_id)
        jsx_props.append(f"{value_prop}={{{safe_name}}}")
    if change_event and ctrl_id:
        setter = "set" + ctrl_id[0].upper() + ctrl_id[1:]
        jsx_props.append(f"{change_event}={{(e) => {setter}(e.target.value)}}")

    if mapping.get("multiple"):
        jsx_props.append("multiple")

    prop_str = " " + " ".join(jsx_props) if jsx_props else ""

    note = mapping.get("note", "")
    comment = f"  {{/* {note} */}}\n" if note else ""

    if mapping.get("complex"):
        return (
            f"{comment}"
            f"      <{element}{prop_str}>\n"
            f"        {{/* TODO: migrate {ctrl_type} (ID: {ctrl_id}) */}}\n"
            f"      </{element}>"
        )

    # Self-closing for void elements
    if element in ("input", "img"):
        return f"{comment}      <{element}{prop_str} />"

    text = props.get("Text", "")
    if text:
        return f"{comment}      <{element}{prop_str}>{text}</{element}>"
    return f"{comment}      <{element}{prop_str} />"


def _namespace_to_path(qualified_name: str) -> str:
    """Convert a dotted namespace/qualified name to a directory path.

    ``MyApp.Models.User`` → ``MyApp/Models``
    """
    parts = qualified_name.split(".")
    if len(parts) > 1:
        return "/".join(parts[:-1])
    return ""


def _extract_namespace(unit: Dict[str, Any]) -> str:
    """Extract the namespace portion of a qualified name."""
    qname = unit.get("qualified_name", "")
    parts = qname.rsplit(".", 1)
    return parts[0] if len(parts) > 1 else ""


# ═══════════════════════════════════════════════════════════════════════
# Lane Implementation
# ═══════════════════════════════════════════════════════════════════════


class VbNetToDotNetCoreLane(MigrationLane):
    """Migration lane for VB.NET → C# on .NET Core.

    Covers three archetypes:

    * **WebForms** code-behind classes → ASP.NET Core Controller + Service
      + optional Repository.  ASPX pages → React components + API + types.
    * **WinForms** Form classes → Blazor component stubs.
    * **Library** modules, classes, interfaces, structs, enums → C#
      equivalents with concern decomposition when mixed logic is detected.
    """

    # ── Identity ────────────────────────────────────────────────

    @property
    def lane_id(self) -> str:
        return "vbnet_to_dotnetcore"

    @property
    def display_name(self) -> str:
        return "VB.NET \u2192 .NET Core (C#)"

    @property
    def source_frameworks(self) -> List[str]:
        return ["vbnet", "aspnet_webforms", "winforms"]

    @property
    def target_frameworks(self) -> List[str]:
        return ["dotnet_core", "aspnetcore", "ef_core"]

    @property
    def version(self) -> str:
        return "1.0.0"

    # ── Applicability ───────────────────────────────────────────

    def detect_applicability(
        self, source_framework: str, target_stack: Dict[str, Any]
    ) -> float:
        source_lower = source_framework.lower()
        if source_lower not in {"vbnet", "aspnet_webforms", "winforms"}:
            return 0.0

        target_fw = str(target_stack.get("framework", "")).lower()
        target_name = str(target_stack.get("name", "")).lower()
        combined = f"{target_fw} {target_name}"

        if "dotnet" in combined or "aspnetcore" in combined or ".net" in combined:
            return 0.95

        # Source matches but target unspecified
        return 0.5

    # ── Archetype Indexing ──────────────────────────────────────

    @staticmethod
    def _build_archetype_index(
        units: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build cross-reference indexes for concern decomposition.

        Returns a dict with:
          ``by_name``       -- simple name → list of unit dicts
          ``by_archetype``  -- archetype constant → list of unit dicts
          ``classified``    -- unit id → archetype constant
          ``by_parent``     -- parent_name → list of child unit dicts
        """
        by_name: Dict[str, List[Dict[str, Any]]] = {}
        by_archetype: Dict[str, List[Dict[str, Any]]] = {
            _ARCH_WEBFORMS: [],
            _ARCH_WINFORMS: [],
            _ARCH_LIBRARY: [],
            _ARCH_UNKNOWN: [],
        }
        classified: Dict[str, str] = {}
        by_parent: Dict[str, List[Dict[str, Any]]] = {}

        for u in units:
            lang = u.get("language", "").lower()
            if lang != "vbnet":
                continue

            ut = u.get("unit_type", "")
            name = u.get("name", "")
            uid = str(u.get("id", name))

            if ut in ("class", "module", "interface"):
                arch = _classify_vb_unit(u)
                by_name.setdefault(name, []).append(u)
                by_archetype[arch].append(u)
                classified[uid] = arch
            elif ut in ("method", "constructor", "property", "event"):
                parent = u.get("parent_name") or u.get("metadata", {}).get("parent_name", "")
                if parent:
                    by_parent.setdefault(parent, []).append(u)

        return {
            "by_name": by_name,
            "by_archetype": by_archetype,
            "classified": classified,
            "by_parent": by_parent,
        }

    # ── Transform Rules ─────────────────────────────────────────

    def get_transform_rules(self) -> List[TransformRule]:
        return [
            # Library transforms
            TransformRule(
                name="module_to_static_class",
                source_pattern={"unit_type": "module", "language": "vbnet"},
                target_template="csharp_static_class",
                confidence=0.95,
                description="Convert VB.NET Module to C# static class.",
            ),
            TransformRule(
                name="class_to_csharp",
                source_pattern={"unit_type": "class", "language": "vbnet"},
                target_template="csharp_class",
                confidence=0.90,
                description=(
                    "Convert VB.NET Class to C# class with concern "
                    "decomposition when mixed logic detected."
                ),
            ),
            TransformRule(
                name="interface_to_csharp",
                source_pattern={"unit_type": "interface", "language": "vbnet"},
                target_template="csharp_interface",
                confidence=0.95,
                description="Convert VB.NET Interface to C# interface.",
            ),
            TransformRule(
                name="struct_to_csharp",
                source_pattern={"unit_type": "struct", "language": "vbnet"},
                target_template="csharp_struct",
                confidence=0.95,
                description="Convert VB.NET Structure to C# struct.",
            ),
            TransformRule(
                name="enum_to_csharp",
                source_pattern={"unit_type": "enum", "language": "vbnet"},
                target_template="csharp_enum",
                confidence=0.98,
                description="Convert VB.NET Enum to C# enum (near 1:1).",
            ),
            # WebForms transforms
            TransformRule(
                name="codebehind_to_controller",
                source_pattern={"unit_type": "class", "archetype": "webforms"},
                target_template="aspnetcore_controller",
                confidence=0.80,
                description=(
                    "Convert WebForms code-behind to ASP.NET Core Controller "
                    "with DI service injection."
                ),
            ),
            TransformRule(
                name="codebehind_to_service",
                source_pattern={"unit_type": "class", "archetype": "webforms"},
                target_template="aspnetcore_service",
                confidence=0.75,
                description=(
                    "Extract business logic from WebForms code-behind into "
                    "service interface + implementation."
                ),
            ),
            TransformRule(
                name="codebehind_to_repository",
                source_pattern={"unit_type": "class", "archetype": "webforms"},
                target_template="ef_core_repository",
                confidence=0.65,
                requires_review=True,
                description=(
                    "Generate EF Core repository from WebForms data access "
                    "patterns (ADO.NET, DataSet, etc.)."
                ),
            ),
            # ASPX view transforms
            TransformRule(
                name="aspx_to_react",
                source_pattern={"unit_type": "aspx_page"},
                target_template="react_functional_component",
                confidence=0.70,
                requires_review=True,
                description=(
                    "Generate React component + API service + types from "
                    "ASPX page server controls and data bindings."
                ),
            ),
            TransformRule(
                name="master_to_layout",
                source_pattern={"unit_type": "aspx_master"},
                target_template="react_layout_component",
                confidence=0.70,
                requires_review=True,
                description="Convert .master page to React layout component.",
            ),
            TransformRule(
                name="ascx_to_component",
                source_pattern={"unit_type": "aspx_control"},
                target_template="react_component",
                confidence=0.75,
                requires_review=True,
                description="Convert .ascx user control to React component.",
            ),
            # WinForms + Data
            TransformRule(
                name="winform_to_stub",
                source_pattern={"unit_type": "class", "archetype": "winforms"},
                target_template="blazor_component",
                confidence=0.55,
                requires_review=True,
                description=(
                    "Generate Blazor component stub from WinForms Form class. "
                    "Requires significant manual redesign."
                ),
            ),
            TransformRule(
                name="ado_to_ef_core",
                source_pattern={"has_db_access": True},
                target_template="ef_core_dbcontext",
                confidence=0.60,
                requires_review=True,
                description=(
                    "Convert ADO.NET data access patterns to EF Core "
                    "DbContext + repository."
                ),
            ),
        ]

    # ── Transform Application ───────────────────────────────────

    def apply_transforms(
        self,
        units: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> List[TransformResult]:
        results: List[TransformResult] = []
        arch_index = self._build_archetype_index(units)
        generated_services: Set[str] = set()
        generated_repos: Set[str] = set()

        for unit in units:
            unit_type = unit.get("unit_type", "")
            language = unit.get("language", "").lower()

            # ── ASPX pages → React (multi-output) ──────────────
            if unit_type in ("aspx_page", "aspx_master", "aspx_control"):
                try:
                    aspx_results = self._transform_aspx_page(unit)
                    results.extend(aspx_results)
                except Exception:
                    logger.warning(
                        "ASPX transform failed for '%s'",
                        unit.get("name", "?"),
                        exc_info=True,
                    )
                continue

            # ── VB.NET units only below ─────────────────────────
            if language != "vbnet":
                continue

            uid = str(unit.get("id", unit.get("name", "")))
            arch = arch_index["classified"].get(uid, _ARCH_LIBRARY)

            try:
                if unit_type == "module":
                    results.append(self._transform_module(unit, arch_index))

                elif unit_type == "class":
                    if arch == _ARCH_WEBFORMS:
                        results.extend(
                            self._transform_webforms_class(
                                unit, arch_index,
                                generated_services, generated_repos,
                            )
                        )
                    elif arch == _ARCH_WINFORMS:
                        results.append(self._transform_winforms_class(unit))
                    else:
                        results.extend(
                            self._transform_class_smart(
                                unit, arch_index,
                                generated_services, generated_repos,
                            )
                        )

                elif unit_type == "interface":
                    results.append(self._transform_interface(unit))

                elif unit_type == "struct":
                    results.append(self._transform_struct(unit))

                elif unit_type == "enum":
                    results.append(self._transform_enum(unit))

            except Exception:
                logger.warning(
                    "Transform failed for '%s' (%s)",
                    unit.get("name", "?"),
                    unit_type,
                    exc_info=True,
                )

        logger.info(
            "VbNetToDotNetCore: applied %d transforms to %d units",
            len(results),
            len(units),
        )
        return results

    # ═══════════════════════════════════════════════════════════════
    # Library Generators
    # ═══════════════════════════════════════════════════════════════

    def _transform_module(
        self, unit: Dict[str, Any], arch_index: Dict[str, Any]
    ) -> TransformResult:
        """Convert VB.NET Module → C# ``public static class``.

        All methods become ``static``.  Sub → ``void``, Function → typed.
        """
        name = unit.get("name", "Unknown")
        unit_id = str(unit.get("id", name))
        meta = unit.get("metadata", {})
        ns = _extract_namespace(unit)

        children = arch_index.get("by_parent", {}).get(name, [])

        lines = ["using System;\n"]
        if ns:
            lines.append(f"namespace {ns}\n{{")

        lines.append(f"public static class {name}\n{{")

        methods_added = False
        for child in children:
            if child.get("unit_type") not in ("method", "constructor"):
                continue
            mname = child.get("name", "Unknown")
            cmeta = child.get("metadata", {})
            params = cmeta.get("parsed_params", [])
            ret = cmeta.get("return_type")

            cs_ret = _vb_type_to_cs(ret) if ret else "void"
            cs_params = ", ".join(_vb_param_to_cs(p) for p in params)

            lines.append(f"    public static {cs_ret} {mname}({cs_params})")
            lines.append("    {")
            lines.append(f"        // TODO: migrate from VB.NET Module {name}.{mname}")
            lines.append("        throw new NotImplementedException();")
            lines.append("    }\n")
            methods_added = True

        if not methods_added:
            # Extract from source
            source = unit.get("source", "")
            for m in _VB_METHOD_SIG_RE.finditer(source):
                mname = m.group(1)
                lines.append(f"    public static void {mname}()")
                lines.append("    {")
                lines.append(f"        // TODO: migrate from VB.NET Module {name}.{mname}")
                lines.append("        throw new NotImplementedException();")
                lines.append("    }\n")
                methods_added = True

        if not methods_added:
            lines.append("    // TODO: add methods from VB.NET Module")

        lines.append("}")
        if ns:
            lines.append("}")

        code = "\n".join(lines)
        ns_path = ns.replace(".", "/") if ns else "src"
        target_path = f"src/{ns_path}/{name}.cs"

        return TransformResult(
            source_unit_id=unit_id,
            target_code=code,
            target_path=target_path,
            rule_name="module_to_static_class",
            confidence=0.95,
            notes=[
                f"Module '{name}' → public static class",
                f"Methods converted: {methods_added}",
                f"Namespace: {ns or '(root)'}",
            ],
        )

    def _transform_class_smart(
        self,
        unit: Dict[str, Any],
        arch_index: Dict[str, Any],
        generated_services: Set[str],
        generated_repos: Set[str],
    ) -> List[TransformResult]:
        """Convert VB.NET Class with concern decomposition.

        If the class has mixed concerns (business logic + DB access),
        splits into Service + Repository.  Otherwise, single C# class.
        """
        name = unit.get("name", "Unknown")
        unit_id = str(unit.get("id", name))
        source = unit.get("source", "")
        meta = unit.get("metadata", {})

        has_db = bool(_DB_ACCESS_RE.search(source))
        children = arch_index.get("by_parent", {}).get(name, [])
        has_business = any(
            c.get("unit_type") == "method"
            and c.get("name", "") not in _WEBFORMS_LIFECYCLE_HANDLERS
            for c in children
        )

        # If class has both business logic and DB access → decompose
        if has_db and has_business:
            results: List[TransformResult] = []
            svc_name = name + "Service"
            repo_name = name + "Repository"

            if svc_name not in generated_services:
                generated_services.add(svc_name)
                results.append(
                    self._gen_service_class(unit_id, unit, children, repo_name if has_db else None)
                )
            if repo_name not in generated_repos:
                generated_repos.add(repo_name)
                results.append(self._gen_repository_class(unit_id, name))

            return results

        # Single class conversion
        return [self._transform_class_single(unit, arch_index)]

    def _transform_class_single(
        self, unit: Dict[str, Any], arch_index: Dict[str, Any]
    ) -> TransformResult:
        """Convert a VB.NET Class to a single C# class."""
        name = unit.get("name", "Unknown")
        unit_id = str(unit.get("id", name))
        meta = unit.get("metadata", {})
        ns = _extract_namespace(unit)
        extends = meta.get("extends", "")
        implements = meta.get("implements", [])
        modifiers = meta.get("modifiers", [])

        access = _vb_access_to_cs(modifiers)
        extra_mods = _vb_nonaccess_mods_to_cs(modifiers)
        mod_str = f"{access} {' '.join(extra_mods)} ".strip() if extra_mods else access

        # Build inheritance clause
        bases: List[str] = []
        if extends:
            bases.append(_vb_type_to_cs(extends))
        for iface in implements:
            bases.append(_vb_type_to_cs(iface.strip()))
        inheritance = f" : {', '.join(bases)}" if bases else ""

        lines = ["using System;\n"]
        if ns:
            lines.append(f"namespace {ns}\n{{")

        lines.append(f"{mod_str} class {name}{inheritance}\n{{")

        children = arch_index.get("by_parent", {}).get(name, [])
        for child in children:
            ct = child.get("unit_type", "")
            cname = child.get("name", "Unknown")
            cmeta = child.get("metadata", {})

            if ct == "property":
                ret = cmeta.get("return_type", "object")
                cs_type = _vb_type_to_cs(ret)
                lines.append(f"    public {cs_type} {cname} {{ get; set; }}\n")

            elif ct == "constructor":
                params = cmeta.get("parsed_params", [])
                cs_params = ", ".join(_vb_param_to_cs(p) for p in params)
                lines.append(f"    public {name}({cs_params})")
                lines.append("    {")
                lines.append(f"        // TODO: migrate constructor logic")
                lines.append("    }\n")

            elif ct == "method":
                params = cmeta.get("parsed_params", [])
                ret = cmeta.get("return_type")
                cs_ret = _vb_type_to_cs(ret) if ret else "void"
                cs_params = ", ".join(_vb_param_to_cs(p) for p in params)

                c_mods = cmeta.get("modifiers", [])
                c_access = _vb_access_to_cs(c_mods)
                c_extra = _vb_nonaccess_mods_to_cs(c_mods)
                m_prefix = f"{c_access} {' '.join(c_extra)} ".strip() if c_extra else c_access

                lines.append(f"    {m_prefix} {cs_ret} {cname}({cs_params})")
                lines.append("    {")
                lines.append(f"        // TODO: migrate from {name}.{cname}")
                lines.append("        throw new NotImplementedException();")
                lines.append("    }\n")

        if not children:
            lines.append("    // TODO: add members from VB.NET class")

        lines.append("}")
        if ns:
            lines.append("}")

        code = "\n".join(lines)
        ns_path = ns.replace(".", "/") if ns else "src"
        target_path = f"src/{ns_path}/{name}.cs"

        return TransformResult(
            source_unit_id=unit_id,
            target_code=code,
            target_path=target_path,
            rule_name="class_to_csharp",
            confidence=0.90,
            notes=[
                f"Class '{name}' → C# class",
                f"Inherits: {extends or 'none'}",
                f"Implements: {', '.join(implements) if implements else 'none'}",
                f"Members converted: {len(children)}",
            ],
        )

    def _transform_interface(self, unit: Dict[str, Any]) -> TransformResult:
        """Convert VB.NET Interface → C# interface."""
        name = unit.get("name", "Unknown")
        unit_id = str(unit.get("id", name))
        meta = unit.get("metadata", {})
        ns = _extract_namespace(unit)

        # Ensure I prefix
        cs_name = name if name.startswith("I") and len(name) > 1 and name[1].isupper() else f"I{name}"

        lines = ["using System;\n"]
        if ns:
            lines.append(f"namespace {ns}\n{{")

        lines.append(f"public interface {cs_name}\n{{")

        # Extract method signatures from source
        source = unit.get("source", "")
        methods_added = False
        for m in _VB_METHOD_SIG_RE.finditer(source):
            mname = m.group(1)
            param_text = m.group(2).strip()
            # Basic param conversion
            lines.append(f"    void {mname}();  // TODO: convert parameters\n")
            methods_added = True

        if not methods_added:
            lines.append("    // TODO: add interface members")

        lines.append("}")
        if ns:
            lines.append("}")

        code = "\n".join(lines)
        ns_path = ns.replace(".", "/") if ns else "src"
        target_path = f"src/{ns_path}/{cs_name}.cs"

        return TransformResult(
            source_unit_id=unit_id,
            target_code=code,
            target_path=target_path,
            rule_name="interface_to_csharp",
            confidence=0.95,
            notes=[
                f"Interface '{name}' → {cs_name}",
                f"Namespace: {ns or '(root)'}",
            ],
        )

    def _transform_struct(self, unit: Dict[str, Any]) -> TransformResult:
        """Convert VB.NET Structure → C# struct."""
        name = unit.get("name", "Unknown")
        unit_id = str(unit.get("id", name))
        meta = unit.get("metadata", {})
        ns = _extract_namespace(unit)
        implements = meta.get("implements", [])

        bases = [_vb_type_to_cs(i.strip()) for i in implements]
        inheritance = f" : {', '.join(bases)}" if bases else ""

        lines = ["using System;\n"]
        if ns:
            lines.append(f"namespace {ns}\n{{")

        lines.append(f"public struct {name}{inheritance}\n{{")

        # Extract fields from source
        source = unit.get("source", "")
        field_re = re.compile(
            r"(?:Public|Private|Friend)\s+(\w+)\s+As\s+(\S+)",
            re.IGNORECASE,
        )
        for fm in field_re.finditer(source):
            fname, ftype = fm.group(1), fm.group(2)
            cs_type = _vb_type_to_cs(ftype)
            lines.append(f"    public {cs_type} {fname} {{ get; set; }}")

        lines.append("}")
        if ns:
            lines.append("}")

        code = "\n".join(lines)
        ns_path = ns.replace(".", "/") if ns else "src"
        target_path = f"src/{ns_path}/{name}.cs"

        return TransformResult(
            source_unit_id=unit_id,
            target_code=code,
            target_path=target_path,
            rule_name="struct_to_csharp",
            confidence=0.95,
            notes=[
                f"Structure '{name}' → C# struct",
                f"Implements: {', '.join(implements) if implements else 'none'}",
            ],
        )

    def _transform_enum(self, unit: Dict[str, Any]) -> TransformResult:
        """Convert VB.NET Enum → C# enum (near 1:1)."""
        name = unit.get("name", "Unknown")
        unit_id = str(unit.get("id", name))
        ns = _extract_namespace(unit)

        lines = []
        if ns:
            lines.append(f"namespace {ns}\n{{")

        lines.append(f"public enum {name}\n{{")

        # Parse enum members from source
        source = unit.get("source", "")
        member_re = re.compile(r"^\s+(\w+)\s*(?:=\s*(.+))?$", re.MULTILINE)
        members: List[str] = []
        for em in member_re.finditer(source):
            mname = em.group(1)
            mval = em.group(2)
            if mname.lower() in ("end", "enum"):
                continue
            if mval:
                members.append(f"    {mname} = {mval.strip()},")
            else:
                members.append(f"    {mname},")

        if members:
            lines.extend(members)
        else:
            lines.append("    // TODO: add enum values")

        lines.append("}")
        if ns:
            lines.append("}")

        code = "\n".join(lines)
        ns_path = ns.replace(".", "/") if ns else "src"
        target_path = f"src/{ns_path}/{name}.cs"

        return TransformResult(
            source_unit_id=unit_id,
            target_code=code,
            target_path=target_path,
            rule_name="enum_to_csharp",
            confidence=0.98,
            notes=[f"Enum '{name}' → C# enum ({len(members)} members)"],
        )

    # ═══════════════════════════════════════════════════════════════
    # Service / Repository Generators (shared by WebForms + Library)
    # ═══════════════════════════════════════════════════════════════

    def _gen_service_class(
        self,
        source_unit_id: str,
        unit: Dict[str, Any],
        children: List[Dict[str, Any]],
        repo_name: Optional[str],
    ) -> TransformResult:
        """Generate a service interface + implementation from VB.NET class."""
        name = unit.get("name", "Unknown")
        svc_name = name + "Service"
        iface_name = f"I{svc_name}"
        ns = _extract_namespace(unit)
        repo_var = _camel_case(name) + "Repository" if repo_name else None

        lines = [
            "using System;",
            "using System.Threading.Tasks;",
            "using Microsoft.Extensions.Logging;\n",
        ]

        # Interface
        lines.append(f"public interface {iface_name}")
        lines.append("{")

        business_methods: List[Dict[str, Any]] = []
        for child in children:
            if child.get("unit_type") != "method":
                continue
            cname = child.get("name", "")
            if cname in _WEBFORMS_LIFECYCLE_HANDLERS:
                continue
            if cname.lower().startswith("_"):
                continue
            business_methods.append(child)

        for bm in business_methods:
            mname = bm.get("name", "Method")
            cmeta = bm.get("metadata", {})
            ret = cmeta.get("return_type")
            cs_ret = _vb_type_to_cs(ret) if ret else "void"
            params = cmeta.get("parsed_params", [])
            cs_params = ", ".join(_vb_param_to_cs(p) for p in params)
            lines.append(f"    Task<{cs_ret}> {mname}({cs_params});")

        if not business_methods:
            lines.append("    Task<object> Execute();")

        lines.append("}\n")

        # Implementation
        lines.append(f"public class {svc_name} : {iface_name}")
        lines.append("{")

        if repo_var:
            lines.append(f"    private readonly {repo_name} _{repo_var};")
            lines.append(f"    private readonly ILogger<{svc_name}> _logger;\n")
            lines.append(f"    public {svc_name}({repo_name} {repo_var}, ILogger<{svc_name}> logger)")
            lines.append("    {")
            lines.append(f"        _{repo_var} = {repo_var};")
            lines.append("        _logger = logger;")
            lines.append("    }\n")
        else:
            lines.append(f"    private readonly ILogger<{svc_name}> _logger;\n")
            lines.append(f"    public {svc_name}(ILogger<{svc_name}> logger)")
            lines.append("    {")
            lines.append("        _logger = logger;")
            lines.append("    }\n")

        for bm in business_methods:
            mname = bm.get("name", "Method")
            cmeta = bm.get("metadata", {})
            ret = cmeta.get("return_type")
            cs_ret = _vb_type_to_cs(ret) if ret else "void"
            params = cmeta.get("parsed_params", [])
            cs_params = ", ".join(_vb_param_to_cs(p) for p in params)
            lines.append(f"    public async Task<{cs_ret}> {mname}({cs_params})")
            lines.append("    {")
            lines.append(f"        // TODO: migrate business logic from {name}.{mname}")
            lines.append("        throw new NotImplementedException();")
            lines.append("    }\n")

        if not business_methods:
            lines.append("    public async Task<object> Execute()")
            lines.append("    {")
            lines.append(f"        // TODO: migrate business logic from {name}")
            lines.append("        throw new NotImplementedException();")
            lines.append("    }\n")

        lines.append("}")

        code = "\n".join(lines)
        ns_path = ns.replace(".", "/") if ns else "Services"
        target_path = f"src/{ns_path}/{svc_name}.cs"

        return TransformResult(
            source_unit_id=source_unit_id,
            target_code=code,
            target_path=target_path,
            rule_name="codebehind_to_service",
            confidence=0.75,
            notes=[
                f"Service layer: extracted from '{name}'",
                f"Interface: {iface_name}",
                f"Business methods: {len(business_methods)}",
                f"Repository: {repo_name or 'none'}",
            ],
        )

    @staticmethod
    def _gen_repository_class(
        source_unit_id: str,
        base_name: str,
    ) -> TransformResult:
        """Generate an EF Core repository from a class with DB access."""
        repo_name = base_name + "Repository"
        iface_name = f"I{repo_name}"
        entity_name = base_name + "Entity"

        lines = [
            "using System;",
            "using System.Collections.Generic;",
            "using System.Threading.Tasks;",
            "using Microsoft.EntityFrameworkCore;\n",
            f"public interface {iface_name}",
            "{",
            f"    Task<{entity_name}?> GetByIdAsync(int id);",
            f"    Task<List<{entity_name}>> GetAllAsync();",
            f"    Task AddAsync({entity_name} entity);",
            "    Task SaveChangesAsync();",
            "}\n",
            f"public class {repo_name} : {iface_name}",
            "{",
            f"    private readonly AppDbContext _context;\n",
            f"    public {repo_name}(AppDbContext context)",
            "    {",
            "        _context = context;",
            "    }\n",
            f"    public async Task<{entity_name}?> GetByIdAsync(int id)",
            "    {",
            f"        return await _context.Set<{entity_name}>().FindAsync(id);",
            "    }\n",
            f"    public async Task<List<{entity_name}>> GetAllAsync()",
            "    {",
            f"        return await _context.Set<{entity_name}>().ToListAsync();",
            "    }\n",
            f"    public async Task AddAsync({entity_name} entity)",
            "    {",
            f"        await _context.Set<{entity_name}>().AddAsync(entity);",
            "    }\n",
            "    public async Task SaveChangesAsync()",
            "    {",
            "        await _context.SaveChangesAsync();",
            "    }",
            "}\n",
            f"// TODO: create {entity_name} class with [Key] and property mappings",
        ]

        code = "\n".join(lines)
        target_path = f"src/Data/{repo_name}.cs"

        return TransformResult(
            source_unit_id=source_unit_id,
            target_code=code,
            target_path=target_path,
            rule_name="codebehind_to_repository",
            confidence=0.65,
            notes=[
                f"Repository: generated for data access in '{base_name}'",
                f"Entity: {entity_name} (needs entity class definition)",
                f"Interface: {iface_name}",
            ],
        )

    # ═══════════════════════════════════════════════════════════════
    # WebForms Generators
    # ═══════════════════════════════════════════════════════════════

    def _transform_webforms_class(
        self,
        unit: Dict[str, Any],
        arch_index: Dict[str, Any],
        generated_services: Set[str],
        generated_repos: Set[str],
    ) -> List[TransformResult]:
        """Transform a WebForms code-behind into layered architecture.

        Produces 2-3 TransformResults:
          1. Controller (always) — thin routing layer
          2. Service (always) — business logic
          3. Repository (conditional) — data access
        """
        name = unit.get("name", "Unknown")
        unit_id = str(unit.get("id", name))
        source = unit.get("source", "")
        children = arch_index.get("by_parent", {}).get(name, [])
        has_db = bool(_DB_ACCESS_RE.search(source))

        svc_name = name + "Service"
        repo_name = name + "Repository"

        results: List[TransformResult] = []

        # 1. Controller — always generated
        results.append(
            self._gen_aspnetcore_controller(unit_id, unit, children, svc_name)
        )

        # 2. Service — always generated
        if svc_name not in generated_services:
            generated_services.add(svc_name)
            results.append(
                self._gen_service_class(
                    unit_id, unit, children,
                    repo_name if has_db else None,
                )
            )

        # 3. Repository — conditional on DB access
        if has_db and repo_name not in generated_repos:
            generated_repos.add(repo_name)
            results.append(self._gen_repository_class(unit_id, name))

        return results

    def _gen_aspnetcore_controller(
        self,
        unit_id: str,
        unit: Dict[str, Any],
        children: List[Dict[str, Any]],
        service_name: str,
    ) -> TransformResult:
        """Generate an ASP.NET Core API Controller with DI service."""
        name = unit.get("name", "Unknown")
        ctrl_name = name + "Controller"
        svc_var = _camel_case(name) + "Service"
        iface_name = f"I{service_name}"

        # Classify handlers
        get_handlers: List[str] = []
        post_handlers: List[str] = []
        for child in children:
            if child.get("unit_type") != "method":
                continue
            cname = child.get("name", "")
            if cname == "Page_Load":
                get_handlers.append(cname)
            elif cname in _WEBFORMS_LIFECYCLE_HANDLERS:
                continue
            elif "_Click" in cname or "_Submit" in cname:
                post_handlers.append(cname)
            elif cname.startswith("btn") or cname.endswith("_Click"):
                post_handlers.append(cname)

        lines = [
            "using Microsoft.AspNetCore.Mvc;",
            "using System.Threading.Tasks;\n",
            "[ApiController]",
            f'[Route("api/[controller]")]',
            f"public class {ctrl_name} : ControllerBase",
            "{",
            f"    private readonly {iface_name} _{svc_var};\n",
            f"    public {ctrl_name}({iface_name} {svc_var})",
            "    {",
            f"        _{svc_var} = {svc_var};",
            "    }\n",
        ]

        # HttpGet from Page_Load
        lines.append("    [HttpGet]")
        lines.append("    public async Task<IActionResult> Index()")
        lines.append("    {")
        lines.append(f"        // Migrated from {name}.Page_Load")
        lines.append(f"        var data = await _{svc_var}.Execute();")
        lines.append("        return Ok(data);")
        lines.append("    }\n")

        # HttpPost from button click handlers
        for handler in post_handlers:
            action_name = re.sub(r"_Click$|_Submit$", "", handler)
            action_name = re.sub(r"^btn", "", action_name)
            if not action_name:
                action_name = "Submit"
            method_name = _pascal_case(action_name)

            lines.append(f'    [HttpPost("{_camel_case(action_name)}")]')
            lines.append(f"    public async Task<IActionResult> {method_name}()")
            lines.append("    {")
            lines.append(f"        // Migrated from {name}.{handler}")
            lines.append(f"        // TODO: add [FromBody] request model")
            lines.append(f"        await _{svc_var}.Execute();")
            lines.append("        return Ok();")
            lines.append("    }\n")

        lines.append("}")

        code = "\n".join(lines)
        target_path = f"src/Controllers/{ctrl_name}.cs"

        return TransformResult(
            source_unit_id=unit_id,
            target_code=code,
            target_path=target_path,
            rule_name="codebehind_to_controller",
            confidence=0.80,
            notes=[
                f"Controller: {ctrl_name} from WebForms '{name}'",
                f"Service layer: delegates to {iface_name}",
                f"GET endpoints: {len(get_handlers) or 1}",
                f"POST endpoints: {len(post_handlers)}",
            ],
        )

    # ═══════════════════════════════════════════════════════════════
    # ASPX → React Generators
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _aspx_name_to_component(aspx_path: str) -> str:
        """Convert an ASPX file path to a React PascalCase component name."""
        cleaned = re.sub(
            r"^(?:App_Code/|Views/|Pages/)+",
            "",
            aspx_path,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\.(aspx|ascx|master)$", "", cleaned, flags=re.IGNORECASE)
        return _pascal_case(cleaned)

    def _transform_aspx_page(
        self, unit: Dict[str, Any]
    ) -> List[TransformResult]:
        """Transform an ASPX page into React component + API + types.

        Always produces a React component; conditionally adds API service
        and TypeScript types depending on metadata.
        """
        meta = unit.get("metadata", {})
        unit_id = str(unit.get("id", unit.get("name", "unknown")))
        aspx_path = unit.get("name", "unknown.aspx")
        unit_type = unit.get("unit_type", "aspx_page")

        server_controls: list = meta.get("server_controls", [])
        validation_controls: list = meta.get("validation_controls", [])
        event_handlers: list = meta.get("event_handlers", [])
        data_bindings: list = meta.get("data_bindings", [])
        viewstate_keys: list = meta.get("viewstate_keys", [])
        session_keys: list = meta.get("session_keys", [])
        has_postback: bool = meta.get("has_postback", False)
        code_behind: str = meta.get("code_behind", "")
        master_page: str = meta.get("master_page", "")

        component_name = self._aspx_name_to_component(aspx_path)
        camel_name = component_name[0].lower() + component_name[1:] if component_name else ""

        gen_meta = {
            "server_controls": server_controls,
            "validation_controls": validation_controls,
            "event_handlers": event_handlers,
            "data_bindings": data_bindings,
            "viewstate_keys": viewstate_keys,
            "session_keys": session_keys,
            "has_postback": has_postback,
            "code_behind": code_behind,
            "master_page": master_page,
            "unit_type": unit_type,
        }

        results: List[TransformResult] = []

        # 1) Always: React component
        results.append(
            self._gen_react_component(
                unit_id, aspx_path, component_name, camel_name, gen_meta
            )
        )

        # 2) API service — when event handlers or postback
        if event_handlers or has_postback:
            results.append(
                self._gen_api_service(
                    unit_id, aspx_path, component_name, camel_name, gen_meta
                )
            )

        # 3) Types file — when data bindings or viewstate/session
        if data_bindings or viewstate_keys or session_keys:
            results.append(
                self._gen_types_file(
                    unit_id, aspx_path, component_name, camel_name, gen_meta
                )
            )

        return results

    def _gen_react_component(
        self,
        unit_id: str,
        aspx_path: str,
        component_name: str,
        camel_name: str,
        meta: Dict[str, Any],
    ) -> TransformResult:
        """Generate a React functional component from ASPX metadata."""
        controls = meta["server_controls"]
        validations = meta["validation_controls"]
        event_handlers = meta["event_handlers"]
        viewstate_keys = meta["viewstate_keys"]
        session_keys = meta["session_keys"]
        has_postback = meta["has_postback"]
        unit_type = meta.get("unit_type", "aspx_page")

        needs_state = bool(viewstate_keys or controls)
        needs_effect = bool(viewstate_keys)
        needs_submit = bool(event_handlers) or has_postback

        # ── Imports ──────────────────────────────────────────
        react_hooks: List[str] = []
        if needs_state:
            react_hooks.append("useState")
        if needs_effect:
            react_hooks.append("useEffect")

        imports: List[str] = []
        if react_hooks:
            imports.append(f"import React, {{ {', '.join(react_hooks)} }} from 'react';")
        else:
            imports.append("import React from 'react';")

        # Check if Link is needed
        has_hyperlink = any(c.get("type") == "HyperLink" for c in controls)
        if has_hyperlink:
            imports.append("import { Link } from 'react-router-dom';")

        if event_handlers or has_postback:
            imports.append(f"import {{ {camel_name}Api }} from '../services/{camel_name}Api';")

        if viewstate_keys or meta.get("data_bindings"):
            imports.append(f"import {{ {component_name}Data }} from '../types/{camel_name}.types';")

        # ── State fields ─────────────────────────────────────
        state_lines: List[str] = []

        # ViewState → useState
        for key in viewstate_keys:
            safe = _camel_case(key) if "_" in key or "-" in key else key
            setter = "set" + safe[0].upper() + safe[1:]
            state_lines.append(f"  const [{safe}, {setter}] = useState<string>('');")

        # Server control state
        seen_ids: Set[str] = set()
        for ctrl in controls:
            ctrl_id = ctrl.get("id", "")
            if not ctrl_id or ctrl_id in seen_ids:
                continue
            seen_ids.add(ctrl_id)
            mapping = _ASP_CONTROL_MAP.get(ctrl.get("type", ""), {})
            if not mapping.get("value_prop"):
                continue
            safe = _camel_case(ctrl_id)
            setter = "set" + ctrl_id[0].upper() + ctrl_id[1:]
            ts_type = "boolean" if mapping.get("type") == "checkbox" else "string"
            default = "false" if ts_type == "boolean" else "''"
            state_lines.append(f"  const [{safe}, {setter}] = useState<{ts_type}>({default});")

        if needs_effect:
            state_lines.append("  const [loading, setLoading] = useState<boolean>(true);")
            state_lines.append("  const [error, setError] = useState<string | null>(null);")

        # ── Handlers ─────────────────────────────────────────
        handlers: List[str] = []
        if needs_submit:
            handlers.append(
                "  const handleSubmit = async (e: React.FormEvent) => {\n"
                "    e.preventDefault();\n"
                "    try {\n"
                f"      await {camel_name}Api.submit({{ /* form data */ }});\n"
                "    } catch (err) {\n"
                "      setError(err instanceof Error ? err.message : 'Submit failed');\n"
                "    }\n"
                "  };"
            )

        # ── useEffect ────────────────────────────────────────
        effects: List[str] = []
        if needs_effect:
            effects.append(
                "  useEffect(() => {\n"
                "    const fetchData = async () => {\n"
                "      try {\n"
                f"        // TODO: fetch initial data\n"
                "        setLoading(false);\n"
                "      } catch (err) {\n"
                "        setError(err instanceof Error ? err.message : 'Load failed');\n"
                "        setLoading(false);\n"
                "      }\n"
                "    };\n"
                "    fetchData();\n"
                "  }, []);"
            )

        # ── JSX body ─────────────────────────────────────────
        jsx_lines: List[str] = []

        # Layout wrapper for master pages
        if unit_type == "aspx_master":
            jsx_lines.append("    <div className=\"layout\">")
            jsx_lines.append("      <header>{/* Master page header */}</header>")
            jsx_lines.append("      <main>{children}</main>")
            jsx_lines.append("      <footer>{/* Master page footer */}</footer>")
            jsx_lines.append("    </div>")
        else:
            if needs_submit:
                jsx_lines.append("    <form onSubmit={handleSubmit}>")

            # Convert each server control to JSX
            for ctrl in controls:
                jsx = _asp_control_to_jsx(ctrl)
                if jsx:
                    jsx_lines.append(f"      {jsx}")

            # Validation messages
            for val in validations:
                val_type = val.get("type", "")
                target = val.get("control_to_validate", "")
                val_info = _VALIDATION_MAP.get(val_type, {"msg": "Invalid"})
                jsx_lines.append(
                    f"      {{/* Validation: {val_type} for {target} — {val_info['msg']} */}}"
                )

            # Session keys as comments
            for key in session_keys:
                jsx_lines.append(f"      {{/* TODO: Session[\"{key}\"] → use auth context */}}")

            if needs_submit:
                jsx_lines.append("      <button type=\"submit\">Submit</button>")
                jsx_lines.append("    </form>")

        if not jsx_lines:
            jsx_lines.append("    <div>")
            jsx_lines.append(f"      {{/* TODO: migrate {aspx_path} content */}}")
            jsx_lines.append("    </div>")

        # ── Assemble component ───────────────────────────────
        all_lines: List[str] = list(imports)
        all_lines.append("")

        # Props interface for master/layout
        if unit_type == "aspx_master":
            all_lines.append(f"interface {component_name}Props {{")
            all_lines.append("  children: React.ReactNode;")
            all_lines.append("}\n")
            all_lines.append(f"const {component_name}: React.FC<{component_name}Props> = ({{ children }}) => {{")
        else:
            all_lines.append(f"const {component_name}: React.FC = () => {{")

        for sl in state_lines:
            all_lines.append(sl)
        if state_lines:
            all_lines.append("")

        for eff in effects:
            all_lines.append(eff)
        if effects:
            all_lines.append("")

        for h in handlers:
            all_lines.append(h)
        if handlers:
            all_lines.append("")

        if needs_effect:
            all_lines.append("  if (loading) return <div>Loading...</div>;")
            all_lines.append("  if (error) return <div className=\"error\">{error}</div>;")
            all_lines.append("")

        all_lines.append("  return (")
        all_lines.extend(jsx_lines)
        all_lines.append("  );")
        all_lines.append("};")
        all_lines.append("")
        all_lines.append(f"export default {component_name};")
        all_lines.append("")

        code = "\n".join(all_lines)
        target_path = f"src/components/{component_name}.tsx"

        return TransformResult(
            source_unit_id=unit_id,
            target_code=code,
            target_path=target_path,
            rule_name="aspx_to_react" if unit_type == "aspx_page" else (
                "master_to_layout" if unit_type == "aspx_master" else "ascx_to_component"
            ),
            confidence=0.70,
            notes=[
                f"ASPX '{aspx_path}' → React component '{component_name}'",
                f"Server controls: {len(controls)}",
                f"ViewState keys → useState: {len(viewstate_keys)}",
                f"Event handlers: {len(meta['event_handlers'])}",
                f"Code-behind: {meta.get('code_behind', 'none')}",
            ],
        )

    @staticmethod
    def _gen_api_service(
        unit_id: str,
        aspx_path: str,
        component_name: str,
        camel_name: str,
        meta: Dict[str, Any],
    ) -> TransformResult:
        """Generate a TypeScript API service for event handlers."""
        event_handlers = meta["event_handlers"]
        has_postback = meta["has_postback"]

        lines = [
            f"// API service for {component_name}",
            f"// Generated from {aspx_path}\n",
            "const API_BASE = '/api';\n",
            f"export const {camel_name}Api = {{",
        ]

        # Submit handler
        lines.append("  submit: async (data: Record<string, unknown>) => {")
        lines.append(f"    const response = await fetch(`${{API_BASE}}/{camel_name}/submit`, {{")
        lines.append("      method: 'POST',")
        lines.append("      headers: { 'Content-Type': 'application/json' },")
        lines.append("      body: JSON.stringify(data),")
        lines.append("    });")
        lines.append("    if (!response.ok) throw new Error(`Submit failed: ${response.statusText}`);")
        lines.append("    return response.json();")
        lines.append("  },\n")

        # Additional handlers
        for handler in event_handlers:
            if handler in _WEBFORMS_LIFECYCLE_HANDLERS:
                continue
            fn_name = _camel_case(re.sub(r"_Click$|_Submit$|^btn", "", handler))
            if not fn_name or fn_name == "submit":
                continue
            lines.append(f"  {fn_name}: async (data?: Record<string, unknown>) => {{")
            lines.append(f"    const response = await fetch(`${{API_BASE}}/{camel_name}/{fn_name}`, {{")
            lines.append("      method: 'POST',")
            lines.append("      headers: { 'Content-Type': 'application/json' },")
            lines.append("      body: data ? JSON.stringify(data) : undefined,")
            lines.append("    });")
            lines.append(f"    if (!response.ok) throw new Error(`{fn_name} failed: ${{response.statusText}}`);")
            lines.append("    return response.json();")
            lines.append("  },\n")

        # Fetch data (for Page_Load equivalent)
        lines.append("  getData: async () => {")
        lines.append(f"    const response = await fetch(`${{API_BASE}}/{camel_name}`);")
        lines.append("    if (!response.ok) throw new Error(`Load failed: ${response.statusText}`);")
        lines.append("    return response.json();")
        lines.append("  },")

        lines.append("};\n")

        code = "\n".join(lines)
        target_path = f"src/services/{camel_name}Api.ts"

        return TransformResult(
            source_unit_id=unit_id,
            target_code=code,
            target_path=target_path,
            rule_name="aspx_to_react",
            confidence=0.70,
            notes=[
                f"API service for {component_name}",
                f"Event handlers: {len(event_handlers)}",
                f"PostBack: {has_postback}",
            ],
        )

    @staticmethod
    def _gen_types_file(
        unit_id: str,
        aspx_path: str,
        component_name: str,
        camel_name: str,
        meta: Dict[str, Any],
    ) -> TransformResult:
        """Generate TypeScript types from ASPX data bindings and state."""
        data_bindings = meta["data_bindings"]
        viewstate_keys = meta["viewstate_keys"]
        session_keys = meta["session_keys"]

        lines = [
            f"// Types for {component_name}",
            f"// Generated from {aspx_path}\n",
            f"export interface {component_name}Data {{",
        ]

        # Fields from data bindings: Eval("Name") → name: string
        seen_fields: Set[str] = set()
        for expr in data_bindings:
            # Extract field name from Eval("FieldName") or Bind("FieldName")
            field_m = re.search(r'(?:Eval|Bind)\s*\(\s*"([^"]+)"', expr)
            if field_m:
                field = field_m.group(1)
                safe = _camel_case(field)
                if safe not in seen_fields:
                    seen_fields.add(safe)
                    lines.append(f"  {safe}: string;")

        # Fields from ViewState keys
        for key in viewstate_keys:
            safe = _camel_case(key) if "_" in key or "-" in key else key
            if safe not in seen_fields:
                seen_fields.add(safe)
                lines.append(f"  {safe}: string;")

        if not seen_fields:
            lines.append("  // TODO: add data fields")

        lines.append("}\n")

        # Session context type
        if session_keys:
            lines.append(f"export interface {component_name}Session {{")
            for key in session_keys:
                safe = _camel_case(key) if "_" in key or "-" in key else key
                lines.append(f"  {safe}: string;  // Session[\"{key}\"]")
            lines.append("}\n")

        code = "\n".join(lines)
        target_path = f"src/types/{camel_name}.types.ts"

        return TransformResult(
            source_unit_id=unit_id,
            target_code=code,
            target_path=target_path,
            rule_name="aspx_to_react",
            confidence=0.70,
            notes=[
                f"Types for {component_name}",
                f"Data binding fields: {len(seen_fields)}",
                f"Session keys: {len(session_keys)}",
            ],
        )

    # ═══════════════════════════════════════════════════════════════
    # WinForms Generator
    # ═══════════════════════════════════════════════════════════════

    def _transform_winforms_class(self, unit: Dict[str, Any]) -> TransformResult:
        """Generate a Blazor component stub from WinForms Form class."""
        name = unit.get("name", "Unknown")
        unit_id = str(unit.get("id", name))

        lines = [
            f"@page \"/{_camel_case(name)}\"",
            f"@* Migrated from WinForms: {name} *@",
            f"@* NOTE: WinForms → web requires significant manual redesign *@\n",
            f"<h1>{name}</h1>\n",
            "<div class=\"container\">",
            "    @* TODO: recreate form layout *@",
            "    @* Original WinForms controls need manual mapping to HTML/Blazor *@",
            "</div>\n",
            "@code {",
            f"    // Migrated from {name}.vb",
            "    // TODO: convert WinForms event handlers to Blazor @onclick",
            "    // TODO: convert WinForms data bindings to Blazor @bind",
            "",
            "    protected override async Task OnInitializedAsync()",
            "    {",
            f"        // TODO: migrate {name}.Load logic",
            "        await base.OnInitializedAsync();",
            "    }",
            "}",
        ]

        code = "\n".join(lines)
        target_path = f"Pages/{name}.razor"

        return TransformResult(
            source_unit_id=unit_id,
            target_code=code,
            target_path=target_path,
            rule_name="winform_to_stub",
            confidence=0.55,
            notes=[
                f"WinForms '{name}' → Blazor component stub",
                "Requires significant manual redesign",
                "Event handlers need conversion to Blazor patterns",
            ],
        )

    # ═══════════════════════════════════════════════════════════════
    # Prompt Augmentation
    # ═══════════════════════════════════════════════════════════════

    def augment_prompt(
        self,
        phase_type: str,
        base_prompt: str,
        context: Dict[str, Any],
    ) -> str:
        if phase_type == "architecture":
            return self._augment_architecture(base_prompt, context)
        if phase_type == "transform":
            return self._augment_transform(base_prompt, context)
        if phase_type == "test":
            return self._augment_test(base_prompt, context)
        return base_prompt

    def _augment_architecture(
        self, base_prompt: str, context: Dict[str, Any]
    ) -> str:
        return base_prompt + """

## VB.NET → C# Keyword Conversion Reference

| VB.NET | C# |
|--------|-----|
| Module | static class |
| Sub | void method |
| Function | typed method |
| Dim | var / type |
| ByVal | (default, omit) |
| ByRef | ref |
| ParamArray | params |
| Optional ... = default | type param = default |
| Shared | static |
| MustOverride | abstract |
| Overridable | virtual |
| NotOverridable | sealed |
| MustInherit | abstract class |
| NotInheritable | sealed class |
| Overrides | override |
| Nothing | null |
| Me | this |
| MyBase | base |
| Handles | += event wire-up |
| RaiseEvent | invoke event |
| WithEvents | event field |
| And / Or / Not | && / || / ! |
| AndAlso / OrElse | && / || (short-circuit) |
| Is Nothing | == null |
| IsNot Nothing | != null |
| TryCast | as |
| CType | (Type) cast |
| DirectCast | (Type) cast |
| GetType() | typeof() |
| Imports | using |
| Namespace ... End Namespace | namespace { } |

## WebForms → ASP.NET Core Architecture

| WebForms Concept | ASP.NET Core Equivalent |
|-----------------|------------------------|
| Code-behind (.aspx.vb) | Controller + Service + Repository |
| Page_Load | [HttpGet] controller action |
| Button_Click | [HttpPost] controller action |
| ViewState | Client-side state (React useState) |
| Session | IDistributedCache / Auth claims |
| Application state | IMemoryCache / DI singleton |
| Global.asax | Startup.cs / Program.cs |
| Web.config | appsettings.json |
| Master pages | React layout components |
| User controls (.ascx) | React components |
| UpdatePanel (AJAX) | React (SPA handles updates) |
| SqlDataSource | EF Core DbContext |

## ASP Server Control → React Component Mapping

| ASP Control | React Element | Notes |
|------------|---------------|-------|
| asp:TextBox | <input type="text"> | value + onChange |
| asp:Button | <button type="submit"> | onClick |
| asp:Label | <span> | {text} |
| asp:DropDownList | <select> | value + onChange |
| asp:CheckBox | <input type="checkbox"> | checked + onChange |
| asp:GridView | <table> + .map() | Data-driven rows |
| asp:Repeater | <div> + .map() | Template items |
| asp:HyperLink | <Link> | react-router-dom |
| asp:Panel | <div> | Container |
| asp:UpdatePanel | <div> | Remove — React handles updates |
| asp:ScriptManager | (remove) | Not needed |
"""

    def _augment_transform(
        self, base_prompt: str, context: Dict[str, Any]
    ) -> str:
        aug = base_prompt + "\n\n## Pre-completed Deterministic Transforms\n\n"
        aug += (
            "The following transforms have been applied deterministically.\n"
            "Review them and fill in business logic marked with TODO.\n\n"
        )

        # Scan context for completed transforms
        transforms = context.get("completed_transforms", [])
        if transforms:
            for t in transforms:
                notes = t.get("notes", [])
                aug += f"- **{t.get('rule_name', '?')}** → `{t.get('target_path', '?')}`\n"
                for note in notes:
                    if note:
                        aug += f"  - {note}\n"

            # Service cross-references
            aug += "\n### Service Layer Cross-References\n\n"
            for t in transforms:
                for note in t.get("notes", []):
                    if "Service layer:" in note or "Repository:" in note or "Entity:" in note:
                        aug += f"- {note}\n"

            # ASPX→React cross-references
            aug += "\n### ASPX → React Cross-References\n\n"
            for t in transforms:
                target = t.get("target_path", "")
                if target.endswith(".tsx") or target.endswith("Api.ts") or target.endswith(".types.ts"):
                    aug += f"- `{target}` (from {t.get('rule_name', '?')})\n"

        aug += "\n### State Migration Notes\n\n"
        aug += "- ViewState → React `useState` hooks (client-side)\n"
        aug += "- Session → Authentication claims or IDistributedCache\n"
        aug += "- Application state → IMemoryCache or DI singleton service\n"
        aug += "- PostBack → React form submission with fetch/axios\n"

        return aug

    def _augment_test(
        self, base_prompt: str, context: Dict[str, Any]
    ) -> str:
        aug = base_prompt + "\n\n## Test Generation Guidance\n\n"

        transforms = context.get("completed_transforms", [])

        # Controller tests
        controllers = [t for t in transforms if "Controller" in t.get("target_path", "")]
        if controllers:
            aug += "### Controller Integration Tests (WebApplicationFactory)\n\n"
            aug += "```csharp\n"
            aug += "// Use WebApplicationFactory<Program> for integration tests\n"
            aug += "[Fact]\n"
            aug += "public async Task Get_ReturnsSuccess()\n"
            aug += "{\n"
            aug += "    var client = _factory.CreateClient();\n"
            aug += "    var response = await client.GetAsync(\"/api/controllername\");\n"
            aug += "    response.EnsureSuccessStatusCode();\n"
            aug += "}\n"
            aug += "```\n\n"
            for c in controllers:
                aug += f"- Test: `{c.get('target_path', '')}`\n"

        # Service tests
        services = [t for t in transforms if "Service" in t.get("target_path", "")]
        if services:
            aug += "\n### Service Unit Tests (xUnit + Moq)\n\n"
            aug += "```csharp\n"
            aug += "// Mock repository, test service business logic\n"
            aug += "var mockRepo = new Mock<IRepository>();\n"
            aug += "var service = new MyService(mockRepo.Object, _logger);\n"
            aug += "var result = await service.Execute();\n"
            aug += "Assert.NotNull(result);\n"
            aug += "```\n\n"
            for s in services:
                aug += f"- Test: `{s.get('target_path', '')}`\n"

        # React component tests
        components = [t for t in transforms if t.get("target_path", "").endswith(".tsx")]
        if components:
            aug += "\n### React Component Tests (React Testing Library)\n\n"
            aug += "```typescript\n"
            aug += "import { render, screen } from '@testing-library/react';\n"
            aug += "// Test rendering, form submission, state management\n"
            aug += "```\n\n"
            for comp in components:
                aug += f"- Test: `{comp.get('target_path', '')}`\n"

        # Repository tests
        repos = [t for t in transforms if "Repository" in t.get("target_path", "")]
        if repos:
            aug += "\n### Repository Tests (EF Core InMemory)\n\n"
            aug += "```csharp\n"
            aug += "// Use InMemory database provider for repository tests\n"
            aug += "var options = new DbContextOptionsBuilder<AppDbContext>()\n"
            aug += "    .UseInMemoryDatabase(\"TestDb\").Options;\n"
            aug += "```\n\n"
            for r in repos:
                aug += f"- Test: `{r.get('target_path', '')}`\n"

        return aug

    # ═══════════════════════════════════════════════════════════════
    # Quality Gates
    # ═══════════════════════════════════════════════════════════════

    def get_gates(self) -> List[GateDefinition]:
        return [
            GateDefinition(
                name="class_parity",
                description=(
                    "Every VB.NET class, module, and interface must have a "
                    "corresponding C# type in the output."
                ),
                blocking=True,
            ),
            GateDefinition(
                name="method_parity",
                description=(
                    "Every public Sub/Function must have a corresponding "
                    "C# method in the output."
                ),
                blocking=True,
            ),
            GateDefinition(
                name="page_component_parity",
                description=(
                    "Every .aspx page must have a corresponding React "
                    "component in the output."
                ),
                blocking=False,
            ),
            GateDefinition(
                name="event_handler_coverage",
                description=(
                    "Every WebForms event handler must have a corresponding "
                    "controller action in the output."
                ),
                blocking=False,
            ),
            GateDefinition(
                name="namespace_preservation",
                description=(
                    "VB.NET namespaces should map to corresponding C# "
                    "namespaces in the output."
                ),
                blocking=False,
            ),
            GateDefinition(
                name="compile_check",
                description="Target code compiles without errors.",
                blocking=True,
                category=GateCategory.COMPILE,
            ),
            GateDefinition(
                name="unit_test_check",
                description="Generated unit tests pass.",
                blocking=False,
                category=GateCategory.UNIT_TEST,
            ),
        ]

    def run_gate(
        self,
        gate_name: str,
        source_units: List[Dict[str, Any]],
        target_outputs: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> GateResult:
        _dispatch = {
            "class_parity": self._gate_class_parity,
            "method_parity": self._gate_method_parity,
            "page_component_parity": self._gate_page_component_parity,
            "event_handler_coverage": self._gate_event_handler_coverage,
            "namespace_preservation": self._gate_namespace_preservation,
            "compile_check": self._gate_build_passthrough,
            "unit_test_check": self._gate_build_passthrough,
        }
        handler = _dispatch.get(gate_name)
        if handler is None:
            return GateResult(
                gate_name=gate_name,
                passed=False,
                details={"error": f"Unknown gate: {gate_name}"},
                blocking=True,
            )
        return handler(source_units, target_outputs, context)

    @staticmethod
    def _gate_build_passthrough(
        source_units: List[Dict[str, Any]],
        target_outputs: List[Dict[str, Any]],
        context: Any,
        *,
        _name: str = "compile_check",
    ) -> GateResult:
        """Pass-through gate for compile/unit-test checks."""
        return GateResult(
            gate_name=_name,
            passed=True,
            details={"note": "Pass-through -- requires external build system"},
            blocking=False,
        )

    @staticmethod
    def _gate_class_parity(
        source_units: List[Dict[str, Any]],
        target_outputs: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> GateResult:
        """Every VB.NET class/module/interface → C# type."""
        source_types: Set[str] = set()
        for u in source_units:
            if u.get("language", "").lower() == "vbnet" and u.get("unit_type") in ("class", "module", "interface"):
                source_types.add(u.get("name", ""))

        target_code = "\n".join(t.get("target_code", "") for t in target_outputs)
        matched = {name for name in source_types if name in target_code}

        coverage = len(matched) / len(source_types) if source_types else 1.0
        return GateResult(
            gate_name="class_parity",
            passed=coverage >= 0.8,
            details={
                "source_count": len(source_types),
                "matched_count": len(matched),
                "coverage": round(coverage, 2),
                "missing": sorted(source_types - matched),
            },
            blocking=True,
        )

    @staticmethod
    def _gate_method_parity(
        source_units: List[Dict[str, Any]],
        target_outputs: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> GateResult:
        """Every public Sub/Function → C# method."""
        source_methods: Set[str] = set()
        for u in source_units:
            if u.get("language", "").lower() != "vbnet":
                continue
            if u.get("unit_type") != "method":
                continue
            mods = u.get("metadata", {}).get("modifiers", [])
            if any(m.lower() in ("public", "friend") for m in mods):
                source_methods.add(u.get("name", ""))

        target_code = "\n".join(t.get("target_code", "") for t in target_outputs)
        matched = {name for name in source_methods if name in target_code}

        coverage = len(matched) / len(source_methods) if source_methods else 1.0
        return GateResult(
            gate_name="method_parity",
            passed=coverage >= 0.7,
            details={
                "source_count": len(source_methods),
                "matched_count": len(matched),
                "coverage": round(coverage, 2),
                "missing": sorted(source_methods - matched),
            },
            blocking=True,
        )

    @staticmethod
    def _gate_page_component_parity(
        source_units: List[Dict[str, Any]],
        target_outputs: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> GateResult:
        """Every .aspx page → React component."""
        aspx_pages: Set[str] = set()
        for u in source_units:
            if u.get("unit_type") in ("aspx_page", "aspx_master", "aspx_control"):
                aspx_pages.add(u.get("name", ""))

        tsx_files = {t.get("target_path", "") for t in target_outputs if t.get("target_path", "").endswith(".tsx")}
        matched = 0
        for page in aspx_pages:
            component = _pascal_case(re.sub(r"\.(aspx|ascx|master)$", "", page, flags=re.IGNORECASE))
            if any(component in tsx for tsx in tsx_files):
                matched += 1

        coverage = matched / len(aspx_pages) if aspx_pages else 1.0
        return GateResult(
            gate_name="page_component_parity",
            passed=coverage >= 0.7,
            details={
                "aspx_count": len(aspx_pages),
                "component_count": matched,
                "coverage": round(coverage, 2),
            },
            blocking=False,
        )

    @staticmethod
    def _gate_event_handler_coverage(
        source_units: List[Dict[str, Any]],
        target_outputs: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> GateResult:
        """Every WebForms event handler → controller action."""
        handlers: Set[str] = set()
        for u in source_units:
            if u.get("unit_type") in ("aspx_page", "aspx_master"):
                for h in u.get("metadata", {}).get("event_handlers", []):
                    if h not in _WEBFORMS_LIFECYCLE_HANDLERS:
                        handlers.add(h)

        target_code = "\n".join(t.get("target_code", "") for t in target_outputs)
        matched = {h for h in handlers if re.sub(r"_Click$|_Submit$|^btn", "", h) in target_code}

        coverage = len(matched) / len(handlers) if handlers else 1.0
        return GateResult(
            gate_name="event_handler_coverage",
            passed=coverage >= 0.6,
            details={
                "handler_count": len(handlers),
                "matched_count": len(matched),
                "coverage": round(coverage, 2),
                "missing": sorted(handlers - matched),
            },
            blocking=False,
        )

    @staticmethod
    def _gate_namespace_preservation(
        source_units: List[Dict[str, Any]],
        target_outputs: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> GateResult:
        """VB.NET namespaces → C# namespaces."""
        source_ns: Set[str] = set()
        for u in source_units:
            if u.get("language", "").lower() == "vbnet":
                qname = u.get("qualified_name", "")
                parts = qname.rsplit(".", 1)
                if len(parts) > 1:
                    source_ns.add(parts[0])

        target_code = "\n".join(t.get("target_code", "") for t in target_outputs)
        matched = {ns for ns in source_ns if ns in target_code}

        coverage = len(matched) / len(source_ns) if source_ns else 1.0
        return GateResult(
            gate_name="namespace_preservation",
            passed=coverage >= 0.5,
            details={
                "source_ns_count": len(source_ns),
                "matched_count": len(matched),
                "coverage": round(coverage, 2),
                "missing": sorted(source_ns - matched),
            },
            blocking=False,
        )

    # ═══════════════════════════════════════════════════════════════
    # Asset Strategy Overrides
    # ═══════════════════════════════════════════════════════════════

    def get_asset_strategy_overrides(self) -> Dict[str, Dict[str, Any]]:
        return {
            "module": {"strategy": "transform", "priority": "high"},
            "class": {"strategy": "transform", "priority": "high"},
            "interface": {"strategy": "transform", "priority": "high"},
            "struct": {"strategy": "transform", "priority": "medium"},
            "enum": {"strategy": "transform", "priority": "medium"},
            "aspx_page": {"strategy": "transform", "priority": "high"},
            "aspx_master": {"strategy": "transform", "priority": "high"},
            "aspx_control": {"strategy": "transform", "priority": "medium"},
            "method": {"strategy": "transform", "priority": "medium"},
            "property": {"strategy": "transform", "priority": "medium"},
            "event": {"strategy": "transform", "priority": "low"},
        }

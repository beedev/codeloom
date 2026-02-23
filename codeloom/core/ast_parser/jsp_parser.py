"""JSP file parser — regex-based.

Extracts a single CodeUnit per JSP file with rich metadata about:
- Taglib directives (prefix + uri)
- Bean references (useBean, html:form)
- Form actions (html:form, s:form, s:url)
- Tile references (tiles:insert, tiles:put)
- Includes (<%@ include, jsp:include)
- EL expressions (${...})
- Struts 1.x tag usage (html:*, bean:*, logic:*, nested:*)
- Struts 2.x tag usage (s:*)

Does NOT use tree-sitter. Follows the regex/fallback parser pattern
with parse_file/parse_source interface.
"""

import logging
import re
from typing import Any, Dict, List, Set

from .models import CodeUnit, ParseResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Taglib directives: <%@ taglib prefix="html" uri="..." %>
_TAGLIB_RE = re.compile(
    r"<%@\s*taglib\s+.*?%>",
    re.IGNORECASE | re.DOTALL,
)
_TAGLIB_PREFIX_RE = re.compile(r'prefix\s*=\s*"([^"]*)"', re.IGNORECASE)
_TAGLIB_URI_RE = re.compile(r'uri\s*=\s*"([^"]*)"', re.IGNORECASE)

# Bean references
_USE_BEAN_RE = re.compile(r"<jsp:useBean[^>]*>", re.IGNORECASE)
_USE_BEAN_CLASS_RE = re.compile(r'class\s*=\s*"([^"]*)"', re.IGNORECASE)
_HTML_FORM_BEAN_RE = re.compile(r"<html:form[^>]*>", re.IGNORECASE)

# Form actions
_HTML_FORM_ACTION_RE = re.compile(r'<html:form\s+[^>]*action\s*=\s*"([^"]*)"', re.IGNORECASE)
_S_FORM_ACTION_RE = re.compile(r'<s:form\s+[^>]*action\s*=\s*"([^"]*)"', re.IGNORECASE)
_S_URL_ACTION_RE = re.compile(r'<s:url\s+[^>]*action\s*=\s*"([^"]*)"', re.IGNORECASE)

# Tile references
_TILES_INSERT_RE = re.compile(
    r"<(?:tiles|tiles-el):insert[^>]*>",
    re.IGNORECASE,
)
_TILES_PUT_RE = re.compile(
    r"<(?:tiles|tiles-el):put[^>]*>",
    re.IGNORECASE,
)
_TILES_ATTR_RE = re.compile(r'(?:attribute|definition|name)\s*=\s*"([^"]*)"', re.IGNORECASE)

# Includes
_DIRECTIVE_INCLUDE_RE = re.compile(r'<%@\s*include\s+file\s*=\s*"([^"]*)"', re.IGNORECASE)
_JSP_INCLUDE_RE = re.compile(r'<jsp:include\s+page\s*=\s*"([^"]*)"', re.IGNORECASE)

# EL expressions: ${...}
_EL_RE = re.compile(r"\$\{([^}]+)\}")

# Struts 1.x tags: <html:...>, <bean:...>, <logic:...>, <nested:...>
_STRUTS1_TAG_RE = re.compile(r"<((?:html|bean|logic|nested):\w+)", re.IGNORECASE)

# Struts 2.x tags: <s:...>
_STRUTS2_TAG_RE = re.compile(r"<(s:\w+)", re.IGNORECASE)


class JspParser:
    """Regex-based JSP parser extracting page-level metadata.

    Each JSP file becomes one CodeUnit of type 'jsp_page' with metadata
    describing taglibs, bean references, form actions, tile references,
    includes, EL expressions, and Struts tag usage.
    """

    def get_language(self) -> str:
        return "jsp"

    def parse_file(self, file_path: str, project_root: str = "") -> ParseResult:
        """Parse a JSP file into a single CodeUnit."""
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
                language="jsp",
                units=[],
                imports=[],
                line_count=0,
            )

        return self.parse_source(source_text, rel_path)

    def parse_source(self, source_text: str, file_path: str) -> ParseResult:
        """Parse JSP source text into a single CodeUnit with rich metadata."""
        line_count = source_text.count("\n") + (
            1 if source_text and not source_text.endswith("\n") else 0
        )

        # --- Taglib directives ---
        taglibs: List[Dict[str, str]] = []
        for m in _TAGLIB_RE.finditer(source_text):
            directive = m.group(0)
            prefix_m = _TAGLIB_PREFIX_RE.search(directive)
            uri_m = _TAGLIB_URI_RE.search(directive)
            taglibs.append({
                "prefix": prefix_m.group(1) if prefix_m else "",
                "uri": uri_m.group(1) if uri_m else "",
            })

        # --- Bean references ---
        bean_refs: Set[str] = set()
        for m in _USE_BEAN_RE.finditer(source_text):
            cls_m = _USE_BEAN_CLASS_RE.search(m.group(0))
            if cls_m:
                bean_refs.add(cls_m.group(1))
        for m in _HTML_FORM_BEAN_RE.finditer(source_text):
            cls_m = _USE_BEAN_CLASS_RE.search(m.group(0))
            if cls_m:
                bean_refs.add(cls_m.group(1))

        # --- Form actions ---
        form_actions: Set[str] = set()
        for m in _HTML_FORM_ACTION_RE.finditer(source_text):
            form_actions.add(m.group(1))
        for m in _S_FORM_ACTION_RE.finditer(source_text):
            form_actions.add(m.group(1))
        for m in _S_URL_ACTION_RE.finditer(source_text):
            form_actions.add(m.group(1))

        # --- Tile references ---
        tile_refs: Set[str] = set()
        for m in _TILES_INSERT_RE.finditer(source_text):
            attr_m = _TILES_ATTR_RE.search(m.group(0))
            if attr_m:
                tile_refs.add(attr_m.group(1))
        for m in _TILES_PUT_RE.finditer(source_text):
            attr_m = _TILES_ATTR_RE.search(m.group(0))
            if attr_m:
                tile_refs.add(attr_m.group(1))

        # --- Includes ---
        includes: Set[str] = set()
        for m in _DIRECTIVE_INCLUDE_RE.finditer(source_text):
            includes.add(m.group(1))
        for m in _JSP_INCLUDE_RE.finditer(source_text):
            includes.add(m.group(1))

        # --- EL expressions ---
        el_refs: Set[str] = set()
        for m in _EL_RE.finditer(source_text):
            el_refs.add(m.group(1).strip())

        # --- Struts 1.x tags ---
        struts_tags: Set[str] = set()
        for m in _STRUTS1_TAG_RE.finditer(source_text):
            struts_tags.add(m.group(1).lower())

        # --- Struts 2.x tags ---
        struts2_tags: Set[str] = set()
        for m in _STRUTS2_TAG_RE.finditer(source_text):
            struts2_tags.add(m.group(1).lower())

        metadata: Dict[str, Any] = {
            "taglibs": taglibs,
            "bean_refs": sorted(bean_refs),
            "form_actions": sorted(form_actions),
            "tile_refs": sorted(tile_refs),
            "includes": sorted(includes),
            "el_refs": sorted(el_refs),
            "struts_tags": sorted(struts_tags),
            "struts2_tags": sorted(struts2_tags),
        }

        # Build a concise signature
        tag_summary_parts: List[str] = []
        if struts_tags:
            tag_summary_parts.append(f"struts1[{len(struts_tags)}]")
        if struts2_tags:
            tag_summary_parts.append(f"struts2[{len(struts2_tags)}]")
        if form_actions:
            tag_summary_parts.append(f"actions={','.join(sorted(form_actions))}")
        tag_summary = " ".join(tag_summary_parts) if tag_summary_parts else "jsp"
        signature = f"jsp_page {file_path} {tag_summary}"

        unit = CodeUnit(
            unit_type="jsp_page",
            name=file_path,
            qualified_name=f"{file_path}:page",
            language="jsp",
            start_line=1,
            end_line=line_count,
            source=source_text,
            file_path=file_path,
            signature=signature,
            metadata=metadata,
        )

        return ParseResult(
            file_path=file_path,
            language="jsp",
            units=[unit],
            imports=[],
            line_count=line_count,
        )

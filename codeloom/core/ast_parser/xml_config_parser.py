"""XML configuration file parser — ElementTree-based.

Extracts structured CodeUnits from Java EE XML config files:
- struts-config.xml (Struts 1.x): actions, form-beans, global-forwards, plugins
- struts.xml (Struts 2.x): actions, results, interceptors
- validation.xml: field validation rules
- tiles-defs.xml: tile definitions
- web.xml: servlets, filters, listeners
- pom.xml: framework-relevant dependencies

Does NOT use tree-sitter. Follows the regex/fallback parser pattern
with parse_file/parse_source interface.
"""

import logging
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

from .models import CodeUnit, ParseError, ParseResult

logger = logging.getLogger(__name__)


def _strip_namespace(tag: str) -> str:
    """Remove XML namespace prefix from a tag.

    '{http://java.sun.com/xml/ns/javaee}web-app' -> 'web-app'
    """
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def _local_find(element: ET.Element, local_name: str) -> Optional[ET.Element]:
    """Find a child element by local name, ignoring namespaces."""
    for child in element:
        if _strip_namespace(child.tag) == local_name:
            return child
    return None


def _local_findall(element: ET.Element, local_name: str) -> List[ET.Element]:
    """Find all child elements by local name, ignoring namespaces."""
    return [
        child for child in element
        if _strip_namespace(child.tag) == local_name
    ]


def _local_findall_recursive(element: ET.Element, local_name: str) -> List[ET.Element]:
    """Find all descendant elements by local name, ignoring namespaces."""
    return [
        node for node in element.iter()
        if _strip_namespace(node.tag) == local_name
    ]


def _get_text(element: ET.Element, child_name: str) -> str:
    """Get the text of a named child element, or empty string."""
    child = _local_find(element, child_name)
    if child is not None and child.text:
        return child.text.strip()
    return ""


def _element_source(element: ET.Element) -> str:
    """Serialize an element back to its XML string representation."""
    try:
        return ET.tostring(element, encoding="unicode", short_empty_elements=True)
    except Exception:
        return ""


def _line_number_of(source_text: str, substring: str, start: int = 0) -> int:
    """Approximate line number for a substring within source text.

    Returns 1-based line number or 0 if not found.
    """
    idx = source_text.find(substring, start)
    if idx < 0:
        return 0
    return source_text[:idx].count("\n") + 1


class XmlConfigParser:
    """Parse Java EE XML configuration files into structured CodeUnit objects.

    Produces CodeUnit objects compatible with the ingestion pipeline.
    Does not subclass BaseLanguageParser (no tree-sitter dependency).
    """

    def get_language(self) -> str:
        return "xml_config"

    def parse_file(self, file_path: str, project_root: str = "") -> ParseResult:
        """Parse an XML config file into structured CodeUnit objects."""
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
                language="xml_config",
                units=[],
                imports=[],
                line_count=0,
            )

        return self.parse_source(source_text, rel_path)

    def parse_source(self, source_text: str, file_path: str) -> ParseResult:
        """Parse XML source text into CodeUnit objects."""
        line_count = source_text.count("\n") + (
            1 if source_text and not source_text.endswith("\n") else 0
        )

        errors: List[ParseError] = []
        units: List[CodeUnit] = []

        try:
            root = ET.fromstring(source_text)
        except ET.ParseError as e:
            logger.warning("Malformed XML in %s: %s", file_path, e)
            errors.append(ParseError(
                file_path=file_path,
                line=0,
                message=f"XML parse error: {e}",
                severity="error",
            ))
            return ParseResult(
                file_path=file_path,
                language="xml_config",
                units=[],
                imports=[],
                line_count=line_count,
                errors=errors,
            )

        root_tag = _strip_namespace(root.tag)

        if root_tag == "struts-config":
            units = self._parse_struts1(root, source_text, file_path)
        elif root_tag == "struts" or _local_findall(root, "package"):
            units = self._parse_struts2(root, source_text, file_path)
        elif root_tag == "form-validation":
            units = self._parse_validation(root, source_text, file_path)
        elif root_tag == "tiles-definitions":
            units = self._parse_tiles(root, source_text, file_path)
        elif root_tag == "web-app":
            units = self._parse_web_xml(root, source_text, file_path)
        elif root_tag == "project" and self._is_maven_pom(root):
            units = self._parse_pom(root, source_text, file_path)
        else:
            logger.debug("Unrecognized XML root element <%s> in %s", root_tag, file_path)

        return ParseResult(
            file_path=file_path,
            language="xml_config",
            units=units,
            imports=[],
            line_count=line_count,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Struts 1.x
    # ------------------------------------------------------------------

    def _parse_struts1(
        self, root: ET.Element, source_text: str, file_path: str
    ) -> List[CodeUnit]:
        """Extract CodeUnits from a Struts 1.x struts-config.xml."""
        units: List[CodeUnit] = []

        # Action mappings
        for mappings in _local_findall(root, "action-mappings"):
            for action in _local_findall(mappings, "action"):
                path = action.get("path", "")
                action_type = action.get("type", "")
                name = action.get("name", "")
                scope = action.get("scope", "")
                validate = action.get("validate", "")
                input_attr = action.get("input", "")
                parameter = action.get("parameter", "")

                forwards: List[Dict[str, str]] = []
                for fwd in _local_findall(action, "forward"):
                    forwards.append({
                        "name": fwd.get("name", ""),
                        "path": fwd.get("path", ""),
                        "redirect": fwd.get("redirect", "false"),
                    })

                metadata: Dict[str, Any] = {
                    "path": path,
                    "type": action_type,
                    "name": name,
                    "scope": scope,
                    "validate": validate,
                    "input": input_attr,
                    "parameter": parameter,
                    "forwards": forwards,
                }

                unit_name = path or action_type
                sig = f"action path={path} type={action_type}"
                start = _line_number_of(source_text, f'path="{path}"') if path else 0

                units.append(CodeUnit(
                    unit_type="struts_action",
                    name=unit_name,
                    qualified_name=f"{file_path}:{unit_name}",
                    language="xml_config",
                    start_line=start,
                    end_line=start,
                    source=_element_source(action),
                    file_path=file_path,
                    signature=sig,
                    metadata=metadata,
                ))

        # Form beans
        for beans in _local_findall(root, "form-beans"):
            for bean in _local_findall(beans, "form-bean"):
                bean_name = bean.get("name", "")
                bean_type = bean.get("type", "")

                metadata = {"name": bean_name, "type": bean_type}
                start = _line_number_of(source_text, f'name="{bean_name}"') if bean_name else 0

                units.append(CodeUnit(
                    unit_type="struts_form_bean",
                    name=bean_name,
                    qualified_name=f"{file_path}:{bean_name}",
                    language="xml_config",
                    start_line=start,
                    end_line=start,
                    source=_element_source(bean),
                    file_path=file_path,
                    signature=f"form-bean name={bean_name} type={bean_type}",
                    metadata=metadata,
                ))

        # Global forwards
        for gf in _local_findall(root, "global-forwards"):
            for fwd in _local_findall(gf, "forward"):
                fwd_name = fwd.get("name", "")
                fwd_path = fwd.get("path", "")
                redirect = fwd.get("redirect", "false")

                metadata = {
                    "name": fwd_name,
                    "path": fwd_path,
                    "redirect": redirect,
                }
                start = _line_number_of(source_text, f'name="{fwd_name}"') if fwd_name else 0

                units.append(CodeUnit(
                    unit_type="struts_forward",
                    name=fwd_name,
                    qualified_name=f"{file_path}:{fwd_name}",
                    language="xml_config",
                    start_line=start,
                    end_line=start,
                    source=_element_source(fwd),
                    file_path=file_path,
                    signature=f"forward name={fwd_name} path={fwd_path}",
                    metadata=metadata,
                ))

        # Plugins
        for plugin in _local_findall(root, "plug-in"):
            class_name = plugin.get("className", "")

            metadata = {"class_name": class_name}
            start = _line_number_of(source_text, f'className="{class_name}"') if class_name else 0

            units.append(CodeUnit(
                unit_type="struts_plugin",
                name=class_name,
                qualified_name=f"{file_path}:{class_name}",
                language="xml_config",
                start_line=start,
                end_line=start,
                source=_element_source(plugin),
                file_path=file_path,
                signature=f"plug-in className={class_name}",
                metadata=metadata,
            ))

        return units

    # ------------------------------------------------------------------
    # Struts 2.x
    # ------------------------------------------------------------------

    def _parse_struts2(
        self, root: ET.Element, source_text: str, file_path: str
    ) -> List[CodeUnit]:
        """Extract CodeUnits from a Struts 2.x struts.xml."""
        units: List[CodeUnit] = []

        for pkg in _local_findall_recursive(root, "package"):
            namespace = pkg.get("namespace", "/")

            for action in _local_findall(pkg, "action"):
                action_name = action.get("name", "")
                action_class = action.get("class", "")
                action_method = action.get("method", "")

                results: List[Dict[str, str]] = []
                for result in _local_findall(action, "result"):
                    results.append({
                        "name": result.get("name", "success"),
                        "type": result.get("type", ""),
                        "value": (result.text or "").strip(),
                    })

                interceptor_refs: List[str] = []
                for iref in _local_findall(action, "interceptor-ref"):
                    ref_name = iref.get("name", "")
                    if ref_name:
                        interceptor_refs.append(ref_name)

                metadata: Dict[str, Any] = {
                    "name": action_name,
                    "class": action_class,
                    "method": action_method,
                    "namespace": namespace,
                    "results": results,
                    "interceptor_refs": interceptor_refs,
                }

                start = _line_number_of(source_text, f'name="{action_name}"') if action_name else 0
                sig = f"struts2 action name={action_name} class={action_class}"

                units.append(CodeUnit(
                    unit_type="struts2_action",
                    name=action_name,
                    qualified_name=f"{file_path}:{action_name}",
                    language="xml_config",
                    start_line=start,
                    end_line=start,
                    source=_element_source(action),
                    file_path=file_path,
                    signature=sig,
                    metadata=metadata,
                ))

        # Standalone result type definitions
        for result_type in _local_findall_recursive(root, "result-type"):
            rt_name = result_type.get("name", "")
            rt_class = result_type.get("class", "")

            metadata = {"name": rt_name, "type": rt_class, "value": ""}
            start = _line_number_of(source_text, f'name="{rt_name}"') if rt_name else 0

            units.append(CodeUnit(
                unit_type="struts2_result",
                name=rt_name,
                qualified_name=f"{file_path}:{rt_name}",
                language="xml_config",
                start_line=start,
                end_line=start,
                source=_element_source(result_type),
                file_path=file_path,
                signature=f"struts2 result-type name={rt_name} class={rt_class}",
                metadata=metadata,
            ))

        # Interceptor definitions
        for interceptor in _local_findall_recursive(root, "interceptor"):
            ic_name = interceptor.get("name", "")
            ic_class = interceptor.get("class", "")

            metadata = {"name": ic_name, "class": ic_class}
            start = _line_number_of(source_text, f'name="{ic_name}"') if ic_name else 0

            units.append(CodeUnit(
                unit_type="struts2_interceptor",
                name=ic_name,
                qualified_name=f"{file_path}:{ic_name}",
                language="xml_config",
                start_line=start,
                end_line=start,
                source=_element_source(interceptor),
                file_path=file_path,
                signature=f"struts2 interceptor name={ic_name} class={ic_class}",
                metadata=metadata,
            ))

        return units

    # ------------------------------------------------------------------
    # Struts validation.xml
    # ------------------------------------------------------------------

    def _parse_validation(
        self, root: ET.Element, source_text: str, file_path: str
    ) -> List[CodeUnit]:
        """Extract CodeUnits from a Struts validation.xml."""
        units: List[CodeUnit] = []

        for formset in _local_findall_recursive(root, "formset"):
            for form in _local_findall(formset, "form"):
                form_name = form.get("name", "")

                for field_elem in _local_findall(form, "field"):
                    field_name = field_elem.get("property", "")
                    depends = field_elem.get("depends", "")

                    # Each depends value is a comma-separated list of validators
                    validator_types = [
                        v.strip() for v in depends.split(",") if v.strip()
                    ]

                    # Collect var params
                    params: Dict[str, str] = {}
                    for var_elem in _local_findall(field_elem, "var"):
                        var_name = _get_text(var_elem, "var-name")
                        var_value = _get_text(var_elem, "var-value")
                        if var_name:
                            params[var_name] = var_value

                    unit_name = f"{form_name}.{field_name}"
                    metadata: Dict[str, Any] = {
                        "form_name": form_name,
                        "field_name": field_name,
                        "validator_type": validator_types,
                        "params": params,
                    }

                    start = _line_number_of(source_text, f'property="{field_name}"') if field_name else 0

                    units.append(CodeUnit(
                        unit_type="struts_validation_rule",
                        name=unit_name,
                        qualified_name=f"{file_path}:{unit_name}",
                        language="xml_config",
                        start_line=start,
                        end_line=start,
                        source=_element_source(field_elem),
                        file_path=file_path,
                        signature=f"validation form={form_name} field={field_name} depends={depends}",
                        metadata=metadata,
                    ))

        return units

    # ------------------------------------------------------------------
    # Tiles definitions
    # ------------------------------------------------------------------

    def _parse_tiles(
        self, root: ET.Element, source_text: str, file_path: str
    ) -> List[CodeUnit]:
        """Extract CodeUnits from a Tiles definitions file."""
        units: List[CodeUnit] = []

        for defn in _local_findall_recursive(root, "definition"):
            def_name = defn.get("name", "")
            def_path = defn.get("path", "") or defn.get("template", "")
            extends = defn.get("extends", "")

            attributes: List[Dict[str, str]] = []
            for put in _local_findall(defn, "put-attribute"):
                attributes.append({
                    "name": put.get("name", ""),
                    "value": put.get("value", ""),
                    "type": put.get("type", ""),
                })
            # Struts 1.x tiles use <put> not <put-attribute>
            for put in _local_findall(defn, "put"):
                attributes.append({
                    "name": put.get("name", ""),
                    "value": put.get("value", ""),
                    "type": put.get("type", ""),
                })

            metadata: Dict[str, Any] = {
                "name": def_name,
                "path": def_path,
                "extends": extends,
                "attributes": attributes,
            }

            start = _line_number_of(source_text, f'name="{def_name}"') if def_name else 0

            units.append(CodeUnit(
                unit_type="struts_tile_def",
                name=def_name,
                qualified_name=f"{file_path}:{def_name}",
                language="xml_config",
                start_line=start,
                end_line=start,
                source=_element_source(defn),
                file_path=file_path,
                signature=f"tiles definition name={def_name} path={def_path}",
                metadata=metadata,
            ))

        return units

    # ------------------------------------------------------------------
    # web.xml
    # ------------------------------------------------------------------

    def _parse_web_xml(
        self, root: ET.Element, source_text: str, file_path: str
    ) -> List[CodeUnit]:
        """Extract CodeUnits from a web.xml deployment descriptor."""
        units: List[CodeUnit] = []

        # Build servlet-name -> url-pattern mapping from <servlet-mapping>
        servlet_mappings: Dict[str, List[str]] = {}
        for mapping in _local_findall_recursive(root, "servlet-mapping"):
            sname = _get_text(mapping, "servlet-name")
            for url_pat in _local_findall(mapping, "url-pattern"):
                pattern = (url_pat.text or "").strip()
                if sname and pattern:
                    servlet_mappings.setdefault(sname, []).append(pattern)

        # Servlets
        for servlet in _local_findall_recursive(root, "servlet"):
            sname = _get_text(servlet, "servlet-name")
            sclass = _get_text(servlet, "servlet-class")
            load = _get_text(servlet, "load-on-startup")

            init_params: Dict[str, str] = {}
            for param in _local_findall(servlet, "init-param"):
                pname = _get_text(param, "param-name")
                pvalue = _get_text(param, "param-value")
                if pname:
                    init_params[pname] = pvalue

            url_patterns = servlet_mappings.get(sname, [])

            metadata: Dict[str, Any] = {
                "name": sname,
                "class_name": sclass,
                "url_patterns": url_patterns,
                "init_params": init_params,
                "load_on_startup": load,
            }

            start = _line_number_of(source_text, sname) if sname else 0

            units.append(CodeUnit(
                unit_type="xml_servlet",
                name=sname,
                qualified_name=f"{file_path}:{sname}",
                language="xml_config",
                start_line=start,
                end_line=start,
                source=_element_source(servlet),
                file_path=file_path,
                signature=f"servlet name={sname} class={sclass}",
                metadata=metadata,
            ))

        # Build filter-name -> url-pattern mapping from <filter-mapping>
        filter_mappings: Dict[str, List[str]] = {}
        for mapping in _local_findall_recursive(root, "filter-mapping"):
            fname = _get_text(mapping, "filter-name")
            for url_pat in _local_findall(mapping, "url-pattern"):
                pattern = (url_pat.text or "").strip()
                if fname and pattern:
                    filter_mappings.setdefault(fname, []).append(pattern)

        # Filters
        for filt in _local_findall_recursive(root, "filter"):
            fname = _get_text(filt, "filter-name")
            fclass = _get_text(filt, "filter-class")

            url_patterns = filter_mappings.get(fname, [])

            metadata = {
                "name": fname,
                "class_name": fclass,
                "url_patterns": url_patterns,
            }

            start = _line_number_of(source_text, fname) if fname else 0

            units.append(CodeUnit(
                unit_type="xml_filter",
                name=fname,
                qualified_name=f"{file_path}:{fname}",
                language="xml_config",
                start_line=start,
                end_line=start,
                source=_element_source(filt),
                file_path=file_path,
                signature=f"filter name={fname} class={fclass}",
                metadata=metadata,
            ))

        # Listeners
        for listener in _local_findall_recursive(root, "listener"):
            lclass = _get_text(listener, "listener-class")
            if not lclass:
                continue

            metadata = {"class_name": lclass}
            start = _line_number_of(source_text, lclass) if lclass else 0

            units.append(CodeUnit(
                unit_type="xml_listener",
                name=lclass,
                qualified_name=f"{file_path}:{lclass}",
                language="xml_config",
                start_line=start,
                end_line=start,
                source=_element_source(listener),
                file_path=file_path,
                signature=f"listener class={lclass}",
                metadata=metadata,
            ))

        return units

    # ------------------------------------------------------------------
    # pom.xml
    # ------------------------------------------------------------------

    def _is_maven_pom(self, root: ET.Element) -> bool:
        """Check if a <project> element is a Maven POM by looking for Maven namespace."""
        tag = root.tag
        if "maven" in tag.lower():
            return True
        # Also detect by presence of typical POM children
        return (
            _local_find(root, "groupId") is not None
            or _local_find(root, "artifactId") is not None
            or _local_find(root, "dependencies") is not None
        )

    def _parse_pom(
        self, root: ET.Element, source_text: str, file_path: str
    ) -> List[CodeUnit]:
        """Extract framework-relevant dependencies from a Maven pom.xml.

        Only extracts dependencies whose groupId contains 'struts' or 'spring'.
        """
        units: List[CodeUnit] = []
        relevant_keywords = ("struts", "spring")

        for deps_elem in _local_findall_recursive(root, "dependencies"):
            for dep in _local_findall(deps_elem, "dependency"):
                group_id = _get_text(dep, "groupId")
                artifact_id = _get_text(dep, "artifactId")
                version = _get_text(dep, "version")
                scope = _get_text(dep, "scope")

                group_lower = group_id.lower()
                if not any(kw in group_lower for kw in relevant_keywords):
                    continue

                dep_name = f"{group_id}:{artifact_id}"
                metadata: Dict[str, Any] = {
                    "group_id": group_id,
                    "artifact_id": artifact_id,
                    "version": version,
                    "scope": scope,
                }

                start = _line_number_of(source_text, artifact_id) if artifact_id else 0

                units.append(CodeUnit(
                    unit_type="xml_dependency",
                    name=dep_name,
                    qualified_name=f"{file_path}:{dep_name}",
                    language="xml_config",
                    start_line=start,
                    end_line=start,
                    source=_element_source(dep),
                    file_path=file_path,
                    signature=f"dependency {group_id}:{artifact_id}:{version}",
                    metadata=metadata,
                ))

        return units

"""Struts 1.x/2.x to Spring Boot migration lane.

Deterministic transforms, prompt augmentation, quality gates, and asset
strategy overrides for migrating legacy Struts applications to Spring Boot.

Supports three view-layer targets:
  * ``rest``       -- ``@RestController`` + DTOs (default)
  * ``thymeleaf``  -- ``@Controller`` + ``@ModelAttribute`` + Thymeleaf
  * ``react``      -- ``@RestController`` + DTOs (frontend separate)
"""

import logging
import re
from typing import Any, Dict, List

from .base import (
    GateDefinition,
    GateResult,
    MigrationLane,
    TransformResult,
    TransformRule,
)

logger = logging.getLogger(__name__)


# ── Helpers ─────────────────────────────────────────────────────────


def _pascal_case(name: str) -> str:
    """Convert a slash/dash/underscore-delimited name to PascalCase."""
    parts = re.split(r"[/\-_.]+", name.strip("/"))
    return "".join(p.capitalize() for p in parts if p)


def _camel_case(name: str) -> str:
    """Convert to camelCase."""
    pascal = _pascal_case(name)
    return pascal[0].lower() + pascal[1:] if pascal else ""


def _extract_action_path(unit: Dict[str, Any]) -> str:
    """Extract the URL path from a Struts action unit's metadata."""
    meta = unit.get("metadata", {})
    return (
        meta.get("path")
        or meta.get("action_path")
        or meta.get("name", "")
        or f"/{unit.get('name', 'unknown')}"
    )


def _extract_fields(unit: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract field definitions from a form-bean or DTO unit."""
    meta = unit.get("metadata", {})
    fields = meta.get("fields", [])
    if isinstance(fields, list):
        return [
            {"name": f.get("name", "field"), "type": f.get("type", "String")}
            for f in fields
            if isinstance(f, dict)
        ]
    return []


def _view_layer(context: Dict[str, Any]) -> str:
    """Resolve the target view layer from migration context."""
    target_stack = context.get("target_stack", {})
    return target_stack.get("view_layer", "rest")


# ── Lane Implementation ────────────────────────────────────────────


class StrutsToSpringBootLane(MigrationLane):
    """Migration lane for Struts 1.x/2.x to Spring Boot.

    Covers actions, form beans, validation, Tiles, interceptors,
    message resources, and web.xml servlet/filter configuration.
    """

    # ── Identity ────────────────────────────────────────────────

    @property
    def lane_id(self) -> str:
        return "struts_to_springboot"

    @property
    def display_name(self) -> str:
        return "Struts 1.x/2.x \u2192 Spring Boot"

    @property
    def source_frameworks(self) -> List[str]:
        return ["struts1", "struts2"]

    @property
    def target_frameworks(self) -> List[str]:
        return ["springboot"]

    # ── Applicability ───────────────────────────────────────────

    def detect_applicability(
        self, source_framework: str, target_stack: Dict[str, Any]
    ) -> float:
        source_lower = source_framework.lower()
        if source_lower not in {"struts1", "struts2"}:
            return 0.0

        # Check if target explicitly mentions Spring Boot
        target_fw = str(target_stack.get("framework", "")).lower()
        target_name = str(target_stack.get("name", "")).lower()
        combined = f"{target_fw} {target_name}"

        if "spring" in combined or "springboot" in combined:
            return 0.95

        # Source matches but no explicit Spring Boot target
        return 0.5

    # ── Transform Rules ─────────────────────────────────────────

    def get_transform_rules(self) -> List[TransformRule]:
        return [
            TransformRule(
                name="action_to_controller",
                source_pattern={"unit_type": "struts_action"},
                target_template="spring_controller_method",
                confidence=0.9,
                description=(
                    "Convert Struts Action (1.x) or Struts2 Action to "
                    "Spring @Controller/@RestController method."
                ),
            ),
            TransformRule(
                name="action_to_controller",
                source_pattern={"unit_type": "struts2_action"},
                target_template="spring_controller_method",
                confidence=0.9,
                description=(
                    "Convert Struts2 Action class to Spring controller method."
                ),
            ),
            TransformRule(
                name="form_bean_to_pojo",
                source_pattern={"unit_type": "struts_form_bean"},
                target_template="spring_pojo_dto",
                confidence=0.95,
                description=(
                    "Convert Struts ActionForm / DynaActionForm to "
                    "Spring POJO DTO."
                ),
            ),
            TransformRule(
                name="validation_to_jsr303",
                source_pattern={"unit_type": "struts_validation_rule"},
                target_template="jsr303_annotation",
                confidence=0.85,
                description=(
                    "Convert Struts validation.xml rules to JSR-303 "
                    "bean-validation annotations."
                ),
            ),
            TransformRule(
                name="tiles_to_thymeleaf",
                source_pattern={"unit_type": "struts_tile_def"},
                target_template="thymeleaf_layout",
                confidence=0.7,
                requires_review=True,
                description=(
                    "Convert Tiles definition to Thymeleaf layout fragment."
                ),
            ),
            TransformRule(
                name="message_resources",
                source_pattern={"unit_type": "properties_file"},
                target_template="spring_message_source",
                confidence=1.0,
                description=(
                    "Map Struts MessageResources .properties to Spring "
                    "MessageSource (files are compatible, just re-register)."
                ),
            ),
            TransformRule(
                name="web_xml_to_boot",
                source_pattern={"unit_type": "xml_servlet"},
                target_template="spring_boot_config",
                confidence=0.8,
                description=(
                    "Convert web.xml servlet declaration to Spring Boot "
                    "@Bean registration."
                ),
            ),
            TransformRule(
                name="web_xml_to_boot",
                source_pattern={"unit_type": "xml_filter"},
                target_template="spring_boot_config",
                confidence=0.8,
                description=(
                    "Convert web.xml filter declaration to Spring Boot "
                    "@Bean filter registration."
                ),
            ),
            TransformRule(
                name="struts2_interceptor_to_filter",
                source_pattern={"unit_type": "struts2_interceptor"},
                target_template="spring_handler_interceptor",
                confidence=0.75,
                requires_review=True,
                description=(
                    "Convert Struts2 Interceptor to Spring "
                    "HandlerInterceptor implementation."
                ),
            ),
            TransformRule(
                name="forward_to_view_resolver",
                source_pattern={"unit_type": "struts_forward"},
                target_template="spring_view_return",
                confidence=0.85,
                description=(
                    "Convert Struts ActionForward to Spring view return "
                    "or redirect."
                ),
            ),
            TransformRule(
                name="forward_to_view_resolver",
                source_pattern={"unit_type": "struts2_result"},
                target_template="spring_view_return",
                confidence=0.85,
                description=(
                    "Convert Struts2 Result to Spring view return or redirect."
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
        view = _view_layer(context)

        # Build a lookup: unit_type -> list of matching rules
        rules_by_type: Dict[str, TransformRule] = {}
        for rule in self.get_transform_rules():
            ut = rule.source_pattern.get("unit_type", "")
            if ut and ut not in rules_by_type:
                rules_by_type[ut] = rule

        for unit in units:
            unit_type = unit.get("unit_type") or unit.get("metadata", {}).get(
                "unit_type", ""
            )
            rule = rules_by_type.get(unit_type)
            if rule is None:
                continue

            result = self._apply_single_transform(unit, unit_type, rule, view)
            if result is not None:
                results.append(result)

        logger.info(
            "StrutsToSpringBoot: applied %d transforms to %d units",
            len(results),
            len(units),
        )
        return results

    def _apply_single_transform(
        self,
        unit: Dict[str, Any],
        unit_type: str,
        rule: TransformRule,
        view: str,
    ) -> TransformResult | None:
        """Dispatch to the appropriate code generator for *unit_type*."""
        unit_id = str(unit.get("id", unit.get("name", "unknown")))
        name = unit.get("name", "Unknown")

        if unit_type in {"struts_action", "struts2_action"}:
            return self._transform_action(unit, unit_id, name, rule, view)
        if unit_type == "struts_form_bean":
            return self._transform_form_bean(unit, unit_id, name, rule)
        if unit_type == "struts_validation_rule":
            return self._transform_validation(unit, unit_id, name, rule)
        if unit_type == "struts_tile_def":
            return self._transform_tile(unit, unit_id, name, rule)
        if unit_type == "properties_file":
            return self._transform_properties(unit, unit_id, name, rule)
        if unit_type in {"xml_servlet", "xml_filter"}:
            return self._transform_web_xml(unit, unit_id, name, rule, unit_type)
        if unit_type == "struts2_interceptor":
            return self._transform_interceptor(unit, unit_id, name, rule)
        if unit_type in {"struts_forward", "struts2_result"}:
            return self._transform_forward(unit, unit_id, name, rule, view)

        return None

    # ── Individual Transform Generators ─────────────────────────

    def _transform_action(
        self,
        unit: Dict[str, Any],
        unit_id: str,
        name: str,
        rule: TransformRule,
        view: str,
    ) -> TransformResult:
        path = _extract_action_path(unit)
        class_name = _pascal_case(name) + "Controller"
        method_name = _camel_case(name)

        if view in {"rest", "react"}:
            annotation = "@RestController"
            return_type = "ResponseEntity<?>"
            return_stmt = "return ResponseEntity.ok(result);"
        else:
            annotation = "@Controller"
            return_type = "String"
            return_stmt = f'return "{_camel_case(name)}";'

        code = (
            f"import org.springframework.web.bind.annotation.*;\n\n"
            f"{annotation}\n"
            f'@RequestMapping("{path}")\n'
            f"public class {class_name} {{\n\n"
            f"    @GetMapping\n"
            f"    public {return_type} {method_name}() {{\n"
            f"        // TODO: migrate business logic from {name}\n"
            f"        {return_stmt}\n"
            f"    }}\n"
            f"}}\n"
        )

        pkg = path.strip("/").replace("/", ".")
        target_path = f"src/main/java/com/app/controller/{class_name}.java"

        return TransformResult(
            source_unit_id=unit_id,
            target_code=code,
            target_path=target_path,
            rule_name=rule.name,
            confidence=rule.confidence,
            notes=[
                f"Mapped action path '{path}' to @RequestMapping",
                f"View layer: {view}",
                f"Package suggestion: com.app.controller.{pkg}"
                if pkg
                else "",
            ],
        )

    def _transform_form_bean(
        self,
        unit: Dict[str, Any],
        unit_id: str,
        name: str,
        rule: TransformRule,
    ) -> TransformResult:
        class_name = _pascal_case(name)
        if not class_name.endswith("Dto") and not class_name.endswith("DTO"):
            class_name += "Dto"
        fields = _extract_fields(unit)

        lines = [
            "import lombok.Data;\n",
            "@Data",
            f"public class {class_name} {{\n",
        ]
        for fld in fields:
            lines.append(f"    private {fld['type']} {fld['name']};")

        if not fields:
            lines.append(
                "    // TODO: add fields from original ActionForm"
            )

        lines.append("}\n")
        code = "\n".join(lines)

        target_path = f"src/main/java/com/app/dto/{class_name}.java"

        return TransformResult(
            source_unit_id=unit_id,
            target_code=code,
            target_path=target_path,
            rule_name=rule.name,
            confidence=rule.confidence,
            notes=[
                f"Converted ActionForm '{name}' to Lombok @Data DTO",
                f"Fields extracted: {len(fields)}",
            ],
        )

    def _transform_validation(
        self,
        unit: Dict[str, Any],
        unit_id: str,
        name: str,
        rule: TransformRule,
    ) -> TransformResult:
        meta = unit.get("metadata", {})
        vtype = meta.get("validation_type", "required")
        field_name = meta.get("field_name", name)

        annotation_map = {
            "required": "@NotNull",
            "minlength": "@Size(min = {min})",
            "maxlength": "@Size(max = {max})",
            "mask": "@Pattern(regexp = \"{pattern}\")",
            "email": "@Email",
            "integer": "@Digits(integer = 10, fraction = 0)",
            "double": "@Digits(integer = 10, fraction = 2)",
            "date": "@DateTimeFormat(pattern = \"{pattern}\")",
            "range": "@Min({min}) @Max({max})",
        }

        annotation = annotation_map.get(vtype, f"@NotNull // was: {vtype}")

        # Substitute params from metadata
        for key in ("min", "max", "pattern"):
            annotation = annotation.replace(
                "{" + key + "}", str(meta.get(key, ""))
            )

        code = (
            f"// Field: {field_name}\n"
            f"// Original Struts validation type: {vtype}\n"
            f"{annotation}\n"
            f"private String {_camel_case(field_name)};\n"
        )

        return TransformResult(
            source_unit_id=unit_id,
            target_code=code,
            target_path="(inline annotation -- merge into DTO class)",
            rule_name=rule.name,
            confidence=rule.confidence,
            notes=[
                f"Validation '{vtype}' -> {annotation}",
                "Merge this snippet into the corresponding DTO field",
            ],
        )

    def _transform_tile(
        self,
        unit: Dict[str, Any],
        unit_id: str,
        name: str,
        rule: TransformRule,
    ) -> TransformResult:
        meta = unit.get("metadata", {})
        template = meta.get("template", "layout")
        puts = meta.get("put_attributes", {})

        # Build a Thymeleaf layout fragment
        sections = "\n".join(
            f'    <div th:replace="~{{{v}}}" />'
            f"  <!-- put: {k} -->"
            for k, v in puts.items()
        ) if isinstance(puts, dict) and puts else (
            "    <!-- TODO: migrate put-attributes -->"
        )

        code = (
            f"<!DOCTYPE html>\n"
            f'<html xmlns:th="http://www.thymeleaf.org"\n'
            f'      xmlns:layout="http://www.ultraq.net.nz/thymeleaf/layout">\n'
            f"<head>\n"
            f"    <title th:text=\"${{title}}\">Page</title>\n"
            f"</head>\n"
            f"<body>\n"
            f'{sections}\n'
            f'    <div layout:fragment="content">\n'
            f"        <!-- page content -->\n"
            f"    </div>\n"
            f"</body>\n"
            f"</html>\n"
        )

        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        target_path = f"src/main/resources/templates/{safe_name}.html"

        return TransformResult(
            source_unit_id=unit_id,
            target_code=code,
            target_path=target_path,
            rule_name=rule.name,
            confidence=rule.confidence,
            notes=[
                f"Tiles definition '{name}' -> Thymeleaf layout",
                f"Original template: {template}",
                "Requires manual review of layout composition",
            ],
        )

    def _transform_properties(
        self,
        unit: Dict[str, Any],
        unit_id: str,
        name: str,
        rule: TransformRule,
    ) -> TransformResult:
        code = (
            f"# Spring MessageSource -- migrated from Struts MessageResources\n"
            f"# Original file: {name}\n"
            f"# No content changes needed; register via:\n"
            f"#   spring.messages.basename={name.replace('.properties', '')}\n"
        )

        target_path = f"src/main/resources/{name}"

        return TransformResult(
            source_unit_id=unit_id,
            target_code=code,
            target_path=target_path,
            rule_name=rule.name,
            confidence=rule.confidence,
            notes=[
                "Properties files are compatible between Struts and Spring",
                "Register basename in application.properties",
            ],
        )

    def _transform_web_xml(
        self,
        unit: Dict[str, Any],
        unit_id: str,
        name: str,
        rule: TransformRule,
        unit_type: str,
    ) -> TransformResult:
        meta = unit.get("metadata", {})
        class_ref = meta.get("class", meta.get("filter_class", name))
        class_name = _pascal_case(name)

        if unit_type == "xml_filter":
            code = (
                f"import org.springframework.boot.web.servlet.FilterRegistrationBean;\n"
                f"import org.springframework.context.annotation.Bean;\n"
                f"import org.springframework.context.annotation.Configuration;\n\n"
                f"@Configuration\n"
                f"public class {class_name}Config {{\n\n"
                f"    @Bean\n"
                f"    public FilterRegistrationBean<{class_name}> "
                f"{_camel_case(name)}Filter() {{\n"
                f"        FilterRegistrationBean<{class_name}> bean =\n"
                f"            new FilterRegistrationBean<>();\n"
                f"        bean.setFilter(new {class_name}());\n"
                f'        bean.addUrlPatterns("/*");\n'
                f"        return bean;\n"
                f"    }}\n"
                f"}}\n"
            )
        else:
            code = (
                f"import org.springframework.boot.web.servlet.ServletRegistrationBean;\n"
                f"import org.springframework.context.annotation.Bean;\n"
                f"import org.springframework.context.annotation.Configuration;\n\n"
                f"@Configuration\n"
                f"public class {class_name}Config {{\n\n"
                f"    @Bean\n"
                f"    public ServletRegistrationBean<{class_name}> "
                f"{_camel_case(name)}Servlet() {{\n"
                f"        return new ServletRegistrationBean<>(\n"
                f'            new {class_name}(), "/*");\n'
                f"    }}\n"
                f"}}\n"
            )

        target_path = f"src/main/java/com/app/config/{class_name}Config.java"

        return TransformResult(
            source_unit_id=unit_id,
            target_code=code,
            target_path=target_path,
            rule_name=rule.name,
            confidence=rule.confidence,
            notes=[
                f"web.xml {unit_type} '{name}' -> Spring @Bean config",
                f"Original class: {class_ref}",
            ],
        )

    def _transform_interceptor(
        self,
        unit: Dict[str, Any],
        unit_id: str,
        name: str,
        rule: TransformRule,
    ) -> TransformResult:
        class_name = _pascal_case(name)
        if not class_name.endswith("Interceptor"):
            class_name += "Interceptor"

        code = (
            f"import javax.servlet.http.HttpServletRequest;\n"
            f"import javax.servlet.http.HttpServletResponse;\n"
            f"import org.springframework.web.servlet.HandlerInterceptor;\n"
            f"import org.springframework.web.servlet.ModelAndView;\n\n"
            f"public class {class_name} implements HandlerInterceptor {{\n\n"
            f"    @Override\n"
            f"    public boolean preHandle(\n"
            f"            HttpServletRequest request,\n"
            f"            HttpServletResponse response,\n"
            f"            Object handler) throws Exception {{\n"
            f"        // TODO: migrate from Struts2 intercept()\n"
            f"        return true;\n"
            f"    }}\n\n"
            f"    @Override\n"
            f"    public void postHandle(\n"
            f"            HttpServletRequest request,\n"
            f"            HttpServletResponse response,\n"
            f"            Object handler,\n"
            f"            ModelAndView modelAndView) throws Exception {{\n"
            f"        // TODO: post-processing logic\n"
            f"    }}\n\n"
            f"    @Override\n"
            f"    public void afterCompletion(\n"
            f"            HttpServletRequest request,\n"
            f"            HttpServletResponse response,\n"
            f"            Object handler,\n"
            f"            Exception ex) throws Exception {{\n"
            f"        // TODO: cleanup logic\n"
            f"    }}\n"
            f"}}\n"
        )

        target_path = (
            f"src/main/java/com/app/interceptor/{class_name}.java"
        )

        return TransformResult(
            source_unit_id=unit_id,
            target_code=code,
            target_path=target_path,
            rule_name=rule.name,
            confidence=rule.confidence,
            notes=[
                f"Struts2 Interceptor '{name}' -> Spring HandlerInterceptor",
                "Register in WebMvcConfigurer.addInterceptors()",
                "Requires manual migration of intercept() body",
            ],
        )

    def _transform_forward(
        self,
        unit: Dict[str, Any],
        unit_id: str,
        name: str,
        rule: TransformRule,
        view: str,
    ) -> TransformResult:
        meta = unit.get("metadata", {})
        target_view = meta.get("path", meta.get("location", name))
        is_redirect = meta.get("redirect", False)

        if is_redirect:
            stmt = f'return "redirect:{target_view}";'
        elif view in {"rest", "react"}:
            stmt = f'return ResponseEntity.ok("{target_view}");'
        else:
            # Strip .jsp extension for Thymeleaf resolution
            view_name = re.sub(r"\.jsp$", "", target_view).strip("/")
            stmt = f'return "{view_name}";'

        code = (
            f"// Forward/Result: {name}\n"
            f"// Original target: {target_view}\n"
            f"{stmt}\n"
        )

        return TransformResult(
            source_unit_id=unit_id,
            target_code=code,
            target_path="(inline -- merge into controller method)",
            rule_name=rule.name,
            confidence=rule.confidence,
            notes=[
                f"Forward '{name}' -> view return statement",
                f"Redirect: {is_redirect}",
            ],
        )

    # ── Prompt Augmentation ─────────────────────────────────────

    def augment_prompt(
        self,
        phase_type: str,
        base_prompt: str,
        context: Dict[str, Any],
    ) -> str:
        if phase_type == "transform":
            return self._augment_transform(base_prompt, context)
        if phase_type == "architecture":
            return self._augment_architecture(base_prompt, context)
        if phase_type == "test":
            return self._augment_test(base_prompt, context)
        return base_prompt

    def _augment_transform(
        self, base_prompt: str, context: Dict[str, Any]
    ) -> str:
        transform_results = context.get("transform_results", [])
        if not transform_results:
            return base_prompt

        summaries: List[str] = []
        for tr in transform_results:
            if isinstance(tr, TransformResult):
                summaries.append(
                    f"- {tr.rule_name}: {tr.source_unit_id} -> "
                    f"{tr.target_path} (confidence {tr.confidence})"
                )
            elif isinstance(tr, dict):
                summaries.append(
                    f"- {tr.get('rule_name', '?')}: "
                    f"{tr.get('source_unit_id', '?')} -> "
                    f"{tr.get('target_path', '?')} "
                    f"(confidence {tr.get('confidence', '?')})"
                )

        block = "\n".join(summaries)
        return (
            f"{base_prompt}\n\n"
            f"## Pre-completed Deterministic Transforms\n\n"
            f"The following transforms have already been applied "
            f"deterministically.  Do NOT regenerate these; focus on the "
            f"remaining units that were not covered.\n\n"
            f"{block}\n"
        )

    def _augment_architecture(
        self, base_prompt: str, context: Dict[str, Any]
    ) -> str:
        mapping_table = (
            "## Struts-to-Spring Mapping Reference\n\n"
            "| Struts Concept | Spring Boot Equivalent |\n"
            "|---|---|\n"
            "| Action / Action class | @Controller / @RestController method |\n"
            "| ActionForm / DynaActionForm | POJO DTO (+ @Data Lombok) |\n"
            "| validation.xml rules | JSR-303 annotations (@NotNull, @Size, etc.) |\n"
            "| Tiles definitions | Thymeleaf layouts / fragments |\n"
            "| MessageResources .properties | Spring MessageSource |\n"
            "| web.xml servlet/filter | @Bean registration in @Configuration |\n"
            "| Struts2 Interceptor | Spring HandlerInterceptor |\n"
            "| ActionForward / Result | Controller return value / redirect |\n"
            "| struts-config.xml | @RequestMapping + application.properties |\n"
            "| struts.xml (Struts2) | @RequestMapping + @Configuration |\n"
        )
        return f"{base_prompt}\n\n{mapping_table}"

    def _augment_test(
        self, base_prompt: str, context: Dict[str, Any]
    ) -> str:
        transform_results = context.get("transform_results", [])
        endpoints: List[str] = []

        for tr in transform_results:
            path = ""
            if isinstance(tr, TransformResult):
                # Try to extract @RequestMapping path from generated code
                match = re.search(
                    r'@RequestMapping\("([^"]+)"\)', tr.target_code
                )
                if match:
                    path = match.group(1)
            elif isinstance(tr, dict):
                path = tr.get("path", "")

            if path:
                endpoints.append(path)

        if not endpoints:
            return base_prompt

        ep_list = "\n".join(f"- `{ep}`" for ep in sorted(set(endpoints)))
        return (
            f"{base_prompt}\n\n"
            f"## Endpoint Coverage Validation\n\n"
            f"Ensure tests exist for each of the following migrated "
            f"endpoints:\n\n{ep_list}\n"
        )

    # ── Quality Gates ───────────────────────────────────────────

    def get_gates(self) -> List[GateDefinition]:
        return [
            GateDefinition(
                name="endpoint_parity",
                description=(
                    "Every Struts action path has a corresponding "
                    "@RequestMapping in the generated Spring code."
                ),
                blocking=True,
            ),
            GateDefinition(
                name="form_field_parity",
                description=(
                    "Every ActionForm field exists in the target "
                    "POJO/DTO class."
                ),
                blocking=True,
            ),
            GateDefinition(
                name="validation_coverage",
                description=(
                    "Every validation.xml rule has a corresponding "
                    "JSR-303 annotation."
                ),
                blocking=False,
            ),
            GateDefinition(
                name="message_key_coverage",
                description=(
                    "Every message resource key from the source is "
                    "preserved in the target."
                ),
                blocking=False,
            ),
        ]

    def run_gate(
        self,
        gate_name: str,
        source_units: List[Dict[str, Any]],
        target_outputs: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> GateResult:
        gate_map = {
            "endpoint_parity": self._gate_endpoint_parity,
            "form_field_parity": self._gate_form_field_parity,
            "validation_coverage": self._gate_validation_coverage,
            "message_key_coverage": self._gate_message_key_coverage,
        }

        gate_def = {g.name: g for g in self.get_gates()}.get(gate_name)
        handler = gate_map.get(gate_name)

        if handler is None or gate_def is None:
            return GateResult(
                gate_name=gate_name,
                passed=False,
                details={"error": f"Unknown gate: {gate_name}"},
                blocking=True,
            )

        return handler(source_units, target_outputs, gate_def)

    def _gate_endpoint_parity(
        self,
        source_units: List[Dict[str, Any]],
        target_outputs: List[Dict[str, Any]],
        gate_def: GateDefinition,
    ) -> GateResult:
        source_paths: set[str] = set()
        for u in source_units:
            ut = u.get("unit_type", "")
            if ut in {"struts_action", "struts2_action"}:
                source_paths.add(_extract_action_path(u))

        target_paths: set[str] = set()
        for t in target_outputs:
            code = t.get("target_code", "")
            for match in re.finditer(r'@RequestMapping\("([^"]+)"\)', code):
                target_paths.add(match.group(1))

        missing = source_paths - target_paths
        return GateResult(
            gate_name=gate_def.name,
            passed=len(missing) == 0,
            details={
                "source_paths": sorted(source_paths),
                "target_paths": sorted(target_paths),
                "missing": sorted(missing),
                "coverage": (
                    len(source_paths - missing) / len(source_paths)
                    if source_paths
                    else 1.0
                ),
            },
            blocking=gate_def.blocking,
        )

    def _gate_form_field_parity(
        self,
        source_units: List[Dict[str, Any]],
        target_outputs: List[Dict[str, Any]],
        gate_def: GateDefinition,
    ) -> GateResult:
        source_fields: set[str] = set()
        for u in source_units:
            if u.get("unit_type") == "struts_form_bean":
                for fld in _extract_fields(u):
                    source_fields.add(fld["name"])

        target_fields: set[str] = set()
        for t in target_outputs:
            code = t.get("target_code", "")
            for match in re.finditer(r"private\s+\S+\s+(\w+);", code):
                target_fields.add(match.group(1))

        missing = source_fields - target_fields
        return GateResult(
            gate_name=gate_def.name,
            passed=len(missing) == 0,
            details={
                "source_fields": sorted(source_fields),
                "target_fields": sorted(target_fields),
                "missing": sorted(missing),
                "coverage": (
                    len(source_fields - missing) / len(source_fields)
                    if source_fields
                    else 1.0
                ),
            },
            blocking=gate_def.blocking,
        )

    def _gate_validation_coverage(
        self,
        source_units: List[Dict[str, Any]],
        target_outputs: List[Dict[str, Any]],
        gate_def: GateDefinition,
    ) -> GateResult:
        source_rules: set[str] = set()
        for u in source_units:
            if u.get("unit_type") == "struts_validation_rule":
                field_name = u.get("metadata", {}).get(
                    "field_name", u.get("name", "")
                )
                source_rules.add(field_name)

        covered: set[str] = set()
        for t in target_outputs:
            code = t.get("target_code", "")
            if "@Not" in code or "@Size" in code or "@Email" in code:
                # Extract the field name from the snippet
                match = re.search(r"private\s+\S+\s+(\w+);", code)
                if match:
                    covered.add(match.group(1))

        missing = source_rules - covered
        return GateResult(
            gate_name=gate_def.name,
            passed=len(missing) == 0,
            details={
                "source_rules": sorted(source_rules),
                "covered": sorted(covered),
                "missing": sorted(missing),
                "coverage": (
                    len(source_rules - missing) / len(source_rules)
                    if source_rules
                    else 1.0
                ),
            },
            blocking=gate_def.blocking,
        )

    def _gate_message_key_coverage(
        self,
        source_units: List[Dict[str, Any]],
        target_outputs: List[Dict[str, Any]],
        gate_def: GateDefinition,
    ) -> GateResult:
        source_keys: set[str] = set()
        for u in source_units:
            if u.get("unit_type") == "properties_file":
                keys = u.get("metadata", {}).get("keys", [])
                if isinstance(keys, list):
                    source_keys.update(keys)

        target_keys: set[str] = set()
        for t in target_outputs:
            keys = t.get("metadata", {}).get("keys", [])
            if isinstance(keys, list):
                target_keys.update(keys)

        missing = source_keys - target_keys
        return GateResult(
            gate_name=gate_def.name,
            passed=len(missing) == 0,
            details={
                "source_keys": sorted(source_keys),
                "target_keys": sorted(target_keys),
                "missing": sorted(missing),
                "coverage": (
                    len(source_keys - missing) / len(source_keys)
                    if source_keys
                    else 1.0
                ),
            },
            blocking=gate_def.blocking,
        )

    # ── Asset Strategy Overrides ────────────────────────────────

    def get_asset_strategy_overrides(self) -> Dict[str, Dict[str, Any]]:
        return {
            "struts_action": {"strategy": "transform", "priority": "high"},
            "struts2_action": {"strategy": "transform", "priority": "high"},
            "struts_form_bean": {"strategy": "transform", "priority": "high"},
            "struts_validation_rule": {
                "strategy": "transform",
                "priority": "medium",
            },
            "struts_tile_def": {
                "strategy": "transform",
                "priority": "medium",
            },
            "xml_servlet": {"strategy": "transform", "priority": "low"},
            "xml_filter": {"strategy": "transform", "priority": "low"},
            "struts2_interceptor": {
                "strategy": "transform",
                "priority": "medium",
            },
            "properties_file": {
                "strategy": "copy_with_edits",
                "priority": "low",
            },
            "jsp_page": {"strategy": "rewrite", "priority": "high"},
        }

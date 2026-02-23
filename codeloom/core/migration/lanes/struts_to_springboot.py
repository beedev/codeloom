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


# ── Service Layer Classification ──────────────────────────────────

_LAYER_ACTION = "action"
_LAYER_SERVICE = "service"
_LAYER_DAO = "dao"
_LAYER_UNKNOWN = "unknown"

_ACTION_BASES = frozenset({
    "Action", "DispatchAction", "MappingDispatchAction",
    "LookupDispatchAction", "ActionSupport", "BaseAction",
    "AbstractAction",
})

_SERVICE_ANNOTATIONS = frozenset({
    "@Service", "@Transactional", "@Component",
})

_DAO_ANNOTATIONS = frozenset({
    "@Repository", "@PersistenceContext",
})

_DAO_BASES = frozenset({
    "HibernateDaoSupport", "JdbcDaoSupport", "SqlMapClientDaoSupport",
    "JpaRepository", "CrudRepository",
})

_SERVICE_NAME_SUFFIXES = ("Service", "ServiceImpl", "Svc", "Manager")
_DAO_NAME_SUFFIXES = ("DAO", "Dao", "Repository", "DaoImpl")

_DB_ACCESS_RE = re.compile(
    r"(?:getSession\b|getHibernateTemplate\b|createQuery\b|createSQLQuery\b"
    r"|PreparedStatement\b|ResultSet\b|Connection\s*="
    r"|EntityManager\b|\.persist\(|\.merge\(|\.find\("
    r"|JdbcTemplate\b|NamedParameterJdbcTemplate\b)",
    re.IGNORECASE,
)

# Regex to extract public method signatures from Java source
_JAVA_METHOD_SIG_RE = re.compile(
    r"public\s+(\S+)\s+(\w+)\s*\(([^)]*)\)",
)

# Struts framework methods to skip when extracting business methods
_FRAMEWORK_METHODS = frozenset({
    "execute", "perform", "doExecute", "unspecified",
    "cancelled", "init", "destroy", "reset", "validate",
})


def _classify_java_unit(unit: Dict[str, Any]) -> str:
    """Classify a Java class/interface unit by architectural layer.

    Priority: annotations > extends > name suffix > unknown.
    """
    if unit.get("unit_type") not in ("class", "interface"):
        return _LAYER_UNKNOWN
    if unit.get("language", "").lower() != "java":
        return _LAYER_UNKNOWN

    meta = unit.get("metadata", {})
    annotations = set(meta.get("annotations", []))
    extends = meta.get("extends") or ""
    name = unit.get("name", "")

    # 1. Annotations
    if annotations & _SERVICE_ANNOTATIONS:
        return _LAYER_SERVICE
    if annotations & _DAO_ANNOTATIONS:
        return _LAYER_DAO

    # 2. Extends chain
    if extends in _ACTION_BASES:
        return _LAYER_ACTION
    if extends in _DAO_BASES:
        return _LAYER_DAO

    # 3. Name suffix heuristic
    if name.endswith(_DAO_NAME_SUFFIXES):
        return _LAYER_DAO
    if name.endswith(_SERVICE_NAME_SUFFIXES):
        return _LAYER_SERVICE

    return _LAYER_UNKNOWN


# ── Lane Implementation ────────────────────────────────────────────


class StrutsToSpringBootLane(MigrationLane):
    """Migration lane for Struts 1.x/2.x to Spring Boot.

    Covers actions, form beans, validation, Tiles, interceptors,
    message resources, web.xml servlet/filter configuration, and
    JSP-to-React component generation.
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

    # ── Service Layer Indexing ──────────────────────────────────

    @staticmethod
    def _build_layer_index(
        units: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build cross-reference indexes for service layer extraction.

        Returns a dict with:
          ``by_name``      -- simple class name → list of unit dicts
          ``by_layer``     -- layer constant → list of unit dicts
          ``classified``   -- unit id → layer constant
        """
        by_name: Dict[str, List[Dict[str, Any]]] = {}
        by_layer: Dict[str, List[Dict[str, Any]]] = {
            _LAYER_ACTION: [],
            _LAYER_SERVICE: [],
            _LAYER_DAO: [],
            _LAYER_UNKNOWN: [],
        }
        classified: Dict[str, str] = {}

        for u in units:
            if u.get("language", "").lower() != "java":
                continue
            if u.get("unit_type") not in ("class", "interface"):
                continue

            layer = _classify_java_unit(u)
            uid = str(u.get("id", u.get("name", "")))
            name = u.get("name", "")

            by_name.setdefault(name, []).append(u)
            by_layer[layer].append(u)
            classified[uid] = layer

        return {
            "by_name": by_name,
            "by_layer": by_layer,
            "classified": classified,
        }

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
                name="action_to_service",
                source_pattern={"unit_type": "struts_action"},
                target_template="spring_service_class",
                confidence=0.80,
                description=(
                    "Extract/generate service layer from Struts Action "
                    "business logic."
                ),
            ),
            TransformRule(
                name="action_to_repository",
                source_pattern={"unit_type": "struts_action"},
                target_template="spring_repository_interface",
                confidence=0.70,
                requires_review=True,
                description=(
                    "Generate JPA repository from Action's data access "
                    "patterns."
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
            TransformRule(
                name="jsp_to_react_component",
                source_pattern={"unit_type": "jsp_page"},
                target_template="react_functional_component",
                confidence=0.75,
                requires_review=True,
                description=(
                    "Generate React functional component with hooks and "
                    "TypeScript from JSP page metadata. Produces component, "
                    "API service, and types files."
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

        # Pre-pass: classify Java units for service layer extraction
        layer_index = self._build_layer_index(units)
        generated_services: set = set()  # dedup: service class names
        generated_repos: set = set()  # dedup: repository class names

        for unit in units:
            unit_type = unit.get("unit_type") or unit.get("metadata", {}).get(
                "unit_type", ""
            )

            # JSP pages produce multiple outputs when view_layer is react/rest
            if unit_type == "jsp_page" and view in {"react", "rest"}:
                jsp_results = self._transform_jsp_page(unit, view)
                results.extend(jsp_results)
                continue

            # Struts actions → layered output (Controller + Service + Repository)
            if unit_type in {"struts_action", "struts2_action"}:
                rule = rules_by_type.get(unit_type)
                if rule:
                    unit_id = str(unit.get("id", unit.get("name", "unknown")))
                    name = unit.get("name", "Unknown")
                    try:
                        layered = self._transform_action_layered(
                            unit, unit_id, name, rule, view,
                            layer_index, generated_services, generated_repos,
                        )
                        results.extend(layered)
                    except Exception:
                        logger.warning(
                            "Layered transform failed for '%s', using fallback",
                            name,
                            exc_info=True,
                        )
                        fallback = self._transform_action(
                            unit, unit_id, name, rule, view,
                        )
                        results.append(fallback)
                continue

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

    # ── Layered Action Transforms ─────────────────────────────

    @staticmethod
    def _find_action_java_class(
        unit: Dict[str, Any],
        layer_index: Dict[str, Any],
    ) -> Dict[str, Any] | None:
        """Resolve the backing Java class for a struts_action / struts2_action."""
        meta = unit.get("metadata", {})
        # Struts 1: metadata["type"], Struts 2: metadata["class"]
        fqcn = meta.get("type") or meta.get("class") or ""
        if not fqcn:
            return None
        simple_name = fqcn.rsplit(".", 1)[-1]
        candidates = layer_index.get("by_name", {}).get(simple_name, [])
        return candidates[0] if candidates else None

    @staticmethod
    def _find_action_dependencies(
        action_class: Dict[str, Any] | None,
        layer_index: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Find service and DAO classes used by an Action.

        Returns ``{"services": [...], "daos": [...], "has_direct_db": bool}``.
        """
        services: List[Dict[str, Any]] = []
        daos: List[Dict[str, Any]] = []
        has_direct_db = False

        if action_class is None:
            return {"services": services, "daos": daos, "has_direct_db": False}

        meta = action_class.get("metadata", {})
        by_name = layer_index.get("by_name", {})

        # Check field types for service/DAO references
        for field in meta.get("fields", []):
            ftype = field.get("type", "")
            if not ftype:
                continue
            matched = by_name.get(ftype, [])
            for m in matched:
                layer = _classify_java_unit(m)
                if layer == _LAYER_SERVICE and m not in services:
                    services.append(m)
                elif layer == _LAYER_DAO and m not in daos:
                    daos.append(m)

        # Check source for direct DB access patterns
        source = action_class.get("source", "")
        if _DB_ACCESS_RE.search(source):
            has_direct_db = True

        return {"services": services, "daos": daos, "has_direct_db": has_direct_db}

    @staticmethod
    def _is_god_class_action(
        action_class: Dict[str, Any] | None,
        deps: Dict[str, Any],
    ) -> bool:
        """True if the Action is a god-class that needs service extraction."""
        if action_class is None:
            return True
        if not deps["services"] and not deps["daos"]:
            return True
        if deps["has_direct_db"] and not deps["daos"]:
            return True
        return False

    def _transform_action_layered(
        self,
        unit: Dict[str, Any],
        unit_id: str,
        name: str,
        rule: TransformRule,
        view: str,
        layer_index: Dict[str, Any],
        generated_services: set,
        generated_repos: set,
    ) -> List[TransformResult]:
        """Transform a Struts action into layered Spring Boot architecture.

        Produces 2-3 TransformResults:
          1. Controller (always) — thin routing layer
          2. Service (always) — business logic layer
          3. Repository (conditional) — data access layer

        Falls back to ``_transform_action()`` on classification failure.
        """
        results: List[TransformResult] = []
        action_class = self._find_action_java_class(unit, layer_index)
        deps = self._find_action_dependencies(action_class, layer_index)
        god_class = self._is_god_class_action(action_class, deps)

        action_path = _extract_action_path(unit)
        service_name = _pascal_case(name) + "Service"
        service_var = _camel_case(name) + "Service"
        repo_name = _pascal_case(name) + "Repository"

        # 1. Controller — always generated
        results.append(
            self._gen_controller(
                unit, unit_id, name, view,
                service_name, service_var, action_path, rule,
            )
        )

        # 2. Service layer
        if not god_class and deps["services"]:
            # Properly layered: convert existing service
            svc = deps["services"][0]
            svc_name = _pascal_case(svc["name"])
            if not svc_name.endswith("Service"):
                svc_name += "Service"
            if svc_name not in generated_services:
                generated_services.add(svc_name)
                results.append(
                    self._gen_service_from_existing(unit_id, svc, deps["daos"])
                )
        else:
            # God-class: generate service stub
            needs_repo = deps["has_direct_db"]
            if service_name not in generated_services:
                generated_services.add(service_name)
                results.append(
                    self._gen_service_stub(
                        unit_id, action_class, name, needs_repo, repo_name,
                    )
                )

        # 3. Repository layer (conditional)
        if not god_class and deps["daos"]:
            # Convert existing DAO
            dao = deps["daos"][0]
            r_name = _pascal_case(dao["name"])
            # Strip DAO/Impl suffix and add Repository
            for suffix in ("DaoImpl", "DAOImpl", "Impl", "DAO", "Dao"):
                if r_name.endswith(suffix):
                    r_name = r_name[: -len(suffix)]
                    break
            r_name += "Repository"
            if r_name not in generated_repos:
                generated_repos.add(r_name)
                results.append(self._gen_repo_from_existing(unit_id, dao))
        elif deps["has_direct_db"]:
            # God-class with DB access: generate repo stub
            if repo_name not in generated_repos:
                generated_repos.add(repo_name)
                results.append(self._gen_repo_stub(unit_id, name))

        return results

    def _gen_controller(
        self,
        unit: Dict[str, Any],
        unit_id: str,
        name: str,
        view: str,
        service_name: str,
        service_var: str,
        action_path: str,
        rule: TransformRule,
    ) -> TransformResult:
        """Generate a thin Spring Controller with constructor-injected service."""
        class_name = _pascal_case(name) + "Controller"
        method_name = _camel_case(name)

        if view in {"rest", "react"}:
            annotation = "@RestController"
            return_type = "ResponseEntity<?>"
            return_stmt = f"return ResponseEntity.ok({service_var}.execute());"
        else:
            annotation = "@Controller"
            return_type = "String"
            return_stmt = f'return "{_camel_case(name)}";'

        code = (
            f"import org.springframework.web.bind.annotation.*;\n\n"
            f"{annotation}\n"
            f'@RequestMapping("{action_path}")\n'
            f"public class {class_name} {{\n\n"
            f"    private final {service_name} {service_var};\n\n"
            f"    public {class_name}({service_name} {service_var}) {{\n"
            f"        this.{service_var} = {service_var};\n"
            f"    }}\n\n"
            f"    @GetMapping\n"
            f"    public {return_type} {method_name}() {{\n"
            f"        {return_stmt}\n"
            f"    }}\n"
            f"}}\n"
        )

        target_path = f"src/main/java/com/app/controller/{class_name}.java"

        return TransformResult(
            source_unit_id=unit_id,
            target_code=code,
            target_path=target_path,
            rule_name=rule.name,
            confidence=rule.confidence,
            notes=[
                f"Mapped action path '{action_path}' to @RequestMapping",
                f"View layer: {view}",
                f"Service layer: delegates to {service_name}",
            ],
        )

    @staticmethod
    def _gen_service_from_existing(
        action_unit_id: str,
        existing_service: Dict[str, Any],
        repo_deps: List[Dict[str, Any]],
    ) -> TransformResult:
        """Convert an existing Java Service class to Spring @Service."""
        svc_name = _pascal_case(existing_service["name"])
        if not svc_name.endswith("Service"):
            svc_name += "Service"

        # Build constructor injection for DAO/repo dependencies
        injections: List[str] = []
        ctor_params: List[str] = []
        ctor_assigns: List[str] = []
        for dao in repo_deps:
            dao_name = _pascal_case(dao["name"])
            for sfx in ("DaoImpl", "DAOImpl", "Impl", "DAO", "Dao"):
                if dao_name.endswith(sfx):
                    dao_name = dao_name[: -len(sfx)]
                    break
            repo_type = dao_name + "Repository"
            repo_var = _camel_case(dao_name) + "Repository"
            injections.append(f"    private final {repo_type} {repo_var};")
            ctor_params.append(f"{repo_type} {repo_var}")
            ctor_assigns.append(f"        this.{repo_var} = {repo_var};")

        lines = [
            "import org.springframework.stereotype.Service;",
            "import org.springframework.transaction.annotation.Transactional;\n",
            "@Service",
            "@Transactional",
            f"public class {svc_name} {{\n",
        ]
        for inj in injections:
            lines.append(inj)
        if injections:
            lines.append("")

        if ctor_params:
            lines.append(f"    public {svc_name}({', '.join(ctor_params)}) {{")
            lines.extend(ctor_assigns)
            lines.append("    }\n")

        # Extract method signatures from source
        source = existing_service.get("source", "")
        for m in _JAVA_METHOD_SIG_RE.finditer(source):
            ret, mname, params = m.group(1), m.group(2), m.group(3).strip()
            if mname in _FRAMEWORK_METHODS:
                continue
            param_str = params if params else ""
            lines.append(f"    public {ret} {mname}({param_str}) {{")
            lines.append(f"        // TODO: migrate business logic from {existing_service['name']}.{mname}")
            lines.append(f"        throw new UnsupportedOperationException(\"Not yet migrated\");")
            lines.append("    }\n")

        if not any("public" in ln and "(" in ln for ln in lines[6:]):
            lines.append("    // TODO: add service methods")

        lines.append("}\n")
        code = "\n".join(lines)

        target_path = f"src/main/java/com/app/service/{svc_name}.java"

        return TransformResult(
            source_unit_id=action_unit_id,
            target_code=code,
            target_path=target_path,
            rule_name="service_conversion",
            confidence=0.85,
            notes=[
                f"Service layer: converted existing {existing_service['name']} to Spring @Service",
                f"Repository deps: {len(repo_deps)}",
            ],
        )

    @staticmethod
    def _gen_service_stub(
        action_unit_id: str,
        action_class: Dict[str, Any] | None,
        action_name: str,
        needs_repo: bool,
        repo_name: str,
    ) -> TransformResult:
        """Generate a service stub for a god-class Action."""
        svc_name = _pascal_case(action_name) + "Service"
        repo_var = _camel_case(action_name) + "Repository"

        lines = [
            "import org.springframework.stereotype.Service;",
            "import org.springframework.transaction.annotation.Transactional;\n",
            "@Service",
            "@Transactional",
            f"public class {svc_name} {{\n",
        ]

        if needs_repo:
            lines.append(f"    private final {repo_name} {repo_var};\n")
            lines.append(f"    public {svc_name}({repo_name} {repo_var}) {{")
            lines.append(f"        this.{repo_var} = {repo_var};")
            lines.append("    }\n")

        # Extract business methods from Action source
        methods_added = False
        if action_class:
            source = action_class.get("source", "")
            for m in _JAVA_METHOD_SIG_RE.finditer(source):
                ret, mname, params = m.group(1), m.group(2), m.group(3).strip()
                if mname in _FRAMEWORK_METHODS:
                    continue
                param_str = params if params else ""
                lines.append(f"    public {ret} {mname}({param_str}) {{")
                lines.append(f"        // TODO: extract business logic from {action_name}.{mname}")
                lines.append(f"        throw new UnsupportedOperationException(\"Not yet migrated\");")
                lines.append("    }\n")
                methods_added = True

        if not methods_added:
            lines.append("    public Object execute() {")
            lines.append(f"        // TODO: extract business logic from {action_name}")
            lines.append("        throw new UnsupportedOperationException(\"Not yet migrated\");")
            lines.append("    }\n")

        lines.append("}\n")
        code = "\n".join(lines)

        target_path = f"src/main/java/com/app/service/{svc_name}.java"

        return TransformResult(
            source_unit_id=action_unit_id,
            target_code=code,
            target_path=target_path,
            rule_name="action_to_service",
            confidence=0.70,
            notes=[
                f"Service layer: generated stub from god-class Action '{action_name}'",
                f"Repository: {'injected ' + repo_name if needs_repo else 'none'}",
            ],
        )

    @staticmethod
    def _gen_repo_from_existing(
        action_unit_id: str,
        existing_dao: Dict[str, Any],
    ) -> TransformResult:
        """Convert an existing DAO class to a JPA @Repository interface."""
        dao_name = _pascal_case(existing_dao["name"])
        # Strip DAO/Impl suffix
        for suffix in ("DaoImpl", "DAOImpl", "Impl", "DAO", "Dao"):
            if dao_name.endswith(suffix):
                dao_name = dao_name[: -len(suffix)]
                break
        repo_name = dao_name + "Repository"
        entity_name = dao_name + "Entity"

        lines = [
            "import org.springframework.data.jpa.repository.JpaRepository;",
            "import org.springframework.stereotype.Repository;\n",
            "@Repository",
            f"public interface {repo_name} extends JpaRepository<{entity_name}, Long> {{\n",
        ]

        # Extract method signatures from DAO source for custom query methods
        source = existing_dao.get("source", "")
        for m in _JAVA_METHOD_SIG_RE.finditer(source):
            ret, mname, params = m.group(1), m.group(2), m.group(3).strip()
            if mname in _FRAMEWORK_METHODS or mname.startswith("get") and not params:
                continue
            param_str = params if params else ""
            lines.append(f"    // TODO: convert to Spring Data query method")
            lines.append(f"    // Original: {ret} {mname}({param_str})")
            lines.append("")

        if not any("TODO" in ln for ln in lines):
            lines.append("    // TODO: add custom query methods from original DAO")

        lines.append("}\n")
        code = "\n".join(lines)

        target_path = f"src/main/java/com/app/repository/{repo_name}.java"

        return TransformResult(
            source_unit_id=action_unit_id,
            target_code=code,
            target_path=target_path,
            rule_name="dao_to_repository",
            confidence=0.80,
            notes=[
                f"Repository: converted existing DAO '{existing_dao['name']}' to JPA interface",
                f"Entity: {entity_name} (needs @Entity class)",
            ],
        )

    @staticmethod
    def _gen_repo_stub(
        action_unit_id: str,
        action_name: str,
    ) -> TransformResult:
        """Generate a repository stub for a god-class Action with DB access."""
        repo_name = _pascal_case(action_name) + "Repository"
        entity_name = _pascal_case(action_name) + "Entity"

        code = (
            "import org.springframework.data.jpa.repository.JpaRepository;\n"
            "import org.springframework.stereotype.Repository;\n\n"
            "@Repository\n"
            f"public interface {repo_name} extends JpaRepository<{entity_name}, Long> {{\n\n"
            f"    // TODO: add query methods for {action_name} data access\n"
            "}\n"
        )

        target_path = f"src/main/java/com/app/repository/{repo_name}.java"

        return TransformResult(
            source_unit_id=action_unit_id,
            target_code=code,
            target_path=target_path,
            rule_name="action_to_repository",
            confidence=0.60,
            notes=[
                f"Repository: generated stub for god-class Action '{action_name}'",
                f"Entity: {entity_name} (needs @Entity class)",
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

    # ── JSP-to-React Transform Generators ──────────────────────

    @staticmethod
    def _jsp_name_to_component(jsp_path: str) -> str:
        """Convert a JSP file path to a React PascalCase component name.

        Strips common web-app prefixes (WEB-INF/, jsp/, pages/, views/) and
        the ``.jsp`` / ``.jspx`` extension, then PascalCases the remainder.

        Examples::

            WEB-INF/jsp/user/login.jsp  ->  UserLogin
            pages/admin/dashboard.jspx  ->  AdminDashboard
            header.jsp                  ->  Header
        """
        cleaned = re.sub(
            r"^(?:WEB-INF/|jsp/|pages/|views/)+",
            "",
            jsp_path,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\.jspx?$", "", cleaned, flags=re.IGNORECASE)
        return _pascal_case(cleaned)

    @staticmethod
    def _has_form_tags(s1_tags: set, s2_tags: set) -> bool:
        """Check whether the JSP uses Struts form tags."""
        return bool({"html:form"} & s1_tags or {"s:form"} & s2_tags)

    @staticmethod
    def _has_iteration_tags(s1_tags: set, s2_tags: set) -> bool:
        """Check whether the JSP uses Struts iteration tags."""
        return bool(
            {"logic:iterate", "nested:iterate"} & s1_tags
            or {"s:iterator"} & s2_tags
        )

    @staticmethod
    def _has_conditional_tags(s1_tags: set, s2_tags: set) -> bool:
        """Check whether the JSP uses Struts conditional tags."""
        return bool(
            {"logic:present", "logic:notpresent", "logic:equal"} & s1_tags
            or {"s:if", "s:elseif"} & s2_tags
        )

    @staticmethod
    def _has_link_tags(s1_tags: set, s2_tags: set) -> bool:
        """Check whether the JSP uses Struts link/URL tags."""
        return bool({"html:link"} & s1_tags or {"s:a", "s:url"} & s2_tags)

    @staticmethod
    def _has_error_tags(s1_tags: set, s2_tags: set) -> bool:
        """Check whether the JSP uses Struts error display tags."""
        return bool(
            {"html:errors"} & s1_tags
            or {"s:fielderror", "s:actionerror"} & s2_tags
        )

    # ── JSP orchestrator ───────────────────────────────────────

    def _transform_jsp_page(
        self, unit: Dict[str, Any], view: str
    ) -> List[TransformResult]:
        """Transform a ``jsp_page`` unit into one or more React outputs.

        Always produces a React component; conditionally adds an API
        service file and a TypeScript types file depending on metadata.
        """
        meta = unit.get("metadata", {})
        unit_id = str(unit.get("id", unit.get("name", "unknown")))
        jsp_path = unit.get("name", "unknown.jsp")

        # Extract metadata sets produced by JspParser
        struts_tags: set = set(meta.get("struts_tags", []))
        struts2_tags: set = set(meta.get("struts2_tags", []))
        el_refs: list = meta.get("el_refs", [])
        form_actions: list = meta.get("form_actions", [])
        bean_refs: list = meta.get("bean_refs", [])
        tile_refs: list = meta.get("tile_refs", [])
        includes: list = meta.get("includes", [])
        form_fields: list = meta.get("form_fields", [])
        angular_patterns: list = meta.get("angular_patterns", [])

        component_name = self._jsp_name_to_component(jsp_path)
        camel_name = component_name[0].lower() + component_name[1:] if component_name else ""

        # Assemble a shared meta dict for the sub-generators
        gen_meta: Dict[str, Any] = {
            "struts_tags": struts_tags,
            "struts2_tags": struts2_tags,
            "el_refs": el_refs,
            "form_actions": form_actions,
            "bean_refs": bean_refs,
            "tile_refs": tile_refs,
            "includes": includes,
            "form_fields": form_fields,
            "angular_patterns": angular_patterns,
        }

        results: List[TransformResult] = []

        # 1) Always generate the React component
        results.append(
            self._gen_react_component(
                unit_id, jsp_path, component_name, camel_name, gen_meta
            )
        )

        # 2) API service -- only when form actions exist
        if form_actions:
            results.append(
                self._gen_api_service(
                    unit_id, jsp_path, component_name, camel_name,
                    form_actions, gen_meta,
                )
            )

        # 3) Types file -- only when bean refs or EL refs exist
        if bean_refs or el_refs:
            results.append(
                self._gen_types_file(
                    unit_id, jsp_path, component_name, camel_name,
                    bean_refs, el_refs, gen_meta,
                )
            )

        return results

    # ── React component generator ──────────────────────────────

    def _gen_react_component(
        self,
        unit_id: str,
        jsp_path: str,
        component_name: str,
        camel_name: str,
        meta: Dict[str, Any],
    ) -> TransformResult:
        """Generate a React functional component ``.tsx`` file.

        Analyses JSP tag metadata to determine imports, state fields,
        event handlers, and JSX structure.
        """
        s1 = meta["struts_tags"]
        s2 = meta["struts2_tags"]
        el_refs: list = meta["el_refs"]
        form_actions: list = meta["form_actions"]
        tile_refs: list = meta["tile_refs"]
        includes: list = meta["includes"]
        form_fields: list = meta["form_fields"]
        angular_patterns: list = meta["angular_patterns"]

        has_form = self._has_form_tags(s1, s2)
        has_iteration = self._has_iteration_tags(s1, s2)
        has_conditionals = self._has_conditional_tags(s1, s2)
        has_links = self._has_link_tags(s1, s2)
        has_errors = self._has_error_tags(s1, s2)
        needs_data_loading = bool(el_refs)
        needs_submit = bool(form_actions)

        # ── Imports ────────────────────────────────────────────
        react_hooks: List[str] = []
        if needs_data_loading or form_fields:
            react_hooks.append("useState")
        if needs_data_loading:
            react_hooks.append("useEffect")

        imports: List[str] = []
        if react_hooks:
            imports.append(
                f"import React, {{ {', '.join(react_hooks)} }} from 'react';"
            )
        else:
            imports.append("import React from 'react';")

        if has_links:
            imports.append("import { Link } from 'react-router-dom';")

        if form_actions:
            imports.append(
                f"import {{ {camel_name}Api }} from '../services/{camel_name}Api';"
            )

        if meta["bean_refs"] or el_refs:
            imports.append(
                f"import {{ {component_name}Data }} from '../types/{camel_name}.types';"
            )

        # Layout / child component imports from tile_refs and includes
        for ref in tile_refs:
            ref_component = _pascal_case(ref)
            ref_camel = _camel_case(ref)
            imports.append(
                f"import {{ {ref_component} }} from './{ref_component}';"
            )
        for inc in includes:
            inc_component = self._jsp_name_to_component(inc)
            imports.append(
                f"import {{ {inc_component} }} from './{inc_component}';"
            )

        # ── State fields ───────────────────────────────────────
        state_lines: List[str] = []

        # State from EL refs: ${user.name} -> name: string
        el_state_fields: Dict[str, str] = {}
        for expr in el_refs:
            # Extract the leaf property: "user.name" -> "name"
            parts = expr.split(".")
            field = parts[-1] if parts else expr
            # Clean up any method calls or brackets
            field = re.sub(r"[\[\]()]+.*$", "", field)
            if field and field not in el_state_fields:
                el_state_fields[field] = "string"

        # State from form fields with type inference
        form_state_fields: Dict[str, str] = {}
        for ff in form_fields:
            prop = ff.get("property", "")
            tag = ff.get("tag", "text").lower()
            if not prop:
                continue
            if tag == "checkbox":
                form_state_fields[prop] = "boolean"
            else:
                form_state_fields[prop] = "string"

        # Merge -- form fields override EL inference if both present
        all_state: Dict[str, str] = {}
        all_state.update(el_state_fields)
        all_state.update(form_state_fields)

        for field_name, ts_type in all_state.items():
            safe_name = _camel_case(field_name) if "_" in field_name or "-" in field_name else field_name
            default = "false" if ts_type == "boolean" else "''"
            setter = "set" + safe_name[0].upper() + safe_name[1:]
            state_lines.append(
                f"  const [{safe_name}, {setter}] = useState<{ts_type}>({default});"
            )

        # Loading and error state when data loading is needed
        if needs_data_loading:
            state_lines.append(
                "  const [loading, setLoading] = useState<boolean>(true);"
            )
            state_lines.append(
                "  const [error, setError] = useState<string | null>(null);"
            )

        # ── Event handlers ─────────────────────────────────────
        handlers: List[str] = []
        if needs_submit:
            action_name = form_actions[0] if form_actions else "submit"
            fn_name = _camel_case(
                re.sub(r"[/.\-]", "_", action_name.strip("/"))
            )
            handlers.append(
                f"  const handleSubmit = async (e: React.FormEvent) => {{\n"
                f"    e.preventDefault();\n"
                f"    try {{\n"
                f"      await {camel_name}Api.submit({{ {', '.join(all_state.keys())} }});\n"
                f"    }} catch (err) {{\n"
                f"      setError(err instanceof Error ? err.message : 'Submit failed');\n"
                f"    }}\n"
                f"  }};"
            )

        # ── useEffect for data loading ─────────────────────────
        effects: List[str] = []
        if needs_data_loading:
            effects.append(
                f"  useEffect(() => {{\n"
                f"    const fetchData = async () => {{\n"
                f"      try {{\n"
                f"        // TODO: fetch initial data for this component\n"
                f"        // const data = await api.getData();\n"
                f"        setLoading(false);\n"
                f"      }} catch (err) {{\n"
                f"        setError(err instanceof Error ? err.message : 'Failed to load data');\n"
                f"        setLoading(false);\n"
                f"      }}\n"
                f"    }};\n"
                f"    fetchData();\n"
                f"  }}, []);"
            )

        # ── JSX body ───────────────────────────────────────────
        jsx_parts: List[str] = []

        # Layout wrapper from tile_refs
        layout_open = ""
        layout_close = ""
        if tile_refs:
            layout_component = _pascal_case(tile_refs[0])
            layout_open = f"    <{layout_component}>"
            layout_close = f"    </{layout_component}>"

        # Error display
        jsx_parts.append(
            "      {error && <div className=\"error-message\">{error}</div>}"
        )

        # Loading / loaded conditional
        if needs_data_loading:
            jsx_parts.append(
                "      {loading ? (\n"
                "        <div>Loading...</div>\n"
                "      ) : ("
            )

        # Form with mapped fields
        if has_form:
            jsx_parts.append(
                "        <form onSubmit={handleSubmit}>"
                if needs_submit
                else "        <form>"
            )
            if has_errors:
                jsx_parts.append(
                    "          {/* TODO: display field-level and action errors */}"
                )
            for ff in form_fields:
                jsx_parts.append(
                    f"          {self._struts_field_to_jsx(ff)}"
                )
            if needs_submit:
                jsx_parts.append(
                    '          <button type="submit">Submit</button>'
                )
            jsx_parts.append("        </form>")

        # Iteration placeholders
        if has_iteration:
            jsx_parts.append(
                "        {/* TODO: replace <logic:iterate> / <s:iterator> with .map() */}\n"
                "        {/* items.map((item) => <div key={item.id}>{item.name}</div>) */}"
            )

        # Conditional render placeholders
        if has_conditionals:
            jsx_parts.append(
                "        {/* TODO: replace <logic:present>/<s:if> with JSX conditionals */}\n"
                "        {/* {condition && <div>...</div>} */}"
            )

        # Child components from includes
        for inc in includes:
            inc_component = self._jsp_name_to_component(inc)
            jsx_parts.append(f"        <{inc_component} />")

        # Close loading conditional
        if needs_data_loading:
            jsx_parts.append("      )}")

        jsx_body = "\n".join(jsx_parts)

        # ── Assemble component ─────────────────────────────────
        lines: List[str] = []
        lines.append(f"// Generated from: {jsp_path}")
        lines.append(f"// Migration rule: jsp_to_react_component")
        lines.append("")
        lines.extend(imports)
        lines.append("")
        lines.append(f"interface {component_name}Props {{")
        lines.append("  // TODO: define component props")
        lines.append("}")
        lines.append("")
        lines.append(
            f"const {component_name}: React.FC<{component_name}Props> = (props) => {{"
        )
        if state_lines:
            lines.extend(state_lines)
            lines.append("")
        if effects:
            lines.extend(effects)
            lines.append("")
        if handlers:
            lines.extend(handlers)
            lines.append("")
        lines.append("  return (")
        if layout_open:
            lines.append(layout_open)
        lines.append("    <div>")
        lines.append(jsx_body)
        lines.append("    </div>")
        if layout_close:
            lines.append(layout_close)
        lines.append("  );")
        lines.append("};")
        lines.append("")
        lines.append(f"export default {component_name};")
        lines.append("")

        code = "\n".join(lines)
        target_path = f"src/components/{component_name}.tsx"

        notes = [
            f"JSP '{jsp_path}' -> React component {component_name}",
            f"State fields: {len(all_state)}",
            f"Form: {has_form}, Links: {has_links}, Iteration: {has_iteration}",
            "Requires review: hook dependencies, API integration, styling",
        ]
        if angular_patterns:
            angular_types = [p.get("type", "unknown") for p in angular_patterns]
            notes.append(
                f"Angular patterns detected: {', '.join(angular_types)} -- "
                f"consider migrating Angular bindings alongside Struts tags"
            )

        return TransformResult(
            source_unit_id=unit_id,
            target_code=code,
            target_path=target_path,
            rule_name="jsp_to_react_component",
            confidence=0.75,
            notes=notes,
        )

    @staticmethod
    def _struts_field_to_jsx(field: Dict[str, str]) -> str:
        """Map a single Struts form field to React JSX markup.

        Uses the field's ``tag`` (html:text, s:textfield, etc.) and
        ``property`` to produce a controlled React input element.
        """
        tag = field.get("tag", "text").lower()
        prop = field.get("property", "field")
        safe_prop = _camel_case(prop) if "_" in prop or "-" in prop else prop
        setter = "set" + safe_prop[0].upper() + safe_prop[1:]

        # Normalize Struts 1 and Struts 2 tag names to a common token
        tag_key = tag.replace("textfield", "text")

        if tag_key in {"text"}:
            return (
                f'<input type="text" name="{prop}" '
                f"value={{{safe_prop}}} "
                f"onChange={{(e) => {setter}(e.target.value)}} />"
            )
        if tag_key == "password":
            return (
                f'<input type="password" name="{prop}" '
                f"value={{{safe_prop}}} "
                f"onChange={{(e) => {setter}(e.target.value)}} />"
            )
        if tag_key == "hidden":
            return (
                f'<input type="hidden" name="{prop}" '
                f"value={{{safe_prop}}} />"
            )
        if tag_key == "checkbox":
            return (
                f'<input type="checkbox" name="{prop}" '
                f"checked={{{safe_prop}}} "
                f"onChange={{(e) => {setter}(e.target.checked)}} />"
            )
        if tag_key == "textarea":
            return (
                f'<textarea name="{prop}" '
                f"value={{{safe_prop}}} "
                f"onChange={{(e) => {setter}(e.target.value)}} />"
            )
        if tag_key == "select":
            return (
                f'<select name="{prop}" '
                f"value={{{safe_prop}}} "
                f"onChange={{(e) => {setter}(e.target.value)}}>"
                f"{{/* TODO: map options */}}"
                f"</select>"
            )
        if tag_key == "submit":
            return '<button type="submit">Submit</button>'

        # Default fallback: render as text input
        return (
            f'<input type="text" name="{prop}" '
            f"value={{{safe_prop}}} "
            f"onChange={{(e) => {setter}(e.target.value)}} />"
        )

    # ── API service generator ──────────────────────────────────

    def _gen_api_service(
        self,
        unit_id: str,
        jsp_path: str,
        component_name: str,
        camel_name: str,
        form_actions: list,
        meta: Dict[str, Any],
    ) -> TransformResult:
        """Generate an API service module ``src/services/{camelName}Api.ts``.

        Creates one async fetch function per form action and exports them
        as a namespaced object.
        """
        lines: List[str] = []
        lines.append(f"// API service generated from: {jsp_path}")
        lines.append(f"// Migration rule: jsp_to_react_component")
        lines.append("")
        lines.append("const API_BASE = '/api';")
        lines.append("")

        fn_names: List[str] = []
        for action in form_actions:
            # Derive function name from the action path
            fn_name = _camel_case(
                re.sub(r"[/.\-]", "_", action.strip("/"))
            )
            if not fn_name:
                fn_name = "submit"
            fn_names.append(fn_name)

            # Map action path to a Spring-style REST endpoint
            endpoint = action.strip("/")
            if not endpoint.startswith("/"):
                endpoint = f"/{endpoint}"

            lines.append(
                f"async function {fn_name}("
                f"data: Record<string, unknown>"
                f"): Promise<Response> {{"
            )
            lines.append(
                f"  return fetch(`${{API_BASE}}{endpoint}`, {{"
            )
            lines.append("    method: 'POST',")
            lines.append("    headers: { 'Content-Type': 'application/json' },")
            lines.append("    body: JSON.stringify(data),")
            lines.append("  });")
            lines.append("}")
            lines.append("")

        # Namespaced export
        export_entries = ", ".join(fn_names)
        # Always alias the first function as 'submit' for convenience
        submit_alias = ""
        if fn_names and fn_names[0] != "submit":
            submit_alias = f", submit: {fn_names[0]}"

        lines.append(
            f"export const {camel_name}Api = {{ {export_entries}{submit_alias} }};"
        )
        lines.append("")

        code = "\n".join(lines)
        target_path = f"src/services/{camel_name}Api.ts"

        return TransformResult(
            source_unit_id=unit_id,
            target_code=code,
            target_path=target_path,
            rule_name="jsp_to_react_component",
            confidence=0.75,
            notes=[
                f"API service for '{jsp_path}' with {len(form_actions)} action(s)",
                "Map endpoints to corresponding Spring @PostMapping controllers",
                f"Actions: {', '.join(form_actions)}",
            ],
        )

    # ── Types file generator ───────────────────────────────────

    def _gen_types_file(
        self,
        unit_id: str,
        jsp_path: str,
        component_name: str,
        camel_name: str,
        bean_refs: list,
        el_refs: list,
        meta: Dict[str, Any],
    ) -> TransformResult:
        """Generate a TypeScript types file ``src/types/{camelName}.types.ts``.

        Builds a data interface from EL expression references and form
        field metadata, with comments referencing original bean classes.
        """
        lines: List[str] = []
        lines.append(f"// Types generated from: {jsp_path}")
        lines.append(f"// Migration rule: jsp_to_react_component")
        lines.append("")

        # Bean reference comments for LLM context
        if bean_refs:
            lines.append("// Original bean references (inspect for full type info):")
            for br in bean_refs:
                lines.append(f"//   - {br}")
            lines.append("")

        lines.append(f"export interface {component_name}Data {{")

        # Fields from EL refs: ${user.name} -> name: string
        seen_fields: set = set()
        for expr in el_refs:
            parts = expr.split(".")
            field = parts[-1] if parts else expr
            field = re.sub(r"[\[\]()]+.*$", "", field)
            if field and field not in seen_fields:
                seen_fields.add(field)
                safe_field = _camel_case(field) if "_" in field or "-" in field else field
                lines.append(f"  {safe_field}: string;")

        # Fields from form_fields metadata
        form_fields: list = meta.get("form_fields", [])
        for ff in form_fields:
            prop = ff.get("property", "")
            tag = ff.get("tag", "text").lower()
            if not prop or prop in seen_fields:
                continue
            seen_fields.add(prop)
            safe_prop = _camel_case(prop) if "_" in prop or "-" in prop else prop
            ts_type = "boolean" if tag == "checkbox" else "string"
            lines.append(f"  {safe_prop}: {ts_type};")

        if not seen_fields:
            lines.append("  // TODO: add fields from bean class inspection")

        lines.append("}")
        lines.append("")

        code = "\n".join(lines)
        target_path = f"src/types/{camel_name}.types.ts"

        return TransformResult(
            source_unit_id=unit_id,
            target_code=code,
            target_path=target_path,
            rule_name="jsp_to_react_component",
            confidence=0.75,
            notes=[
                f"Types for '{jsp_path}' with {len(seen_fields)} field(s)",
                f"Bean refs: {', '.join(bean_refs) if bean_refs else 'none'}",
                "Review generated types against original Java bean classes",
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

        # Angular detection context for hybrid JSP+Angular pages
        angular_notes: List[str] = []
        for tr in transform_results:
            notes = (
                tr.notes
                if isinstance(tr, TransformResult)
                else (tr.get("notes") or [])
            )
            for note in notes:
                if "Angular" in note:
                    angular_notes.append(note)

        angular_block = ""
        if angular_notes:
            items = "\n".join(f"- {n}" for n in angular_notes)
            angular_block = (
                f"\n\n## Angular Hybrid Pages Detected\n\n"
                f"The following JSP pages contain embedded Angular patterns. "
                f"These have been merged into React scaffolding. Pay special "
                f"attention to data binding and component communication that "
                f"was previously handled by Angular.\n\n{items}\n"
            )

        # API endpoint cross-reference (JSP form_actions → Spring endpoints)
        api_notes: List[str] = []
        for tr in transform_results:
            notes = (
                tr.notes
                if isinstance(tr, TransformResult)
                else (tr.get("notes") or [])
            )
            for note in notes:
                if "Form actions:" in note:
                    api_notes.append(note)

        api_block = ""
        if api_notes:
            items = "\n".join(f"- {n}" for n in api_notes)
            api_block = (
                f"\n\n## JSP Form Action → Spring Endpoint Cross-Reference\n\n"
                f"Verify that each React API service call maps to a generated "
                f"Spring @RestController endpoint:\n\n{items}\n"
            )

        # Service layer cross-reference (Controller → Service → Repository)
        svc_notes: List[str] = []
        repo_notes: List[str] = []
        entity_notes: List[str] = []
        for tr in transform_results:
            notes = (
                tr.notes
                if isinstance(tr, TransformResult)
                else (tr.get("notes") or [])
            )
            for note in notes:
                if note.startswith("Service layer:"):
                    svc_notes.append(note)
                elif note.startswith("Repository:"):
                    repo_notes.append(note)
                elif note.startswith("Entity:"):
                    entity_notes.append(note)

        svc_block = ""
        if svc_notes or repo_notes:
            parts: List[str] = []
            if svc_notes:
                parts.append("### Generated Services\n")
                parts.extend(f"- {n}" for n in svc_notes)
            if repo_notes:
                parts.append("\n### Generated Repositories\n")
                parts.extend(f"- {n}" for n in repo_notes)
            if entity_notes:
                parts.append("\n### Entity References\n")
                parts.extend(f"- {n}" for n in entity_notes)

            items = "\n".join(parts)
            svc_block = (
                f"\n\n## Service Layer Architecture\n\n"
                f"The deterministic transforms above produced the following "
                f"layered Spring Boot components. Complete the method bodies, "
                f"add JPA @Entity mappings for referenced entities, and wire "
                f"any missing dependencies via constructor injection.\n\n"
                f"{items}\n"
            )

        return (
            f"{base_prompt}\n\n"
            f"## Pre-completed Deterministic Transforms\n\n"
            f"The following transforms have already been applied "
            f"deterministically.  Do NOT regenerate these; focus on the "
            f"remaining units that were not covered.\n\n"
            f"{block}\n{angular_block}{api_block}{svc_block}"
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
            "| Action + Service class | @Controller + @Service (constructor injection) |\n"
            "| Action + DAO class | @Controller + @Service + @Repository (JPA) |\n"
            "| God-class Action (mixed) | @Controller + @Service stub + @Repository stub |\n"
        )

        # Append JSP → React mapping when view layer is react/rest
        view = _view_layer(context)
        if view in {"react", "rest"}:
            mapping_table += (
                "\n## JSP to React Component Mapping Reference\n\n"
                "| JSP / Struts Tag | React Equivalent |\n"
                "|---|---|\n"
                "| html:form / s:form | `<form onSubmit={handleSubmit}>` |\n"
                "| html:text / s:textfield | `<input type=\"text\" value={state} onChange={...} />` |\n"
                "| html:password / s:password | `<input type=\"password\" .../>` |\n"
                "| html:select / s:select | `<select value={state} onChange={...}>` |\n"
                "| html:checkbox / s:checkbox | `<input type=\"checkbox\" checked={state} .../>` |\n"
                "| html:textarea / s:textarea | `<textarea value={state} onChange={...} />` |\n"
                "| logic:iterate / s:iterator | `{items.map(item => <Component key={...} />)}` |\n"
                "| logic:present / s:if | `{condition && <Element />}` |\n"
                "| bean:write / s:property | `{variable}` (JSX interpolation) |\n"
                "| bean:message / s:text | `{t('key')}` (i18n hook) |\n"
                "| html:errors / s:fielderror | Error display component |\n"
                "| html:link / s:a | `<Link to=\"...\">` (react-router-dom) |\n"
                "| tiles:insert | Layout component wrapper |\n"
                "| jsp:include / <%@ include | Child component import |\n"
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

        # Collect React component names for test coverage
        react_components: List[str] = []
        for tr in transform_results:
            tp = (
                tr.target_path
                if isinstance(tr, TransformResult)
                else tr.get("target_path", "")
            )
            if tp.startswith("src/components/") and tp.endswith(".tsx"):
                react_components.append(tp)

        result = base_prompt

        if endpoints:
            ep_list = "\n".join(
                f"- `{ep}`" for ep in sorted(set(endpoints))
            )
            result += (
                f"\n\n## Endpoint Coverage Validation\n\n"
                f"Ensure tests exist for each of the following migrated "
                f"endpoints:\n\n{ep_list}\n"
            )

        if react_components:
            comp_list = "\n".join(
                f"- `{c}`" for c in sorted(set(react_components))
            )
            result += (
                f"\n\n## React Component Test Coverage\n\n"
                f"Ensure render tests and API service mocks exist for:\n\n"
                f"{comp_list}\n"
            )

        # Service and Repository test coverage
        service_paths: List[str] = []
        repo_paths: List[str] = []
        for tr in transform_results:
            tp = (
                tr.target_path
                if isinstance(tr, TransformResult)
                else tr.get("target_path", "")
            )
            if "Service" in tp and tp.endswith(".java"):
                service_paths.append(tp)
            elif "Repository" in tp and tp.endswith(".java"):
                repo_paths.append(tp)

        if service_paths:
            svc_list = "\n".join(
                f"- `{s}` — unit test with mocked repository"
                for s in sorted(set(service_paths))
            )
            result += (
                f"\n\n## Service Layer Test Coverage\n\n"
                f"Generate unit tests for each @Service class using "
                f"@MockBean for repository dependencies:\n\n{svc_list}\n"
            )

        if repo_paths:
            repo_list = "\n".join(
                f"- `{r}` — integration test with @DataJpaTest"
                for r in sorted(set(repo_paths))
            )
            result += (
                f"\n\n## Repository Layer Test Coverage\n\n"
                f"Generate integration tests for each @Repository interface "
                f"using @DataJpaTest:\n\n{repo_list}\n"
            )

        return result

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
            GateDefinition(
                name="view_component_parity",
                description=(
                    "Every JSP page has a corresponding React component "
                    "in the generated target when view_layer is 'react'."
                ),
                blocking=False,
            ),
            GateDefinition(
                name="service_layer_parity",
                description=(
                    "Every Struts action has a corresponding @Service "
                    "class in the target output."
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
            "view_component_parity": self._gate_view_component_parity,
            "service_layer_parity": self._gate_service_layer_parity,
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

    def _gate_view_component_parity(
        self,
        source_units: List[Dict[str, Any]],
        target_outputs: List[Dict[str, Any]],
        gate_def: GateDefinition,
    ) -> GateResult:
        """Check that every JSP page has a corresponding React component."""
        source_jsps: set[str] = set()
        for u in source_units:
            if u.get("unit_type") == "jsp_page":
                source_jsps.add(u.get("name", u.get("id", "")))

        covered_sources: set[str] = set()
        for t in target_outputs:
            tp = t.get("target_path", "")
            if tp.startswith("src/components/") and tp.endswith(".tsx"):
                src_id = t.get("source_unit_id", "")
                if src_id:
                    covered_sources.add(src_id)

        missing = source_jsps - covered_sources
        return GateResult(
            gate_name=gate_def.name,
            passed=len(missing) == 0,
            details={
                "source_jsps": sorted(source_jsps),
                "target_components": sorted(covered_sources),
                "missing": sorted(missing),
                "coverage": (
                    len(source_jsps - missing) / len(source_jsps)
                    if source_jsps
                    else 1.0
                ),
            },
            blocking=gate_def.blocking,
        )

    def _gate_service_layer_parity(
        self,
        source_units: List[Dict[str, Any]],
        target_outputs: List[Dict[str, Any]],
        gate_def: GateDefinition,
    ) -> GateResult:
        """Check that every Struts action has a corresponding @Service."""
        action_names: set[str] = set()
        for u in source_units:
            ut = u.get("unit_type", "")
            if ut in {"struts_action", "struts2_action"}:
                name = u.get("name", "")
                # Derive expected service name from action name
                simple = name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
                for suffix in ("Action",):
                    if simple.endswith(suffix):
                        simple = simple[: -len(suffix)]
                if simple:
                    action_names.add(simple)

        service_names: set[str] = set()
        for t in target_outputs:
            code = t.get("target_code", "")
            tp = t.get("target_path", "")
            if "Service" in tp and tp.endswith(".java"):
                match = re.search(
                    r"public\s+class\s+(\w+)Service\b", code
                )
                if match:
                    service_names.add(match.group(1))

        covered = action_names & service_names
        missing = action_names - service_names
        return GateResult(
            gate_name=gate_def.name,
            passed=len(missing) == 0,
            details={
                "expected_services": sorted(action_names),
                "found_services": sorted(service_names),
                "missing": sorted(missing),
                "coverage": (
                    len(covered) / len(action_names)
                    if action_names
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
            "jsp_page": {"strategy": "transform", "priority": "high"},
        }

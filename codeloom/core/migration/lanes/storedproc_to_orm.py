"""Stored Procedure to Service + Data Access Layer migration lane.

Target-aware polyglot lane that converts SQL stored procedures, functions,
triggers, and views into application-layer equivalents.  Reads ``target_stack``
from plan context to dispatch to the correct language/framework generator set:

  * ``spring``  -- Spring Data JPA repositories + Spring ``@Service`` classes
  * ``ef_core`` -- Entity Framework Core DbContext + services  (future)
  * ``django``  -- Django models + service layer  (future)

Only Spring/JPA generators are implemented in this version.  EF Core and
Django generators raise ``NotImplementedError``; adding a new target is
purely additive (new ``_gen_*`` methods, no changes to rules, gates, or
prompts).

SQL parser metadata consumed per unit (from ``sql_parser.py``):
    parameters   -- [{name, type, direction}, ...]
    tables_referenced -- [str, ...]
    sp_calls     -- [str, ...]
    return_type  -- "TABLE" | scalar type  (functions only)
    trigger_table -- str  (triggers only)
    schema       -- str
    is_deprecated -- bool
"""

import logging
import re
from typing import Any, Dict, List, Optional

from .base import (
    GateDefinition,
    GateResult,
    MigrationLane,
    TransformResult,
    TransformRule,
)

logger = logging.getLogger(__name__)


# ── Helpers ─────────────────────────────────────────────────────────


def _ucfirst(s: str) -> str:
    """Uppercase only the first character, preserving the rest."""
    return s[0].upper() + s[1:] if s else ""


def _pascal_case(name: str) -> str:
    """Convert a SQL object name to PascalCase, stripping common prefixes.

    Splits on underscores/dashes and uppercases the first letter of each
    segment while **preserving** existing internal casing (e.g. camelCase
    segments within the name are not flattened).

    Examples::

        usp_GetUserOrders  ->  GetUserOrders
        fn_CalcTotal       ->  CalcTotal
        trg_AuditLog       ->  AuditLog
        vw_ActiveUsers     ->  ActiveUsers
        get_user_by_id     ->  GetUserById
    """
    stripped = re.sub(r"^(usp_|sp_|fn_|trg_|vw_)", "", name, flags=re.IGNORECASE)
    parts = re.split(r"[_\-]+", stripped)
    return "".join(_ucfirst(p) for p in parts if p)


def _camel_case(name: str) -> str:
    """Convert a SQL object name to camelCase."""
    pascal = _pascal_case(name)
    return pascal[0].lower() + pascal[1:] if pascal else ""


def _sql_type_to_java(sql_type: str) -> str:
    """Map a SQL parameter type to its Java equivalent.

    Strips parenthesised precision/scale (e.g. ``VARCHAR(255)`` -> ``String``)
    and falls back to ``Object`` for unknown types.
    """
    _TYPE_MAP = {
        "INT": "Integer",
        "BIGINT": "Long",
        "SMALLINT": "Short",
        "BIT": "Boolean",
        "VARCHAR": "String",
        "NVARCHAR": "String",
        "CHAR": "String",
        "TEXT": "String",
        "DATETIME": "LocalDateTime",
        "DATE": "LocalDate",
        "DECIMAL": "BigDecimal",
        "NUMERIC": "BigDecimal",
        "FLOAT": "Double",
        "MONEY": "BigDecimal",
        "UNIQUEIDENTIFIER": "UUID",
    }
    base = re.sub(r"\(.*\)", "", sql_type.upper()).strip()
    return _TYPE_MAP.get(base, "Object")


def _entity_name(table_name: str) -> str:
    """Derive a JPA entity class name from a SQL table name.

    Examples::

        Orders         ->  OrderEntity
        user_accounts  ->  UserAccountEntity
        Users          ->  UserEntity
    """
    pascal = _pascal_case(table_name)
    # Naive singularisation: strip trailing 's' if long enough and not 'ss'
    if len(pascal) > 4 and pascal.endswith("s") and not pascal.endswith("ss"):
        pascal = pascal[:-1]
    return pascal + "Entity"


# ── Target dispatch map ─────────────────────────────────────────────

_TARGET_MAP: Dict[str, set] = {
    "spring": {"springboot", "spring", "spring_data", "jpa", "hibernate"},
    "ef_core": {"dotnet_core", "ef_core", "entity_framework", "dotnet"},
    "django": {"django", "sqlalchemy"},
}


# ── Lane Implementation ────────────────────────────────────────────


class StoredProcToOrmLane(MigrationLane):
    """Migration lane: Stored Procedures to Service + Data Access Layer.

    Produces TWO output layers per stored procedure:

    1. **Repository / Data Access interface** -- thin query interface
       (e.g. Spring ``@Repository``, EF Core ``DbContext``).
    2. **Service class** -- encapsulates the SP's business logic and
       orchestrates repository calls (e.g. Spring ``@Service``).

    SQL functions, triggers, and views each produce a single output.
    """

    # ── Identity ────────────────────────────────────────────────

    @property
    def lane_id(self) -> str:
        return "storedproc_to_orm"

    @property
    def display_name(self) -> str:
        return "Stored Procedures \u2192 Service + Data Access Layer"

    @property
    def source_frameworks(self) -> List[str]:
        return ["sql_stored_procs"]

    @property
    def target_frameworks(self) -> List[str]:
        return [
            "springboot",
            "spring_data",
            "jpa",
            "hibernate",
            "dotnet_core",
            "ef_core",
            "django",
            "sqlalchemy",
        ]

    # ── Applicability ───────────────────────────────────────────

    def detect_applicability(
        self, source_framework: str, target_stack: Dict[str, Any]
    ) -> float:
        """Score applicability for a source/target combination.

        Returns a high score when the source contains SQL stored procedures
        and the target explicitly mentions a supported ORM framework.
        """
        if source_framework.lower() not in {"sql_stored_procs", "sql"}:
            return 0.0

        target_fw = str(target_stack.get("framework", "")).lower()
        if any(kw in target_fw for kw in ("spring", "jpa", "hibernate")):
            return 0.95
        if any(kw in target_fw for kw in ("dotnet", "ef_core", "entity_framework")):
            return 0.90  # future
        return 0.4  # source matches but no explicit target

    # ── Target Resolution ───────────────────────────────────────

    def _resolve_target(self, context: Dict[str, Any]) -> str:
        """Read ``target_stack`` from plan context and resolve the generator family.

        Returns one of ``"spring"``, ``"ef_core"``, or ``"django"``.
        Defaults to ``"spring"`` when no explicit match is found.
        """
        target_stack = context.get("target_stack", {})
        fw = str(target_stack.get("framework", "")).lower()
        for target_key, keywords in _TARGET_MAP.items():
            if any(kw in fw for kw in keywords):
                return target_key
        return "spring"  # default

    # ── Transform Rules ─────────────────────────────────────────

    def get_transform_rules(self) -> List[TransformRule]:
        """Return all deterministic transform rules.

        ``sp_to_repository`` and ``sp_to_service`` both match
        ``stored_procedure`` -- ``apply_transforms`` intentionally produces
        TWO ``TransformResult`` objects per SP (one per layer).
        """
        return [
            TransformRule(
                name="sp_to_repository",
                source_pattern={"unit_type": "stored_procedure"},
                target_template="data_access_interface",
                confidence=0.85,
                description=(
                    "Convert stored procedure to repository/data-access "
                    "interface with query methods."
                ),
            ),
            TransformRule(
                name="sp_to_service",
                source_pattern={"unit_type": "stored_procedure"},
                target_template="service_class",
                confidence=0.80,
                description=(
                    "Extract business logic from stored procedure into "
                    "service class."
                ),
            ),
            TransformRule(
                name="function_to_service",
                source_pattern={"unit_type": "sql_function"},
                target_template="service_utility_method",
                confidence=0.80,
                description=(
                    "Convert SQL function to service utility method or query."
                ),
            ),
            TransformRule(
                name="trigger_to_lifecycle",
                source_pattern={"unit_type": "trigger"},
                target_template="entity_lifecycle_callback",
                confidence=0.65,
                requires_review=True,
                description=(
                    "Convert trigger to entity lifecycle event handler."
                ),
            ),
            TransformRule(
                name="view_to_projection",
                source_pattern={"unit_type": "view"},
                target_template="readonly_entity",
                confidence=0.75,
                description=(
                    "Convert SQL view to read-only entity/projection."
                ),
            ),
        ]

    # ── Transform Application ───────────────────────────────────

    def apply_transforms(
        self,
        units: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> List[TransformResult]:
        """Apply deterministic transforms to matching SQL units.

        Stored procedures produce TWO results (repository + service).
        Functions, triggers, and views each produce one result.
        """
        target = self._resolve_target(context)
        results: List[TransformResult] = []

        for unit in units:
            unit_type = unit.get("unit_type") or unit.get("metadata", {}).get(
                "unit_type", ""
            )
            meta = unit.get("metadata", {})
            unit_id = str(unit.get("id", unit.get("name", "unknown")))
            name = unit.get("name", "Unknown")

            try:
                if unit_type == "stored_procedure":
                    # TWO outputs: repository + service
                    results.append(
                        self._transform_sp_repository(unit, unit_id, name, meta, target)
                    )
                    results.append(
                        self._transform_sp_service(unit, unit_id, name, meta, target)
                    )
                elif unit_type == "sql_function":
                    results.append(
                        self._transform_function(unit, unit_id, name, meta, target)
                    )
                elif unit_type == "trigger":
                    results.append(
                        self._transform_trigger(unit, unit_id, name, meta, target)
                    )
                elif unit_type == "view":
                    results.append(
                        self._transform_view(unit, unit_id, name, meta, target)
                    )
            except NotImplementedError as exc:
                logger.warning(
                    "StoredProcToOrm: skipping unit '%s' (%s) -- %s",
                    name,
                    unit_type,
                    exc,
                )
            except Exception:
                logger.error(
                    "StoredProcToOrm: failed to transform unit '%s' (%s)",
                    name,
                    unit_type,
                    exc_info=True,
                )

        logger.info(
            "StoredProcToOrm: applied %d transforms to %d units",
            len(results),
            len(units),
        )
        return [r for r in results if r is not None]

    # ── Spring/JPA Generators ───────────────────────────────────

    def _transform_sp_repository(
        self,
        unit: Dict[str, Any],
        unit_id: str,
        name: str,
        meta: Dict[str, Any],
        target: str,
    ) -> TransformResult:
        """Generate a data-access interface for a stored procedure.

        For Spring: a ``@Repository`` interface extending ``JpaRepository``
        with either ``@Procedure`` (simple SP) or ``@Query(nativeQuery)``
        (complex SP with OUTPUT params, multi-table joins, or SP-to-SP calls).
        """
        if target != "spring":
            raise NotImplementedError(
                f"{target} repository generators not yet implemented"
            )

        params = meta.get("parameters", [])
        tables = meta.get("tables_referenced", [])
        class_name = _pascal_case(name) + "Repository"

        # Primary entity = first table referenced (or generic)
        primary_table = tables[0] if tables else "Unknown"
        entity = _entity_name(primary_table)

        # Build method signature from IN params
        in_params = [p for p in params if p.get("direction") != "OUTPUT"]
        out_params = [p for p in params if p.get("direction") == "OUTPUT"]
        method_name = _camel_case(name)

        param_str = ", ".join(
            f"@Param(\"{p['name'].lstrip('@')}\") "
            f"{_sql_type_to_java(p['type'])} "
            f"{_camel_case(p['name'].lstrip('@'))}"
            for p in in_params
        )

        # Determine annotation strategy based on SP complexity
        has_output = len(out_params) > 0
        has_multiple_tables = len(tables) > 2
        sp_calls_other = len(meta.get("sp_calls", [])) > 0

        if has_output or has_multiple_tables or sp_calls_other:
            # Complex SP -> @Query(nativeQuery=true)
            schema = meta.get("schema", "dbo")
            annotation = (
                f'@Query(value = "EXEC {schema}.{name} ...", nativeQuery = true)'
            )
            return_type = "List<Object[]>"
        else:
            # Simple SP -> @Procedure
            annotation = f'@Procedure(name = "{name}")'
            return_type = f"List<{entity}>"

        code = (
            f"import org.springframework.data.jpa.repository.JpaRepository;\n"
            f"import org.springframework.data.jpa.repository.Query;\n"
            f"import org.springframework.data.jpa.repository.query.Procedure;\n"
            f"import org.springframework.data.repository.query.Param;\n\n"
            f"public interface {class_name} extends JpaRepository<{entity}, Long> {{\n\n"
            f"    {annotation}\n"
            f"    {return_type} {method_name}({param_str});\n"
            f"}}\n"
        )

        # Entity stub notes for architecture-phase prompt augmentation
        entity_stubs: List[str] = []
        for tbl in tables:
            ent = _entity_name(tbl)
            entity_stubs.append(f"  - {ent} (from table: {tbl})")

        target_path = f"src/main/java/com/app/repository/{class_name}.java"

        notes = [
            f"SP '{name}' \u2192 {class_name}",
            f"IN params: {len(in_params)}, OUT params: {len(out_params)}",
            f"Tables: {', '.join(tables)}",
        ]
        if entity_stubs:
            notes.append("Entity stubs needed:\n" + "\n".join(entity_stubs))

        return TransformResult(
            source_unit_id=unit_id,
            target_code=code,
            target_path=target_path,
            rule_name="sp_to_repository",
            confidence=0.85,
            notes=notes,
        )

    def _transform_sp_service(
        self,
        unit: Dict[str, Any],
        unit_id: str,
        name: str,
        meta: Dict[str, Any],
        target: str,
    ) -> TransformResult:
        """Generate a service class for a stored procedure.

        For Spring: a ``@Service`` class that injects the corresponding
        repository and encapsulates the SP's business logic with
        ``@Transactional`` semantics.
        """
        if target != "spring":
            raise NotImplementedError(
                f"{target} service generators not yet implemented"
            )

        params = meta.get("parameters", [])
        tables = meta.get("tables_referenced", [])
        sp_calls = meta.get("sp_calls", [])
        repo_name = _pascal_case(name) + "Repository"
        service_name = _pascal_case(name) + "Service"
        method_name = _camel_case(name)

        in_params = [p for p in params if p.get("direction") != "OUTPUT"]
        param_str = ", ".join(
            f"{_sql_type_to_java(p['type'])} {_camel_case(p['name'].lstrip('@'))}"
            for p in in_params
        )

        # Build injection fields -- one for primary repo, plus any called SPs
        injections = [f"    private final {repo_name} {_camel_case(repo_name)};"]
        for sp in sp_calls:
            dep_repo = _pascal_case(sp) + "Repository"
            injections.append(
                f"    private final {dep_repo} {_camel_case(dep_repo)};"
            )

        inject_block = "\n".join(injections)

        code = (
            f"import org.springframework.stereotype.Service;\n"
            f"import org.springframework.transaction.annotation.Transactional;\n\n"
            f"@Service\n"
            f"public class {service_name} {{\n\n"
            f"{inject_block}\n\n"
            f"    public {service_name}({repo_name} {_camel_case(repo_name)}) {{\n"
            f"        this.{_camel_case(repo_name)} = {_camel_case(repo_name)};\n"
            f"    }}\n\n"
            f"    @Transactional\n"
            f"    public void {method_name}({param_str}) {{\n"
            f"        // TODO: migrate business logic from SP '{name}'\n"
            f"        // Original SP references tables: {', '.join(tables)}\n"
        )
        if sp_calls:
            code += (
                f"        // SP calls other SPs: {', '.join(sp_calls)}\n"
            )
        code += (
            f"        {_camel_case(repo_name)}.{method_name}(...);\n"
            f"    }}\n"
            f"}}\n"
        )

        target_path = f"src/main/java/com/app/service/{service_name}.java"

        notes = [
            f"SP '{name}' \u2192 {service_name} (business logic layer)",
            f"Injects: {repo_name}"
            + (f" + {len(sp_calls)} SP dependencies" if sp_calls else ""),
            "Requires manual migration of procedural logic",
        ]

        return TransformResult(
            source_unit_id=unit_id,
            target_code=code,
            target_path=target_path,
            rule_name="sp_to_service",
            confidence=0.80,
            notes=notes,
        )

    def _transform_function(
        self,
        unit: Dict[str, Any],
        unit_id: str,
        name: str,
        meta: Dict[str, Any],
        target: str,
    ) -> TransformResult:
        """Generate a service class for a SQL function.

        Table-valued functions become ``JdbcTemplate.query()`` calls;
        scalar functions become ``JdbcTemplate.queryForObject()`` calls.
        """
        if target != "spring":
            raise NotImplementedError(
                f"{target} function generators not yet implemented"
            )

        return_type = meta.get("return_type", "UNKNOWN")
        params = meta.get("parameters", [])
        is_table_valued = return_type.upper() == "TABLE"
        service_name = _pascal_case(name) + "Service"
        method_name = _camel_case(name)

        in_params = [p for p in params if p.get("direction") != "OUTPUT"]
        param_str = ", ".join(
            f"{_sql_type_to_java(p['type'])} {_camel_case(p['name'].lstrip('@'))}"
            for p in in_params
        )

        if is_table_valued:
            java_return = "List<Object[]>"
            body = (
                f"        // Table-valued function -> use @Query or JdbcTemplate\n"
                f'        return jdbcTemplate.query("SELECT * FROM {name}(...)", ...);\n'
            )
        else:
            java_return = _sql_type_to_java(return_type)
            schema = meta.get("schema", "dbo")
            body = (
                f'        return jdbcTemplate.queryForObject('
                f'"SELECT {schema}.{name}(...)", {java_return}.class);\n'
            )

        code = (
            f"import org.springframework.jdbc.core.JdbcTemplate;\n"
            f"import org.springframework.stereotype.Service;\n\n"
            f"@Service\n"
            f"public class {service_name} {{\n\n"
            f"    private final JdbcTemplate jdbcTemplate;\n\n"
            f"    public {service_name}(JdbcTemplate jdbcTemplate) {{\n"
            f"        this.jdbcTemplate = jdbcTemplate;\n"
            f"    }}\n\n"
            f"    public {java_return} {method_name}({param_str}) {{\n"
            f"{body}"
            f"    }}\n"
            f"}}\n"
        )

        target_path = f"src/main/java/com/app/service/{service_name}.java"

        return TransformResult(
            source_unit_id=unit_id,
            target_code=code,
            target_path=target_path,
            rule_name="function_to_service",
            confidence=0.80,
            notes=[
                f"Function '{name}' \u2192 {service_name}",
                f"Return type: {return_type} "
                f"({'table-valued' if is_table_valued else 'scalar'})",
            ],
        )

    def _transform_trigger(
        self,
        unit: Dict[str, Any],
        unit_id: str,
        name: str,
        meta: Dict[str, Any],
        target: str,
    ) -> TransformResult:
        """Generate a JPA entity listener for a SQL trigger.

        Detects INSERT/UPDATE/DELETE references in the trigger source to
        map to ``@PrePersist``, ``@PreUpdate``, ``@PreRemove`` callbacks.
        """
        if target != "spring":
            raise NotImplementedError(
                f"{target} trigger generators not yet implemented"
            )

        trigger_table = meta.get("trigger_table", "Unknown")
        entity = _entity_name(trigger_table)
        listener_name = _pascal_case(name) + "Listener"
        source = unit.get("source", "")

        # Detect trigger events from source SQL
        events: List[str] = []
        if re.search(r"\bINSERT\b", source, re.IGNORECASE):
            events.append("@PrePersist")
        if re.search(r"\bUPDATE\b", source, re.IGNORECASE):
            events.append("@PreUpdate")
        if re.search(r"\bDELETE\b", source, re.IGNORECASE):
            events.append("@PreRemove")

        callbacks: List[str] = []
        for evt in events:
            method = _camel_case(evt.lstrip("@"))
            callbacks.append(
                f"    {evt}\n"
                f"    public void {method}({entity} entity) {{\n"
                f"        // TODO: migrate trigger logic from '{name}'\n"
                f"    }}\n"
            )

        code = (
            f"import javax.persistence.*;\n\n"
            f"public class {listener_name} {{\n\n"
            + "\n".join(callbacks)
            + f"}}\n\n"
            f"// Register on entity:\n"
            f"// @EntityListeners({listener_name}.class)\n"
            f'// @Entity @Table(name = "{trigger_table}")\n'
            f"// public class {entity} {{ ... }}\n"
        )

        target_path = f"src/main/java/com/app/listener/{listener_name}.java"

        return TransformResult(
            source_unit_id=unit_id,
            target_code=code,
            target_path=target_path,
            rule_name="trigger_to_lifecycle",
            confidence=0.65,
            notes=[
                f"Trigger '{name}' on table '{trigger_table}' \u2192 {listener_name}",
                f"Events detected: {', '.join(events) or 'none (manual review needed)'}",
                "Requires human review -- trigger logic may not map cleanly to JPA lifecycle",
            ],
        )

    def _transform_view(
        self,
        unit: Dict[str, Any],
        unit_id: str,
        name: str,
        meta: Dict[str, Any],
        target: str,
    ) -> TransformResult:
        """Generate a read-only ``@Immutable`` entity for a SQL view."""
        if target != "spring":
            raise NotImplementedError(
                f"{target} view generators not yet implemented"
            )

        entity = _entity_name(name)
        schema = meta.get("schema", "dbo")
        tables = meta.get("tables_referenced", [])

        code = (
            f"import javax.persistence.*;\n"
            f"import org.hibernate.annotations.Immutable;\n\n"
            f"@Entity\n"
            f"@Immutable\n"
            f'@Table(name = "{name}", schema = "{schema}")\n'
            f"public class {entity} {{\n\n"
            f"    @Id\n"
            f"    private Long id;  // TODO: identify primary key from view definition\n\n"
            f"    // TODO: add fields matching view columns\n"
            f"    // Source tables: {', '.join(tables)}\n"
            f"}}\n"
        )

        target_path = f"src/main/java/com/app/entity/{entity}.java"

        return TransformResult(
            source_unit_id=unit_id,
            target_code=code,
            target_path=target_path,
            rule_name="view_to_projection",
            confidence=0.75,
            notes=[
                f"View '{name}' \u2192 {entity} (read-only @Immutable entity)",
                f"Underlying tables: {', '.join(tables)}",
            ],
        )

    # ── Prompt Augmentation ─────────────────────────────────────

    def augment_prompt(
        self,
        phase_type: str,
        base_prompt: str,
        context: Dict[str, Any],
    ) -> str:
        """Augment a phase prompt with SP-to-ORM domain context.

        Supported phases: ``architecture``, ``transform``, ``test``.
        Other phases return the base prompt unchanged.
        """
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
        """Inject concept-mapping table and entity list for architecture phase.

        The mapping table is target-aware (Spring Boot for now).
        """
        target = self._resolve_target(context)

        if target == "spring":
            mapping_table = (
                "## Stored Procedure \u2192 Spring Boot Mapping Reference\n\n"
                "| SQL Concept | Spring Boot Equivalent |\n"
                "|---|---|\n"
                "| Stored Procedure | @Repository interface + @Service class |\n"
                "| SP Parameters (IN) | Method parameters with @Param |\n"
                "| SP Parameters (OUTPUT) | Return type / DTO fields |\n"
                "| EXEC/CALL other SP | Injected @Service dependency |\n"
                "| Transaction (BEGIN TRAN) | @Transactional on service method |\n"
                "| Temp tables / cursors | Java collections + streams |\n"
                "| SQL Function (scalar) | @Service method with JdbcTemplate |\n"
                "| SQL Function (table) | @Query returning List<DTO> |\n"
                "| Trigger | @EntityListeners + lifecycle callback |\n"
                "| View | @Entity + @Immutable read-only mapping |\n"
            )
        else:
            # Placeholder for future targets
            mapping_table = (
                f"## Stored Procedure \u2192 {target} Mapping Reference\n\n"
                f"_(Detailed mapping table for {target} will be added "
                f"in a future release.)_\n"
            )

        # Collect entity stubs from transform results if available
        transform_results = context.get("transform_results", [])
        entity_lines: List[str] = []
        seen_entities: set = set()

        for tr in transform_results:
            notes = []
            if isinstance(tr, TransformResult):
                notes = tr.notes
            elif isinstance(tr, dict):
                notes = tr.get("notes", [])

            for note in notes:
                if isinstance(note, str) and "Entity stubs needed:" in note:
                    for line in note.split("\n"):
                        line = line.strip()
                        if line.startswith("- ") and line not in seen_entities:
                            entity_lines.append(line)
                            seen_entities.add(line)

        entity_block = ""
        if entity_lines:
            entity_block = (
                "\nEntity classes needed (generate with proper JPA annotations):\n"
                + "\n".join(entity_lines)
                + "\n"
            )

        return f"{base_prompt}\n\n{mapping_table}{entity_block}"

    def _augment_transform(
        self, base_prompt: str, context: Dict[str, Any]
    ) -> str:
        """List pre-completed deterministic transforms for the transform phase.

        Instructs the LLM to skip already-generated code and focus on
        remaining complex SPs, entity field completion, and service bodies.
        """
        transform_results = context.get("transform_results", [])
        if not transform_results:
            return base_prompt

        summaries: List[str] = []
        for tr in transform_results:
            if isinstance(tr, TransformResult):
                summaries.append(
                    f"- {tr.rule_name}: {tr.source_unit_id} \u2192 "
                    f"{tr.target_path} (confidence {tr.confidence})"
                )
            elif isinstance(tr, dict):
                summaries.append(
                    f"- {tr.get('rule_name', '?')}: "
                    f"{tr.get('source_unit_id', '?')} \u2192 "
                    f"{tr.get('target_path', '?')} "
                    f"(confidence {tr.get('confidence', '?')})"
                )

        block = "\n".join(summaries)
        return (
            f"{base_prompt}\n\n"
            f"## Pre-completed Deterministic Transforms\n\n"
            f"The following transforms have already been applied. "
            f"Do NOT regenerate.\n"
            f"Focus on the remaining units and on completing the entity "
            f"class fields, service method bodies, and complex SP logic "
            f"translation.\n\n"
            f"{block}\n"
        )

    def _augment_test(
        self, base_prompt: str, context: Dict[str, Any]
    ) -> str:
        """List repository and service methods that need tests.

        Repository tests use ``@DataJpaTest``; service tests use
        ``@SpringBootTest`` with mocked repositories.
        """
        target = self._resolve_target(context)
        transform_results = context.get("transform_results", [])

        repo_methods: List[str] = []
        service_methods: List[str] = []

        for tr in transform_results:
            rule_name = ""
            target_path = ""
            source_code = ""
            if isinstance(tr, TransformResult):
                rule_name = tr.rule_name
                target_path = tr.target_path
                source_code = tr.target_code
            elif isinstance(tr, dict):
                rule_name = tr.get("rule_name", "")
                target_path = tr.get("target_path", "")
                source_code = tr.get("target_code", "")

            # Extract method signatures from generated code
            method_match = re.search(
                r"(?:public\s+\S+\s+)(\w+)\s*\(([^)]*)\)", source_code
            )
            if not method_match:
                continue

            method_sig = f"{method_match.group(1)}({method_match.group(2)})"

            # Classify by class name extracted from target_path
            class_match = re.search(r"/(\w+)\.java$", target_path)
            class_name = class_match.group(1) if class_match else "Unknown"

            if rule_name == "sp_to_repository" or "Repository" in target_path:
                repo_methods.append(f"- {class_name}.{method_sig}")
            elif rule_name in {"sp_to_service", "function_to_service"} or "Service" in target_path:
                service_methods.append(f"- {class_name}.{method_sig}")

        if not repo_methods and not service_methods:
            return base_prompt

        sections: List[str] = ["## Test Coverage Required\n"]

        if target == "spring":
            if repo_methods:
                sections.append("Repository integration tests (@DataJpaTest):")
                sections.extend(repo_methods)
                sections.append("")
            if service_methods:
                sections.append("Service unit tests (@SpringBootTest):")
                sections.extend(service_methods)
        else:
            # Generic test listing for future targets
            all_methods = repo_methods + service_methods
            if all_methods:
                sections.append("Methods requiring test coverage:")
                sections.extend(all_methods)

        test_block = "\n".join(sections)
        return f"{base_prompt}\n\n{test_block}\n"

    # ── Quality Gates ───────────────────────────────────────────

    def get_gates(self) -> List[GateDefinition]:
        """Return quality gate definitions for SP-to-ORM migration.

        ``sp_parity`` and ``table_entity_coverage`` are blocking;
        ``parameter_mapping`` and ``service_coverage`` are advisory.
        """
        return [
            GateDefinition(
                name="sp_parity",
                description=(
                    "Every stored_procedure has a corresponding "
                    "repository + service method."
                ),
                blocking=True,
            ),
            GateDefinition(
                name="table_entity_coverage",
                description=(
                    "Tables referenced by SPs have @Entity classes "
                    "in target code."
                ),
                blocking=True,
            ),
            GateDefinition(
                name="parameter_mapping",
                description=(
                    "SP parameters mapped to method parameters "
                    "(count match)."
                ),
                blocking=False,
            ),
            GateDefinition(
                name="service_coverage",
                description=(
                    "Business logic SPs have a service method, "
                    "not just data access."
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
        """Run a specific quality gate against source units and target outputs."""
        gate_map = {
            "sp_parity": self._gate_sp_parity,
            "table_entity_coverage": self._gate_table_entity_coverage,
            "parameter_mapping": self._gate_parameter_mapping,
            "service_coverage": self._gate_service_coverage,
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

    def _gate_sp_parity(
        self,
        source_units: List[Dict[str, Any]],
        target_outputs: List[Dict[str, Any]],
        gate_def: GateDefinition,
    ) -> GateResult:
        """Verify every stored_procedure has a corresponding repository class.

        Collects SP names from source, scans target for ``*Repository``
        class names, and reports any SPs without a matching repository.
        """
        source_sps: set = set()
        for u in source_units:
            if u.get("unit_type") == "stored_procedure":
                source_sps.add(u.get("name", ""))

        # Build set of SP names that have a generated repository
        covered_sps: set = set()
        for t in target_outputs:
            code = t.get("target_code", "")
            # Match repository interface declarations
            for match in re.finditer(
                r"public\s+interface\s+(\w+)Repository\b", code
            ):
                repo_pascal = match.group(1)
                # Find matching SP by comparing pascal-cased names
                for sp in source_sps:
                    if _pascal_case(sp) == repo_pascal:
                        covered_sps.add(sp)

        missing = source_sps - covered_sps
        return GateResult(
            gate_name=gate_def.name,
            passed=len(missing) == 0,
            details={
                "source_sps": sorted(source_sps),
                "covered_sps": sorted(covered_sps),
                "missing": sorted(missing),
                "coverage": (
                    len(covered_sps) / len(source_sps)
                    if source_sps
                    else 1.0
                ),
            },
            blocking=gate_def.blocking,
        )

    def _gate_table_entity_coverage(
        self,
        source_units: List[Dict[str, Any]],
        target_outputs: List[Dict[str, Any]],
        gate_def: GateDefinition,
    ) -> GateResult:
        """Verify tables referenced by SPs have ``@Entity`` classes.

        Collects all ``tables_referenced`` across SPs and checks target
        code for ``@Entity`` + ``@Table(name = "...")`` declarations.
        """
        required_tables: set = set()
        for u in source_units:
            if u.get("unit_type") in {"stored_procedure", "sql_function"}:
                tables = u.get("metadata", {}).get("tables_referenced", [])
                required_tables.update(tables)

        # Scan target code for @Entity + @Table declarations
        covered_tables: set = set()
        for t in target_outputs:
            code = t.get("target_code", "")
            for match in re.finditer(
                r'@Table\s*\(\s*name\s*=\s*"(\w+)"', code
            ):
                covered_tables.add(match.group(1))

        missing = required_tables - covered_tables
        return GateResult(
            gate_name=gate_def.name,
            passed=len(missing) == 0,
            details={
                "required_tables": sorted(required_tables),
                "covered_tables": sorted(covered_tables),
                "missing": sorted(missing),
                "coverage": (
                    len(required_tables - missing) / len(required_tables)
                    if required_tables
                    else 1.0
                ),
            },
            blocking=gate_def.blocking,
        )

    def _gate_parameter_mapping(
        self,
        source_units: List[Dict[str, Any]],
        target_outputs: List[Dict[str, Any]],
        gate_def: GateDefinition,
    ) -> GateResult:
        """Verify SP IN-parameter counts match generated method parameters.

        For each SP, counts IN params from metadata and compares against
        ``@Param`` annotations in the generated repository code.
        """
        mismatches: List[Dict[str, Any]] = []

        # Build target param counts: repo class name -> param count
        target_param_counts: Dict[str, int] = {}
        for t in target_outputs:
            code = t.get("target_code", "")
            target_path = t.get("target_path", "")
            if "Repository" not in target_path:
                continue
            # Count @Param annotations
            param_count = len(re.findall(r"@Param\(", code))
            class_match = re.search(r"/(\w+Repository)\.java$", target_path)
            if class_match:
                target_param_counts[class_match.group(1)] = param_count

        for u in source_units:
            if u.get("unit_type") != "stored_procedure":
                continue
            name = u.get("name", "")
            params = u.get("metadata", {}).get("parameters", [])
            in_count = sum(
                1 for p in params if p.get("direction") != "OUTPUT"
            )

            expected_repo = _pascal_case(name) + "Repository"
            actual_count = target_param_counts.get(expected_repo)

            if actual_count is not None and actual_count != in_count:
                mismatches.append({
                    "sp": name,
                    "expected_in_params": in_count,
                    "actual_method_params": actual_count,
                })

        return GateResult(
            gate_name=gate_def.name,
            passed=len(mismatches) == 0,
            details={
                "mismatches": mismatches,
                "total_sps_checked": len(target_param_counts),
            },
            blocking=gate_def.blocking,
        )

    def _gate_service_coverage(
        self,
        source_units: List[Dict[str, Any]],
        target_outputs: List[Dict[str, Any]],
        gate_def: GateDefinition,
    ) -> GateResult:
        """Verify every SP has both a repository AND a service in target outputs.

        SPs with only a repository but no service represent incomplete
        migrations where business logic has not been extracted.
        """
        source_sps: set = set()
        for u in source_units:
            if u.get("unit_type") == "stored_procedure":
                source_sps.add(u.get("name", ""))

        has_repo: set = set()
        has_service: set = set()

        for t in target_outputs:
            target_path = t.get("target_path", "")
            code = t.get("target_code", "")

            for sp in source_sps:
                pascal = _pascal_case(sp)
                if f"{pascal}Repository" in target_path or f"{pascal}Repository" in code:
                    has_repo.add(sp)
                if f"{pascal}Service" in target_path or f"{pascal}Service" in code:
                    has_service.add(sp)

        repo_only = has_repo - has_service
        return GateResult(
            gate_name=gate_def.name,
            passed=len(repo_only) == 0,
            details={
                "source_sps": sorted(source_sps),
                "has_repo": sorted(has_repo),
                "has_service": sorted(has_service),
                "repo_only_missing_service": sorted(repo_only),
                "coverage": (
                    len(has_service) / len(source_sps)
                    if source_sps
                    else 1.0
                ),
            },
            blocking=gate_def.blocking,
        )

    # ── Asset Strategy Overrides ────────────────────────────────

    def get_asset_strategy_overrides(self) -> Dict[str, Dict[str, Any]]:
        """Return default strategies for SQL asset sub-types.

        Stored procedures, functions, and triggers are transformed;
        views are lower priority; generic SQL blocks are kept as-is.
        """
        return {
            "stored_procedure": {"strategy": "transform", "priority": "high"},
            "sql_function": {"strategy": "transform", "priority": "medium"},
            "trigger": {"strategy": "transform", "priority": "medium"},
            "view": {"strategy": "transform", "priority": "low"},
            "block": {"strategy": "keep_as_is", "priority": "low"},
        }

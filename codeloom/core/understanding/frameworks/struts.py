"""Struts 1.x / Struts 2.x framework analyzer.

Detects:
- Struts action mappings (1.x struts-config.xml and 2.x struts.xml)
- Form-bean declarations (DI registrations)
- Servlet filter pipeline from web.xml
- Security-related action attributes (roles, validate)
- Struts-specific analysis hints for LLM context
"""

import logging
from typing import Any, Dict, List
from uuid import UUID

from sqlalchemy import text

from ...db import DatabaseManager
from .base import FrameworkAnalyzer, FrameworkContext

logger = logging.getLogger(__name__)


class StrutsAnalyzer(FrameworkAnalyzer):

    def detect(self, project_id: str) -> bool:
        """Return True if Struts action units exist in the project."""
        pid = UUID(project_id)
        with self._db.get_session() as session:
            result = session.execute(
                text("""
                    SELECT COUNT(*) AS cnt FROM code_units
                    WHERE project_id = :pid
                      AND unit_type IN ('struts_action', 'struts2_action')
                """),
                {"pid": pid},
            )
            return result.fetchone().cnt > 0

    def analyze(self, project_id: str) -> FrameworkContext:
        """Analyze Struts-specific patterns and return framework context."""
        pid = UUID(project_id)

        # Determine framework version from unit types present
        framework_type = self._detect_version(pid)
        framework_name = (
            "Apache Struts 2" if framework_type == "struts2" else "Apache Struts 1"
        )

        ctx = FrameworkContext(
            framework_name=framework_name,
            framework_type=framework_type,
        )

        with self._db.get_session() as session:
            # DI registrations: form-bean declarations
            fb_result = session.execute(
                text("""
                    SELECT name, unit_metadata->>'type' AS bean_type
                    FROM code_units
                    WHERE project_id = :pid
                      AND unit_type = 'struts_form_bean'
                    ORDER BY name LIMIT 50
                """),
                {"pid": pid},
            )
            ctx.di_registrations = [
                f"{row.name} -> {row.bean_type}" if row.bean_type else row.name
                for row in fb_result.fetchall()
            ]

            # Middleware pipeline: servlet filters from web.xml
            filter_result = session.execute(
                text("""
                    SELECT name, unit_metadata->>'class_name' AS filter_class
                    FROM code_units
                    WHERE project_id = :pid
                      AND unit_type = 'xml_filter'
                    ORDER BY name LIMIT 20
                """),
                {"pid": pid},
            )
            ctx.middleware_pipeline = [
                f"{row.name} ({row.filter_class})" if row.filter_class else row.name
                for row in filter_result.fetchall()
            ]

            # Security config: actions with validate=true or roles attribute
            sec_result = session.execute(
                text("""
                    SELECT name,
                           unit_metadata->>'validate' AS validate_attr,
                           unit_metadata->>'input' AS input_attr
                    FROM code_units
                    WHERE project_id = :pid
                      AND unit_type IN ('struts_action', 'struts2_action')
                      AND (
                          unit_metadata->>'validate' = 'true'
                          OR unit_metadata->>'input' IS NOT NULL
                      )
                    ORDER BY name LIMIT 30
                """),
                {"pid": pid},
            )
            for row in sec_result.fetchall():
                info: Dict[str, Any] = {}
                if row.validate_attr == "true":
                    info["validation"] = "enabled"
                if row.input_attr:
                    info["input_page"] = row.input_attr
                if info:
                    ctx.security_config[row.name] = info

            # Version detection from pom.xml dependencies
            version = self._detect_dependency_version(pid, session)
            if version:
                ctx.version = version

        # Struts-specific hints for LLM
        if framework_type == "struts2":
            ctx.analysis_hints = [
                "Struts 2 uses ValueStack/OGNL for data binding -- watch for OGNL injection vulnerabilities",
                "Interceptor stack ordering matters -- security interceptors must precede action execution",
                "Struts 2 actions are instantiated per-request (unlike Struts 1 singletons)",
                "Check for wildcard mappings and dynamic method invocation (DMI) security risks",
            ]
        else:
            ctx.analysis_hints = [
                "Struts 1 Actions are singletons -- instance variables cause thread-safety issues",
                "ActionForm validation runs before Action.execute() -- check validate() and reset() methods",
                "Global forwards and module-relative forwards have different path resolution semantics",
                "Struts 1 uses JSTL/EL; check for scriptlet usage that should be migrated to EL expressions",
            ]

        return ctx

    def _detect_version(self, pid: UUID) -> str:
        """Determine whether project uses Struts 1 or Struts 2."""
        with self._db.get_session() as session:
            result = session.execute(
                text("""
                    SELECT
                        SUM(CASE WHEN unit_type = 'struts2_action' THEN 1 ELSE 0 END) AS s2,
                        SUM(CASE WHEN unit_type = 'struts_action' THEN 1 ELSE 0 END) AS s1
                    FROM code_units
                    WHERE project_id = :pid
                      AND unit_type IN ('struts_action', 'struts2_action')
                """),
                {"pid": pid},
            )
            row = result.fetchone()
            if row and (row.s2 or 0) > 0:
                return "struts2"
            return "struts1"

    @staticmethod
    def _detect_dependency_version(pid: UUID, session) -> str:
        """Try to extract Struts version from pom.xml dependency units."""
        result = session.execute(
            text("""
                SELECT unit_metadata->>'version' AS version
                FROM code_units
                WHERE project_id = :pid
                  AND unit_type = 'xml_dependency'
                  AND lower(name) LIKE '%struts%'
                LIMIT 1
            """),
            {"pid": pid},
        )
        row = result.fetchone()
        return row.version if row and row.version else ""

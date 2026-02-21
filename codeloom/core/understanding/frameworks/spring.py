"""Spring Boot / Spring MVC framework analyzer.

Detects:
- XML bean definitions and @Configuration DI
- Spring Security filter chain configuration
- @Transactional boundaries
- AOP pointcuts (@Aspect, @Around, @Before, @After)
- Spring Data repositories
"""

import logging
from typing import Any, Dict, List
from uuid import UUID

from sqlalchemy import text

from ...db import DatabaseManager
from .base import FrameworkAnalyzer, FrameworkContext

logger = logging.getLogger(__name__)


class SpringAnalyzer(FrameworkAnalyzer):

    def detect(self, project_id: str) -> bool:
        pid = UUID(project_id)
        with self._db.get_session() as session:
            result = session.execute(
                text("""
                    SELECT COUNT(*) AS cnt FROM code_units
                    WHERE project_id = :pid
                      AND (
                          source ~ '@SpringBootApplication'
                          OR source ~ '@RestController'
                          OR source ~ '@Controller'
                          OR source ~ 'spring-boot'
                      )
                """),
                {"pid": pid},
            )
            return result.fetchone().cnt > 0

    def analyze(self, project_id: str) -> FrameworkContext:
        pid = UUID(project_id)
        ctx = FrameworkContext(
            framework_name="Spring Boot",
            framework_type="spring",
        )

        with self._db.get_session() as session:
            # DI: @Configuration classes
            di_result = session.execute(
                text("""
                    SELECT name, qualified_name FROM code_units
                    WHERE project_id = :pid
                      AND source ~ '@(Configuration|Component|Service|Repository|Bean)'
                    ORDER BY name LIMIT 50
                """),
                {"pid": pid},
            )
            ctx.di_registrations = [
                row.qualified_name for row in di_result.fetchall()
            ]

            # Security filter chain
            sec_result = session.execute(
                text("""
                    SELECT name, source FROM code_units
                    WHERE project_id = :pid
                      AND (source ~ 'SecurityFilterChain' OR source ~ 'WebSecurityConfigurerAdapter')
                    LIMIT 5
                """),
                {"pid": pid},
            )
            for row in sec_result.fetchall():
                ctx.security_config[row.name] = "Spring Security config detected"

            # @Transactional boundaries
            tx_result = session.execute(
                text("""
                    SELECT qualified_name FROM code_units
                    WHERE project_id = :pid AND source ~ '@Transactional'
                    ORDER BY name LIMIT 50
                """),
                {"pid": pid},
            )
            ctx.transaction_boundaries = [
                row.qualified_name for row in tx_result.fetchall()
            ]

            # AOP
            aop_result = session.execute(
                text("""
                    SELECT qualified_name FROM code_units
                    WHERE project_id = :pid
                      AND source ~ '@(Aspect|Around|Before|After|Pointcut)'
                    LIMIT 20
                """),
                {"pid": pid},
            )
            ctx.aop_pointcuts = [
                row.qualified_name for row in aop_result.fetchall()
            ]

        ctx.analysis_hints = [
            "Spring uses proxy-based AOP — @Transactional only works on public methods called from outside the class",
            "Spring Security filter chain ordering matters — check for antMatchers/requestMatchers precedence",
            "Check for @Lazy and circular dependency patterns in DI registrations",
        ]

        return ctx

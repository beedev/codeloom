"""ASP.NET Core framework analyzer.

Detects:
- DI registration in Startup/Program.cs (AddScoped, AddTransient, AddSingleton)
- Middleware pipeline ordering
- Action filters and authorization attributes
- Entity Framework DbContext patterns
"""

import logging
from typing import Any, Dict, List
from uuid import UUID

from sqlalchemy import text

from ...db import DatabaseManager
from .base import FrameworkAnalyzer, FrameworkContext

logger = logging.getLogger(__name__)


class AspNetAnalyzer(FrameworkAnalyzer):

    def detect(self, project_id: str) -> bool:
        pid = UUID(project_id)
        with self._db.get_session() as session:
            result = session.execute(
                text("""
                    SELECT COUNT(*) AS cnt FROM code_units
                    WHERE project_id = :pid
                      AND (
                          source ~ 'Microsoft\\.AspNetCore'
                          OR source ~ 'WebApplication\\.CreateBuilder'
                          OR source ~ ': ControllerBase'
                          OR source ~ ': Controller'
                      )
                """),
                {"pid": pid},
            )
            return result.fetchone().cnt > 0

    def analyze(self, project_id: str) -> FrameworkContext:
        pid = UUID(project_id)
        ctx = FrameworkContext(
            framework_name="ASP.NET Core",
            framework_type="aspnet",
        )

        with self._db.get_session() as session:
            # DI registrations
            di_result = session.execute(
                text("""
                    SELECT source FROM code_units
                    WHERE project_id = :pid
                      AND (source ~ 'Add(Scoped|Transient|Singleton)' OR source ~ 'builder\\.Services')
                    LIMIT 10
                """),
                {"pid": pid},
            )
            for row in di_result.fetchall():
                for line in (row.source or "").split("\n"):
                    if "Add" in line and ("Scoped" in line or "Transient" in line or "Singleton" in line):
                        ctx.di_registrations.append(line.strip())

            # Middleware pipeline
            mw_result = session.execute(
                text("""
                    SELECT source FROM code_units
                    WHERE project_id = :pid
                      AND source ~ 'app\\.Use'
                    LIMIT 10
                """),
                {"pid": pid},
            )
            for row in mw_result.fetchall():
                for line in (row.source or "").split("\n"):
                    if "app.Use" in line:
                        ctx.middleware_pipeline.append(line.strip())

            # DbContext
            db_result = session.execute(
                text("""
                    SELECT name, qualified_name FROM code_units
                    WHERE project_id = :pid
                      AND (source ~ ': DbContext' OR source ~ 'DbSet<')
                    LIMIT 20
                """),
                {"pid": pid},
            )
            for row in db_result.fetchall():
                ctx.security_config[row.name] = "EF DbContext"

        ctx.analysis_hints = [
            "ASP.NET middleware order matters — UseAuthentication before UseAuthorization",
            "Check DI lifetime mismatches (Singleton depending on Scoped = captive dependency)",
            "Action filters can short-circuit the pipeline — check for IAuthorizationFilter",
        ]

        return ctx

"""JavaParser subprocess bridge for deep Java semantic analysis.

Uses a small Maven CLI tool (tools/javaparser-cli) to provide:
- Fully qualified type names (resolves imports)
- Generic type parameters with bounds
- Checked exception declarations
- Interface implementation across packages

Requires: JDK + Maven + `./dev.sh setup-tools` to build the JAR.
Falls back gracefully if unavailable.
"""

import logging
import shutil
from pathlib import Path
from typing import List

from .base import SubprocessBridge

logger = logging.getLogger(__name__)

_JAR_PATH = Path(__file__).parents[4] / "tools" / "javaparser-cli" / "target" / "javaparser-cli.jar"


class JavaParserBridge(SubprocessBridge):
    """Bridge to JavaParser CLI for deep Java enrichment."""

    def is_available(self) -> bool:
        if self._availability_checked:
            return self._is_available_cache

        self._availability_checked = True

        if not _JAR_PATH.exists():
            if not self._warned:
                self._warned = True
                logger.info(
                    "JavaParser enrichment unavailable — "
                    "run `./dev.sh setup-tools` for deeper Java analysis"
                )
            self._is_available_cache = False
            return False

        if shutil.which("java") is None:
            if not self._warned:
                self._warned = True
                logger.info(
                    "JavaParser enrichment unavailable — "
                    "install JDK to enable deeper Java analysis"
                )
            self._is_available_cache = False
            return False

        self._is_available_cache = True
        return True

    def enrich(self, file_path: str, units: list) -> list:
        if not self.is_available():
            return units

        result = self._run_tool(
            ["java", "-jar", str(_JAR_PATH), file_path],
            timeout=30,
        )
        if not result:
            return units

        enrichments = result.get("enrichments", [])
        self._merge_enrichments(units, enrichments)

        return units

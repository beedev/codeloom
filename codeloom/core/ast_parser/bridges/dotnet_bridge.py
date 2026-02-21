"""Roslyn subprocess bridge for deep C# semantic analysis.

Uses a small .NET console app (tools/roslyn-analyzer) to provide:
- Fully qualified type names
- Nullable reference type annotations
- Generic constraints
- Override chains
- LINQ expression types

Requires: .NET SDK + `./dev.sh setup-tools` to build the DLL.
Falls back gracefully if unavailable.
"""

import logging
import shutil
from pathlib import Path
from typing import List

from .base import SubprocessBridge

logger = logging.getLogger(__name__)

_PROJECT_DIR = Path(__file__).parents[4] / "tools" / "roslyn-analyzer"
_DLL_PATH = _PROJECT_DIR / "bin" / "Release" / "net8.0" / "roslyn-analyzer.dll"


class RoslynBridge(SubprocessBridge):
    """Bridge to Roslyn Analyzer CLI for deep C# enrichment."""

    def is_available(self) -> bool:
        if self._availability_checked:
            return self._is_available_cache

        self._availability_checked = True

        if not _DLL_PATH.exists():
            if not self._warned:
                self._warned = True
                logger.info(
                    "Roslyn enrichment unavailable — "
                    "run `./dev.sh setup-tools` for deeper C# analysis"
                )
            self._is_available_cache = False
            return False

        if shutil.which("dotnet") is None:
            if not self._warned:
                self._warned = True
                logger.info(
                    "Roslyn enrichment unavailable — "
                    "install .NET SDK to enable deeper C# analysis"
                )
            self._is_available_cache = False
            return False

        self._is_available_cache = True
        return True

    def enrich(self, file_path: str, units: list) -> list:
        if not self.is_available():
            return units

        result = self._run_tool(
            ["dotnet", str(_DLL_PATH), file_path],
            timeout=30,
        )
        if not result:
            return units

        enrichments = result.get("enrichments", [])
        self._merge_enrichments(units, enrichments)

        return units

"""Base class for subprocess-based semantic analysis bridges.

Each bridge wraps an external CLI tool (JavaParser, Roslyn) that provides
deeper semantic analysis than tree-sitter alone. Bridges are designed to
fail gracefully — if the runtime or tool is unavailable, enrichment is
silently skipped.
"""

import json
import logging
import subprocess
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SubprocessBridge(ABC):
    """Abstract base for subprocess-based semantic enrichment bridges.

    Subclasses wrap a CLI tool that reads a source file and outputs
    enrichment JSON to stdout.
    """

    # Cache availability check per session to avoid repeated subprocess calls
    _availability_checked: bool = False
    _is_available_cache: bool = False
    _warned: bool = False

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the runtime and compiled tool exist.

        Returns:
            True if the bridge can run enrichment
        """
        ...

    @abstractmethod
    def enrich(self, file_path: str, units: list) -> list:
        """Enrich code units with deeper semantic metadata.

        Merges additional metadata into existing CodeUnit.metadata dicts.
        Returns units unchanged if tool is unavailable.

        Args:
            file_path: Absolute path to the source file
            units: List of CodeUnit objects from tree-sitter parsing

        Returns:
            The same list of units (modified in-place)
        """
        ...

    def _run_tool(self, cmd: List[str], timeout: int = 30) -> Optional[Dict[str, Any]]:
        """Run a CLI tool and parse JSON output from stdout.

        Args:
            cmd: Command and arguments to execute
            timeout: Maximum execution time in seconds

        Returns:
            Parsed JSON dict, or None on any failure
        """
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if proc.returncode != 0:
                logger.debug(f"Bridge tool failed (exit {proc.returncode}): {proc.stderr[:200]}")
                return None

            if not proc.stdout.strip():
                return None

            return json.loads(proc.stdout)

        except subprocess.TimeoutExpired:
            logger.debug(f"Bridge tool timed out after {timeout}s: {cmd[0]}")
            return None
        except json.JSONDecodeError as e:
            logger.debug(f"Bridge tool produced invalid JSON: {e}")
            return None
        except FileNotFoundError:
            logger.debug(f"Bridge tool not found: {cmd[0]}")
            return None
        except Exception as e:
            logger.debug(f"Bridge tool error: {e}")
            return None

    def _merge_enrichments(
        self,
        units: list,
        enrichments: List[Dict[str, Any]],
    ) -> None:
        """Merge enrichment data from tool output into matching CodeUnit.metadata.

        Matches enrichments to units by qualified_name.
        """
        if not enrichments:
            return

        # Build lookup by qualified name
        enrich_by_qn: Dict[str, Dict] = {}
        for e in enrichments:
            qn = e.get("qualified_name", "")
            if qn:
                enrich_by_qn[qn] = e

        for unit in units:
            match = enrich_by_qn.get(unit.qualified_name)
            if not match:
                continue

            meta = unit.metadata

            # Merge resolved params (prefer bridge over tree-sitter)
            if "resolved_params" in match:
                meta["parsed_params"] = match["resolved_params"]

            if "resolved_return_type" in match:
                meta["return_type"] = match["resolved_return_type"]

            if "thrown_exceptions" in match:
                meta["thrown_exceptions"] = match["thrown_exceptions"]

            if "implements_interfaces" in match:
                meta["implements"] = match["implements_interfaces"]

            if "generic_type_params" in match:
                meta["generic_params"] = match["generic_type_params"]

            if "nullable_annotations" in match:
                meta["nullable_annotations"] = match["nullable_annotations"]

"""Java properties file parser — line-based.

Extracts a single CodeUnit per .properties file with metadata about
keys, categorized into message keys (UI-facing) and config keys.

Does NOT use tree-sitter. Follows the regex/fallback parser pattern
with parse_file/parse_source interface.
"""

import logging
import re
from typing import Any, Dict, List

from .models import CodeUnit, ParseResult

logger = logging.getLogger(__name__)

# Words that indicate a key is a UI message rather than configuration
_MESSAGE_INDICATORS = frozenset({
    "title", "label", "error", "message", "prompt",
    "button", "heading", "header", "footer", "text",
    "tooltip", "placeholder", "hint", "caption", "description",
    "warning", "info", "confirm", "success", "fail",
})

# Matches key=value or key: value (with optional leading whitespace)
_KV_RE = re.compile(r"^\s*([^#!\s][^=:\s]*)\s*[=:]\s*(.*)", re.MULTILINE)


def _is_message_key(key: str) -> bool:
    """Check if a property key looks like a UI message key."""
    key_lower = key.lower()
    return any(indicator in key_lower for indicator in _MESSAGE_INDICATORS)


class PropertiesParser:
    """Line-based parser for Java .properties files.

    Each properties file becomes one CodeUnit of type 'properties_file'
    with metadata listing all keys, categorized as message or config.
    """

    def get_language(self) -> str:
        return "properties"

    def parse_file(self, file_path: str, project_root: str = "") -> ParseResult:
        """Parse a .properties file into a single CodeUnit."""
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
                language="properties",
                units=[],
                imports=[],
                line_count=0,
            )

        return self.parse_source(source_text, rel_path)

    def parse_source(self, source_text: str, file_path: str) -> ParseResult:
        """Parse properties source text into a single CodeUnit."""
        line_count = source_text.count("\n") + (
            1 if source_text and not source_text.endswith("\n") else 0
        )

        keys: List[str] = []
        message_keys: List[str] = []
        config_keys: List[str] = []

        for m in _KV_RE.finditer(source_text):
            key = m.group(1).strip()
            if not key:
                continue
            keys.append(key)
            if _is_message_key(key):
                message_keys.append(key)
            else:
                config_keys.append(key)

        metadata: Dict[str, Any] = {
            "key_count": len(keys),
            "keys": keys,
            "message_keys": message_keys,
            "config_keys": config_keys,
        }

        signature = f"properties {file_path} keys={len(keys)}"

        unit = CodeUnit(
            unit_type="properties_file",
            name=file_path,
            qualified_name=f"{file_path}:properties",
            language="properties",
            start_line=1,
            end_line=line_count,
            source=source_text,
            file_path=file_path,
            signature=signature,
            metadata=metadata,
        )

        return ParseResult(
            file_path=file_path,
            language="properties",
            units=[unit],
            imports=[],
            line_count=line_count,
        )

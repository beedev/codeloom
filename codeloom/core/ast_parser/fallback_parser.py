"""Fallback parser for unsupported file types.

Splits source code at blank lines into ~1024 token blocks.
No structural extraction — just embeddable text blocks.
"""

import logging
from typing import List

from .models import CodeUnit, ParseResult

logger = logging.getLogger(__name__)

# Approximate characters per token (conservative for code)
CHARS_PER_TOKEN = 4
DEFAULT_MAX_TOKENS = 1024


class FallbackParser:
    """Blank-line splitting parser for unsupported languages.

    Produces CodeUnit objects with unit_type="block" that can be
    chunked and embedded, but lack structural metadata.
    """

    def __init__(self, max_tokens: int = DEFAULT_MAX_TOKENS):
        self._max_chars = max_tokens * CHARS_PER_TOKEN

    def parse_file(self, file_path: str, project_root: str = "") -> ParseResult:
        """Parse a file by splitting on blank lines."""
        if project_root and file_path.startswith(project_root):
            rel_path = file_path[len(project_root):].lstrip("/")
        else:
            rel_path = file_path

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                source_text = f.read()
        except OSError as e:
            logger.error(f"Cannot read {file_path}: {e}")
            return ParseResult(
                file_path=rel_path,
                language="unknown",
                units=[],
                imports=[],
                line_count=0,
            )

        return self.parse_source(source_text, rel_path)

    def parse_source(self, source_text: str, file_path: str) -> ParseResult:
        """Parse source text by splitting on blank lines."""
        lines = source_text.split("\n")
        line_count = len(lines)

        # Split into blocks at blank lines
        blocks: List[tuple[int, int, str]] = []
        current_lines: list[str] = []
        block_start = 1

        for i, line in enumerate(lines, start=1):
            if line.strip() == "" and current_lines:
                block_text = "\n".join(current_lines)
                if block_text.strip():
                    blocks.append((block_start, i - 1, block_text))
                current_lines = []
                block_start = i + 1
            else:
                if not current_lines:
                    block_start = i
                current_lines.append(line)

        # Last block
        if current_lines:
            block_text = "\n".join(current_lines)
            if block_text.strip():
                blocks.append((block_start, block_start + len(current_lines) - 1, block_text))

        # Merge small blocks, split large ones
        units: List[CodeUnit] = []
        merged_text = ""
        merged_start = 1
        merged_end = 1
        block_idx = 0

        for start, end, text in blocks:
            if merged_text and len(merged_text) + len(text) + 1 > self._max_chars:
                # Emit current merged block
                block_idx += 1
                units.append(self._make_unit(merged_text, merged_start, merged_end, file_path, block_idx))
                merged_text = text
                merged_start = start
                merged_end = end
            elif not merged_text:
                merged_text = text
                merged_start = start
                merged_end = end
            else:
                merged_text += "\n\n" + text
                merged_end = end

        # Emit last block
        if merged_text.strip():
            block_idx += 1
            units.append(self._make_unit(merged_text, merged_start, merged_end, file_path, block_idx))

        return ParseResult(
            file_path=file_path,
            language="unknown",
            units=units,
            imports=[],
            line_count=line_count,
        )

    @staticmethod
    def _make_unit(text: str, start: int, end: int, file_path: str, idx: int) -> CodeUnit:
        return CodeUnit(
            unit_type="block",
            name=f"block_{idx}",
            qualified_name=f"{file_path}::block_{idx}",
            language="unknown",
            start_line=start,
            end_line=end,
            source=text,
            file_path=file_path,
        )

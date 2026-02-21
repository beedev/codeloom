"""Code chunker — converts ParseResult into embeddable TextNode objects.

Each CodeUnit maps 1:1 to a TextNode with preamble injection.
Large units are split at logical boundaries.
"""

import logging
from typing import List

from llama_index.core.schema import TextNode

from ..ast_parser.models import CodeUnit, ParseResult
from .preamble import PreambleBuilder
from .token_counter import TokenCounter

logger = logging.getLogger(__name__)


class CodeChunker:
    """Convert parsed code units into embeddable TextNode objects.

    Each chunk = preamble + code unit source.
    Metadata matches PGVectorStore expectations.
    """

    def __init__(self, max_tokens: int = 1024, encoding_name: str = "cl100k_base"):
        self._max_tokens = max_tokens
        self._preamble_builder = PreambleBuilder()
        self._token_counter = TokenCounter(encoding_name)

    def chunk_file(
        self,
        parse_result: ParseResult,
        project_id: str,
        file_id: str,
    ) -> List[TextNode]:
        """Convert all CodeUnits from a parsed file into TextNodes.

        Args:
            parse_result: Output from AST parser
            project_id: Project UUID string
            file_id: CodeFile UUID string (used as source_id)

        Returns:
            List of TextNode objects ready for embedding
        """
        nodes: List[TextNode] = []

        for unit in parse_result.units:
            preamble = self._preamble_builder.build(
                file_path=parse_result.file_path,
                imports=parse_result.imports,
                parent_class=unit.parent_name,
            )

            text = f"{preamble}\n\n{unit.source}"
            token_count = self._token_counter.count(text)

            if token_count <= self._max_tokens:
                node = self._create_text_node(unit, text, project_id, file_id, parse_result)
                nodes.append(node)
            else:
                # Split oversized units
                split_nodes = self._split_unit(unit, preamble, project_id, file_id, parse_result)
                nodes.extend(split_nodes)

        return nodes

    def _create_text_node(
        self,
        unit: CodeUnit,
        text: str,
        project_id: str,
        file_id: str,
        parse_result: ParseResult,
    ) -> TextNode:
        """Create a TextNode with PGVectorStore-compatible metadata."""
        return TextNode(
            text=text,
            metadata={
                "unit_id": unit.metadata.get("unit_id"),
                "project_id": project_id,
                "source_id": file_id,
                "file_name": parse_result.file_path,
                "node_type": "code",
                "unit_type": unit.unit_type,
                "unit_name": unit.name,
                "qualified_name": unit.qualified_name,
                "class_name": unit.parent_name,
                "language": unit.language,
                "start_line": unit.start_line,
                "end_line": unit.end_line,
                "signature": unit.signature,
                "has_docstring": bool(unit.docstring),
            },
            excluded_embed_metadata_keys=[
                "unit_id",
                "project_id",
                "source_id",
                "node_type",
                "start_line",
                "end_line",
                "has_docstring",
            ],
        )

    def _split_unit(
        self,
        unit: CodeUnit,
        preamble: str,
        project_id: str,
        file_id: str,
        parse_result: ParseResult,
    ) -> List[TextNode]:
        """Split an oversized code unit into smaller chunks.

        Strategy: split at blank lines or function boundaries within
        the source. Each sub-chunk gets the same preamble.
        """
        lines = unit.source.split("\n")
        chunks: List[str] = []
        current_lines: list[str] = []
        preamble_tokens = self._token_counter.count(preamble + "\n\n")
        max_chunk_tokens = self._max_tokens - preamble_tokens

        for line in lines:
            current_lines.append(line)
            candidate = "\n".join(current_lines)
            if self._token_counter.count(candidate) > max_chunk_tokens:
                # Emit current chunk (without the last line)
                if len(current_lines) > 1:
                    chunk_text = "\n".join(current_lines[:-1])
                    if chunk_text.strip():
                        chunks.append(chunk_text)
                    current_lines = [line]
                else:
                    # Single line exceeds limit — include it anyway
                    chunks.append(candidate)
                    current_lines = []

        # Remaining lines
        if current_lines:
            chunk_text = "\n".join(current_lines)
            if chunk_text.strip():
                chunks.append(chunk_text)

        # Create TextNodes for each sub-chunk
        nodes: List[TextNode] = []
        for i, chunk in enumerate(chunks):
            text = f"{preamble}\n\n{chunk}"
            # Create a modified unit for metadata
            sub_name = f"{unit.name}_part{i + 1}" if len(chunks) > 1 else unit.name
            sub_unit = CodeUnit(
                unit_type=unit.unit_type,
                name=sub_name,
                qualified_name=f"{unit.qualified_name}_part{i + 1}" if len(chunks) > 1 else unit.qualified_name,
                language=unit.language,
                start_line=unit.start_line,
                end_line=unit.end_line,
                source=chunk,
                file_path=unit.file_path,
                signature=unit.signature,
                docstring=unit.docstring if i == 0 else None,
                parent_name=unit.parent_name,
            )
            node = self._create_text_node(sub_unit, text, project_id, file_id, parse_result)
            nodes.append(node)

        if not nodes:
            # Fallback: emit the whole thing even if oversized
            text = f"{preamble}\n\n{unit.source}"
            node = self._create_text_node(unit, text, project_id, file_id, parse_result)
            nodes.append(node)

        return nodes

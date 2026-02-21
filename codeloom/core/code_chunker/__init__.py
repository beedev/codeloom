"""CodeLoom Code Chunker — AST-informed code chunking.

Converts parsed code units into embeddable TextNode objects
with preamble injection for self-contained context.

Public API:
    chunk_file(parse_result, project_id, file_id) → List[TextNode]
"""

from .chunker import CodeChunker

__all__ = ["CodeChunker", "chunk_file"]

# Module-level convenience instance
_default_chunker = None


def chunk_file(parse_result, project_id: str, file_id: str):
    """Convenience function using default chunker settings."""
    global _default_chunker
    if _default_chunker is None:
        _default_chunker = CodeChunker()
    return _default_chunker.chunk_file(parse_result, project_id, file_id)

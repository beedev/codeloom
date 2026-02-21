"""CodeLoom AST Parser — tree-sitter based code parsing.

Public API:
    parse_file(path, project_root) → ParseResult
    parse_source(source, file_path, language) → ParseResult
    detect_language(file_path) → str | None
"""

from .models import CodeUnit, ParseError, ParseResult
from .utils import detect_language, get_parser, is_supported_file, should_skip_directory

__all__ = [
    "parse_file",
    "parse_source",
    "detect_language",
    "is_supported_file",
    "should_skip_directory",
    "CodeUnit",
    "ParseError",
    "ParseResult",
]


def parse_file(file_path: str, project_root: str = "") -> ParseResult:
    """Parse a source file into structured code units.

    Detects language from file extension and uses the appropriate
    tree-sitter parser. Falls back to blank-line splitting for
    unsupported languages.

    Args:
        file_path: Absolute path to the source file
        project_root: Project root for computing relative paths

    Returns:
        ParseResult containing extracted code units
    """
    language = detect_language(file_path)

    if language:
        parser = get_parser(language)
        return parser.parse_file(file_path, project_root)
    else:
        from .fallback_parser import FallbackParser
        return FallbackParser().parse_file(file_path, project_root)


def parse_source(source_text: str, file_path: str, language: str | None = None) -> ParseResult:
    """Parse source code string into structured code units.

    Args:
        source_text: Source code as string
        file_path: Relative file path (for metadata)
        language: Language identifier. If None, detected from file_path.

    Returns:
        ParseResult containing extracted code units
    """
    if language is None:
        language = detect_language(file_path)

    if language:
        parser = get_parser(language)
        return parser.parse_source(source_text, file_path)
    else:
        from .fallback_parser import FallbackParser
        return FallbackParser().parse_source(source_text, file_path)

"""AST Parser utilities.

Language detection, parser registry, and helper functions.
"""

import os
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseLanguageParser

# Extension → language mapping
SUPPORTED_EXTENSIONS: Dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".cs": "csharp",
    ".sql": "sql",
    ".vb": "vbnet",
}

# Directories to skip during file walking
SKIP_DIRECTORIES = frozenset({
    "__pycache__",
    ".git",
    "venv",
    "env",
    ".venv",
    ".env",
    "node_modules",
    ".tox",
    "__pypackages__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".eggs",
    "*.egg-info",
    # C# / .NET
    "bin",
    "obj",
    "packages",
})

# Parser registry — lazy-loaded to avoid import overhead
_parser_registry: Dict[str, "BaseLanguageParser"] = {}


def detect_language(file_path: str) -> Optional[str]:
    """Detect programming language from file extension.

    Args:
        file_path: Path to the source file

    Returns:
        Language identifier string or None if unsupported
    """
    _, ext = os.path.splitext(file_path)
    return SUPPORTED_EXTENSIONS.get(ext.lower())


def get_parser(language: str) -> "BaseLanguageParser":
    """Get a parser instance for the given language.

    Uses a lazy-initialized registry to avoid importing
    all parser modules at startup.

    Args:
        language: Language identifier (e.g., "python")

    Returns:
        Parser instance

    Raises:
        ValueError: If language is not supported
    """
    if language not in _parser_registry:
        if language == "python":
            from .python_parser import PythonParser
            _parser_registry["python"] = PythonParser()
        elif language == "javascript":
            from .javascript_parser import JavaScriptParser
            _parser_registry["javascript"] = JavaScriptParser()
        elif language == "typescript":
            from .typescript_parser import TypeScriptParser
            _parser_registry["typescript"] = TypeScriptParser()
        elif language == "java":
            from .java_parser import JavaParser
            _parser_registry["java"] = JavaParser()
        elif language == "csharp":
            from .csharp_parser import CSharpParser
            _parser_registry["csharp"] = CSharpParser()
        elif language == "sql":
            from .sql_parser import SqlParser
            _parser_registry["sql"] = SqlParser()
        elif language == "vbnet":
            from .vbnet_parser import VbNetParser
            _parser_registry["vbnet"] = VbNetParser()
        else:
            raise ValueError(
                f"Unsupported language: {language}. "
                f"Supported: {list(SUPPORTED_EXTENSIONS.values())}"
            )

    return _parser_registry[language]


def should_skip_directory(dir_name: str) -> bool:
    """Check if a directory should be skipped during file walking.

    Args:
        dir_name: Directory name (not full path)

    Returns:
        True if directory should be skipped
    """
    return dir_name in SKIP_DIRECTORIES or dir_name.startswith(".")


def is_supported_file(file_path: str) -> bool:
    """Check if a file has a supported language extension.

    Args:
        file_path: Path to the file

    Returns:
        True if file type is supported for AST parsing
    """
    return detect_language(file_path) is not None

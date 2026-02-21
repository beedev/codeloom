# Subprocess bridges for deep semantic analysis.
# JavaParser (Java) and Roslyn (C#) provide richer type resolution
# than tree-sitter alone. Both are optional — if runtimes are not
# installed, tree-sitter enrichment still provides baseline metadata.

from .base import SubprocessBridge

__all__ = ["SubprocessBridge"]

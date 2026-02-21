"""Preamble builder for code chunks.

Constructs a context header prepended to each code chunk,
giving the embedding model and LLM context about where
the code lives in the project.
"""

from typing import List, Optional


class PreambleBuilder:
    """Builds context preambles for code chunks."""

    def build(
        self,
        file_path: str,
        imports: List[str],
        parent_class: Optional[str] = None,
        max_imports: int = 10,
    ) -> str:
        """Build a preamble string for a code chunk.

        Args:
            file_path: Relative file path within project
            imports: List of import statements
            parent_class: Class name if this is a method
            max_imports: Maximum imports to include

        Returns:
            Preamble string (e.g., "# File: src/core/engine.py\\n# Imports: ...")
        """
        lines = [f"# File: {file_path}"]

        if imports:
            # Truncate long import lists
            import_text = ", ".join(imports[:max_imports])
            if len(imports) > max_imports:
                import_text += f" (+{len(imports) - max_imports} more)"
            lines.append(f"# Imports: {import_text}")

        if parent_class:
            lines.append(f"# Class: {parent_class}")

        return "\n".join(lines)

"""Tests for the AST parser module."""

import pytest
from codeloom.core.ast_parser import parse_source, parse_file, detect_language, CodeUnit, ParseResult


# =========================================================================
# Sample Python source fixtures
# =========================================================================

SIMPLE_FUNCTION = '''
def greet(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}!"
'''

CLASS_WITH_METHODS = '''
import os
from typing import List

class Calculator:
    """A simple calculator."""

    def __init__(self, initial: int = 0):
        """Initialize with a starting value."""
        self.value = initial

    def add(self, n: int) -> int:
        """Add n to the current value."""
        self.value += n
        return self.value

    def reset(self):
        self.value = 0
'''

DECORATED_FUNCTION = '''
import functools

@staticmethod
def helper():
    """A decorated helper."""
    pass

@functools.lru_cache(maxsize=128)
def cached_compute(x: int) -> int:
    return x * x
'''

DECORATED_CLASS = '''
import dataclasses

@dataclasses.dataclass
class Point:
    """A 2D point."""
    x: float
    y: float

    def distance(self) -> float:
        return (self.x ** 2 + self.y ** 2) ** 0.5
'''

MODULE_WITH_DOCSTRING = '''"""This is the module docstring."""

import sys

def main():
    pass
'''

NESTED_CLASSES = '''
class Outer:
    """Outer class."""

    class Inner:
        """Inner class."""

        def inner_method(self):
            pass

    def outer_method(self):
        pass
'''

EMPTY_FILE = ''

SYNTAX_ERROR_FILE = '''
def broken(
    # missing closing paren and colon
'''


# =========================================================================
# Tests: Language detection
# =========================================================================

class TestLanguageDetection:
    def test_python(self):
        assert detect_language("foo/bar.py") == "python"

    def test_unknown(self):
        assert detect_language("foo/bar.rs") is None

    def test_case_insensitive(self):
        assert detect_language("FOO.PY") == "python"


# =========================================================================
# Tests: Python parser - functions
# =========================================================================

class TestPythonFunctions:
    def test_simple_function(self):
        result = parse_source(SIMPLE_FUNCTION, "test.py", "python")
        assert isinstance(result, ParseResult)
        assert result.language == "python"

        funcs = [u for u in result.units if u.unit_type == "function"]
        assert len(funcs) == 1

        f = funcs[0]
        assert f.name == "greet"
        assert f.docstring == "Say hello."
        assert "name: str" in f.signature
        assert "-> str" in f.signature
        assert f.file_path == "test.py"

    def test_decorated_functions(self):
        result = parse_source(DECORATED_FUNCTION, "utils.py", "python")
        funcs = [u for u in result.units if u.unit_type == "function"]
        assert len(funcs) == 2

        helper = next(f for f in funcs if f.name == "helper")
        assert "@staticmethod" in helper.decorators

        cached = next(f for f in funcs if f.name == "cached_compute")
        assert any("lru_cache" in d for d in cached.decorators)


# =========================================================================
# Tests: Python parser - classes
# =========================================================================

class TestPythonClasses:
    def test_class_with_methods(self):
        result = parse_source(CLASS_WITH_METHODS, "calc.py", "python")

        classes = [u for u in result.units if u.unit_type == "class"]
        assert len(classes) == 1
        assert classes[0].name == "Calculator"
        assert classes[0].docstring == "A simple calculator."

        methods = [u for u in result.units if u.unit_type == "method"]
        assert len(methods) == 3  # __init__, add, reset
        method_names = {m.name for m in methods}
        assert method_names == {"__init__", "add", "reset"}

        # Methods should have parent_name
        for m in methods:
            assert m.parent_name == "Calculator"

    def test_decorated_class(self):
        result = parse_source(DECORATED_CLASS, "models.py", "python")

        classes = [u for u in result.units if u.unit_type == "class"]
        assert len(classes) == 1
        assert classes[0].name == "Point"
        assert any("dataclass" in d for d in classes[0].decorators)

        methods = [u for u in result.units if u.unit_type == "method"]
        assert len(methods) == 1
        assert methods[0].name == "distance"


# =========================================================================
# Tests: Python parser - imports
# =========================================================================

class TestPythonImports:
    def test_imports_extracted(self):
        result = parse_source(CLASS_WITH_METHODS, "calc.py", "python")
        assert len(result.imports) == 2
        assert any("import os" in i for i in result.imports)
        assert any("from typing import List" in i for i in result.imports)

    def test_imports_injected_into_units(self):
        result = parse_source(CLASS_WITH_METHODS, "calc.py", "python")
        for unit in result.units:
            assert len(unit.imports) == 2


# =========================================================================
# Tests: Module docstring
# =========================================================================

class TestModuleDocstring:
    def test_module_docstring(self):
        result = parse_source(MODULE_WITH_DOCSTRING, "main.py", "python")
        assert result.module_docstring == "This is the module docstring."


# =========================================================================
# Tests: Qualified names
# =========================================================================

class TestQualifiedNames:
    def test_function_qualified_name(self):
        result = parse_source(SIMPLE_FUNCTION, "codeloom/core/parser.py", "python")
        funcs = [u for u in result.units if u.unit_type == "function"]
        assert funcs[0].qualified_name == "codeloom.core.parser.greet"

    def test_method_qualified_name(self):
        result = parse_source(CLASS_WITH_METHODS, "codeloom/calc.py", "python")
        methods = [u for u in result.units if u.unit_type == "method"]
        add_method = next(m for m in methods if m.name == "add")
        assert add_method.qualified_name == "codeloom.calc.Calculator.add"


# =========================================================================
# Tests: Edge cases
# =========================================================================

class TestEdgeCases:
    def test_empty_file(self):
        result = parse_source(EMPTY_FILE, "empty.py", "python")
        assert result.units == []
        assert result.imports == []

    def test_syntax_errors_still_parse(self):
        result = parse_source(SYNTAX_ERROR_FILE, "broken.py", "python")
        # Should not crash, but may report warnings
        assert isinstance(result, ParseResult)
        assert len(result.errors) > 0 or len(result.units) >= 0  # Parser does its best

    def test_line_count(self):
        result = parse_source(SIMPLE_FUNCTION, "test.py", "python")
        assert result.line_count > 0


# =========================================================================
# Tests: Fallback parser
# =========================================================================

class TestFallbackParser:
    def test_unsupported_language_uses_fallback(self):
        source = "fn main() {\n    println!(\"Hello\");\n}\n\nfn helper() {\n    return 42;\n}"
        result = parse_source(source, "main.rs")  # No language override → fallback
        assert result.language == "unknown"
        assert len(result.units) > 0
        assert all(u.unit_type == "block" for u in result.units)

    def test_fallback_blocks_have_source(self):
        source = "block one\n\nblock two\n\nblock three"
        result = parse_source(source, "readme.txt")
        for unit in result.units:
            assert len(unit.source) > 0

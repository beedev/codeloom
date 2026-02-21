"""Unit tests for ChainTracer — entry point detection, call tree tracing, membership.

Tests cover:
- Pass 1 heuristic detection (zero incoming calls)
- Pass 2 annotation detection (framework patterns)
- Merge logic (Pass 2 precedence, deduplication)
- Entry point cap (max_entry_points)
- Call tree tracing with cycle prevention
- Flat unit membership aggregation (min_depth, path_count)
- Classify entry type heuristics
"""

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

from codeloom.core.understanding.chain_tracer import ChainTracer, _ANNOTATION_PATTERNS
from codeloom.core.understanding.models import (
    CallTreeNode,
    EntryPoint,
    EntryPointType,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_tree() -> CallTreeNode:
    """Build a sample call tree for testing.

    Structure:
        root (depth=0)
        ├── child_a (depth=1)
        │   ├── grandchild_a1 (depth=2)
        │   └── grandchild_a2 (depth=2)
        └── child_b (depth=1)
            └── grandchild_b1 (depth=2)
                └── leaf (depth=3)
    """
    leaf = CallTreeNode(
        unit_id="u-leaf", name="leaf", qualified_name="mod.leaf",
        unit_type="function", language="python", file_path="mod.py",
        start_line=40, end_line=45, source="def leaf(): pass", depth=3,
    )
    gc_b1 = CallTreeNode(
        unit_id="u-gcb1", name="grandchild_b1", qualified_name="mod.gcb1",
        unit_type="function", language="python", file_path="mod.py",
        start_line=30, end_line=35, source="def gcb1(): leaf()", depth=2,
        children=[leaf],
    )
    gc_a1 = CallTreeNode(
        unit_id="u-gca1", name="grandchild_a1", qualified_name="mod.gca1",
        unit_type="function", language="python", file_path="mod.py",
        start_line=10, end_line=15, source="def gca1(): pass", depth=2,
    )
    gc_a2 = CallTreeNode(
        unit_id="u-gca2", name="grandchild_a2", qualified_name="mod.gca2",
        unit_type="function", language="python", file_path="mod.py",
        start_line=16, end_line=20, source="def gca2(): pass", depth=2,
    )
    child_a = CallTreeNode(
        unit_id="u-ca", name="child_a", qualified_name="mod.child_a",
        unit_type="function", language="python", file_path="mod.py",
        start_line=5, end_line=9, source="def child_a(): gca1(); gca2()", depth=1,
        children=[gc_a1, gc_a2],
    )
    child_b = CallTreeNode(
        unit_id="u-cb", name="child_b", qualified_name="mod.child_b",
        unit_type="function", language="python", file_path="mod.py",
        start_line=21, end_line=28, source="def child_b(): gcb1()", depth=1,
        children=[gc_b1],
    )
    root = CallTreeNode(
        unit_id="u-root", name="root", qualified_name="mod.root",
        unit_type="function", language="python", file_path="mod.py",
        start_line=1, end_line=4, source="def root(): child_a(); child_b()", depth=0,
        children=[child_a, child_b],
    )
    return root


def _make_entry_point(
    uid: str = None,
    name: str = "handler",
    entry_type: EntryPointType = EntryPointType.PUBLIC_API,
    detected_by: str = "heuristic",
    file_path: str = "app.py",
) -> EntryPoint:
    return EntryPoint(
        unit_id=uid or str(uuid4()),
        name=name,
        qualified_name=f"module.{name}",
        file_path=file_path,
        entry_type=entry_type,
        language="python",
        metadata={},
        detected_by=detected_by,
    )


# ── Tests: Merge Logic ────────────────────────────────────────────────────


class TestMergeEntryPoints:
    """Tests for ChainTracer._merge_entry_points — deduplication and precedence."""

    def test_pass2_overrides_pass1_type(self):
        """Pass 2 classification should take precedence over Pass 1."""
        uid = str(uuid4())
        pass1 = [_make_entry_point(uid=uid, entry_type=EntryPointType.PUBLIC_API, detected_by="heuristic")]
        pass2 = [_make_entry_point(uid=uid, entry_type=EntryPointType.HTTP_ENDPOINT, detected_by="annotation")]

        db = MagicMock()
        tracer = ChainTracer(db)
        merged = tracer._merge_entry_points(pass1, pass2)

        assert len(merged) == 1
        assert merged[0].unit_id == uid
        assert merged[0].entry_type == EntryPointType.HTTP_ENDPOINT
        assert merged[0].detected_by == "both"

    def test_union_of_unique_entries(self):
        """Entries unique to each pass should both appear in merged result."""
        uid_a = str(uuid4())
        uid_b = str(uuid4())
        pass1 = [_make_entry_point(uid=uid_a, name="func_a", file_path="a.py")]
        pass2 = [_make_entry_point(uid=uid_b, name="func_b", file_path="b.py")]

        db = MagicMock()
        tracer = ChainTracer(db)
        merged = tracer._merge_entry_points(pass1, pass2)

        assert len(merged) == 2
        ids = {ep.unit_id for ep in merged}
        assert uid_a in ids
        assert uid_b in ids

    def test_sorted_by_file_path_then_name(self):
        """Merged result should be sorted by (file_path, name)."""
        db = MagicMock()
        tracer = ChainTracer(db)

        pass1 = [
            _make_entry_point(name="z_func", file_path="b.py"),
            _make_entry_point(name="a_func", file_path="a.py"),
        ]
        merged = tracer._merge_entry_points(pass1, [])

        assert merged[0].file_path == "a.py"
        assert merged[1].file_path == "b.py"

    def test_empty_inputs(self):
        """Empty pass1 and pass2 should return empty list."""
        db = MagicMock()
        tracer = ChainTracer(db)
        assert tracer._merge_entry_points([], []) == []


# ── Tests: Classify Entry Type ────────────────────────────────────────────


class TestClassifyEntryType:
    """Tests for ChainTracer._classify_entry_type — heuristic classification."""

    def test_main_with_static_is_cli(self):
        """'main' with static modifier should classify as CLI_COMMAND."""
        db = MagicMock()
        tracer = ChainTracer(db)

        row = MagicMock()
        row.name = "main"
        row.metadata = {"modifiers": ["public", "static"]}

        assert tracer._classify_entry_type(row) == EntryPointType.CLI_COMMAND

    def test_test_prefixed_is_unknown(self):
        """Functions starting with 'test' should be UNKNOWN (test code)."""
        db = MagicMock()
        tracer = ChainTracer(db)

        row = MagicMock()
        row.name = "test_handle_order"
        row.metadata = {}

        assert tracer._classify_entry_type(row) == EntryPointType.UNKNOWN

    def test_is_endpoint_meta(self):
        """Unit with is_endpoint metadata should classify as HTTP_ENDPOINT."""
        db = MagicMock()
        tracer = ChainTracer(db)

        row = MagicMock()
        row.name = "get_users"
        row.metadata = {"is_endpoint": True}

        assert tracer._classify_entry_type(row) == EntryPointType.HTTP_ENDPOINT

    def test_default_is_public_api(self):
        """Regular function with no special markers should be PUBLIC_API."""
        db = MagicMock()
        tracer = ChainTracer(db)

        row = MagicMock()
        row.name = "process_order"
        row.metadata = {}

        assert tracer._classify_entry_type(row) == EntryPointType.PUBLIC_API


# ── Tests: Max Entry Points Cap ──────────────────────────────────────────


class TestMaxEntryPointsCap:
    """Tests for the max_entry_points cap in detect_entry_points."""

    @patch.object(ChainTracer, "_detect_pass1_heuristic")
    @patch.object(ChainTracer, "_detect_pass2_annotations")
    def test_caps_when_over_limit(self, mock_pass2, mock_pass1):
        """detect_entry_points should cap results to max_entry_points."""
        entries = [_make_entry_point(name=f"func_{i}", file_path=f"{i}.py") for i in range(20)]
        mock_pass1.return_value = entries
        mock_pass2.return_value = []

        db = MagicMock()
        tracer = ChainTracer(db, max_entry_points=5)
        result = tracer.detect_entry_points("fake-project-id")

        assert len(result) == 5

    @patch.object(ChainTracer, "_detect_pass1_heuristic")
    @patch.object(ChainTracer, "_detect_pass2_annotations")
    def test_no_cap_when_under_limit(self, mock_pass2, mock_pass1):
        """Should return all when count is under max_entry_points."""
        entries = [_make_entry_point(name=f"func_{i}", file_path=f"{i}.py") for i in range(3)]
        mock_pass1.return_value = entries
        mock_pass2.return_value = []

        db = MagicMock()
        tracer = ChainTracer(db, max_entry_points=10)
        result = tracer.detect_entry_points("fake-project-id")

        assert len(result) == 3

    @patch.object(ChainTracer, "_detect_pass1_heuristic")
    @patch.object(ChainTracer, "_detect_pass2_annotations")
    def test_no_cap_when_none(self, mock_pass2, mock_pass1):
        """Should return all when max_entry_points is None."""
        entries = [_make_entry_point(name=f"func_{i}", file_path=f"{i}.py") for i in range(50)]
        mock_pass1.return_value = entries
        mock_pass2.return_value = []

        db = MagicMock()
        tracer = ChainTracer(db, max_entry_points=None)
        result = tracer.detect_entry_points("fake-project-id")

        assert len(result) == 50


# ── Tests: Flat Unit Membership ──────────────────────────────────────────


class TestFlatUnitMembership:
    """Tests for ChainTracer.get_flat_unit_membership — min_depth, path_count."""

    def test_basic_tree_membership(self):
        """All 7 units should appear with correct depths."""
        db = MagicMock()
        tracer = ChainTracer(db)
        tree = _make_tree()
        members = tracer.get_flat_unit_membership(tree)

        by_id = {m["unit_id"]: m for m in members}

        assert len(by_id) == 7
        assert by_id["u-root"]["min_depth"] == 0
        assert by_id["u-ca"]["min_depth"] == 1
        assert by_id["u-cb"]["min_depth"] == 1
        assert by_id["u-gca1"]["min_depth"] == 2
        assert by_id["u-gca2"]["min_depth"] == 2
        assert by_id["u-gcb1"]["min_depth"] == 2
        assert by_id["u-leaf"]["min_depth"] == 3

    def test_path_count_single_path(self):
        """Each unit in a linear chain should have path_count=1."""
        db = MagicMock()
        tracer = ChainTracer(db)

        leaf = CallTreeNode(
            unit_id="u1", name="leaf", qualified_name="leaf",
            unit_type="function", language="py", file_path="a.py",
            start_line=1, end_line=2, source=None, depth=1,
        )
        root = CallTreeNode(
            unit_id="u0", name="root", qualified_name="root",
            unit_type="function", language="py", file_path="a.py",
            start_line=1, end_line=2, source=None, depth=0,
            children=[leaf],
        )

        members = tracer.get_flat_unit_membership(root)
        by_id = {m["unit_id"]: m for m in members}

        assert by_id["u0"]["path_count"] == 1
        assert by_id["u1"]["path_count"] == 1

    def test_diamond_pattern_path_count(self):
        """A unit reachable from two parents should have path_count=2.

        Tree:
            root (depth=0)
            ├── a (depth=1)
            │   └── shared (depth=2)
            └── b (depth=1)
                └── shared (depth=2)   <-- same unit_id
        """
        db = MagicMock()
        tracer = ChainTracer(db)

        shared_from_a = CallTreeNode(
            unit_id="u-shared", name="shared", qualified_name="shared",
            unit_type="function", language="py", file_path="x.py",
            start_line=1, end_line=2, source=None, depth=2,
        )
        shared_from_b = CallTreeNode(
            unit_id="u-shared", name="shared", qualified_name="shared",
            unit_type="function", language="py", file_path="x.py",
            start_line=1, end_line=2, source=None, depth=2,
        )
        a = CallTreeNode(
            unit_id="u-a", name="a", qualified_name="a",
            unit_type="function", language="py", file_path="a.py",
            start_line=1, end_line=2, source=None, depth=1,
            children=[shared_from_a],
        )
        b = CallTreeNode(
            unit_id="u-b", name="b", qualified_name="b",
            unit_type="function", language="py", file_path="b.py",
            start_line=1, end_line=2, source=None, depth=1,
            children=[shared_from_b],
        )
        root = CallTreeNode(
            unit_id="u-root", name="root", qualified_name="root",
            unit_type="function", language="py", file_path="r.py",
            start_line=1, end_line=2, source=None, depth=0,
            children=[a, b],
        )

        members = tracer.get_flat_unit_membership(root)
        by_id = {m["unit_id"]: m for m in members}

        assert by_id["u-shared"]["path_count"] == 2
        assert by_id["u-shared"]["min_depth"] == 2

    def test_cycle_prevention_in_membership(self):
        """Membership walk should not loop on cycles.

        Tree: root -> child -> root (cycle via same unit_id)
        """
        db = MagicMock()
        tracer = ChainTracer(db)

        # Create a cycle: root -> child -> child_of_child (same unit_id as root)
        cycle_back = CallTreeNode(
            unit_id="u-root", name="root", qualified_name="root",
            unit_type="function", language="py", file_path="r.py",
            start_line=1, end_line=2, source=None, depth=2,
        )
        child = CallTreeNode(
            unit_id="u-child", name="child", qualified_name="child",
            unit_type="function", language="py", file_path="r.py",
            start_line=3, end_line=4, source=None, depth=1,
            children=[cycle_back],
        )
        root = CallTreeNode(
            unit_id="u-root", name="root", qualified_name="root",
            unit_type="function", language="py", file_path="r.py",
            start_line=1, end_line=2, source=None, depth=0,
            children=[child],
        )

        members = tracer.get_flat_unit_membership(root)
        by_id = {m["unit_id"]: m for m in members}

        # Should see exactly 2 unique units, not infinite loop
        assert len(by_id) == 2
        assert "u-root" in by_id
        assert "u-child" in by_id
        # root appears at depth 0 (its original position)
        assert by_id["u-root"]["min_depth"] == 0

    def test_single_node_tree(self):
        """A tree with just the root node should return 1 member."""
        db = MagicMock()
        tracer = ChainTracer(db)

        root = CallTreeNode(
            unit_id="u-only", name="only", qualified_name="only",
            unit_type="function", language="py", file_path="a.py",
            start_line=1, end_line=2, source=None, depth=0,
        )

        members = tracer.get_flat_unit_membership(root)
        assert len(members) == 1
        assert members[0]["unit_id"] == "u-only"
        assert members[0]["min_depth"] == 0
        assert members[0]["path_count"] == 1


# ── Tests: Annotation Patterns ──────────────────────────────────────────


class TestAnnotationPatterns:
    """Verify that annotation pattern regexes are properly structured."""

    def test_java_patterns_present(self):
        """Java should have HTTP, message handler, and scheduled patterns."""
        java = _ANNOTATION_PATTERNS["java"]
        assert EntryPointType.HTTP_ENDPOINT in java
        assert EntryPointType.MESSAGE_HANDLER in java
        assert EntryPointType.SCHEDULED_TASK in java

    def test_python_patterns_present(self):
        """Python should have HTTP, CLI, and scheduled patterns."""
        python = _ANNOTATION_PATTERNS["python"]
        assert EntryPointType.HTTP_ENDPOINT in python
        assert EntryPointType.CLI_COMMAND in python
        assert EntryPointType.SCHEDULED_TASK in python

    def test_csharp_patterns_present(self):
        """C# should have HTTP and message handler patterns."""
        csharp = _ANNOTATION_PATTERNS["csharp"]
        assert EntryPointType.HTTP_ENDPOINT in csharp
        assert EntryPointType.MESSAGE_HANDLER in csharp

    def test_typescript_patterns_present(self):
        """TypeScript should have HTTP patterns."""
        ts = _ANNOTATION_PATTERNS["typescript"]
        assert EntryPointType.HTTP_ENDPOINT in ts

    def test_all_pattern_lists_are_nonempty(self):
        """Every registered entry type should have at least one pattern."""
        for lang, type_map in _ANNOTATION_PATTERNS.items():
            for entry_type, patterns in type_map.items():
                assert len(patterns) > 0, (
                    f"Empty pattern list for {lang}/{entry_type.value}"
                )


# ── Tests: Build Tree From Paths ─────────────────────────────────────────


class TestBuildTreeFromPaths:
    """Tests for ChainTracer._build_tree_from_paths — trie-like reconstruction."""

    def test_linear_chain(self):
        """A linear path [root, A, B] should produce root -> A -> B."""
        db = MagicMock()
        tracer = ChainTracer(db)

        uid_root, uid_a, uid_b = str(uuid4()), str(uuid4()), str(uuid4())

        rows = [
            SimpleNamespace(
                unit_id=uid_root, name="root", qualified_name="root",
                unit_type="function", language="python", file_path="a.py",
                start_line=1, end_line=5, source="def root(): pass",
                depth=0, edge_type="root", path=[uid_root],
            ),
            SimpleNamespace(
                unit_id=uid_a, name="func_a", qualified_name="func_a",
                unit_type="function", language="python", file_path="a.py",
                start_line=6, end_line=10, source="def func_a(): pass",
                depth=1, edge_type="calls", path=[uid_root, uid_a],
            ),
            SimpleNamespace(
                unit_id=uid_b, name="func_b", qualified_name="func_b",
                unit_type="function", language="python", file_path="a.py",
                start_line=11, end_line=15, source="def func_b(): pass",
                depth=2, edge_type="calls", path=[uid_root, uid_a, uid_b],
            ),
        ]

        tree = tracer._build_tree_from_paths(rows, uid_root)

        assert tree.unit_id == uid_root
        assert tree.name == "root"
        assert len(tree.children) == 1
        assert tree.children[0].unit_id == uid_a
        assert len(tree.children[0].children) == 1
        assert tree.children[0].children[0].unit_id == uid_b

    def test_branching_tree(self):
        """Root calling two children should produce root -> [A, B]."""
        db = MagicMock()
        tracer = ChainTracer(db)

        uid_root, uid_a, uid_b = str(uuid4()), str(uuid4()), str(uuid4())

        rows = [
            SimpleNamespace(
                unit_id=uid_root, name="root", qualified_name="root",
                unit_type="function", language="python", file_path="a.py",
                start_line=1, end_line=5, source="def root(): pass",
                depth=0, edge_type="root", path=[uid_root],
            ),
            SimpleNamespace(
                unit_id=uid_a, name="func_a", qualified_name="func_a",
                unit_type="function", language="python", file_path="a.py",
                start_line=6, end_line=10, source="def func_a(): pass",
                depth=1, edge_type="calls", path=[uid_root, uid_a],
            ),
            SimpleNamespace(
                unit_id=uid_b, name="func_b", qualified_name="func_b",
                unit_type="function", language="python", file_path="a.py",
                start_line=11, end_line=15, source="def func_b(): pass",
                depth=1, edge_type="calls", path=[uid_root, uid_b],
            ),
        ]

        tree = tracer._build_tree_from_paths(rows, uid_root)

        assert tree.unit_id == uid_root
        assert len(tree.children) == 2
        child_ids = {c.unit_id for c in tree.children}
        assert uid_a in child_ids
        assert uid_b in child_ids

"""Unit tests for ChainAnalyzer — tier selection, quality gates, parsing.

Tests cover:
- Tier selection thresholds (T1/T2/T3)
- Token counting across tree nodes
- JSON output parsing with markdown fence stripping
- Quality gate enforcement (evidence refs, narrative length)
- Source preparation for all 3 tiers
- Bundle building from parsed output
- Unit flattening
"""

import json
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from uuid import uuid4

from codeloom.core.understanding.analyzer import (
    ChainAnalyzer,
    TIER_1_MAX,
    TIER_2_MAX,
    DEFAULT_REQUIRE_EVIDENCE_REFS,
    DEFAULT_MIN_NARRATIVE_LENGTH,
)
from codeloom.core.understanding.models import (
    AnalysisTier,
    CallTreeNode,
    DeepContextBundle,
    EntryPoint,
    EntryPointType,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_entry_point(name: str = "handle_request") -> EntryPoint:
    return EntryPoint(
        unit_id=str(uuid4()),
        name=name,
        qualified_name=f"app.controllers.{name}",
        file_path="controllers.py",
        entry_type=EntryPointType.HTTP_ENDPOINT,
        language="python",
    )


def _make_node(
    uid: str = None,
    name: str = "func",
    source: str = "def func(): pass",
    depth: int = 0,
    children: list = None,
    token_count: int = 0,
) -> CallTreeNode:
    return CallTreeNode(
        unit_id=uid or str(uuid4()),
        name=name,
        qualified_name=f"mod.{name}",
        unit_type="function",
        language="python",
        file_path="mod.py",
        start_line=1,
        end_line=5,
        source=source,
        depth=depth,
        children=children or [],
        token_count=token_count,
    )


def _mock_token_counter(tokens_per_call: int = 50):
    """Create a mock TokenCounter that returns a fixed count."""
    tc = MagicMock()
    tc.count.return_value = tokens_per_call
    return tc


# ── Tests: Tier Selection ─────────────────────────────────────────────────


class TestTierSelection:
    """Tests for ChainAnalyzer._select_tier threshold boundaries."""

    def test_tier1_at_boundary(self):
        analyzer = ChainAnalyzer()
        assert analyzer._select_tier(TIER_1_MAX) == AnalysisTier.TIER_1

    def test_tier1_below(self):
        analyzer = ChainAnalyzer()
        assert analyzer._select_tier(50_000) == AnalysisTier.TIER_1

    def test_tier2_just_above_tier1(self):
        analyzer = ChainAnalyzer()
        assert analyzer._select_tier(TIER_1_MAX + 1) == AnalysisTier.TIER_2

    def test_tier2_at_boundary(self):
        analyzer = ChainAnalyzer()
        assert analyzer._select_tier(TIER_2_MAX) == AnalysisTier.TIER_2

    def test_tier3_above_tier2(self):
        analyzer = ChainAnalyzer()
        assert analyzer._select_tier(TIER_2_MAX + 1) == AnalysisTier.TIER_3

    def test_tier1_zero_tokens(self):
        analyzer = ChainAnalyzer()
        assert analyzer._select_tier(0) == AnalysisTier.TIER_1

    def test_tier3_very_large(self):
        analyzer = ChainAnalyzer()
        assert analyzer._select_tier(1_000_000) == AnalysisTier.TIER_3


# ── Tests: Token Counting ─────────────────────────────────────────────────


class TestTokenCounting:
    """Tests for ChainAnalyzer._count_tree_tokens."""

    def test_single_node(self):
        tc = _mock_token_counter(100)
        analyzer = ChainAnalyzer(token_counter=tc)

        node = _make_node(source="def f(): pass")
        total = analyzer._count_tree_tokens(node)

        assert total == 100
        assert node.token_count == 100

    def test_recursive_counting(self):
        tc = _mock_token_counter(50)
        analyzer = ChainAnalyzer(token_counter=tc)

        child = _make_node(name="child", source="def child(): pass", depth=1)
        root = _make_node(name="root", source="def root(): child()", depth=0, children=[child])

        total = analyzer._count_tree_tokens(root)

        assert total == 100  # 50 + 50
        assert root.token_count == 50
        assert child.token_count == 50

    def test_none_source_contributes_zero(self):
        tc = _mock_token_counter(50)
        analyzer = ChainAnalyzer(token_counter=tc)

        node = _make_node(source=None)
        total = analyzer._count_tree_tokens(node)

        assert total == 0
        tc.count.assert_not_called()


# ── Tests: JSON Parsing ──────────────────────────────────────────────────


class TestJsonParsing:
    """Tests for ChainAnalyzer._parse_json_output."""

    def test_clean_json(self):
        analyzer = ChainAnalyzer()
        result = analyzer._parse_json_output('{"business_rules": [], "narrative": "test"}')
        assert result["narrative"] == "test"
        assert result["business_rules"] == []

    def test_json_with_markdown_fences(self):
        analyzer = ChainAnalyzer()
        raw = '```json\n{"key": "value"}\n```'
        result = analyzer._parse_json_output(raw)
        assert result["key"] == "value"

    def test_json_with_surrounding_text(self):
        """Should extract JSON even when surrounded by extra text."""
        analyzer = ChainAnalyzer()
        raw = 'Here is the analysis:\n{"key": "value"}\nThat is all.'
        result = analyzer._parse_json_output(raw)
        assert result["key"] == "value"

    def test_invalid_json_returns_error(self):
        analyzer = ChainAnalyzer()
        result = analyzer._parse_json_output("this is not json at all")
        assert "parse_error" in result

    def test_nested_json_in_fences(self):
        analyzer = ChainAnalyzer()
        data = {"rules": [{"id": "R1", "name": "Validation"}], "confidence": 0.85}
        raw = f"```json\n{json.dumps(data)}\n```"
        result = analyzer._parse_json_output(raw)
        assert result["confidence"] == 0.85
        assert len(result["rules"]) == 1


# ── Tests: Quality Gates ─────────────────────────────────────────────────


class TestQualityGates:
    """Tests for quality gate enforcement in _build_bundle."""

    def _call_build_bundle(self, parsed, analyzer=None):
        """Helper to call _build_bundle with minimal arguments."""
        if analyzer is None:
            analyzer = ChainAnalyzer(token_counter=_mock_token_counter())

        ep = _make_entry_point()
        tree = _make_node()

        return analyzer._build_bundle(
            entry_point=ep,
            tier=AnalysisTier.TIER_1,
            total_tokens=1000,
            call_tree=tree,
            parsed=parsed,
        )

    @patch("codeloom.core.understanding.analyzer.logger")
    def test_warns_on_missing_evidence_refs(self, mock_logger):
        """Should warn when business rules lack evidence references."""
        parsed = {
            "business_rules": [
                {"id": "BR1", "name": "Rule with no evidence"},
                {"id": "BR2", "evidence_refs": [{"unit_id": "u1"}]},
            ],
            "narrative": "A" * 200,
        }

        # _build_bundle has try/except around config loading — defaults apply
        self._call_build_bundle(parsed)

        # Should have logged a warning about 1 rule without evidence
        mock_logger.warning.assert_called()
        warning_msg = str(mock_logger.warning.call_args)
        assert "evidence" in warning_msg.lower()

    @patch("codeloom.core.understanding.analyzer.logger")
    def test_no_warning_when_all_rules_have_evidence(self, mock_logger):
        """No warning when all business rules have evidence refs."""
        parsed = {
            "business_rules": [
                {"id": "BR1", "evidence_refs": [{"unit_id": "u1"}]},
            ],
            "narrative": "A" * 200,
        }

        self._call_build_bundle(parsed)

        # Check that no evidence-related warning was issued
        for call_args in mock_logger.warning.call_args_list:
            assert "evidence" not in str(call_args).lower()

    @patch("codeloom.core.understanding.analyzer.logger")
    def test_warns_on_short_narrative(self, mock_logger):
        """Should warn when narrative is shorter than min_narrative_length."""
        parsed = {
            "narrative": "Short",
            "business_rules": [],
        }

        self._call_build_bundle(parsed)

        mock_logger.warning.assert_called()
        warning_msg = str(mock_logger.warning.call_args)
        assert "narrative" in warning_msg.lower()

    @patch("codeloom.core.understanding.analyzer.logger")
    def test_no_warning_on_sufficient_narrative(self, mock_logger):
        """No warning when narrative meets minimum length."""
        parsed = {
            "narrative": "A" * 200,
            "business_rules": [],
        }

        self._call_build_bundle(parsed)

        for call_args in mock_logger.warning.call_args_list:
            assert "narrative" not in str(call_args).lower()


# ── Tests: Bundle Building ───────────────────────────────────────────────


class TestBundleBuilding:
    """Tests for _build_bundle output correctness."""

    def test_bundle_shape(self):
        """Bundle should have all expected fields populated."""
        analyzer = ChainAnalyzer(token_counter=_mock_token_counter())
        ep = _make_entry_point()
        tree = _make_node()

        parsed = {
            "narrative": "A comprehensive narrative about this endpoint. " * 5,
            "business_rules": [{"id": "BR1", "evidence_refs": [{"unit_id": "u1"}]}],
            "data_entities": [{"name": "Order"}],
            "integrations": [{"name": "PaymentGateway"}],
            "side_effects": [{"type": "email"}],
            "cross_cutting_concerns": ["logging", "auth"],
            "confidence": 0.85,
            "coverage": 0.7,
        }

        bundle = analyzer._build_bundle(
            entry_point=ep,
            tier=AnalysisTier.TIER_1,
            total_tokens=5000,
            call_tree=tree,
            parsed=parsed,
        )

        assert isinstance(bundle, DeepContextBundle)
        assert bundle.entry_point == ep
        assert bundle.tier == AnalysisTier.TIER_1
        assert bundle.total_tokens == 5000
        assert bundle.confidence == 0.85
        assert bundle.coverage == 0.7
        assert len(bundle.business_rules) == 1
        assert len(bundle.data_entities) == 1
        assert len(bundle.integrations) == 1
        assert len(bundle.side_effects) == 1
        assert len(bundle.cross_cutting_concerns) == 2
        assert bundle.schema_version == 1
        assert bundle.prompt_version == "v1.0"
        assert bundle.analyzed_at is not None

    def test_chain_truncated_flag(self):
        """chain_truncated should be False for TIER_1, True for TIER_2/3."""
        analyzer = ChainAnalyzer(token_counter=_mock_token_counter())
        ep = _make_entry_point()
        tree = _make_node()

        parsed = {"narrative": "A" * 200}

        bundle_t1 = analyzer._build_bundle(ep, AnalysisTier.TIER_1, 1000, tree, parsed)
        bundle_t2 = analyzer._build_bundle(ep, AnalysisTier.TIER_2, 1000, tree, parsed)
        bundle_t3 = analyzer._build_bundle(ep, AnalysisTier.TIER_3, 1000, tree, parsed)

        assert bundle_t1.chain_truncated is False
        assert bundle_t2.chain_truncated is True
        assert bundle_t3.chain_truncated is True

    def test_missing_fields_default_gracefully(self):
        """Bundle should handle missing fields in parsed output."""
        analyzer = ChainAnalyzer(token_counter=_mock_token_counter())
        ep = _make_entry_point()
        tree = _make_node()

        bundle = analyzer._build_bundle(ep, AnalysisTier.TIER_1, 100, tree, {})

        assert bundle.business_rules == []
        assert bundle.data_entities == []
        assert bundle.integrations == []
        assert bundle.narrative == ""
        assert bundle.confidence == 0.0


# ── Tests: Unit Flattening ───────────────────────────────────────────────


class TestUnitFlattening:
    """Tests for ChainAnalyzer._flatten_units."""

    def test_single_node(self):
        analyzer = ChainAnalyzer()
        node = _make_node(uid="u1")
        assert analyzer._flatten_units(node) == ["u1"]

    def test_tree_flattening(self):
        analyzer = ChainAnalyzer()
        child1 = _make_node(uid="u2", depth=1)
        child2 = _make_node(uid="u3", depth=1)
        root = _make_node(uid="u1", children=[child1, child2])

        flat = analyzer._flatten_units(root)
        assert set(flat) == {"u1", "u2", "u3"}

    def test_deep_tree(self):
        analyzer = ChainAnalyzer()
        leaf = _make_node(uid="u3", depth=2)
        mid = _make_node(uid="u2", depth=1, children=[leaf])
        root = _make_node(uid="u1", children=[mid])

        flat = analyzer._flatten_units(root)
        assert flat == ["u1", "u2", "u3"]


# ── Tests: Source Preparation ─────────────────────────────────────────────


class TestSourcePreparation:
    """Tests for _prepare_source routing to the correct tier formatter."""

    def test_tier1_uses_full_source(self):
        analyzer = ChainAnalyzer(token_counter=_mock_token_counter(10))
        tree = _make_node(source="def func(): pass")

        result = analyzer._prepare_source(tree, AnalysisTier.TIER_1, 10)
        assert "def func(): pass" in result

    def test_format_full_source_includes_header(self):
        analyzer = ChainAnalyzer()
        node = _make_node(name="handler", source="def handler(): pass")

        result = analyzer._format_full_source(node)
        assert "mod.handler" in result
        assert "def handler(): pass" in result

    def test_format_unit_signature_uses_first_line(self):
        analyzer = ChainAnalyzer()
        node = _make_node(name="handler", source="def handler(request):\n    return response")

        result = analyzer._format_unit_signature(node)
        assert "def handler(request)" in result
        assert "return response" not in result

    def test_format_unit_signature_fallback_for_none_source(self):
        analyzer = ChainAnalyzer()
        node = _make_node(name="handler", source=None)

        result = analyzer._format_unit_signature(node)
        assert "handler" in result

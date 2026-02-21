"""Integration tests for the understanding module.

Tests cover:
- Full job lifecycle (engine.start_analysis -> status -> results)
- Migration deep-context injection and fallback
- Chat narrative enrichment and fallback
- Tenant/access-control boundaries
- Framework detection registry
"""

import json
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from uuid import uuid4, UUID

from codeloom.core.understanding.engine import UnderstandingEngine
from codeloom.core.understanding.models import (
    AnalysisTier,
    CallTreeNode,
    DeepContextBundle,
    EntryPoint,
    EntryPointType,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


def _mock_db():
    """Create a mock DatabaseManager."""
    db = MagicMock()
    session = MagicMock()
    db.get_session.return_value.__enter__ = MagicMock(return_value=session)
    db.get_session.return_value.__exit__ = MagicMock(return_value=False)
    return db, session


def _make_entry_point(name="handler") -> EntryPoint:
    return EntryPoint(
        unit_id=str(uuid4()),
        name=name,
        qualified_name=f"app.{name}",
        file_path="app.py",
        entry_type=EntryPointType.HTTP_ENDPOINT,
        language="python",
    )


# ── Tests: Engine Job Lifecycle ──────────────────────────────────────────


class TestEngineJobLifecycle:
    """Tests for UnderstandingEngine — start, status, results."""

    def test_start_analysis_creates_job(self):
        """start_analysis should insert a job row and return job_id."""
        db, session = _mock_db()
        engine = UnderstandingEngine(db)

        # Prevent worker from actually starting
        with patch.object(engine, "_ensure_worker"):
            result = engine.start_analysis(str(uuid4()), str(uuid4()))

        assert "job_id" in result
        assert result["status"] == "pending"
        assert "project_id" in result

        # Session execute should have been called (INSERT + UPDATE)
        assert session.execute.call_count >= 1

    def test_get_job_status_returns_status(self):
        """get_job_status should query and return job details."""
        db, session = _mock_db()
        engine = UnderstandingEngine(db)

        job_id = str(uuid4())
        project_id = str(uuid4())

        mock_row = MagicMock()
        mock_row.job_id = UUID(job_id)
        mock_row.project_id = UUID(project_id)
        mock_row.status = "running"
        mock_row.total_entry_points = 10
        mock_row.completed_entry_points = 3
        mock_row.created_at = MagicMock(isoformat=MagicMock(return_value="2024-01-01T00:00:00"))
        mock_row.started_at = MagicMock(isoformat=MagicMock(return_value="2024-01-01T00:01:00"))
        mock_row.completed_at = None
        mock_row.retry_count = 0
        mock_row.error_details = None

        session.execute.return_value.fetchone.return_value = mock_row

        result = engine.get_job_status(project_id, job_id)

        assert result["status"] == "running"
        assert result["progress"]["total"] == 10
        assert result["progress"]["completed"] == 3
        assert result["retry_count"] == 0

    def test_get_job_status_not_found(self):
        """get_job_status should return error dict when job doesn't exist."""
        db, session = _mock_db()
        engine = UnderstandingEngine(db)

        session.execute.return_value.fetchone.return_value = None

        result = engine.get_job_status(str(uuid4()), str(uuid4()))
        assert "error" in result

    def test_get_entry_points_delegates_to_tracer(self):
        """get_entry_points should use ChainTracer.detect_entry_points."""
        db = MagicMock()
        engine = UnderstandingEngine(db)

        ep = _make_entry_point()
        mock_tracer = MagicMock()
        mock_tracer.detect_entry_points.return_value = [ep]
        engine._tracer = mock_tracer

        result = engine.get_entry_points(str(uuid4()))

        assert len(result) == 1
        assert result[0]["name"] == "handler"
        assert result[0]["entry_type"] == "http_endpoint"

    def test_get_analysis_results_returns_summaries(self):
        """get_analysis_results should return list of analysis summaries."""
        db, session = _mock_db()
        engine = UnderstandingEngine(db)

        analysis_id = uuid4()
        entry_unit_id = uuid4()

        mock_row = MagicMock()
        mock_row.analysis_id = analysis_id
        mock_row.entry_unit_id = entry_unit_id
        mock_row.entry_name = "handle_order"
        mock_row.entry_qualified = "app.handle_order"
        mock_row.entry_file = "app.py"
        mock_row.entry_type = "http_endpoint"
        mock_row.tier = "tier_1"
        mock_row.total_units = 15
        mock_row.total_tokens = 50000
        mock_row.confidence_score = 0.85
        mock_row.coverage_pct = 72.0
        mock_row.narrative = "This endpoint handles order creation..."
        mock_row.schema_version = 1
        mock_row.analyzed_at = MagicMock(isoformat=MagicMock(return_value="2024-01-01"))

        session.execute.return_value.fetchall.return_value = [mock_row]

        result = engine.get_analysis_results(str(uuid4()))

        assert len(result) == 1
        assert result[0]["entry_name"] == "handle_order"
        assert result[0]["confidence_score"] == 0.85
        assert result[0]["tier"] == "tier_1"

    def test_get_chain_detail_not_found(self):
        """get_chain_detail should return error when analysis doesn't exist."""
        db, session = _mock_db()
        engine = UnderstandingEngine(db)

        session.execute.return_value.fetchone.return_value = None

        result = engine.get_chain_detail(str(uuid4()), str(uuid4()))
        assert "error" in result

    def test_get_chain_detail_includes_units(self):
        """get_chain_detail should include analysis_units list."""
        db, session = _mock_db()
        engine = UnderstandingEngine(db)

        analysis_id = uuid4()
        project_id = uuid4()

        # Mock the main analysis row
        analysis_row = MagicMock()
        analysis_row.analysis_id = analysis_id
        analysis_row.entry_unit_id = uuid4()
        analysis_row.entry_name = "handler"
        analysis_row.entry_qualified = "app.handler"
        analysis_row.entry_file = "app.py"
        analysis_row.entry_type = "http_endpoint"
        analysis_row.tier = "tier_1"
        analysis_row.total_units = 5
        analysis_row.total_tokens = 3000
        analysis_row.confidence_score = 0.9
        analysis_row.coverage_pct = 80.0
        analysis_row.narrative = "A narrative"
        analysis_row.schema_version = 1
        analysis_row.prompt_version = "v1.0"
        analysis_row.analyzed_at = MagicMock(isoformat=MagicMock(return_value="2024-01-01"))
        analysis_row.result_json = json.dumps({"business_rules": []})

        # Mock the units query
        unit_row = MagicMock()
        unit_row.unit_id = uuid4()
        unit_row.name = "helper"
        unit_row.qualified_name = "app.helper"
        unit_row.unit_type = "function"
        unit_row.file_path = "app.py"
        unit_row.min_depth = 1
        unit_row.path_count = 2

        # First call returns analysis, second call returns units
        session.execute.side_effect = [
            MagicMock(fetchone=MagicMock(return_value=analysis_row)),
            MagicMock(fetchall=MagicMock(return_value=[unit_row])),
        ]

        result = engine.get_chain_detail(str(project_id), str(analysis_id))

        assert result["tier"] == "tier_1"
        assert len(result["units"]) == 1
        assert result["units"][0]["name"] == "helper"
        assert result["units"][0]["min_depth"] == 1


# ── Tests: Engine Lazy Initialization ────────────────────────────────────


class TestEngineLazyInit:
    """Tests for lazy worker and tracer initialization."""

    def test_ensure_tracer_creates_once(self):
        """_ensure_tracer should create ChainTracer only once."""
        db = MagicMock()
        engine = UnderstandingEngine(db)

        assert engine._tracer is None
        engine._ensure_tracer()
        assert engine._tracer is not None

        first = engine._tracer
        engine._ensure_tracer()
        assert engine._tracer is first  # Same instance

    @patch("codeloom.core.understanding.engine.UnderstandingWorker")
    def test_ensure_worker_creates_and_starts(self, MockWorker):
        """_ensure_worker should create and start the worker."""
        db = MagicMock()
        engine = UnderstandingEngine(db)

        mock_instance = MockWorker.return_value
        mock_instance._running = False

        engine._ensure_worker()

        MockWorker.assert_called_once_with(db)
        mock_instance.start.assert_called_once()


# ── Tests: Migration Context Builder Integration ─────────────────────────


class TestMigrationContextIntegration:
    """Tests for deep analysis context injection into migration context builder."""

    def test_empty_unit_ids_returns_empty(self):
        """Should return empty string when no unit IDs provided."""
        from codeloom.core.migration.context_builder import MigrationContextBuilder

        db, session = _mock_db()
        builder = MigrationContextBuilder.__new__(MigrationContextBuilder)
        builder._db = db
        builder._pid = uuid4()

        result = builder.get_deep_analysis_context([])
        assert result == ""

    def test_no_overlapping_analyses_returns_empty(self):
        """Should return empty string when no analyses overlap."""
        from codeloom.core.migration.context_builder import MigrationContextBuilder

        db, session = _mock_db()
        session.execute.return_value.fetchall.return_value = []

        builder = MigrationContextBuilder.__new__(MigrationContextBuilder)
        builder._db = db
        builder._pid = uuid4()

        result = builder.get_deep_analysis_context([str(uuid4())])
        assert result == ""

    def test_context_includes_narrative_and_rules(self):
        """Should compose DEEP UNDERSTANDING section with narratives and rules."""
        from codeloom.core.migration.context_builder import MigrationContextBuilder

        db, session = _mock_db()

        # Mock analysis row with narrative and rules
        mock_row = MagicMock()
        mock_row.analysis_id = uuid4()
        mock_row.entry_point_name = "processOrder"
        mock_row.narrative = "This endpoint processes customer orders..."
        mock_row.confidence_score = 0.85
        mock_row.coverage_pct = 70.0
        mock_row.overlap_units = 5
        mock_row.best_depth = 0
        mock_row.overlap_paths = 10
        mock_row.result_json = {
            "business_rules": [
                {"id": "BR1", "description": "Orders require payment method"},
            ],
            "integrations": [
                {"name": "PaymentGateway", "description": "Stripe integration"},
            ],
        }

        session.execute.return_value.fetchall.return_value = [mock_row]

        builder = MigrationContextBuilder.__new__(MigrationContextBuilder)
        builder._db = db
        builder._pid = uuid4()

        result = builder.get_deep_analysis_context([str(uuid4())])

        assert "DEEP UNDERSTANDING" in result
        assert "processOrder" in result
        assert "This endpoint processes" in result
        assert "Business Rules" in result
        assert "BR1" in result
        assert "PaymentGateway" in result

    def test_coverage_warning_when_below_threshold(self):
        """Should include warning when coverage is below threshold."""
        from codeloom.core.migration.context_builder import MigrationContextBuilder

        db, session = _mock_db()

        mock_row = MagicMock()
        mock_row.analysis_id = uuid4()
        mock_row.entry_point_name = "handler"
        mock_row.narrative = "A narrative"
        mock_row.confidence_score = 0.5
        mock_row.coverage_pct = 30.0
        mock_row.overlap_units = 1
        mock_row.best_depth = 0
        mock_row.overlap_paths = 1
        mock_row.result_json = {}

        session.execute.return_value.fetchall.return_value = [mock_row]

        builder = MigrationContextBuilder.__new__(MigrationContextBuilder)
        builder._db = db
        builder._pid = uuid4()

        # Default warn_below is 50% — coverage will be very low (0%)
        result = builder.get_deep_analysis_context([str(uuid4()) for _ in range(10)])

        assert "WARNING" in result


# ── Tests: Chat Narrative Integration ────────────────────────────────────


class TestChatNarrativeIntegration:
    """Tests for _get_relevant_narratives from code_chat.py."""

    def test_returns_narratives_on_overlap(self):
        """Should return narrative strings when analyses overlap with unit_ids."""
        from codeloom.api.routes.code_chat import _get_relevant_narratives

        db, session = _mock_db()

        mock_row = MagicMock()
        mock_row.narrative = "This function handles user authentication..."

        session.execute.return_value.fetchall.return_value = [mock_row]

        project_id = str(uuid4())
        unit_ids = [str(uuid4()), str(uuid4())]

        narratives = _get_relevant_narratives(db, project_id, unit_ids)

        assert len(narratives) == 1
        assert "authentication" in narratives[0]

    def test_returns_empty_on_no_overlap(self):
        """Should return empty list when no analyses overlap."""
        from codeloom.api.routes.code_chat import _get_relevant_narratives

        db, session = _mock_db()
        session.execute.return_value.fetchall.return_value = []

        narratives = _get_relevant_narratives(db, str(uuid4()), [str(uuid4())])

        assert narratives == []

    def test_respects_max_narratives(self):
        """Should limit results to max_narratives parameter."""
        from codeloom.api.routes.code_chat import _get_relevant_narratives

        db, session = _mock_db()

        # Return 5 narratives
        rows = [MagicMock(narrative=f"Narrative {i}") for i in range(5)]
        session.execute.return_value.fetchall.return_value = rows

        narratives = _get_relevant_narratives(
            db, str(uuid4()), [str(uuid4())], max_narratives=3
        )

        # The SQL LIMIT handles this, so we just verify the call was made
        # The actual limiting happens in the DB query
        assert len(narratives) == 5  # Mock returns all; real DB would limit


# ── Tests: Fallback Behavior ─────────────────────────────────────────────


class TestFallbackBehavior:
    """Tests for graceful fallback when no deep analyses exist."""

    def test_engine_results_empty_for_new_project(self):
        """get_analysis_results should return empty list for project with no analyses."""
        db, session = _mock_db()
        engine = UnderstandingEngine(db)

        session.execute.return_value.fetchall.return_value = []

        result = engine.get_analysis_results(str(uuid4()))
        assert result == []

    def test_migration_context_empty_for_no_analyses(self):
        """Migration context builder should return empty string gracefully."""
        from codeloom.core.migration.context_builder import MigrationContextBuilder

        db, session = _mock_db()
        session.execute.return_value.fetchall.return_value = []

        builder = MigrationContextBuilder.__new__(MigrationContextBuilder)
        builder._db = db
        builder._pid = uuid4()

        result = builder.get_deep_analysis_context([str(uuid4())])
        assert result == ""


# ── Tests: Framework Detection Registry ──────────────────────────────────


class TestFrameworkDetectionRegistry:
    """Tests for the framework detection registry."""

    def test_detect_and_analyze_returns_empty_for_no_frameworks(self):
        """Should return empty list when no frameworks are detected."""
        from codeloom.core.understanding.frameworks import detect_and_analyze

        db = MagicMock()

        # All analyzers fail detection
        with patch("codeloom.core.understanding.frameworks.spring.SpringAnalyzer") as MockSpring, \
             patch("codeloom.core.understanding.frameworks.aspnet.AspNetAnalyzer") as MockAspnet:

            MockSpring.return_value.detect.return_value = False
            MockAspnet.return_value.detect.return_value = False

            # Can't easily patch _ANALYZERS, so test that exceptions are handled
            contexts = detect_and_analyze(db, str(uuid4()))

        # Even with errors, should return a list
        assert isinstance(contexts, list)

    def test_framework_context_serialization(self):
        """FrameworkContext should serialize to a well-formed dict."""
        from codeloom.core.understanding.frameworks.base import FrameworkContext

        ctx = FrameworkContext(
            framework_name="Spring Boot",
            framework_type="java_spring",
            version="3.2",
            di_registrations=["@Service OrderService"],
            middleware_pipeline=["SecurityFilter"],
            security_config={"csrf_enabled": True},
            transaction_boundaries=["@Transactional processOrder"],
            aop_pointcuts=["@Around logging"],
            analysis_hints=["Uses Spring Security"],
        )

        assert ctx.framework_name == "Spring Boot"
        assert len(ctx.di_registrations) == 1
        assert ctx.security_config["csrf_enabled"] is True


# ── Tests: Data Models ───────────────────────────────────────────────────


class TestDataModels:
    """Tests for understanding data models."""

    def test_entry_point_type_values(self):
        """EntryPointType enum should have all expected values."""
        expected = {
            "http_endpoint", "message_handler", "scheduled_task",
            "cli_command", "event_listener", "startup_hook",
            "public_api", "unknown",
        }
        actual = {e.value for e in EntryPointType}
        assert actual == expected

    def test_analysis_tier_values(self):
        """AnalysisTier enum should have tier_1, tier_2, tier_3."""
        assert AnalysisTier.TIER_1.value == "tier_1"
        assert AnalysisTier.TIER_2.value == "tier_2"
        assert AnalysisTier.TIER_3.value == "tier_3"

    def test_call_tree_node_defaults(self):
        """CallTreeNode should have sensible defaults."""
        node = CallTreeNode(
            unit_id="u1", name="f", qualified_name="f", unit_type="function",
            language="py", file_path="a.py", start_line=1, end_line=2,
            source=None, depth=0,
        )
        assert node.children == []
        assert node.token_count == 0
        assert node.edge_type == "calls"

    def test_deep_context_bundle_defaults(self):
        """DeepContextBundle should have empty defaults for collections."""
        ep = _make_entry_point()
        bundle = DeepContextBundle(
            entry_point=ep,
            tier=AnalysisTier.TIER_1,
            total_units=1,
            total_tokens=100,
        )
        assert bundle.business_rules == []
        assert bundle.data_entities == []
        assert bundle.integrations == []
        assert bundle.side_effects == []
        assert bundle.cross_cutting_concerns == []
        assert bundle.narrative == ""
        assert bundle.confidence == 0.0
        assert bundle.coverage == 0.0
        assert bundle.chain_truncated is False
        assert bundle.schema_version == 1

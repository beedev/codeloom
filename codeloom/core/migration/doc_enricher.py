"""Fetch target framework documentation for migration blueprint.

Topics are generated dynamically from detected source patterns --
NOT from a hardcoded framework list.  Works for any framework.

Uses Tavily search API for targeted documentation retrieval.
Gracefully degrades if Tavily key is missing or network unavailable.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DocEnricher:
    """Fetches target framework documentation for migration blueprint.

    Topics are generated dynamically from detected source patterns --
    NOT from a hardcoded framework list.  Works for any framework.
    """

    # Generic concern categories that apply to any framework.
    # Used as fallback when source patterns are unavailable.
    CONCERN_CATEGORIES = [
        "dependency injection",
        "data access ORM repository",
        "web routing controllers API",
        "configuration settings",
        "error handling exceptions",
        "testing unit integration",
        "project structure conventions",
        "middleware pipeline",
        "authentication authorization",
        "logging observability",
    ]

    # Source pattern -> target search topic mapping.
    # Pattern-driven, not framework-driven.
    PATTERN_TO_TOPIC: Dict[str, str] = {
        # DI patterns
        "di_field_injection": "dependency injection constructor injection service registration",
        "di_constructor_injection": "dependency injection IoC container service lifetime",
        "di_inject_annotation": "dependency injection constructor injection configuration",
        "di_flask_di": "dependency injection service layer pattern",
        # Data layer
        "data_spring_data_jpa": "ORM entity mapping repository pattern data access",
        "data_jpa": "ORM entity mapping repository pattern data access",
        "data_orm_repository_pattern": "ORM entity mapping repository pattern",
        "data_sqlalchemy_or_django_orm": "ORM entity mapping migrations relationships",
        "data_jdbc": "database access connection pooling query builder",
        # Web layer
        "web_spring_mvc": "REST API controller routing request handling response",
        "web_express_or_koa": "REST API routing middleware request handling",
        "web_flask": "REST API routing request handling response",
        "web_fastapi": "REST API routing dependency injection response models",
        # Config
        "config_spring_properties": "configuration settings environment variables profiles",
        "config_spring_config_properties": "configuration type-safe settings binding",
        "config_file_based_config": "configuration settings environment variables",
        # Testing
        "test_junit": "unit testing mocking assertions test framework",
        "test_junit_spring": "integration testing dependency injection test containers",
        "test_pytest": "unit testing fixtures mocking assertions",
        "test_jest_or_mocha": "unit testing mocking assertions test framework",
        "test_xunit_or_nunit": "unit testing mocking assertions xUnit patterns",
    }

    # Phase -> relevant concern keywords for filtering cached docs.
    _PHASE_FOCUS: Dict[int, List[str]] = {
        2: [
            "dependency injection", "data access", "ORM", "repository",
            "routing", "controller", "configuration", "project structure",
            "conventions", "middleware", "service registration",
        ],
        3: [
            "business logic", "domain", "validation", "entity",
            "integration", "service", "repository", "data model",
        ],
        4: [
            "interface", "type system", "dependency injection", "pattern",
            "error handling", "design", "architecture", "contract",
        ],
        5: [
            "code idiom", "import", "annotation", "decorator",
            "async", "migration", "implementation", "syntax",
        ],
        6: [
            "test framework", "assertion", "mock", "integration test",
            "unit test", "test container", "fixture", "setup",
        ],
    }

    def __init__(self) -> None:
        self._client: Optional[Any] = None

    def _get_client(self) -> Optional[Any]:
        """Lazy-init Tavily client.  Returns None if key missing."""
        if self._client is not None:
            return self._client

        api_key = os.environ.get("TAVILY_API_KEY", "").strip()
        if not api_key:
            logger.warning(
                "TAVILY_API_KEY not set -- framework doc enrichment disabled"
            )
            return None

        try:
            from tavily import TavilyClient  # type: ignore[import-untyped]
            self._client = TavilyClient(api_key=api_key)
            return self._client
        except ImportError:
            logger.warning("tavily-python not installed -- doc enrichment disabled")
            return None

    # ── Public API ────────────────────────────────────────────────────

    def enrich_plan(
        self,
        frameworks: List[str],
        source_patterns: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch documentation for each target framework.

        Args:
            frameworks: Target framework names (e.g. ["ASP.NET Core", "Entity Framework"])
            source_patterns: Output of ``get_source_patterns()`` from context_builder.
                If provided, search topics are tailored to source patterns.

        Returns:
            Dict keyed by framework name::

                {
                    "ASP.NET Core": {
                        "content": "## ASP.NET Core Best Practices\\n...",
                        "source": "tavily-search",
                        "fetched_at": "2026-02-18T10:00:00+00:00"
                    },
                    ...
                }
        """
        client = self._get_client()
        if client is None:
            return {}

        result: Dict[str, Dict[str, Any]] = {}
        for fw in frameworks:
            topics = self._build_search_topics(fw, source_patterns)
            content = self._fetch_docs(fw, topics)
            if content:
                result[fw] = {
                    "content": content,
                    "source": "tavily-search",
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                }
            else:
                logger.info("No docs fetched for framework %s", fw)

        return result

    def get_phase_docs(
        self,
        framework_docs: Dict[str, Dict[str, Any]],
        phase_number: int,
        budget: int = 3000,
    ) -> str:
        """Extract phase-relevant documentation from cached framework docs.

        Filters cached content by keywords relevant to the given phase,
        then truncates to stay within ``budget`` tokens (~4 chars/token).

        Args:
            framework_docs: Cached docs from ``enrich_plan()``
            phase_number: Migration phase (2-6)
            budget: Approximate token budget for the returned text

        Returns:
            Formatted markdown string with relevant framework docs
        """
        if not framework_docs:
            return ""

        focus_keywords = self._PHASE_FOCUS.get(phase_number, [])
        char_budget = budget * 4  # approximate chars-per-token

        sections: List[str] = []
        remaining = char_budget

        for fw_name, fw_data in framework_docs.items():
            content = fw_data.get("content", "")
            if not content:
                continue

            # Filter paragraphs by relevance to phase focus keywords
            relevant = self._filter_by_keywords(content, focus_keywords)
            if not relevant:
                # Fall back to full content if nothing matched
                relevant = content

            header = f"### {fw_name}\n\n"
            chunk = header + relevant

            if len(chunk) > remaining:
                chunk = chunk[:remaining]
            sections.append(chunk)
            remaining -= len(chunk)

            if remaining <= 0:
                break

        if not sections:
            return ""

        return "## Target Framework Best Practices\n\n" + "\n\n".join(sections)

    # ── Internal ──────────────────────────────────────────────────────

    def _build_search_topics(
        self,
        framework: str,
        source_patterns: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Build search topics dynamically from source patterns.

        If source patterns are available, map detected patterns to targeted
        search topics.  Otherwise fall back to generic concern categories.
        """
        topics: List[str] = []

        if source_patterns:
            # Map DI pattern
            di = source_patterns.get("di_pattern", {})
            if isinstance(di, dict):
                for pattern_name, detected in di.items():
                    if detected:
                        key = f"di_{pattern_name}"
                        topic = self.PATTERN_TO_TOPIC.get(key)
                        if topic:
                            topics.append(f'"{framework}" {topic}')

            # Map data layer
            data = source_patterns.get("data_layer", {})
            if isinstance(data, dict):
                for pattern_name, detected in data.items():
                    if detected:
                        key = f"data_{pattern_name}"
                        topic = self.PATTERN_TO_TOPIC.get(key)
                        if topic:
                            topics.append(f'"{framework}" {topic}')

            # Map web layer
            web = source_patterns.get("web_layer", {})
            if isinstance(web, dict):
                for pattern_name, detected in web.items():
                    if detected:
                        key = f"web_{pattern_name}"
                        topic = self.PATTERN_TO_TOPIC.get(key)
                        if topic:
                            topics.append(f'"{framework}" {topic}')

            # Map config pattern
            cfg = source_patterns.get("config_pattern", {})
            if isinstance(cfg, dict):
                for pattern_name, detected in cfg.items():
                    if detected:
                        key = f"config_{pattern_name}"
                        topic = self.PATTERN_TO_TOPIC.get(key)
                        if topic:
                            topics.append(f'"{framework}" {topic}')

            # Map test framework
            test = source_patterns.get("test_framework", {})
            if isinstance(test, dict):
                for pattern_name, detected in test.items():
                    if detected:
                        key = f"test_{pattern_name}"
                        topic = self.PATTERN_TO_TOPIC.get(key)
                        if topic:
                            topics.append(f'"{framework}" {topic}')

        # Fallback to generic concerns if nothing detected
        if not topics:
            topics = [
                f'"{framework}" {concern}' for concern in self.CONCERN_CATEGORIES[:5]
            ]

        # Always include baseline query
        topics.append(f'"{framework}" best practices project structure getting started')

        # Deduplicate and limit to 3 topics (API budget)
        seen: set[str] = set()
        unique: List[str] = []
        for t in topics:
            if t not in seen:
                seen.add(t)
                unique.append(t)

        return unique[:3]

    def _fetch_docs(self, framework: str, topics: List[str]) -> Optional[str]:
        """Execute Tavily searches and combine results into markdown."""
        client = self._get_client()
        if client is None:
            return None

        all_content: List[str] = []

        for topic in topics:
            try:
                response = client.search(
                    query=topic,
                    search_depth="advanced",
                    max_results=3,
                    include_answer=True,
                )

                # Use the AI-generated answer if available
                answer = response.get("answer", "")
                if answer:
                    all_content.append(answer)

                # Also collect result snippets
                for result in response.get("results", []):
                    content = result.get("content", "")
                    if content and len(content) > 50:
                        all_content.append(content)

            except Exception as e:
                logger.warning("Tavily search failed for topic '%s': %s", topic, e)
                continue

        if not all_content:
            return None

        # Deduplicate similar content (simple length-based dedup)
        unique: List[str] = []
        for c in all_content:
            if not any(c[:100] == existing[:100] for existing in unique):
                unique.append(c)

        return "\n\n".join(unique)

    @staticmethod
    def _filter_by_keywords(content: str, keywords: List[str]) -> str:
        """Filter content paragraphs by keyword relevance."""
        if not keywords:
            return content

        paragraphs = content.split("\n\n")
        relevant: List[str] = []

        for para in paragraphs:
            lower = para.lower()
            if any(kw.lower() in lower for kw in keywords):
                relevant.append(para)

        return "\n\n".join(relevant)

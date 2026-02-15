"""Tavily Web Search and Scraping provider implementation.

Tavily is an AI-optimized search API designed specifically for LLM and RAG applications.
It provides clean, relevant search results and content extraction.

API Documentation: https://docs.tavily.com/
"""

import logging
import os
from typing import List, Dict, Any, Optional

from ..interfaces.web_content import (
    WebSearchProvider,
    WebScraperProvider,
    WebSearchResult,
    ScrapedContent
)

logger = logging.getLogger(__name__)


class TavilyProvider(WebSearchProvider, WebScraperProvider):
    """
    Tavily Web Search and Scraping provider.

    Provides both web search and content extraction capabilities using Tavily's API.
    Tavily is specifically designed for AI applications with LLM-optimized results.

    Features:
    - AI-optimized search results (clean, relevant content)
    - Content extraction from URLs
    - Fast response times
    - Built-in answer generation (optional)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
    ):
        """
        Initialize Tavily provider.

        Args:
            api_key: Tavily API key (uses TAVILY_API_KEY env var if not provided)
        """
        self._api_key = api_key or os.getenv("TAVILY_API_KEY")

        if not self._api_key:
            raise ValueError(
                "Tavily API key required. Set TAVILY_API_KEY environment variable."
            )

        self._client = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the Tavily client."""
        try:
            from tavily import TavilyClient

            self._client = TavilyClient(api_key=self._api_key)
            logger.info("Initialized Tavily provider")
        except ImportError:
            raise ImportError(
                "tavily-python not installed. "
                "Run: pip install tavily-python"
            )

    def search(
        self,
        query: str,
        num_results: int = 5,
        **kwargs
    ) -> List[WebSearchResult]:
        """
        Search the web for a given query using Tavily.

        Args:
            query: Search query string
            num_results: Number of results to return (1-20)
            **kwargs: Additional options:
                - search_depth: "basic" or "advanced" (default: "basic")
                - include_answer: Include AI-generated answer (default: False)
                - include_raw_content: Include raw HTML content (default: False)
                - include_domains: List of domains to include
                - exclude_domains: List of domains to exclude

        Returns:
            List of WebSearchResult objects
        """
        if self._client is None:
            raise RuntimeError("Tavily client not initialized")

        num_results = max(1, min(num_results, 20))

        try:
            # Extract kwargs
            search_depth = kwargs.get("search_depth", "basic")
            include_answer = kwargs.get("include_answer", False)
            include_raw_content = kwargs.get("include_raw_content", False)
            include_domains = kwargs.get("include_domains", None)
            exclude_domains = kwargs.get("exclude_domains", None)

            # Tavily search API
            response = self._client.search(
                query=query,
                max_results=num_results,
                search_depth=search_depth,
                include_answer=include_answer,
                include_raw_content=include_raw_content,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
            )

            results = []
            search_results = response.get("results", [])

            for idx, item in enumerate(search_results):
                result = WebSearchResult(
                    url=item.get("url", ""),
                    title=item.get("title", "Untitled"),
                    description=item.get("content", ""),
                    score=item.get("score", 1.0 - (idx * 0.1)),
                    metadata={
                        "raw_content": item.get("raw_content"),
                        "published_date": item.get("published_date"),
                        "answer": response.get("answer") if include_answer else None,
                    }
                )
                results.append(result)

            logger.info(f"Tavily search returned {len(results)} results for: {query}")
            return results

        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            raise RuntimeError(f"Failed to search: {e}") from e

    def _format_as_markdown(self, title: str, url: str, content: str) -> str:
        """
        Format extracted content as a well-structured markdown document.

        Args:
            title: Page title
            url: Source URL
            content: Raw extracted content

        Returns:
            Markdown-formatted string
        """
        # Build markdown document
        md_parts = []

        # Title as H1
        if title:
            md_parts.append(f"# {title}\n")

        # Source metadata
        md_parts.append(f"> **Source:** [{url}]({url})\n")

        # Separator
        md_parts.append("---\n")

        # Content body - preserve any existing structure
        if content:
            # Clean up content: normalize whitespace but preserve paragraphs
            paragraphs = content.split('\n\n')
            cleaned_paragraphs = []
            for p in paragraphs:
                # Normalize internal whitespace in each paragraph
                cleaned = ' '.join(p.split())
                if cleaned:
                    cleaned_paragraphs.append(cleaned)

            md_parts.append('\n\n'.join(cleaned_paragraphs))

        return '\n'.join(md_parts)

    def scrape(
        self,
        url: str,
        **kwargs
    ) -> ScrapedContent:
        """
        Extract content from a URL using Tavily's extract API.

        Args:
            url: URL to scrape
            **kwargs: Additional options:
                - format_markdown: Format output as markdown (default: True)

        Returns:
            ScrapedContent object with extracted text in markdown format
        """
        if self._client is None:
            raise RuntimeError("Tavily client not initialized")

        format_markdown = kwargs.get("format_markdown", True)

        try:
            # Tavily extract API
            response = self._client.extract(urls=[url])

            results = response.get("results", [])

            if not results:
                raise RuntimeError(f"No content extracted from {url}")

            # Get first result (we only requested one URL)
            item = results[0]
            raw_content = item.get("raw_content", "")
            title = item.get("title", url)

            # Format as markdown if requested (default)
            if format_markdown:
                content = self._format_as_markdown(title, url, raw_content)
            else:
                content = raw_content

            # Calculate word count
            word_count = len(content.split()) if content else 0

            scraped = ScrapedContent(
                url=url,
                title=title,
                content=content,
                word_count=word_count,
                metadata={
                    "failed_results": response.get("failed_results", []),
                    "format": "markdown" if format_markdown else "raw",
                }
            )

            logger.info(f"Tavily extracted {word_count} words from: {url} (format: {'markdown' if format_markdown else 'raw'})")
            return scraped

        except Exception as e:
            logger.error(f"Tavily extract failed for {url}: {e}")
            raise RuntimeError(f"Failed to scrape {url}: {e}") from e

    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about this provider.

        Returns:
            Dictionary with provider capabilities and settings
        """
        return {
            "name": "tavily",
            "display_name": "Tavily AI Search",
            "available": self._client is not None,
            "max_results": 20,
            "capabilities": [
                "web_search",
                "content_extraction",
                "ai_answer_generation",
                "domain_filtering",
            ],
            "search_depths": ["basic", "advanced"],
            "documentation": "https://docs.tavily.com/",
        }

    @property
    def name(self) -> str:
        """Get the provider name identifier."""
        return "tavily"

    def validate(self) -> bool:
        """
        Validate provider configuration and connectivity.

        Returns:
            True if provider is properly configured
        """
        try:
            if self._client is None:
                return False

            # Try a simple search to verify API key works
            self._client.search(query="test", max_results=1)
            return True
        except Exception as e:
            logger.warning(f"Tavily validation failed: {e}")
            return False

    def search_with_answer(
        self,
        query: str,
        num_results: int = 5,
        search_depth: str = "advanced"
    ) -> Dict[str, Any]:
        """
        Search and get an AI-generated answer along with results.

        This is a convenience method that combines search with answer generation.

        Args:
            query: Search query string
            num_results: Number of results to return
            search_depth: "basic" or "advanced"

        Returns:
            Dictionary with 'answer' and 'results' keys
        """
        if self._client is None:
            raise RuntimeError("Tavily client not initialized")

        try:
            response = self._client.search(
                query=query,
                max_results=num_results,
                search_depth=search_depth,
                include_answer=True,
            )

            return {
                "answer": response.get("answer", ""),
                "results": [
                    WebSearchResult(
                        url=item.get("url", ""),
                        title=item.get("title", "Untitled"),
                        description=item.get("content", ""),
                        score=item.get("score", 0.0),
                    )
                    for item in response.get("results", [])
                ]
            }

        except Exception as e:
            logger.error(f"Tavily search_with_answer failed: {e}")
            raise RuntimeError(f"Failed to search: {e}") from e

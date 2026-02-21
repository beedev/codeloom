"""Web Content Ingestion service for scraping and embedding web content."""

import logging
import hashlib
import os
import re
from typing import List, Dict, Any, Optional
from datetime import datetime

from llama_index.core import Document
from llama_index.core.schema import BaseNode
from llama_index.core.node_parser import SentenceSplitter

from ..interfaces.web_content import WebSearchProvider, WebScraperProvider, WebSearchResult, ScrapedContent
from ..providers.tavily import TavilyProvider
from ..project import ProjectManager
from ...setting import get_settings, RAGSettings

logger = logging.getLogger(__name__)


class WebContentIngestion:
    """
    Service for ingesting web content into projects.

    Provides:
    - Web search via Firecrawl
    - Content scraping via Jina Reader
    - Document creation and embedding
    """

    def __init__(
        self,
        search_provider: Optional[WebSearchProvider] = None,
        scraper_provider: Optional[WebScraperProvider] = None,
        project_manager: Optional[ProjectManager] = None,
        setting: Optional[RAGSettings] = None,
    ):
        """
        Initialize web content ingestion service.

        Args:
            search_provider: Web search provider (default: Firecrawl)
            scraper_provider: Web scraper provider (default: Jina Reader)
            project_manager: Project manager for document tracking
            setting: RAG settings instance
        """
        self._setting = setting or get_settings()
        self._project_manager = project_manager

        # Initialize providers with lazy loading
        self._search_provider = search_provider
        self._scraper_provider = scraper_provider

        # Node parser for chunking
        self._node_parser = SentenceSplitter(
            chunk_size=self._setting.ingestion.chunk_size,
            chunk_overlap=self._setting.ingestion.chunk_overlap,
        )

        logger.info("WebContentIngestion initialized")

    def _get_search_provider(self) -> WebSearchProvider:
        """Get or create search provider (lazy initialization).

        Uses Tavily for AI-optimized web search.
        """
        if self._search_provider is None:
            if not os.getenv("TAVILY_API_KEY"):
                raise RuntimeError(
                    "TAVILY_API_KEY environment variable required for web search. "
                    "Get your API key at https://tavily.com"
                )
            self._search_provider = TavilyProvider()
            logger.info("Using Tavily for web search")
        return self._search_provider

    def _get_scraper_provider(self) -> WebScraperProvider:
        """Get or create scraper provider (lazy initialization).

        Uses Tavily for AI-optimized content extraction.
        """
        if self._scraper_provider is None:
            if not os.getenv("TAVILY_API_KEY"):
                raise RuntimeError(
                    "TAVILY_API_KEY environment variable required for web scraping. "
                    "Get your API key at https://tavily.com"
                )
            self._scraper_provider = TavilyProvider()
            logger.info("Using Tavily for web scraping")
        return self._scraper_provider

    def search(
        self,
        query: str,
        num_results: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search the web for content.

        Args:
            query: Search query
            num_results: Number of results to return

        Returns:
            List of search results with url, title, description
        """
        provider = self._get_search_provider()
        results = provider.search(query, num_results=num_results, **kwargs)

        return [
            {
                "url": r.url,
                "title": r.title,
                "description": r.description,
                "score": r.score,
            }
            for r in results
        ]

    def preview_url(
        self,
        url: str,
        max_chars: int = 500
    ) -> Dict[str, Any]:
        """
        Get a preview of URL content.

        Args:
            url: URL to preview
            max_chars: Maximum characters in preview

        Returns:
            Preview with title, content_preview, word_count
        """
        provider = self._get_scraper_provider()
        result = provider.preview(url, max_chars=max_chars)

        return {
            "url": result.url,
            "title": result.title,
            "content_preview": result.content,
            "word_count": result.word_count,
        }

    def scrape_url(self, url: str) -> ScrapedContent:
        """
        Scrape full content from a URL.

        Args:
            url: URL to scrape

        Returns:
            ScrapedContent with full text
        """
        provider = self._get_scraper_provider()
        return provider.scrape(url)

    def scrape_urls(self, urls: List[str]) -> List[ScrapedContent]:
        """
        Scrape content from multiple URLs.

        Args:
            urls: List of URLs to scrape

        Returns:
            List of ScrapedContent objects
        """
        results = []
        for url in urls:
            try:
                result = self.scrape_url(url)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to scrape {url}: {e}")
                # Continue with other URLs
        return results

    def create_nodes_from_content(
        self,
        content: ScrapedContent
    ) -> List[BaseNode]:
        """
        Create document nodes from scraped content.

        Args:
            content: Scraped web content

        Returns:
            List of parsed nodes
        """
        # Create a Document from the scraped content
        doc = Document(
            text=content.content,
            metadata={
                "source": content.url,
                "title": content.title,
                "source_type": "web",
                "word_count": content.word_count,
                "scraped_at": datetime.utcnow().isoformat(),
            }
        )

        # Parse into nodes
        nodes = self._node_parser.get_nodes_from_documents([doc])

        logger.info(f"Created {len(nodes)} nodes from {content.url}")
        return nodes

    def ingest_urls_to_project(
        self,
        project_id: str,
        urls: List[str],
        source_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Ingest web content from URLs into a project.

        This is the main method for adding web content to a project.
        It scrapes the URLs, creates nodes, and registers them.

        Args:
            project_id: Target project ID
            urls: List of URLs to ingest
            source_name: Optional custom name for the source (e.g., search query)

        Returns:
            List of ingested sources with source_id, url, title, chunk_count
        """
        if not self._project_manager:
            raise RuntimeError("ProjectManager not configured for web ingestion")

        ingested_sources = []

        for url in urls:
            try:
                # Scrape the URL
                scraped = self.scrape_url(url)

                # Create nodes from content
                nodes = self.create_nodes_from_content(scraped)

                # Generate a content hash for duplicate detection
                content_hash = hashlib.sha256(scraped.content.encode()).hexdigest()

                # Register with project manager
                # Use source_name (search query) if provided, otherwise use page title
                display_name = source_name or scraped.title
                # Clean the name for file system (remove special chars, limit length)
                clean_name = re.sub(r'[^\w\s-]', '', display_name)[:50].strip()
                clean_name = re.sub(r'\s+', '_', clean_name)

                source_id = self._project_manager.add_document(
                    project_id=project_id,
                    file_name=f"web_{clean_name}_{content_hash[:8]}.md",
                    file_content=scraped.content.encode(),
                    file_type="web",
                    chunk_count=len(nodes),
                )

                ingested_sources.append({
                    "source_id": source_id,
                    "url": url,
                    "title": scraped.title,
                    "chunk_count": len(nodes),
                    "word_count": scraped.word_count,
                    "nodes": nodes,  # For embedding by the caller
                })

                logger.info(f"Ingested web content: {url} -> {source_id}")

            except ValueError as e:
                # Duplicate content
                logger.warning(f"Duplicate content from {url}: {e}")
            except Exception as e:
                logger.error(f"Failed to ingest {url}: {e}")
                # Continue with other URLs

        return ingested_sources

    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about available providers.

        Returns:
            Dictionary with provider status and capabilities
        """
        search_info = {}
        scraper_info = {}

        try:
            search_provider = self._get_search_provider()
            search_info = search_provider.get_provider_info()
        except Exception as e:
            search_info = {"available": False, "error": str(e)}

        try:
            scraper_provider = self._get_scraper_provider()
            scraper_info = scraper_provider.get_provider_info()
        except Exception as e:
            scraper_info = {"available": False, "error": str(e)}

        return {
            "search": search_info,
            "scraper": scraper_info,
        }

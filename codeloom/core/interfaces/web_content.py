"""Abstract interfaces for web content providers (search and scraping)."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class WebSearchResult:
    """Represents a single web search result."""
    url: str
    title: str
    description: str
    score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ScrapedContent:
    """Represents scraped content from a web page."""
    url: str
    title: str
    content: str
    word_count: int
    metadata: Optional[Dict[str, Any]] = None


class WebSearchProvider(ABC):
    """Abstract base class for web search providers.

    All web search providers must implement this interface
    to be compatible with the plugin system.
    """

    @abstractmethod
    def search(
        self,
        query: str,
        num_results: int = 5,
        **kwargs
    ) -> List[WebSearchResult]:
        """Search the web for a given query.

        Args:
            query: Search query string
            num_results: Number of results to return (1-20)
            **kwargs: Provider-specific options

        Returns:
            List of WebSearchResult objects
        """
        pass

    @abstractmethod
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about this provider.

        Returns:
            Dictionary with provider capabilities and settings
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the provider name identifier."""
        pass

    def validate(self) -> bool:
        """Validate provider configuration and connectivity.

        Returns:
            True if provider is properly configured
        """
        try:
            info = self.get_provider_info()
            return info.get("available", False)
        except Exception:
            return False


class WebScraperProvider(ABC):
    """Abstract base class for web scraping providers.

    All web scraping providers must implement this interface
    to be compatible with the plugin system.
    """

    @abstractmethod
    def scrape(
        self,
        url: str,
        **kwargs
    ) -> ScrapedContent:
        """Scrape content from a URL.

        Args:
            url: URL to scrape
            **kwargs: Provider-specific options

        Returns:
            ScrapedContent object with extracted text
        """
        pass

    @abstractmethod
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about this provider.

        Returns:
            Dictionary with provider capabilities and settings
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the provider name identifier."""
        pass

    def validate(self) -> bool:
        """Validate provider configuration and connectivity.

        Returns:
            True if provider is properly configured
        """
        try:
            info = self.get_provider_info()
            return info.get("available", False)
        except Exception:
            return False

    def preview(self, url: str, max_chars: int = 500) -> ScrapedContent:
        """Get a preview of scraped content.

        Args:
            url: URL to preview
            max_chars: Maximum characters in preview

        Returns:
            ScrapedContent with truncated content
        """
        result = self.scrape(url)
        if len(result.content) > max_chars:
            result.content = result.content[:max_chars] + "..."
        return result

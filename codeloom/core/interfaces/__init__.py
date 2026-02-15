"""Abstract interfaces for swappable RAG components."""

from .retrieval import RetrievalStrategy
from .llm import LLMProvider
from .embedding import EmbeddingProvider
from .processor import ContentProcessor
from .image_generation import ImageGenerationProvider
from .web_content import WebSearchProvider, WebScraperProvider, WebSearchResult, ScrapedContent
from .vision import VisionProvider, VisionAnalysisResult
from .services import (
    IChatService,
    IImageService,
    IDocumentService,
    INotebookService,
    IVisionService,
)
from .routing import (
    IDocumentRoutingService,
    IRoutingPrompts,
    RoutingStrategy,
    RoutingResult,
    DocumentSummary,
)

__all__ = [
    "RetrievalStrategy",
    "LLMProvider",
    "EmbeddingProvider",
    "ContentProcessor",
    "ImageGenerationProvider",
    "WebSearchProvider",
    "WebScraperProvider",
    "WebSearchResult",
    "ScrapedContent",
    "VisionProvider",
    "VisionAnalysisResult",
    # Service interfaces
    "IChatService",
    "IImageService",
    "IDocumentService",
    "INotebookService",
    "IVisionService",
    # Routing interfaces
    "IDocumentRoutingService",
    "IRoutingPrompts",
    "RoutingStrategy",
    "RoutingResult",
    "DocumentSummary",
]

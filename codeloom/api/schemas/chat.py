"""Chat request/response schemas."""

from typing import List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., description="User query/message", min_length=1)
    project_id: Optional[str] = Field(None, description="Project ID to query (optional)")
    mode: str = Field("chat", description="Chat mode: 'chat' or 'QA'")
    stream: bool = Field(False, description="Stream response token-by-token")


class SourceReference(BaseModel):
    """Source document reference."""
    document_name: str = Field(..., description="Name of source document")
    chunk_location: Optional[str] = Field(None, description="Location in document (page, section, etc.)")
    relevance_score: float = Field(..., description="Relevance score 0.0-1.0")
    excerpt: str = Field(..., description="Text excerpt from chunk")
    project_id: Optional[str] = Field(None, description="Project this source belongs to")
    source_id: Optional[str] = Field(None, description="Document UUID")


class ChatResponse(BaseModel):
    """Chat response model."""
    success: bool = Field(..., description="Whether request succeeded")
    response: str = Field(..., description="LLM-generated response")
    sources: List[SourceReference] = Field(default_factory=list, description="Source documents used")
    project_ids: List[str] = Field(default_factory=list, description="Projects queried")
    retrieval_strategy: Optional[str] = Field(None, description="Retrieval method used")
    error: Optional[str] = Field(None, description="Error message if failed")

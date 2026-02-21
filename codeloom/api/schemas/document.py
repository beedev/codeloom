"""Document request/response schemas."""

from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    """Document information."""
    source_id: str = Field(..., description="Document UUID")
    file_name: str = Field(..., description="Original filename")
    file_size: int = Field(..., description="File size in bytes")
    chunk_count: int = Field(0, description="Number of chunks/nodes")
    uploaded_at: str = Field(..., description="Upload timestamp")
    project_id: Optional[str] = Field(None, description="Parent project ID")


class DocumentUploadResponse(BaseModel):
    """Document upload response."""
    success: bool = Field(..., description="Whether upload succeeded")
    uploaded: List[dict] = Field(default_factory=list, description="Uploaded documents info")
    count: int = Field(0, description="Number of documents uploaded")
    message: str = Field(..., description="Status message")
    error: Optional[str] = Field(None, description="Error message if failed")


class DocumentList(BaseModel):
    """List of documents response."""
    success: bool = Field(..., description="Whether request succeeded")
    documents: List[DocumentInfo] = Field(default_factory=list, description="List of documents")
    project_id: Optional[str] = Field(None, description="Parent project ID")
    error: Optional[str] = Field(None, description="Error message if failed")

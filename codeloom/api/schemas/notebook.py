"""Project request/response schemas."""

from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class ProjectCreate(BaseModel):
    """Create project request."""
    name: str = Field(..., description="Project name", min_length=1)
    description: Optional[str] = Field(None, description="Project description")


class ProjectResponse(BaseModel):
    """Project response model."""
    success: bool = Field(..., description="Whether request succeeded")
    project: Optional[dict] = Field(None, description="Project data")
    data: Optional[dict] = Field(None, description="Project data (alternative key)")
    error: Optional[str] = Field(None, description="Error message if failed")


class ProjectInfo(BaseModel):
    """Project information."""
    id: str = Field(..., description="Project UUID")
    name: str = Field(..., description="Project name")
    description: Optional[str] = Field(None, description="Project description")
    document_count: int = Field(0, description="Number of documents")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    updated_at: Optional[str] = Field(None, description="Last update timestamp")


class ProjectList(BaseModel):
    """List of projects response."""
    success: bool = Field(..., description="Whether request succeeded")
    projects: List[dict] = Field(default_factory=list, description="List of projects")
    count: int = Field(0, description="Total project count")
    error: Optional[str] = Field(None, description="Error message if failed")

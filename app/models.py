from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class PatientNote(BaseModel):
    """Domain model for a patient note that we index in Qdrant."""

    id: Optional[str] = Field(
        default=None,
        description="Stable unique identifier for the note. If not provided, a deterministic ID is generated.",
    )
    patient_id: str = Field(..., description="Unique identifier of the patient")
    note: str = Field(..., description="Raw clinical note text")
    diagnosis: Optional[str] = Field(
        default=None,
        description="Primary diagnosis mentioned in the note, if any",
    )
    medications: Optional[List[str]] = Field(
        default=None,
        description="List of medications referenced in the note",
    )
    timestamp: Optional[datetime] = Field(
        default=None,
        description="Timestamp of when the note was written or recorded",
    )


class SearchRequest(BaseModel):
    """Request body for semantic search over patient notes."""

    query: str = Field(..., description="Free-text search query from a clinician")
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of search results to return (1-100)",
    )


class SearchResult(BaseModel):
    """Single search hit enriched with metadata from the stored note."""

    id: str
    patient_id: str
    score: float
    note: str
    diagnosis: Optional[str] = None
    medications: Optional[List[str]] = None
    timestamp: Optional[datetime] = None


class SearchResponse(BaseModel):
    """Response body for semantic search."""

    query: str
    limit: int
    results: List[SearchResult]


class RebuildIndexResponse(BaseModel):
    """Response model for index rebuild operations."""

    success: bool
    used_fallback: bool
    message: str

"""Pydantic schemas for API and internal use."""

from typing import List, Optional

from pydantic import BaseModel, Field


class SourceCitation(BaseModel):
    """Citation for a document source."""

    document_id: str = Field(..., description="Document identifier")
    document_name: str = Field(..., description="Human-readable document name")
    chunk_id: str = Field(..., description="Chunk identifier")
    content: str = Field(..., description="Relevant text excerpt")
    score: float = Field(..., description="Relevance score (0-1)")


class QueryRequest(BaseModel):
    """Request schema for /query endpoint (frontend compatibility)."""

    question: str = Field(..., min_length=1, max_length=2000)
    history: Optional[List[str]] = Field(
        default=None,
        description="Optional list of previous user/assistant messages for context",
    )


class ChatRequest(BaseModel):
    """Request schema for chat endpoint."""

    query: str = Field(..., min_length=1, max_length=2000)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    include_sources: bool = Field(default=True)
    history: Optional[List[str]] = Field(
        default=None,
        description="Optional list of previous user/assistant messages for context",
    )


class ChatResponse(BaseModel):
    """Response schema for chat endpoint."""

    answer: str = Field(..., description="Generated answer")
    sources: List[SourceCitation] = Field(
        default_factory=list,
        description="Cited document sources",
    )
    query: str = Field(..., description="Original query")


class DocumentChunk(BaseModel):
    """Internal representation of a document chunk."""

    chunk_id: str
    document_id: str
    document_name: str
    content: str
    metadata: dict = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    index_loaded: bool = False
    embedding_model: str = ""
    llm_model: str = ""

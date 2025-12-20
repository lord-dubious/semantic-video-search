"""Pydantic models for Semantic Video Search."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class VideoStatus(str, Enum):
    """Video processing status."""

    PENDING = "pending"
    EXTRACTING = "extracting"
    EMBEDDING = "embedding"
    INDEXED = "indexed"
    ERROR = "error"


class VideoMetadata(BaseModel):
    """Metadata for a video file."""

    video_id: str = Field(description="Unique video identifier")
    filename: str = Field(description="Original filename")
    filepath: str = Field(description="Full path to video file")
    duration: float = Field(default=0.0, description="Video duration in seconds")
    fps: float = Field(default=0.0, description="Frames per second")
    width: int = Field(default=0, description="Video width in pixels")
    height: int = Field(default=0, description="Video height in pixels")
    codec: str = Field(default="", description="Video codec")
    size_bytes: int = Field(default=0, description="File size in bytes")
    created_at: datetime = Field(default_factory=datetime.now)
    status: VideoStatus = Field(default=VideoStatus.PENDING)


class FrameData(BaseModel):
    """Data for an extracted video frame."""

    frame_id: str = Field(description="Unique frame identifier")
    video_id: str = Field(description="Parent video ID")
    timestamp: float = Field(description="Frame timestamp in seconds")
    frame_number: int = Field(description="Frame number in video")
    filepath: str = Field(description="Path to saved frame image")
    width: int = Field(default=0, description="Frame width")
    height: int = Field(default=0, description="Frame height")
    description: str = Field(default="", description="AI-generated frame description")
    extracted_at: datetime = Field(default_factory=datetime.now)


class FrameEmbedding(BaseModel):
    """Embedding data for a video frame."""

    frame_id: str = Field(description="Frame identifier")
    video_id: str = Field(description="Parent video ID")
    timestamp: float = Field(description="Frame timestamp")
    description: str = Field(description="Frame description")
    embedding: list[float] = Field(description="Vector embedding")
    created_at: datetime = Field(default_factory=datetime.now)


class SearchQuery(BaseModel):
    """A search query for finding video moments."""

    query: str = Field(description="Natural language search query")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of results to return")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Similarity threshold")
    video_ids: list[str] | None = Field(default=None, description="Filter to specific videos")


class SearchResult(BaseModel):
    """A single search result."""

    frame_id: str = Field(description="Frame identifier")
    video_id: str = Field(description="Video identifier")
    timestamp: float = Field(description="Frame timestamp in seconds")
    description: str = Field(description="Frame description")
    similarity: float = Field(description="Similarity score")
    verified: bool = Field(default=False, description="Whether verified by LLM")
    verification_note: str = Field(default="", description="LLM verification note")


class SearchResults(BaseModel):
    """Collection of search results."""

    query: str = Field(description="Original query")
    results: list[SearchResult] = Field(default_factory=list)
    total_found: int = Field(default=0, description="Total matches found")
    search_time_ms: float = Field(default=0.0, description="Search time in milliseconds")


class ExtractionConfig(BaseModel):
    """Configuration for frame extraction."""

    interval_seconds: float = Field(
        default=2.0, ge=0.1, description="Extract frame every N seconds"
    )
    format: str = Field(default="jpg", description="Output image format")
    quality: int = Field(default=85, ge=1, le=100, description="JPEG quality")
    max_width: int | None = Field(default=None, description="Max width (preserve aspect ratio)")
    max_height: int | None = Field(default=None, description="Max height (preserve aspect ratio)")


class EmbeddingConfig(BaseModel):
    """Configuration for embeddings."""

    model_name: str = Field(default="models/text-embedding-004", description="Embedding model")
    dimension: int = Field(default=768, description="Embedding dimension")
    batch_size: int = Field(default=10, ge=1, le=100, description="Batch size for embedding")


class VectorStoreConfig(BaseModel):
    """Configuration for vector store."""

    db_path: str = Field(default="./data/lancedb", description="Path to LanceDB database")
    table_name: str = Field(default="video_frames", description="Table name")


class VideoSearchConfig(BaseModel):
    """Main configuration for video search system."""

    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    frames_dir: str = Field(default="./data/frames", description="Directory for extracted frames")


class IndexingProgress(BaseModel):
    """Progress tracking for video indexing."""

    video_id: str = Field(description="Video being indexed")
    total_frames: int = Field(default=0, description="Total frames to process")
    extracted_frames: int = Field(default=0, description="Frames extracted")
    embedded_frames: int = Field(default=0, description="Frames embedded")
    status: VideoStatus = Field(default=VideoStatus.PENDING)
    error_message: str | None = Field(default=None)
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: datetime | None = Field(default=None)


class VerificationRequest(BaseModel):
    """Request to verify a search result with the LLM."""

    frame_id: str = Field(description="Frame to verify")
    query: str = Field(description="Original search query")
    frame_path: str = Field(description="Path to frame image")
    description: str = Field(description="Frame description")


class VerificationResult(BaseModel):
    """Result from LLM verification."""

    frame_id: str = Field(description="Frame verified")
    is_match: bool = Field(description="Whether frame matches query")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score")
    explanation: str = Field(default="", description="Explanation of verification")

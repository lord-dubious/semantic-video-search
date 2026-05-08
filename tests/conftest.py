"""Test fixtures for Semantic Video Search."""

from __future__ import annotations

import pytest

from video_search.models import (
    EmbeddingConfig,
    ExtractionConfig,
    FrameData,
    FrameEmbedding,
    IndexingProgress,
    SearchQuery,
    SearchResult,
    SearchResults,
    VectorStoreConfig,
    VideoMetadata,
    VideoSearchConfig,
    VideoStatus,
)


@pytest.fixture
def sample_extraction_config() -> ExtractionConfig:
    """Create sample extraction config."""
    return ExtractionConfig(
        interval_seconds=2.0,
        format="jpg",
        quality=85,
    )


@pytest.fixture
def sample_embedding_config() -> EmbeddingConfig:
    """Create sample embedding config."""
    return EmbeddingConfig(
        model_name="models/text-embedding-004",
        dimension=768,
        batch_size=10,
    )


@pytest.fixture
def sample_vector_store_config(tmp_path) -> VectorStoreConfig:
    """Create sample vector store config."""
    return VectorStoreConfig(
        db_path=str(tmp_path / "lancedb"),
        table_name="test_frames",
    )


@pytest.fixture
def sample_video_search_config(tmp_path) -> VideoSearchConfig:
    """Create sample video search config."""
    return VideoSearchConfig(
        frames_dir=str(tmp_path / "frames"),
    )


@pytest.fixture
def sample_video_metadata() -> VideoMetadata:
    """Create sample video metadata."""
    return VideoMetadata(
        video_id="abc123def456",
        filename="test_video.mp4",
        filepath="/path/to/test_video.mp4",
        duration=120.0,
        fps=30.0,
        width=1920,
        height=1080,
        codec="h264",
        size_bytes=50_000_000,
        status=VideoStatus.PENDING,
    )


@pytest.fixture
def sample_frame_data() -> FrameData:
    """Create sample frame data."""
    return FrameData(
        frame_id="abc123_000001",
        video_id="abc123def456",
        timestamp=2.0,
        frame_number=1,
        filepath="/path/to/frames/frame_000001.jpg",
        width=1920,
        height=1080,
    )


@pytest.fixture
def sample_frame_embedding() -> FrameEmbedding:
    """Create sample frame embedding."""
    return FrameEmbedding(
        frame_id="abc123_000001",
        video_id="abc123def456",
        timestamp=2.0,
        description="A person walking through a park with trees in the background.",
        embedding=[0.1] * 768,
    )


@pytest.fixture
def sample_search_query() -> SearchQuery:
    """Create sample search query."""
    return SearchQuery(
        query="person walking in park",
        top_k=5,
        threshold=0.5,
    )


@pytest.fixture
def sample_search_result() -> SearchResult:
    """Create sample search result."""
    return SearchResult(
        frame_id="abc123_000001",
        video_id="abc123def456",
        timestamp=2.0,
        description="A person walking through a park.",
        similarity=0.85,
    )


@pytest.fixture
def sample_search_results(sample_search_result) -> SearchResults:
    """Create sample search results."""
    return SearchResults(
        query="person walking in park",
        results=[sample_search_result],
        total_found=1,
        search_time_ms=15.5,
    )


@pytest.fixture
def sample_indexing_progress() -> IndexingProgress:
    """Create sample indexing progress."""
    return IndexingProgress(
        video_id="abc123def456",
        total_frames=60,
        extracted_frames=30,
        embedded_frames=25,
        status=VideoStatus.EMBEDDING,
    )


@pytest.fixture
def mock_ffprobe_output():
    """Create mock ffprobe output."""
    return {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "r_frame_rate": "30/1",
            }
        ],
        "format": {
            "duration": "120.0",
            "size": "50000000",
        },
    }


@pytest.fixture
def temp_video_dir(tmp_path):
    """Create a temporary directory for video tests."""
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    return video_dir


@pytest.fixture
def temp_frames_dir(tmp_path):
    """Create a temporary directory for frames."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    return frames_dir


@pytest.fixture(autouse=True)
def set_test_env(monkeypatch):
    """Set test environment variables."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LANCEDB_PATH", "/tmp/test_lancedb")

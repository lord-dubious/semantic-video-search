"""Tests for Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

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
    VerificationRequest,
    VerificationResult,
    VideoMetadata,
    VideoSearchConfig,
    VideoStatus,
)


class TestVideoStatus:
    """Tests for VideoStatus enum."""

    def test_video_statuses(self):
        """Test all video statuses exist."""
        assert VideoStatus.PENDING == "pending"
        assert VideoStatus.EXTRACTING == "extracting"
        assert VideoStatus.EMBEDDING == "embedding"
        assert VideoStatus.INDEXED == "indexed"
        assert VideoStatus.ERROR == "error"


class TestExtractionConfig:
    """Tests for ExtractionConfig."""

    def test_create_extraction_config(self):
        """Test creating extraction config."""
        config = ExtractionConfig(
            interval_seconds=1.0,
            format="png",
            quality=90,
        )
        assert config.interval_seconds == 1.0
        assert config.format == "png"
        assert config.quality == 90

    def test_extraction_config_defaults(self):
        """Test default values."""
        config = ExtractionConfig()
        assert config.interval_seconds == 2.0
        assert config.format == "jpg"
        assert config.quality == 85

    def test_extraction_config_validation(self):
        """Test validation constraints."""
        with pytest.raises(ValidationError):
            ExtractionConfig(interval_seconds=0.0)

        with pytest.raises(ValidationError):
            ExtractionConfig(quality=101)


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig."""

    def test_create_embedding_config(self):
        """Test creating embedding config."""
        config = EmbeddingConfig(
            model_name="test-model",
            dimension=512,
            batch_size=20,
        )
        assert config.model_name == "test-model"
        assert config.dimension == 512
        assert config.batch_size == 20

    def test_embedding_config_defaults(self):
        """Test default values."""
        config = EmbeddingConfig()
        assert config.model_name == "models/text-embedding-004"
        assert config.dimension == 768
        assert config.batch_size == 10


class TestVectorStoreConfig:
    """Tests for VectorStoreConfig."""

    def test_create_vector_store_config(self):
        """Test creating vector store config."""
        config = VectorStoreConfig(
            db_path="/custom/path",
            table_name="custom_table",
        )
        assert config.db_path == "/custom/path"
        assert config.table_name == "custom_table"

    def test_vector_store_config_defaults(self):
        """Test default values."""
        config = VectorStoreConfig()
        assert config.db_path == "./data/lancedb"
        assert config.table_name == "video_frames"


class TestVideoMetadata:
    """Tests for VideoMetadata."""

    def test_create_video_metadata(self, sample_video_metadata):
        """Test creating video metadata."""
        assert sample_video_metadata.video_id == "abc123def456"
        assert sample_video_metadata.duration == 120.0
        assert sample_video_metadata.fps == 30.0

    def test_video_metadata_status(self):
        """Test video status field."""
        metadata = VideoMetadata(
            video_id="test",
            filename="test.mp4",
            filepath="/path/test.mp4",
            status=VideoStatus.INDEXED,
        )
        assert metadata.status == VideoStatus.INDEXED


class TestFrameData:
    """Tests for FrameData."""

    def test_create_frame_data(self, sample_frame_data):
        """Test creating frame data."""
        assert sample_frame_data.frame_id == "abc123_000001"
        assert sample_frame_data.timestamp == 2.0
        assert sample_frame_data.frame_number == 1

    def test_frame_data_defaults(self):
        """Test default values."""
        frame = FrameData(
            frame_id="test",
            video_id="video1",
            timestamp=0.0,
            frame_number=0,
            filepath="/path/frame.jpg",
        )
        assert frame.description == ""
        assert frame.width == 0


class TestFrameEmbedding:
    """Tests for FrameEmbedding."""

    def test_create_frame_embedding(self, sample_frame_embedding):
        """Test creating frame embedding."""
        assert sample_frame_embedding.frame_id == "abc123_000001"
        assert len(sample_frame_embedding.embedding) == 768

    def test_frame_embedding_description(self):
        """Test description field."""
        embedding = FrameEmbedding(
            frame_id="test",
            video_id="video1",
            timestamp=0.0,
            description="Test description",
            embedding=[0.0] * 768,
        )
        assert embedding.description == "Test description"


class TestSearchQuery:
    """Tests for SearchQuery."""

    def test_create_search_query(self, sample_search_query):
        """Test creating search query."""
        assert sample_search_query.query == "person walking in park"
        assert sample_search_query.top_k == 5

    def test_search_query_validation(self):
        """Test validation constraints."""
        with pytest.raises(ValidationError):
            SearchQuery(query="test", top_k=0)

        with pytest.raises(ValidationError):
            SearchQuery(query="test", threshold=1.5)

    def test_search_query_video_filter(self):
        """Test video ID filter."""
        query = SearchQuery(
            query="test",
            video_ids=["vid1", "vid2"],
        )
        assert query.video_ids == ["vid1", "vid2"]


class TestSearchResult:
    """Tests for SearchResult."""

    def test_create_search_result(self, sample_search_result):
        """Test creating search result."""
        assert sample_search_result.similarity == 0.85
        assert not sample_search_result.verified

    def test_search_result_verification(self):
        """Test verification fields."""
        result = SearchResult(
            frame_id="test",
            video_id="video1",
            timestamp=0.0,
            description="Test",
            similarity=0.9,
            verified=True,
            verification_note="Confirmed match",
        )
        assert result.verified
        assert result.verification_note == "Confirmed match"


class TestSearchResults:
    """Tests for SearchResults."""

    def test_create_search_results(self, sample_search_results):
        """Test creating search results."""
        assert sample_search_results.total_found == 1
        assert len(sample_search_results.results) == 1

    def test_empty_search_results(self):
        """Test empty results."""
        results = SearchResults(query="test")
        assert results.total_found == 0
        assert len(results.results) == 0


class TestVideoSearchConfig:
    """Tests for VideoSearchConfig."""

    def test_create_config(self):
        """Test creating video search config."""
        config = VideoSearchConfig(
            frames_dir="/custom/frames",
        )
        assert config.frames_dir == "/custom/frames"

    def test_nested_configs(self):
        """Test nested configurations."""
        config = VideoSearchConfig()
        assert isinstance(config.extraction, ExtractionConfig)
        assert isinstance(config.embedding, EmbeddingConfig)
        assert isinstance(config.vector_store, VectorStoreConfig)


class TestIndexingProgress:
    """Tests for IndexingProgress."""

    def test_create_indexing_progress(self, sample_indexing_progress):
        """Test creating indexing progress."""
        assert sample_indexing_progress.total_frames == 60
        assert sample_indexing_progress.embedded_frames == 25

    def test_progress_status(self):
        """Test progress status tracking."""
        progress = IndexingProgress(
            video_id="test",
            status=VideoStatus.ERROR,
            error_message="Something went wrong",
        )
        assert progress.status == VideoStatus.ERROR
        assert progress.error_message == "Something went wrong"


class TestVerificationRequest:
    """Tests for VerificationRequest."""

    def test_create_verification_request(self):
        """Test creating verification request."""
        request = VerificationRequest(
            frame_id="test",
            query="find the cat",
            frame_path="/path/to/frame.jpg",
            description="A cat sitting on a couch",
        )
        assert request.query == "find the cat"


class TestVerificationResult:
    """Tests for VerificationResult."""

    def test_create_verification_result(self):
        """Test creating verification result."""
        result = VerificationResult(
            frame_id="test",
            is_match=True,
            confidence=0.95,
            explanation="Frame clearly shows a cat",
        )
        assert result.is_match
        assert result.confidence == 0.95

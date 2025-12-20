"""Tests for search engine."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

import pytest

from video_search.models import (
    FrameData,
    FrameEmbedding,
    SearchResults,
    VideoMetadata,
    VideoSearchConfig,
    VideoStatus,
)
from video_search.search import (
    VideoSearchEngine,
    create_search_engine,
)


class TestVideoSearchEngine:
    """Tests for VideoSearchEngine class."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        engine = create_search_engine()
        assert engine.config is not None
        assert engine.extractor is not None
        assert engine.embedder is not None
        assert engine.store is not None

    def test_init_custom_config(self, sample_video_search_config):
        """Test initialization with custom config."""
        engine = create_search_engine(sample_video_search_config)
        assert engine.config == sample_video_search_config

    def test_on_progress_callback(self):
        """Test progress callback registration."""
        engine = create_search_engine()

        progress_updates = []
        engine.on_progress(lambda p: progress_updates.append(p))

        assert engine._progress_callback is not None

    def test_list_indexed_videos_empty(self):
        """Test listing videos when none indexed."""
        engine = create_search_engine()
        engine.store._db = "mock"
        engine.store._mock_data = []

        videos = engine.list_indexed_videos()

        assert len(videos) == 0

    def test_list_indexed_videos(self):
        """Test listing indexed videos."""
        engine = create_search_engine()
        engine.store._db = "mock"
        engine.store._mock_data = [
            {"video_id": "vid1", "frame_id": "f1"},
            {"video_id": "vid2", "frame_id": "f2"},
        ]

        videos = engine.list_indexed_videos()

        assert "vid1" in videos
        assert "vid2" in videos

    def test_get_video_info(self):
        """Test getting video info."""
        engine = create_search_engine()
        engine.store._db = "mock"
        engine.store._mock_data = [
            {
                "video_id": "vid1",
                "frame_id": "f1",
                "timestamp": 0.0,
                "description": "",
                "vector": [],
            },
            {
                "video_id": "vid1",
                "frame_id": "f2",
                "timestamp": 2.0,
                "description": "",
                "vector": [],
            },
        ]

        info = engine.get_video_info("vid1")

        assert info["video_id"] == "vid1"
        assert info["frame_count"] == 2

    def test_delete_video(self):
        """Test deleting a video."""
        engine = create_search_engine()
        engine.store._db = "mock"
        engine.store._mock_data = [
            {"video_id": "vid1", "frame_id": "f1"},
            {"video_id": "vid1", "frame_id": "f2"},
        ]

        deleted = engine.delete_video("vid1")

        assert deleted == 2

    def test_get_stats(self):
        """Test getting engine statistics."""
        engine = create_search_engine()
        engine.store._db = "mock"
        engine.store._mock_data = [
            {"video_id": "vid1", "frame_id": "f1"},
            {"video_id": "vid2", "frame_id": "f2"},
        ]

        stats = engine.get_stats()

        assert stats["total_frames"] == 2
        assert stats["indexed_videos"] == 2
        assert "config" in stats

    @pytest.mark.asyncio
    async def test_search_empty_store(self):
        """Test searching empty store."""
        engine = create_search_engine()
        engine.store._db = "mock"
        engine.store._mock_data = []
        engine.store._table = None

        # Mock the embedder
        engine.embedder._model = "mock"
        engine.embedder._embed_model = "mock"

        results = await engine.search("test query")

        assert isinstance(results, SearchResults)
        assert len(results.results) == 0

    @pytest.mark.asyncio
    async def test_search_with_results(self):
        """Test searching with mock results."""
        engine = create_search_engine()
        engine.store._db = "mock"
        engine.store._mock_data = []
        engine.store._table = None
        engine.embedder._model = "mock"
        engine.embedder._embed_model = "mock"

        results = await engine.search(
            query="find a cat",
            top_k=5,
            threshold=0.5,
        )

        assert results.query == "find a cat"

    def test_search_sync(self):
        """Test synchronous search wrapper."""
        engine = create_search_engine()
        engine.store._db = "mock"
        engine.store._mock_data = []
        engine.store._table = None
        engine.embedder._model = "mock"
        engine.embedder._embed_model = "mock"

        results = engine.search_sync("test query")

        assert isinstance(results, SearchResults)


class TestCreateSearchEngine:
    """Tests for create_search_engine factory."""

    def test_create_engine_default(self):
        """Test creating engine with defaults."""
        engine = create_search_engine()
        assert isinstance(engine, VideoSearchEngine)

    def test_create_engine_with_config(self, sample_video_search_config):
        """Test creating engine with config."""
        engine = create_search_engine(sample_video_search_config)
        assert engine.config == sample_video_search_config


class TestSearchParameters:
    """Tests for search parameter handling."""

    @pytest.mark.asyncio
    async def test_search_top_k(self):
        """Test top_k parameter."""
        engine = create_search_engine()
        engine.store._db = "mock"
        engine.store._mock_data = []
        engine.store._table = None
        engine.embedder._model = "mock"
        engine.embedder._embed_model = "mock"

        results = await engine.search("query", top_k=10)

        assert results is not None

    @pytest.mark.asyncio
    async def test_search_threshold(self):
        """Test threshold parameter."""
        engine = create_search_engine()
        engine.store._db = "mock"
        engine.store._mock_data = []
        engine.store._table = None
        engine.embedder._model = "mock"
        engine.embedder._embed_model = "mock"

        results = await engine.search("query", threshold=0.8)

        assert results is not None

    @pytest.mark.asyncio
    async def test_search_video_filter(self):
        """Test video_ids filter."""
        engine = create_search_engine()
        engine.store._db = "mock"
        engine.store._mock_data = []
        engine.store._table = None
        engine.embedder._model = "mock"
        engine.embedder._embed_model = "mock"

        results = await engine.search(
            "query",
            video_ids=["vid1", "vid2"],
        )

        assert results is not None


class TestProgressTracking:
    """Tests for progress tracking."""

    def test_progress_callback_called(self):
        """Test that progress callback is called."""
        engine = create_search_engine()

        progress_updates = []
        engine.on_progress(lambda p: progress_updates.append(p))

        # Manually trigger progress update
        from video_search.models import IndexingProgress, VideoStatus

        progress = IndexingProgress(
            video_id="test",
            status=VideoStatus.EXTRACTING,
        )
        engine._update_progress(progress)

        assert len(progress_updates) == 1
        assert progress_updates[0].video_id == "test"

    def test_progress_without_callback(self):
        """Test progress update without callback doesn't crash."""
        engine = create_search_engine()

        from video_search.models import IndexingProgress, VideoStatus

        progress = IndexingProgress(
            video_id="test",
            status=VideoStatus.EXTRACTING,
        )

        # Should not raise
        engine._update_progress(progress)

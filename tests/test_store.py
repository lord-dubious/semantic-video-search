"""Tests for vector store."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from video_search.models import (
    FrameEmbedding,
    SearchQuery,
    VectorStoreConfig,
)
from video_search.store import VectorStore, create_vector_store


class TestVectorStore:
    """Tests for VectorStore class."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        store = VectorStore()
        assert store.config.db_path == "./data/lancedb"
        assert store.config.table_name == "video_frames"

    def test_init_custom_config(self, sample_vector_store_config):
        """Test initialization with custom config."""
        store = VectorStore(config=sample_vector_store_config)
        assert store.config == sample_vector_store_config

    def test_add_embedding_mock_mode(self, sample_frame_embedding):
        """Test adding embedding in mock mode."""
        store = VectorStore()
        store._db = "mock"
        store._mock_data = []

        store.add_embedding(sample_frame_embedding)

        assert len(store._mock_data) == 1
        assert store._mock_data[0]["frame_id"] == sample_frame_embedding.frame_id

    def test_add_embeddings_batch(self, sample_frame_embedding):
        """Test adding multiple embeddings."""
        store = VectorStore()
        store._db = "mock"
        store._mock_data = []

        embeddings = [
            sample_frame_embedding,
            FrameEmbedding(
                frame_id="frame_2",
                video_id="video1",
                timestamp=4.0,
                description="Second frame",
                embedding=[0.2] * 768,
            ),
        ]

        store.add_embeddings(embeddings)

        assert len(store._mock_data) == 2

    def test_add_empty_embeddings(self):
        """Test adding empty list."""
        store = VectorStore()
        store._db = "mock"
        store._mock_data = []

        store.add_embeddings([])

        assert len(store._mock_data) == 0

    def test_search_mock_mode(self, sample_frame_embedding):
        """Test search in mock mode."""
        store = VectorStore()
        store._db = "mock"
        store._mock_data = []
        store._table = None

        query = SearchQuery(query="test", top_k=5)
        query_embedding = [0.1] * 768

        results = store.search(query_embedding, query)

        assert results.query == "test"
        assert len(results.results) == 0

    def test_get_frame_mock_mode(self, sample_frame_embedding):
        """Test getting a frame in mock mode."""
        store = VectorStore()
        store._db = "mock"
        store._mock_data = [
            {
                "frame_id": sample_frame_embedding.frame_id,
                "video_id": sample_frame_embedding.video_id,
                "timestamp": sample_frame_embedding.timestamp,
                "description": sample_frame_embedding.description,
                "vector": sample_frame_embedding.embedding,
            }
        ]

        frame = store.get_frame(sample_frame_embedding.frame_id)

        assert frame is not None
        assert frame.frame_id == sample_frame_embedding.frame_id

    def test_get_frame_not_found(self):
        """Test getting non-existent frame."""
        store = VectorStore()
        store._db = "mock"
        store._mock_data = []

        frame = store.get_frame("nonexistent")

        assert frame is None

    def test_get_video_frames_mock_mode(self, sample_frame_embedding):
        """Test getting all frames for a video."""
        store = VectorStore()
        store._db = "mock"
        store._mock_data = [
            {
                "frame_id": f"frame_{i}",
                "video_id": "video1",
                "timestamp": float(i * 2),
                "description": f"Frame {i}",
                "vector": [0.1] * 768,
            }
            for i in range(5)
        ]

        frames = store.get_video_frames("video1")

        assert len(frames) == 5

    def test_get_video_frames_empty(self):
        """Test getting frames for non-existent video."""
        store = VectorStore()
        store._db = "mock"
        store._mock_data = []

        frames = store.get_video_frames("nonexistent")

        assert len(frames) == 0

    def test_delete_video_mock_mode(self):
        """Test deleting a video in mock mode."""
        store = VectorStore()
        store._db = "mock"
        store._mock_data = [
            {
                "frame_id": "f1",
                "video_id": "video1",
                "timestamp": 0,
                "description": "",
                "vector": [],
            },
            {
                "frame_id": "f2",
                "video_id": "video1",
                "timestamp": 2,
                "description": "",
                "vector": [],
            },
            {
                "frame_id": "f3",
                "video_id": "video2",
                "timestamp": 0,
                "description": "",
                "vector": [],
            },
        ]

        deleted = store.delete_video("video1")

        assert deleted == 2
        assert len(store._mock_data) == 1

    def test_delete_video_not_found(self):
        """Test deleting non-existent video."""
        store = VectorStore()
        store._db = "mock"
        store._mock_data = []

        deleted = store.delete_video("nonexistent")

        assert deleted == 0

    def test_count_mock_mode(self):
        """Test counting frames in mock mode."""
        store = VectorStore()
        store._db = "mock"
        store._mock_data = [{"frame_id": f"f{i}"} for i in range(10)]

        count = store.count()

        assert count == 10

    def test_count_empty(self):
        """Test counting empty store."""
        store = VectorStore()
        store._db = "mock"
        store._mock_data = []

        count = store.count()

        assert count == 0

    def test_list_videos_mock_mode(self):
        """Test listing videos in mock mode."""
        store = VectorStore()
        store._db = "mock"
        store._mock_data = [
            {"video_id": "video1"},
            {"video_id": "video1"},
            {"video_id": "video2"},
            {"video_id": "video3"},
        ]

        videos = store.list_videos()

        assert len(videos) == 3
        assert "video1" in videos
        assert "video2" in videos
        assert "video3" in videos

    def test_list_videos_empty(self):
        """Test listing videos when empty."""
        store = VectorStore()
        store._db = "mock"
        store._mock_data = []

        videos = store.list_videos()

        assert len(videos) == 0

    def test_clear_mock_mode(self):
        """Test clearing store in mock mode."""
        store = VectorStore()
        store._db = "mock"
        store._mock_data = [{"frame_id": f"f{i}"} for i in range(10)]

        store.clear()

        assert len(store._mock_data) == 0


class TestCreateVectorStore:
    """Tests for create_vector_store factory."""

    def test_create_store_default(self):
        """Test creating store with defaults."""
        store = create_vector_store()
        assert isinstance(store, VectorStore)

    def test_create_store_with_config(self, sample_vector_store_config):
        """Test creating store with config."""
        store = create_vector_store(config=sample_vector_store_config)
        assert store.config == sample_vector_store_config


class TestVectorStoreQueries:
    """Tests for query functionality."""

    def test_search_with_video_filter(self):
        """Test search with video ID filter."""
        store = VectorStore()
        store._db = "mock"
        store._mock_data = []
        store._table = None

        query = SearchQuery(
            query="test",
            top_k=5,
            video_ids=["video1", "video2"],
        )

        results = store.search([0.1] * 768, query)

        assert results.query == "test"

    def test_search_with_threshold(self):
        """Test search with similarity threshold."""
        store = VectorStore()
        store._db = "mock"
        store._mock_data = []
        store._table = None

        query = SearchQuery(
            query="test",
            top_k=10,
            threshold=0.8,
        )

        results = store.search([0.1] * 768, query)

        assert results.query == "test"

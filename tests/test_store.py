"""Tests for vector store implementations."""

from __future__ import annotations

import builtins

import pytest

from video_search.models import FrameEmbedding, SearchQuery
from video_search.store import (
    InMemoryVectorStore,
    VectorStore,
    VectorStoreDependencyError,
    create_vector_store,
)


class TestVectorStore:
    """Tests for the LanceDB-backed vector store."""

    def test_init_default_config(self):
        store = VectorStore()
        assert store.config.db_path == "./data/lancedb"
        assert store.config.table_name == "video_frames"

    def test_init_custom_config(self, sample_vector_store_config):
        store = VectorStore(config=sample_vector_store_config)
        assert store.config == sample_vector_store_config

    def test_missing_lancedb_raises_explicit_error(self, monkeypatch):
        real_import = builtins.__import__

        def import_without_lancedb(name, *args, **kwargs):
            if name == "lancedb":
                raise ImportError("lancedb unavailable")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", import_without_lancedb)

        with pytest.raises(VectorStoreDependencyError, match="LanceDB is required"):
            VectorStore().count()


class TestInMemoryVectorStore:
    """Tests for the explicit in-memory test/demo store."""

    def test_add_embedding(self, sample_frame_embedding):
        store = InMemoryVectorStore()

        store.add_embedding(sample_frame_embedding)

        frame = store.get_frame(sample_frame_embedding.frame_id)

        assert store.count() == 1
        assert frame is not None
        assert frame.frame_id == sample_frame_embedding.frame_id
        assert frame.video_id == sample_frame_embedding.video_id
        assert frame.embedding == sample_frame_embedding.embedding

    def test_add_embeddings_batch(self, sample_frame_embedding):
        store = InMemoryVectorStore()
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

        assert store.count() == 2

    def test_add_empty_embeddings(self):
        store = InMemoryVectorStore()

        store.add_embeddings([])

        assert store.count() == 0

    def test_search_returns_empty_results_without_silent_lancedb_fallback(self):
        store = InMemoryVectorStore()
        query = SearchQuery(query="test", top_k=5)

        results = store.search([0.1] * 768, query)

        assert results.query == "test"
        assert results.results == []
        assert results.total_found == 0

    def test_get_frame_not_found(self):
        store = InMemoryVectorStore()

        frame = store.get_frame("nonexistent")

        assert frame is None

    def test_get_video_frames(self):
        store = InMemoryVectorStore()
        store.add_embeddings(
            [
                FrameEmbedding(
                    frame_id=f"frame_{i}",
                    video_id="video1",
                    timestamp=float(i * 2),
                    description=f"Frame {i}",
                    embedding=[0.1] * 768,
                )
                for i in range(5)
            ]
        )

        frames = store.get_video_frames("video1")

        assert len(frames) == 5

    def test_delete_video(self):
        store = InMemoryVectorStore()
        store.add_embeddings(
            [
                FrameEmbedding(
                    frame_id="f1",
                    video_id="video1",
                    timestamp=0,
                    description="",
                    embedding=[],
                ),
                FrameEmbedding(
                    frame_id="f2",
                    video_id="video1",
                    timestamp=2,
                    description="",
                    embedding=[],
                ),
                FrameEmbedding(
                    frame_id="f3",
                    video_id="video2",
                    timestamp=0,
                    description="",
                    embedding=[],
                ),
            ]
        )

        deleted = store.delete_video("video1")

        assert deleted == 2
        assert store.count() == 1
        assert store.get_video_frames("video1") == []

    def test_list_videos(self):
        store = InMemoryVectorStore()
        store.add_embeddings(
            [
                FrameEmbedding(
                    frame_id="f1", video_id="video1", timestamp=0, description="", embedding=[]
                ),
                FrameEmbedding(
                    frame_id="f2", video_id="video1", timestamp=1, description="", embedding=[]
                ),
                FrameEmbedding(
                    frame_id="f3", video_id="video2", timestamp=2, description="", embedding=[]
                ),
            ]
        )

        videos = store.list_videos()

        assert videos == ["video1", "video2"]

    def test_clear(self, sample_frame_embedding):
        store = InMemoryVectorStore()
        store.add_embedding(sample_frame_embedding)

        store.clear()

        assert store.count() == 0


class TestCreateVectorStore:
    """Tests for create_vector_store factory."""

    def test_create_store_default(self):
        store = create_vector_store()
        assert isinstance(store, VectorStore)

    def test_create_store_with_config(self, sample_vector_store_config):
        store = create_vector_store(config=sample_vector_store_config)
        assert store.config == sample_vector_store_config

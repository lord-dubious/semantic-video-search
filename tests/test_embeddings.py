"""Tests for embedding generation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

import pytest

from video_search.models import EmbeddingConfig, FrameData, FrameEmbedding
from video_search.embeddings import (
    EmbeddingGenerator,
    create_embedding_generator,
)


class TestEmbeddingGenerator:
    """Tests for EmbeddingGenerator class."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        generator = EmbeddingGenerator()
        assert generator.config.dimension == 768
        assert generator.config.model_name == "models/text-embedding-004"

    def test_init_custom_config(self, sample_embedding_config):
        """Test initialization with custom config."""
        generator = EmbeddingGenerator(config=sample_embedding_config)
        assert generator.config == sample_embedding_config

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        generator = EmbeddingGenerator(api_key="custom-key")
        assert generator.api_key == "custom-key"

    @pytest.mark.asyncio
    async def test_describe_frame_mock_mode(self, tmp_path):
        """Test frame description in mock mode."""
        # Create a test image file
        frame_path = tmp_path / "test_frame.jpg"
        frame_path.touch()

        generator = EmbeddingGenerator()
        generator._model = "mock"
        generator._embed_model = "mock"

        description = await generator.describe_frame(frame_path)

        assert "Mock description" in description

    @pytest.mark.asyncio
    async def test_describe_frame_not_found(self):
        """Test error when frame not found."""
        generator = EmbeddingGenerator()

        with pytest.raises(FileNotFoundError):
            await generator.describe_frame("/nonexistent/frame.jpg")

    @pytest.mark.asyncio
    async def test_embed_text_mock_mode(self):
        """Test text embedding in mock mode."""
        generator = EmbeddingGenerator()
        generator._model = "mock"
        generator._embed_model = "mock"

        embedding = await generator.embed_text("Test text for embedding")

        assert isinstance(embedding, list)
        assert len(embedding) == 768

    @pytest.mark.asyncio
    async def test_embed_query_mock_mode(self):
        """Test query embedding in mock mode."""
        generator = EmbeddingGenerator()
        generator._model = "mock"
        generator._embed_model = "mock"

        embedding = await generator.embed_query("find a cat")

        assert isinstance(embedding, list)
        assert len(embedding) == 768

    @pytest.mark.asyncio
    async def test_embed_text_deterministic(self):
        """Test that same text produces same embedding."""
        generator = EmbeddingGenerator()
        generator._model = "mock"
        generator._embed_model = "mock"

        embedding1 = await generator.embed_text("same text")
        embedding2 = await generator.embed_text("same text")

        assert embedding1 == embedding2

    @pytest.mark.asyncio
    async def test_embed_different_texts(self):
        """Test that different texts produce different embeddings."""
        generator = EmbeddingGenerator()
        generator._model = "mock"
        generator._embed_model = "mock"

        embedding1 = await generator.embed_text("text one")
        embedding2 = await generator.embed_text("text two")

        assert embedding1 != embedding2

    @pytest.mark.asyncio
    async def test_process_frame(self, sample_frame_data, tmp_path):
        """Test processing a single frame."""
        # Create test frame file
        frame_path = tmp_path / "frame.jpg"
        frame_path.touch()
        sample_frame_data.filepath = str(frame_path)

        generator = EmbeddingGenerator()
        generator._model = "mock"
        generator._embed_model = "mock"

        result = await generator.process_frame(sample_frame_data)

        assert isinstance(result, FrameEmbedding)
        assert result.frame_id == sample_frame_data.frame_id
        assert len(result.embedding) == 768

    @pytest.mark.asyncio
    async def test_process_frames_batch(self, tmp_path):
        """Test processing multiple frames."""
        frames = []
        for i in range(5):
            frame_path = tmp_path / f"frame_{i}.jpg"
            frame_path.touch()
            frames.append(
                FrameData(
                    frame_id=f"frame_{i}",
                    video_id="test_video",
                    timestamp=float(i * 2),
                    frame_number=i,
                    filepath=str(frame_path),
                )
            )

        generator = EmbeddingGenerator()
        generator._model = "mock"
        generator._embed_model = "mock"

        results = []
        async for embedding in generator.process_frames(frames):
            results.append(embedding)

        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_verify_frame_mock_mode(self, tmp_path):
        """Test frame verification in mock mode."""
        frame_path = tmp_path / "frame.jpg"
        frame_path.touch()

        generator = EmbeddingGenerator()
        generator._model = "mock"
        generator._embed_model = "mock"

        is_match, confidence, explanation = await generator.verify_frame(
            frame_path,
            "find a cat",
            "A cat sitting on a couch",
        )

        assert isinstance(is_match, bool)
        assert 0 <= confidence <= 1
        assert isinstance(explanation, str)

    @pytest.mark.asyncio
    async def test_verify_frame_not_found(self):
        """Test verification with missing frame."""
        generator = EmbeddingGenerator()

        is_match, confidence, explanation = await generator.verify_frame(
            "/nonexistent/frame.jpg",
            "query",
            "description",
        )

        assert is_match == False
        assert confidence == 0.0
        assert "not found" in explanation


class TestCreateEmbeddingGenerator:
    """Tests for create_embedding_generator factory."""

    def test_create_generator_default(self):
        """Test creating generator with defaults."""
        generator = create_embedding_generator()
        assert isinstance(generator, EmbeddingGenerator)

    def test_create_generator_with_config(self, sample_embedding_config):
        """Test creating generator with config."""
        generator = create_embedding_generator(config=sample_embedding_config)
        assert generator.config == sample_embedding_config

    def test_create_generator_with_api_key(self):
        """Test creating generator with API key."""
        generator = create_embedding_generator(api_key="test-key")
        assert generator.api_key == "test-key"


class TestEmbeddingDimensions:
    """Tests for embedding dimension handling."""

    @pytest.mark.asyncio
    async def test_custom_dimension(self):
        """Test custom embedding dimension."""
        config = EmbeddingConfig(dimension=512)
        generator = EmbeddingGenerator(config=config)
        generator._model = "mock"
        generator._embed_model = "mock"

        embedding = await generator.embed_text("test")

        assert len(embedding) == 512

    @pytest.mark.asyncio
    async def test_batch_size_respected(self, tmp_path):
        """Test batch size is respected during processing."""
        config = EmbeddingConfig(batch_size=2)
        generator = EmbeddingGenerator(config=config)
        generator._model = "mock"
        generator._embed_model = "mock"

        frames = []
        for i in range(5):
            frame_path = tmp_path / f"frame_{i}.jpg"
            frame_path.touch()
            frames.append(
                FrameData(
                    frame_id=f"frame_{i}",
                    video_id="test",
                    timestamp=float(i),
                    frame_number=i,
                    filepath=str(frame_path),
                )
            )

        results = []
        async for embedding in generator.process_frames(frames):
            results.append(embedding)

        assert len(results) == 5

"""Embedding generation using Google Gemini."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from video_search.models import (
    EmbeddingConfig,
    FrameData,
    FrameEmbedding,
)

load_dotenv()


class EmbeddingGenerator:
    """Generate embeddings and descriptions for video frames using Gemini."""

    def __init__(self, config: EmbeddingConfig | None = None, api_key: str | None = None):
        """Initialize the embedding generator.

        Args:
            config: Embedding configuration.
            api_key: Gemini API key (defaults to GEMINI_API_KEY env var).
        """
        self.config = config or EmbeddingConfig()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self._model: Any | None = None
        self._embed_model: Any | None = None

    def _init_models(self) -> None:
        """Initialize Gemini models lazily."""
        if self._model is not None:
            return

        try:
            import google.generativeai as genai

            if self.api_key:
                genai.configure(api_key=self.api_key)

            self._model = genai.GenerativeModel("gemini-2.5-flash")
            self._embed_model = "models/text-embedding-004"
        except ImportError:
            # Mock mode for testing
            self._model = "mock"
            self._embed_model = "mock"

    async def describe_frame(self, frame_path: str | Path) -> str:
        """Generate a description for a video frame.

        Args:
            frame_path: Path to the frame image.

        Returns:
            Text description of the frame.
        """
        self._init_models()
        frame_path = Path(frame_path)

        if not frame_path.exists():
            raise FileNotFoundError(f"Frame not found: {frame_path}")

        if self._model == "mock":
            return f"Mock description for frame: {frame_path.name}"

        try:
            from PIL import Image

            # Load and prepare image
            image = Image.open(frame_path)
            model = self._model
            if model is None:
                raise RuntimeError("Gemini model is not initialized")

            # Generate description
            prompt = """Describe this video frame in detail. Include:
- Main subjects and actions
- Setting and environment
- Notable objects or text visible
- Any motion or activity apparent
Keep the description concise but comprehensive (2-3 sentences)."""

            response = await asyncio.to_thread(
                model.generate_content,
                [prompt, image],
            )

            return response.text.strip()

        except Exception as e:
            # Return basic description on error
            return f"Frame from video (description unavailable: {e})"

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        self._init_models()

        if self._embed_model == "mock":
            # Return mock embedding for testing
            import hashlib

            hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
            return [(hash_val >> i) % 1000 / 1000.0 for i in range(self.config.dimension)]

        try:
            import google.generativeai as genai

            result = await asyncio.to_thread(
                genai.embed_content,
                model=self._embed_model,
                content=text,
                task_type="retrieval_document",
            )

            return result["embedding"]

        except Exception:
            # Return zero vector on error
            return [0.0] * self.config.dimension

    async def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a search query.

        Args:
            query: Search query text.

        Returns:
            Embedding vector.
        """
        self._init_models()

        if self._embed_model == "mock":
            import hashlib

            hash_val = int(hashlib.md5(query.encode()).hexdigest(), 16)
            return [(hash_val >> i) % 1000 / 1000.0 for i in range(self.config.dimension)]

        try:
            import google.generativeai as genai

            result = await asyncio.to_thread(
                genai.embed_content,
                model=self._embed_model,
                content=query,
                task_type="retrieval_query",
            )

            return result["embedding"]

        except Exception:
            return [0.0] * self.config.dimension

    async def process_frame(self, frame: FrameData) -> FrameEmbedding:
        """Process a single frame: describe and embed.

        Args:
            frame: Frame data.

        Returns:
            FrameEmbedding with description and vector.
        """
        # Generate description
        description = await self.describe_frame(frame.filepath)

        # Generate embedding from description
        embedding = await self.embed_text(description)

        return FrameEmbedding(
            frame_id=frame.frame_id,
            video_id=frame.video_id,
            timestamp=frame.timestamp,
            description=description,
            embedding=embedding,
        )

    async def process_frames(
        self,
        frames: list[FrameData],
    ) -> AsyncGenerator[FrameEmbedding, None]:
        """Process multiple frames with batching.

        Args:
            frames: List of frames to process.

        Yields:
            FrameEmbedding for each processed frame.
        """
        for i in range(0, len(frames), self.config.batch_size):
            batch = frames[i : i + self.config.batch_size]
            tasks = [self.process_frame(frame) for frame in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, BaseException):
                    # Skip failed frames
                    continue
                yield result

    async def verify_frame(
        self,
        frame_path: str | Path,
        query: str,
        description: str,
    ) -> tuple[bool, float, str]:
        """Verify if a frame matches a search query.

        Args:
            frame_path: Path to frame image.
            query: Original search query.
            description: Frame description.

        Returns:
            Tuple of (is_match, confidence, explanation).
        """
        self._init_models()
        frame_path = Path(frame_path)

        if not frame_path.exists():
            return False, 0.0, "Frame not found"

        if self._model == "mock":
            return True, 0.8, "Mock verification"

        try:
            from PIL import Image

            image = Image.open(frame_path)
            model = self._model
            if model is None:
                raise RuntimeError("Gemini model is not initialized")

            prompt = f"""Verify if this video frame matches the search query.

Query: "{query}"
Frame description: "{description}"

Analyze the image and determine:
1. Does this frame contain what the user is looking for?
2. How confident are you (0-100%)?
3. Brief explanation.

Respond in this exact format:
MATCH: yes/no
CONFIDENCE: [0-100]
EXPLANATION: [brief explanation]"""

            response = await asyncio.to_thread(
                model.generate_content,
                [prompt, image],
            )

            text = response.text.strip()

            # Parse response
            is_match = "MATCH: yes" in text.lower()
            confidence = 0.5

            for line in text.split("\n"):
                if "CONFIDENCE:" in line.upper():
                    try:
                        conf_str = line.split(":")[-1].strip().replace("%", "")
                        confidence = float(conf_str) / 100.0
                    except ValueError:
                        pass

            explanation = ""
            for line in text.split("\n"):
                if "EXPLANATION:" in line.upper():
                    explanation = line.split(":", 1)[-1].strip()
                    break

            return is_match, min(1.0, max(0.0, confidence)), explanation

        except Exception as e:
            return False, 0.0, f"Verification failed: {e}"


def create_embedding_generator(
    config: EmbeddingConfig | None = None,
    api_key: str | None = None,
) -> EmbeddingGenerator:
    """Create an EmbeddingGenerator instance.

    Args:
        config: Embedding configuration.
        api_key: Gemini API key.

    Returns:
        EmbeddingGenerator instance.
    """
    return EmbeddingGenerator(config, api_key)

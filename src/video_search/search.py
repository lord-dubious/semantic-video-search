"""Main search functionality for semantic video search."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from video_search.embeddings import create_embedding_generator
from video_search.extractor import create_extractor
from video_search.models import (
    FrameEmbedding,
    IndexingProgress,
    SearchQuery,
    SearchResults,
    VideoMetadata,
    VideoSearchConfig,
    VideoStatus,
)
from video_search.store import create_vector_store


class SearchStore(Protocol):
    """Store interface used by the search engine."""

    def add_embeddings(self, embeddings: list[FrameEmbedding]) -> None: ...

    def search(self, query_embedding: list[float], query: SearchQuery) -> SearchResults: ...

    def get_frame(self, frame_id: str) -> FrameEmbedding | None: ...

    def get_video_frames(self, video_id: str) -> list[FrameEmbedding]: ...

    def list_videos(self) -> list[str]: ...

    def delete_video(self, video_id: str) -> int: ...

    def count(self) -> int: ...


class VideoSearchEngine:
    """Main engine for semantic video search."""

    def __init__(self, config: VideoSearchConfig | None = None, store: SearchStore | None = None):
        """Initialize the search engine.

        Args:
            config: Search engine configuration.
            store: Optional vector store implementation for tests or demos.
        """
        self.config = config or VideoSearchConfig()

        # Initialize components
        self.extractor = create_extractor(
            self.config.extraction,
            self.config.frames_dir,
        )
        self.embedder = create_embedding_generator(self.config.embedding)
        self.store = store or create_vector_store(self.config.vector_store)

        # Progress callback
        self._progress_callback: Callable[[IndexingProgress], None] | None = None

    def on_progress(self, callback: Callable[[IndexingProgress], None]) -> None:
        """Register a progress callback.

        Args:
            callback: Function to call with progress updates.
        """
        self._progress_callback = callback

    def _update_progress(self, progress: IndexingProgress) -> None:
        """Update progress and notify callback.

        Args:
            progress: Current progress.
        """
        if self._progress_callback:
            self._progress_callback(progress)

    async def index_video(
        self,
        video_path: str | Path,
        verify: bool = False,
    ) -> VideoMetadata:
        """Index a video for searching.

        Args:
            video_path: Path to video file.
            verify: Whether to verify extracted frames.

        Returns:
            VideoMetadata for the indexed video.
        """
        video_path = Path(video_path)

        # Get metadata
        metadata = self.extractor.get_metadata(video_path)
        expected_frames = self.extractor.get_expected_frames(video_path)

        # Initialize progress
        progress = IndexingProgress(
            video_id=metadata.video_id,
            total_frames=expected_frames,
            status=VideoStatus.EXTRACTING,
        )
        self._update_progress(progress)

        # Extract frames
        frames = self.extractor.extract(video_path)
        progress.extracted_frames = len(frames)
        progress.status = VideoStatus.EMBEDDING
        self._update_progress(progress)

        # Process frames with embeddings
        embeddings: list[FrameEmbedding] = []
        async for embedding in self.embedder.process_frames(frames):
            embeddings.append(embedding)
            progress.embedded_frames = len(embeddings)
            self._update_progress(progress)

        # Store embeddings
        self.store.add_embeddings(embeddings)

        # Update metadata status
        metadata.status = VideoStatus.INDEXED
        progress.status = VideoStatus.INDEXED
        self._update_progress(progress)

        return metadata

    def index_video_sync(self, video_path: str | Path) -> VideoMetadata:
        """Synchronous wrapper for index_video.

        Args:
            video_path: Path to video file.

        Returns:
            VideoMetadata.
        """
        return asyncio.run(self.index_video(video_path))

    async def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.5,
        video_ids: list[str] | None = None,
        verify: bool = False,
    ) -> SearchResults:
        """Search for video moments matching a query.

        Args:
            query: Natural language search query.
            top_k: Number of results to return.
            threshold: Similarity threshold.
            video_ids: Filter to specific videos.
            verify: Whether to verify results with LLM.

        Returns:
            SearchResults with matching frames.
        """
        # Create search query
        search_query = SearchQuery(
            query=query,
            top_k=top_k,
            threshold=threshold,
            video_ids=video_ids,
        )

        # Generate query embedding
        query_embedding = await self.embedder.embed_query(query)

        # Search vector store
        results = self.store.search(query_embedding, search_query)

        # Optionally verify results with LLM
        if verify and results.results:
            verified_results = []
            for result in results.results:
                frame = self.store.get_frame(result.frame_id)
                if frame:
                    is_match, confidence, explanation = await self.embedder.verify_frame(
                        Path(self.config.frames_dir) / result.video_id / f"{result.frame_id}.jpg",
                        query,
                        result.description,
                    )
                    result.verified = is_match
                    result.verification_note = explanation
                    if is_match:
                        verified_results.append(result)

            results.results = verified_results
            results.total_found = len(verified_results)

        return results

    def search_sync(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.5,
        video_ids: list[str] | None = None,
        verify: bool = False,
    ) -> SearchResults:
        """Synchronous wrapper for search.

        Args:
            query: Search query.
            top_k: Number of results.
            threshold: Similarity threshold.
            video_ids: Filter to videos.
            verify: Verify results.

        Returns:
            SearchResults.
        """
        return asyncio.run(self.search(query, top_k, threshold, video_ids, verify))

    def get_video_info(self, video_id: str) -> dict:
        """Get information about an indexed video.

        Args:
            video_id: Video identifier.

        Returns:
            Dictionary with video info.
        """
        frames = self.store.get_video_frames(video_id)
        return {
            "video_id": video_id,
            "frame_count": len(frames),
            "timestamps": sorted([f.timestamp for f in frames]),
        }

    def list_indexed_videos(self) -> list[str]:
        """List all indexed videos.

        Returns:
            List of video IDs.
        """
        return self.store.list_videos()

    def delete_video(self, video_id: str) -> int:
        """Delete a video from the index.

        Args:
            video_id: Video identifier.

        Returns:
            Number of frames deleted.
        """
        return self.store.delete_video(video_id)

    def get_stats(self) -> dict:
        """Get search engine statistics.

        Returns:
            Dictionary with stats.
        """
        return {
            "total_frames": self.store.count(),
            "indexed_videos": len(self.store.list_videos()),
            "config": {
                "frame_interval": self.config.extraction.interval_seconds,
                "embedding_model": self.config.embedding.model_name,
                "db_path": self.config.vector_store.db_path,
            },
        }


def create_search_engine(
    config: VideoSearchConfig | None = None,
    store: SearchStore | None = None,
) -> VideoSearchEngine:
    """Create a VideoSearchEngine instance.

    Args:
        config: Configuration.
        store: Optional vector store implementation.

    Returns:
        VideoSearchEngine instance.
    """
    return VideoSearchEngine(config, store)

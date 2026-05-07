"""Vector store using LanceDB for semantic search."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from video_search.models import (
    FrameEmbedding,
    SearchQuery,
    SearchResult,
    SearchResults,
    VectorStoreConfig,
)


class VectorStoreDependencyError(RuntimeError):
    """Raised when the LanceDB vector store cannot be initialized."""


RowData = dict[str, Any]


def _embedding_to_row(embedding: FrameEmbedding) -> RowData:
    return {
        "frame_id": embedding.frame_id,
        "video_id": embedding.video_id,
        "timestamp": embedding.timestamp,
        "description": embedding.description,
        "vector": embedding.embedding,
    }


def _row_to_embedding(row: RowData) -> FrameEmbedding:
    return FrameEmbedding(
        frame_id=row["frame_id"],
        video_id=row["video_id"],
        timestamp=row["timestamp"],
        description=row["description"],
        embedding=list(row["vector"]),
    )


class VectorStore:
    """LanceDB-based vector store for video frame embeddings."""

    def __init__(self, config: VectorStoreConfig | None = None):
        """Initialize the vector store.

        Args:
            config: Vector store configuration.
        """
        self.config = config or VectorStoreConfig()
        self._db = None
        self._table = None

    @property
    def is_ready(self) -> bool:
        """Whether the backing LanceDB table has been opened or created."""
        self._init_db()
        return self._table is not None

    def _init_db(self) -> None:
        """Initialize LanceDB connection lazily."""
        if self._db is not None:
            return

        try:
            lancedb = importlib.import_module("lancedb")
        except ImportError as exc:
            raise VectorStoreDependencyError(
                "LanceDB is required for VectorStore. Install the project dependencies or use "
                "InMemoryVectorStore explicitly for tests and demos."
            ) from exc

        db_path = Path(self.config.db_path)
        db_path.mkdir(parents=True, exist_ok=True)

        self._db = lancedb.connect(str(db_path))

        if self.config.table_name in self._db.table_names():
            self._table = self._db.open_table(self.config.table_name)
        else:
            self._table = None

    def _create_table_if_needed(self, embedding_dim: int = 768) -> None:
        """Create the table if it doesn't exist.

        Args:
            embedding_dim: Dimension of embeddings.
        """
        self._init_db()

        if self._table is None:
            try:
                pa = importlib.import_module("pyarrow")
            except ImportError as exc:
                raise VectorStoreDependencyError(
                    "PyArrow is required to create the LanceDB frame table. Install the project "
                    "dependencies before indexing videos."
                ) from exc

            # Define schema
            schema = pa.schema(
                [
                    pa.field("frame_id", pa.string()),
                    pa.field("video_id", pa.string()),
                    pa.field("timestamp", pa.float64()),
                    pa.field("description", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), embedding_dim)),
                ]
            )

            db = self._db
            assert db is not None
            self._table = db.create_table(
                self.config.table_name,
                schema=schema,
            )

    def add_embedding(self, embedding: FrameEmbedding) -> None:
        """Add a single embedding to the store.

        Args:
            embedding: Frame embedding to add.
        """
        self._init_db()
        self._create_table_if_needed(len(embedding.embedding))
        table = self._table
        assert table is not None

        table.add([_embedding_to_row(embedding)])

    def add_embeddings(self, embeddings: list[FrameEmbedding]) -> None:
        """Add multiple embeddings to the store.

        Args:
            embeddings: List of frame embeddings to add.
        """
        if not embeddings:
            return

        self._init_db()
        self._create_table_if_needed(len(embeddings[0].embedding))
        table = self._table
        assert table is not None

        table.add([_embedding_to_row(embedding) for embedding in embeddings])

    def search(
        self,
        query_embedding: list[float],
        query: SearchQuery,
    ) -> SearchResults:
        """Search for similar frames.

        Args:
            query_embedding: Query vector.
            query: Search query parameters.

        Returns:
            SearchResults with matching frames.
        """
        import time

        start_time = time.time()

        self._init_db()

        if self._table is None:
            return SearchResults(
                query=query.query,
                results=[],
                total_found=0,
                search_time_ms=(time.time() - start_time) * 1000,
            )

        # Build search query
        search = self._table.search(query_embedding).limit(query.top_k)

        # Filter by video IDs if specified
        if query.video_ids:
            video_filter = " OR ".join([f"video_id = '{vid}'" for vid in query.video_ids])
            search = search.where(video_filter)

        # Execute search
        results_df = search.to_pandas()

        # Convert to SearchResult objects
        results = []
        for _, row in results_df.iterrows():
            # LanceDB returns _distance, convert to similarity
            distance = row.get("_distance", 0)
            similarity = 1.0 / (1.0 + distance)

            if similarity >= query.threshold:
                results.append(
                    SearchResult(
                        frame_id=row["frame_id"],
                        video_id=row["video_id"],
                        timestamp=row["timestamp"],
                        description=row["description"],
                        similarity=similarity,
                    )
                )

        search_time = (time.time() - start_time) * 1000

        return SearchResults(
            query=query.query,
            results=results,
            total_found=len(results),
            search_time_ms=search_time,
        )

    def get_frame(self, frame_id: str) -> FrameEmbedding | None:
        """Get a specific frame embedding.

        Args:
            frame_id: Frame identifier.

        Returns:
            FrameEmbedding or None if not found.
        """
        self._init_db()

        if self._table is None:
            return None

        results = self._table.search().where(f"frame_id = '{frame_id}'").limit(1).to_pandas()

        if results.empty:
            return None

        row = results.iloc[0]
        return FrameEmbedding(
            frame_id=row["frame_id"],
            video_id=row["video_id"],
            timestamp=row["timestamp"],
            description=row["description"],
            embedding=list(row["vector"]),
        )

    def get_video_frames(self, video_id: str) -> list[FrameEmbedding]:
        """Get all frames for a video.

        Args:
            video_id: Video identifier.

        Returns:
            List of FrameEmbeddings for the video.
        """
        self._init_db()

        if self._table is None:
            return []

        results = self._table.search().where(f"video_id = '{video_id}'").to_pandas()

        return [
            FrameEmbedding(
                frame_id=row["frame_id"],
                video_id=row["video_id"],
                timestamp=row["timestamp"],
                description=row["description"],
                embedding=list(row["vector"]),
            )
            for _, row in results.iterrows()
        ]

    def delete_video(self, video_id: str) -> int:
        """Delete all frames for a video.

        Args:
            video_id: Video identifier.

        Returns:
            Number of frames deleted.
        """
        self._init_db()

        if self._table is None:
            return 0

        # Count before deletion
        count_before = len(self._table.search().where(f"video_id = '{video_id}'").to_pandas())

        # Delete
        self._table.delete(f"video_id = '{video_id}'")

        return count_before

    def count(self) -> int:
        """Get total number of frames indexed.

        Returns:
            Total frame count.
        """
        self._init_db()

        if self._table is None:
            return 0

        return self._table.count_rows()

    def list_videos(self) -> list[str]:
        """List all indexed video IDs.

        Returns:
            List of video IDs.
        """
        self._init_db()

        if self._table is None:
            return []

        df = self._table.to_pandas()
        return list(df["video_id"].unique())

    def clear(self) -> None:
        """Clear all data from the store."""
        self._init_db()

        if self._table is not None:
            db = self._db
            assert db is not None
            db.drop_table(self.config.table_name)
            self._table = None


class InMemoryVectorStore:
    """Explicit in-memory vector store for tests and demos."""

    def __init__(self, config: VectorStoreConfig | None = None):
        self.config = config or VectorStoreConfig()
        self._rows: list[RowData] = []

    @property
    def is_ready(self) -> bool:
        return True

    def add_embedding(self, embedding: FrameEmbedding) -> None:
        self._rows.append(_embedding_to_row(embedding))

    def add_embeddings(self, embeddings: list[FrameEmbedding]) -> None:
        self._rows.extend(_embedding_to_row(embedding) for embedding in embeddings)

    def search(self, query_embedding: list[float], query: SearchQuery) -> SearchResults:
        import time

        start_time = time.time()
        return SearchResults(
            query=query.query,
            results=[],
            total_found=0,
            search_time_ms=(time.time() - start_time) * 1000,
        )

    def get_frame(self, frame_id: str) -> FrameEmbedding | None:
        for row in self._rows:
            if row["frame_id"] == frame_id:
                return _row_to_embedding(row)
        return None

    def get_video_frames(self, video_id: str) -> list[FrameEmbedding]:
        return [_row_to_embedding(row) for row in self._rows if row["video_id"] == video_id]

    def delete_video(self, video_id: str) -> int:
        original_len = len(self._rows)
        self._rows = [row for row in self._rows if row["video_id"] != video_id]
        return original_len - len(self._rows)

    def count(self) -> int:
        return len(self._rows)

    def list_videos(self) -> list[str]:
        return sorted({row["video_id"] for row in self._rows})

    def clear(self) -> None:
        self._rows = []


def create_vector_store(config: VectorStoreConfig | None = None) -> VectorStore:
    """Create a VectorStore instance.

    Args:
        config: Vector store configuration.

    Returns:
        VectorStore instance.
    """
    return VectorStore(config)

"""Semantic Video Search - Multimodal RAG with LanceDB and Gemini AI."""

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
    VerificationRequest,
    VerificationResult,
)
from video_search.extractor import (
    VideoExtractor,
    create_extractor,
    extract_frames,
    get_video_metadata,
)
from video_search.embeddings import (
    EmbeddingGenerator,
    create_embedding_generator,
)
from video_search.store import (
    VectorStore,
    create_vector_store,
)
from video_search.search import (
    VideoSearchEngine,
    create_search_engine,
)

__version__ = "1.0.0"

__all__ = [
    # Models
    "EmbeddingConfig",
    "ExtractionConfig",
    "FrameData",
    "FrameEmbedding",
    "IndexingProgress",
    "SearchQuery",
    "SearchResult",
    "SearchResults",
    "VectorStoreConfig",
    "VideoMetadata",
    "VideoSearchConfig",
    "VideoStatus",
    "VerificationRequest",
    "VerificationResult",
    # Extractor
    "VideoExtractor",
    "create_extractor",
    "extract_frames",
    "get_video_metadata",
    # Embeddings
    "EmbeddingGenerator",
    "create_embedding_generator",
    # Store
    "VectorStore",
    "create_vector_store",
    # Search
    "VideoSearchEngine",
    "create_search_engine",
]

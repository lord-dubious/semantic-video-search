# Semantic Video Search

Indexes video frames with ffmpeg, describes them with Gemini, and stores searchable frame embeddings in LanceDB.

## Portfolio Review

- [Architecture](docs/ARCHITECTURE.md) - component boundaries, data flow, external dependencies, and degraded-mode behavior.
- [Demo Guide](docs/DEMO.md) - safe local walkthrough commands and recruiter-facing talking points.

## What Works

- Extracts frames from local videos at configurable intervals using ffmpeg
- Generates frame descriptions with Gemini 2.5 Flash
- Searches indexed moments with natural-language queries and vector similarity
- Stores frame embeddings in LanceDB without a separate vector database service
- Optionally asks Gemini to verify candidate results against the original frame
- Provides a Typer/Rich CLI for indexing, search, listing, deletion, and stats

## Current Limits

- Requires a valid `GEMINI_API_KEY`; local tests use explicit in-memory fakes instead of silently switching production code into mock mode.
- Search quality depends on frame interval, caption quality, and the embedding model. This is a reference implementation, not a benchmarked video retrieval product.
- LanceDB is required for the default `VectorStore`. Use `InMemoryVectorStore` only for tests, demos, or examples where persistence is not expected.

## Features

- **Frame Extraction**: Extract keyframes from videos at configurable intervals using ffmpeg
- **AI-Powered Descriptions**: Generate rich textual descriptions of frames using Gemini 2.5 Flash
- **Semantic Search**: Find video moments using natural language queries
- **Vector Store**: Efficient storage and retrieval with LanceDB (embedded, no server needed)
- **Verification**: Optional LLM verification of search results for accuracy
- **CLI Interface**: Full-featured command line tool for all operations

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Video Ingestion                              │
├─────────────┬───────────────┬────────────────┬─────────────────────┤
│   Video     │    ffmpeg     │    Gemini      │     LanceDB         │
│   Input     │   Extract     │   Describe     │   Store             │
│   (MP4)  ──▶│   Frames   ──▶│   Frames    ──▶│   Embeddings        │
└─────────────┴───────────────┴────────────────┴─────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         Semantic Search                              │
├─────────────┬───────────────┬────────────────┬─────────────────────┤
│   Query     │    Gemini     │    LanceDB     │    Results          │
│   "find..." │   Embed    ──▶│   Search    ──▶│   + Verify          │
└─────────────┴───────────────┴────────────────┴─────────────────────┘
```

## Installation

### Prerequisites

- Python 3.11+
- ffmpeg installed and in PATH
- Google Gemini API key

### Quick Start

```bash
# Clone the repository
git clone https://github.com/lord-dubious/semantic-video-search.git
cd semantic-video-search

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Set up environment variables
cp .env.example .env
# Edit .env with your API key

# Index a video
video-search index /path/to/video.mp4

# Search for moments
video-search search "person walking in the park"
```

## Usage

### CLI Commands

```bash
# Index a video file
video-search index video.mp4 --interval 2.0

# Search for video moments
video-search search "chef cutting onions" --top 5 --verify

# List all indexed videos
video-search list-videos

# Get statistics
video-search stats

# Delete a video from index
video-search delete <video-id> --force

# Run demo
video-search demo
```

### Python API

```python
import asyncio
from video_search import VideoSearchEngine, VideoSearchConfig

async def main():
    # Create search engine
    config = VideoSearchConfig(
        frames_dir="./data/frames",
    )
    engine = VideoSearchEngine(config)
    
    # Index a video
    metadata = await engine.index_video("/path/to/video.mp4")
    print(f"Indexed: {metadata.video_id}")
    
    # Search for moments
    results = await engine.search(
        query="person walking in park",
        top_k=5,
        threshold=0.5,
        verify=True,  # Optional LLM verification
    )
    
    for result in results.results:
        print(f"Found at {result.timestamp}s: {result.description}")
        print(f"Similarity: {result.similarity:.2f}")

asyncio.run(main())
```

### Configuration

```python
from video_search import (
    VideoSearchConfig,
    ExtractionConfig,
    EmbeddingConfig,
    VectorStoreConfig,
)

config = VideoSearchConfig(
    # Frame extraction settings
    extraction=ExtractionConfig(
        interval_seconds=2.0,  # Extract frame every 2 seconds
        format="jpg",
        quality=85,
        max_width=1280,  # Resize for faster processing
    ),
    
    # Embedding settings
    embedding=EmbeddingConfig(
        model_name="models/text-embedding-004",
        dimension=768,
        batch_size=10,
    ),
    
    # Vector store settings
    vector_store=VectorStoreConfig(
        db_path="./data/lancedb",
        table_name="video_frames",
    ),
    
    frames_dir="./data/frames",
)
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | Required |
| `LANCEDB_PATH` | Path to LanceDB database | `./data/lancedb` |
| `LANCEDB_TABLE` | Table name for frames | `video_frames` |
| `FRAME_INTERVAL` | Frame extraction interval | `2.0` |

## Dependency Behavior

The default store fails fast when LanceDB is unavailable. That keeps production runs from appearing successful while dropping data into a temporary mock store. Tests and examples that do not need persistence should instantiate `InMemoryVectorStore` directly.

```python
from video_search.store import InMemoryVectorStore

store = InMemoryVectorStore()
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=video_search

# Run specific test file
pytest tests/test_search.py -v
```

### Code Quality

```bash
# Lint code
ruff check src tests

# Format code
ruff format src tests
```

### Project Structure

```
semantic-video-search/
├── src/video_search/
│   ├── __init__.py      # Package exports
│   ├── models.py        # Pydantic data models
│   ├── extractor.py     # ffmpeg frame extraction
│   ├── embeddings.py    # Gemini embedding generation
│   ├── store.py         # LanceDB and in-memory vector stores
│   ├── search.py        # Main search engine
│   └── cli.py           # Typer CLI
├── tests/
│   ├── conftest.py      # Test fixtures
│   ├── test_models.py   # Model tests
│   ├── test_extractor.py # Extraction tests
│   ├── test_embeddings.py # Embedding tests
│   ├── test_store.py    # Store tests
│   └── test_search.py   # Search tests
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## How It Works

### 1. Video Ingestion

When you index a video:
1. **Frame Extraction**: ffmpeg extracts frames at regular intervals (default: every 2 seconds)
2. **Description Generation**: Gemini 2.5 Flash analyzes each frame and generates a rich textual description
3. **Embedding Creation**: The descriptions are converted to vector embeddings
4. **Storage**: Embeddings are stored in LanceDB for efficient similarity search

### 2. Semantic Search

When you search:
1. **Query Embedding**: Your natural language query is converted to a vector
2. **Similarity Search**: LanceDB finds the most similar frame embeddings
3. **Optional Verification**: Gemini can verify if the actual frame matches the query
4. **Results**: You get timestamps and descriptions of matching moments

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Open focused pull requests with a short problem statement, test evidence, and any known limitations. This repository favors honest maintenance history over large unreviewable drops.

## Acknowledgments

- [LanceDB](https://lancedb.com/) - Embedded vector database
- [Google Gemini](https://ai.google.dev/) - Multimodal AI
- [ffmpeg](https://ffmpeg.org/) - Video processing

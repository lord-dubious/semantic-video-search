# Semantic Video Search Dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash searchuser
WORKDIR /app

# Install uv for faster package management
RUN pip install uv

# Copy dependency files first (for caching)
COPY pyproject.toml requirements.txt requirements-dev.txt ./

# Install dependencies
RUN uv pip install --system -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY tests/ ./tests/

# Install the package
RUN uv pip install --system -e .

# Create data directories
RUN mkdir -p /app/data/frames /app/data/lancedb && \
    chown -R searchuser:searchuser /app

# Switch to non-root user
USER searchuser

# Default command
CMD ["video-search", "demo"]

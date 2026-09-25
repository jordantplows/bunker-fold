# Multi-stage build for Bunker API
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Install Python 3.11
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# Set working directory
WORKDIR /app

# Install dependencies
COPY pyproject.toml /app/
RUN pip install --upgrade pip && \
    pip install -e . && \
    pip install ".[esm,boltz]"

# Copy application code
COPY src/ /app/src/
COPY LICENSE CHANGELOG.md /app/

# Create cache directory
RUN mkdir -p /root/.cache/bunker

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the server
CMD ["bunker", "serve", "--host", "0.0.0.0", "--port", "8000"]

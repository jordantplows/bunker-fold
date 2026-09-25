# Multi-stage build for Bunker API
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Ubuntu 22.04 provides a stable Python 3.10 runtime.
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy everything required by Hatchling before installing the package.
COPY pyproject.toml README.md /app/
COPY src/ /app/src/
COPY LICENSE CHANGELOG.md /app/
RUN python -m pip install --upgrade pip && \
    python -m pip install ".[esm,esmfold,boltz]"

# Create cache directory
RUN mkdir -p /root/.cache/bunker

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the server
CMD ["bunker", "serve", "--host", "0.0.0.0", "--port", "8000"]

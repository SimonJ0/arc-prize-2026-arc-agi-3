# Multi-stage production container for ARC-AGI-3 Agent & W&B Launch
FROM python:3.12-slim

# Set environment flags
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy packaging configuration first for caching
COPY pyproject.toml .

# Install Python package and runtime dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install wandb pandas pyarrow scipy && \
    pip install .

# Copy agent and application codebase
COPY src/ /app/src/
COPY agent/ /app/agent/
COPY configs/ /app/configs/
COPY cli.py /app/cli.py

# Default entrypoint runs agent diagnostics
ENTRYPOINT ["python", "cli.py"]
CMD ["--help"]

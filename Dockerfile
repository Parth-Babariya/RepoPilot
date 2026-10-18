# RepoPilot image: Python 3.12 + Node.js (for the MCP servers) + local models.
# Usage: see "Run with Docker" in README.md, or just `docker compose run --rm repopilot`.
FROM python:3.12-slim

# Node.js/npm run the MCP servers (GitHub, filesystem) via npx; git is handy for the shell tool
RUN apt-get update \
 && apt-get install -y --no-install-recommends nodejs npm git \
 && rm -rf /var/lib/apt/lists/*

# The mounted repo is owned by the host user, not the container's root, and git refuses
# to work in it ("dubious ownership"). /plan would then find no repo and skip its worktree,
# writing straight into your checkout. The container only sees the repo you mount.
RUN git config --system --add safe.directory '*'

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FASTEMBED_CACHE_PATH=/cache/fastembed \
    npm_config_cache=/cache/npm

WORKDIR /app

# Dependencies first, so code changes don't reinstall them
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps -e .

# Download the default embedding model and reranker now, so the first start is fast
RUN python -c "from fastembed import TextEmbedding; \
from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextEmbedding('BAAI/bge-small-en-v1.5'); \
TextCrossEncoder('jinaai/jina-reranker-v1-tiny-en')"

# The repository to analyse is mounted here
WORKDIR /workspace
ENTRYPOINT ["repopilot"]

FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy source
COPY src/ ./src/
COPY ui/ ./ui/
COPY monitoring/ ./monitoring/

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

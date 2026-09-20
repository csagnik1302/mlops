FROM python:3.12-slim

# Copy uv binary from astral-sh official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Set uv timeout and concurrency settings for downloading large packages
ENV UV_HTTP_TIMEOUT=600 \
    UV_CONCURRENT_DOWNLOADS=1

# Copy dependency definition files first for optimal layer caching
COPY pyproject.toml uv.lock ./

# Install project dependencies into /app/.venv
RUN uv sync --frozen --no-install-project

# Ensure virtual environment binaries are on PATH
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# Copy project source code
COPY . .

# Install the project itself into the venv
RUN uv sync --frozen

EXPOSE 8000

CMD ["uvicorn", "src.app.app:app", "--host", "0.0.0.0", "--port", "8000"]


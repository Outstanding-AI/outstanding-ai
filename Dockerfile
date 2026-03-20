# =============================================================================
# Outstanding AI Engine Dockerfile
# FastAPI service for AI-powered email classification and draft generation
# =============================================================================
# Note: Multi-stage build NOT needed - all dependencies are pure Python
# (no build-essential, no native extensions to compile)
# =============================================================================

FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Create non-root user early (before installing dependencies)
# Also give appuser ownership of /app so uv can create .venv
RUN useradd --create-home --shell /bin/bash appuser \
    && chown appuser:appuser /app

# Install system dependencies and uv (as root, then make accessible)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && cp /root/.local/bin/uv /usr/local/bin/uv \
    && chmod +x /usr/local/bin/uv

# Copy dependency files first (for layer caching)
COPY --chown=appuser:appuser pyproject.toml uv.lock ./

# Switch to non-root user for dependency installation
USER appuser

# Install dependencies using uv (production only, no dev deps)
# Cache mount avoids re-downloading packages on every rebuild
RUN --mount=type=cache,target=/home/appuser/.cache/uv,uid=1000 \
    uv sync --no-dev --frozen

# Copy source code (only src/ needed in production)
COPY --chown=appuser:appuser src/ src/

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8001/ping || exit 1

# Run uvicorn directly from venv (faster startup, no uv overhead)
# --timeout-graceful-shutdown: Allow in-flight LLM requests to complete
ENV PATH="/app/.venv/bin:$PATH"
CMD ["uvicorn", "src.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8001", \
     "--timeout-graceful-shutdown", "30"]

.PHONY: help install run dev test test-cov test-live lint format docker-build docker-run docker-up docker-down docker-logs pre-commit-install clean

help:
	@echo "Outstanding AI Engine Commands (uses uv)"
	@echo "===================================="
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install          - Install dependencies (uses uv)"
	@echo "  make pre-commit-install - Install pre-commit hooks"
	@echo ""
	@echo "Development:"
	@echo "  make run              - Run the API server"
	@echo "  make dev              - Run with auto-reload"
	@echo ""
	@echo "Testing:"
	@echo "  make test             - Run unit tests (mocked, no API calls)"
	@echo "  make test-cov         - Run tests with coverage report"
	@echo "  make test-live        - Print the manual live-validation workflow"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             - Run linter (ruff)"
	@echo "  make format           - Format code (ruff)"
	@echo "  make clean            - Remove cache files"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build     - Build Docker image"
	@echo "  make docker-run       - Run Docker container"
	@echo "  make docker-up        - Start with docker-compose"
	@echo "  make docker-down      - Stop docker-compose"
	@echo "  make docker-logs      - View docker-compose logs"
	@echo ""
	@echo "URLs:"
	@echo "  API:     http://localhost:8001"
	@echo "  Health:  http://localhost:8001/health"

# =============================================================================
# SETUP & INSTALLATION
# =============================================================================

install:
	uv sync --all-extras

pre-commit-install:
	uv run pre-commit install
	@echo "Pre-commit hooks installed!"

# =============================================================================
# DEVELOPMENT
# =============================================================================

run:
	uv run uvicorn src.main:app --host 0.0.0.0 --port 8001

dev:
	uv run uvicorn src.main:app --host 0.0.0.0 --port 8001 --reload

# =============================================================================
# TESTING
# =============================================================================

test:
	uv run pytest tests/ -v

test-cov:
	uv run pytest tests/ --cov=src --cov-report=html

test-live:
	@echo "No dedicated live test file is checked in."
	@echo "Validate live on a running service with:"
	@echo "  1. GET /health/llm"
	@echo "  2. POST /classify with a representative payload"
	@echo "  3. POST /generate-draft with a representative payload"

# =============================================================================
# CODE QUALITY
# =============================================================================

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov/ 2>/dev/null || true
	@echo "Cleaned cache and temp files"

# =============================================================================
# DOCKER
# =============================================================================

docker-build:
	docker build -t outstandingai-ai:latest .

docker-run:
	docker run -p 8001:8001 --env-file .env outstandingai-ai:latest

docker-up:
	docker-compose up -d
	@echo "AI Engine started at http://localhost:8001"

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

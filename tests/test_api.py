"""API integration tests for Outstanding AI Engine."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture
def client():
    """Unauthenticated test client. Auth is ON (token set in conftest.py)."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self, client):
        """Test shallow health endpoint returns probe-safe response."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "uptime_seconds" in data
        assert "provider" not in data
        assert "model" not in data

    @patch("src.api.routes.health.llm_client")
    def test_llm_health_check(self, mock_llm_client, authed_client):
        """Test deep LLM health endpoint returns provider-aware status."""
        mock_llm_client.health_check = AsyncMock(
            return_value={
                "primary": {"status": "healthy"},
                "fallback": {"status": "healthy"},
            }
        )
        mock_llm_client.provider_name = "openai"
        mock_llm_client.model_name = "gpt-5"
        mock_fallback = type(
            "Fallback", (), {"provider_name": "gemini", "model_name": "gemini-2.5-flash"}
        )()
        mock_llm_client.fallback = mock_fallback
        mock_llm_client.fallback_count = 0

        response = authed_client.get("/health/llm")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["provider"] == "openai"
        assert data["model"] == "gpt-5"
        assert data["fallback_provider"] == "gemini"
        assert data["fallback_model"] == "gemini-2.5-flash"
        assert "uptime_seconds" in data


class TestClassifyEndpoint:
    """Tests for /classify endpoint."""

    def test_classify_requires_auth(self, client):
        """Test classify endpoint rejects unauthenticated requests."""
        response = client.post("/classify", json={})
        assert response.status_code == 401

    def test_classify_requires_email(self, authed_client):
        """Test classify endpoint requires email field."""
        response = authed_client.post("/classify", json={})

        assert response.status_code == 422

    def test_classify_requires_context(self, authed_client):
        """Test classify endpoint requires context field."""
        response = authed_client.post(
            "/classify",
            json={
                "email": {
                    "subject": "Test",
                    "body": "Test body",
                    "from_address": "test@example.com",
                }
            },
        )

        assert response.status_code == 422

    @patch("src.api.routes.classify.classifier")
    def test_classify_success(self, mock_classifier, authed_client, sample_classify_request):
        """Test successful classification."""
        from src.api.models.responses import ClassifyResponse

        mock_response = ClassifyResponse(
            classification="HARDSHIP", confidence=0.92, reasoning="Job loss mentioned"
        )
        mock_classifier.classify = AsyncMock(return_value=mock_response)

        response = authed_client.post(
            "/classify", json=sample_classify_request.model_dump(mode="json")
        )

        assert response.status_code == 200
        data = response.json()
        assert data["classification"] == "HARDSHIP"


class TestGenerateEndpoint:
    """Tests for /generate-draft endpoint."""

    def test_generate_requires_auth(self, client):
        """Test generate endpoint rejects unauthenticated requests."""
        response = client.post("/generate-draft", json={})
        assert response.status_code == 401

    def test_generate_requires_context(self, authed_client):
        """Test generate endpoint requires context field."""
        response = authed_client.post("/generate-draft", json={"tone": "firm"})

        assert response.status_code == 422

    @patch("src.api.routes.generate.generator")
    def test_generate_success(self, mock_generator, authed_client, sample_generate_draft_request):
        """Test successful draft generation."""
        from src.api.models.responses import GenerateDraftResponse

        mock_response = GenerateDraftResponse(
            subject="Re: Your Account",
            body="Dear Customer,\n\nThank you for reaching out.",
            tone_used="concerned_inquiry",
            invoices_referenced=["INV-123"],
        )
        mock_generator.generate = AsyncMock(return_value=mock_response)

        response = authed_client.post(
            "/generate-draft", json=sample_generate_draft_request.model_dump(mode="json")
        )

        assert response.status_code == 200
        data = response.json()
        assert data["subject"] == "Re: Your Account"
        assert data["body"] == "Dear Customer,\n\nThank you for reaching out."


class TestGatesEndpoint:
    """Tests for /evaluate-gates endpoint."""

    def test_gates_requires_auth(self, client):
        """Test gates endpoint rejects unauthenticated requests."""
        response = client.post("/evaluate-gates", json={})
        assert response.status_code == 401

    def test_gates_requires_context(self, authed_client):
        """Test gates endpoint requires context field."""
        response = authed_client.post("/evaluate-gates", json={"proposed_action": "send_email"})

        assert response.status_code == 422

    @patch("src.api.routes.gates.gate_evaluator")
    def test_gates_success(self, mock_evaluator, authed_client, sample_evaluate_gates_request):
        """Test successful gate evaluation."""
        from src.api.models.responses import EvaluateGatesResponse

        mock_response = EvaluateGatesResponse(
            allowed=True, gate_results={}, recommended_action=None
        )
        mock_evaluator.evaluate = AsyncMock(return_value=mock_response)

        response = authed_client.post(
            "/evaluate-gates", json=sample_evaluate_gates_request.model_dump(mode="json")
        )

        assert response.status_code == 200
        data = response.json()
        assert data["allowed"] is True

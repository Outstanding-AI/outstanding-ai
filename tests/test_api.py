"""API integration tests for Outstanding AI Engine."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self, client):
        """Test health endpoint returns valid response.

        Note: Status may be 'healthy' or 'degraded' depending on
        LLM provider availability (rate limits, API keys, etc.).
        """
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        # Status can be healthy or degraded depending on LLM availability
        assert data["status"] in ["healthy", "degraded"]
        assert "version" in data
        assert "provider" in data
        assert "model" in data
        assert "uptime_seconds" in data


class TestClassifyEndpoint:
    """Tests for /classify endpoint."""

    def test_classify_requires_email(self, client):
        """Test classify endpoint requires email field."""
        response = client.post("/classify", json={})

        assert response.status_code == 422

    def test_classify_requires_context(self, client):
        """Test classify endpoint requires context field."""
        response = client.post(
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
    def test_classify_success(self, mock_classifier, client, sample_classify_request):
        """Test successful classification."""
        from src.api.models.responses import ClassifyResponse

        mock_response = ClassifyResponse(
            classification="HARDSHIP", confidence=0.92, reasoning="Job loss mentioned"
        )
        # Use AsyncMock for the async classify method
        mock_classifier.classify = AsyncMock(return_value=mock_response)

        response = client.post("/classify", json=sample_classify_request.model_dump(mode="json"))

        assert response.status_code == 200
        data = response.json()
        assert data["classification"] == "HARDSHIP"


class TestGenerateEndpoint:
    """Tests for /generate-draft endpoint."""

    def test_generate_requires_context(self, client):
        """Test generate endpoint requires context field."""
        response = client.post("/generate-draft", json={"tone": "firm"})

        assert response.status_code == 422

    @patch("src.api.routes.generate.generator")
    def test_generate_success(self, mock_generator, client, sample_generate_draft_request):
        """Test successful draft generation."""
        from src.api.models.responses import GenerateDraftResponse

        mock_response = GenerateDraftResponse(
            subject="Re: Your Account",
            body="Dear Customer,\n\nThank you for reaching out.",
            tone_used="concerned_inquiry",
            invoices_referenced=["INV-123"],
        )
        # Use AsyncMock for the async generate method
        mock_generator.generate = AsyncMock(return_value=mock_response)

        response = client.post(
            "/generate-draft", json=sample_generate_draft_request.model_dump(mode="json")
        )

        assert response.status_code == 200
        data = response.json()
        assert data["subject"] == "Re: Your Account"
        assert data["body"] == "Dear Customer,\n\nThank you for reaching out."


class TestGatesEndpoint:
    """Tests for /evaluate-gates endpoint."""

    def test_gates_requires_context(self, client):
        """Test gates endpoint requires context field."""
        response = client.post("/evaluate-gates", json={"proposed_action": "send_email"})

        assert response.status_code == 422

    @patch("src.api.routes.gates.gate_evaluator")
    def test_gates_success(self, mock_evaluator, client, sample_evaluate_gates_request):
        """Test successful gate evaluation."""
        from src.api.models.responses import EvaluateGatesResponse

        mock_response = EvaluateGatesResponse(
            allowed=True, gate_results={}, recommended_action=None
        )
        # Use AsyncMock for the async evaluate method
        mock_evaluator.evaluate = AsyncMock(return_value=mock_response)

        response = client.post(
            "/evaluate-gates", json=sample_evaluate_gates_request.model_dump(mode="json")
        )

        assert response.status_code == 200
        data = response.json()
        assert data["allowed"] is True

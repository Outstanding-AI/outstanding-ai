"""Unit tests for DraftGenerator."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from src.api.models.responses import GenerateDraftResponse
from src.engine.generator import DraftGenerator
from src.llm.base import LLMResponse


def _make_llm_response(content: dict, tokens: int = 100) -> LLMResponse:
    """Helper to create mock LLMResponse objects."""
    return LLMResponse(
        content=json.dumps(content),
        model="test-model",
        provider="test",
        usage={"total_tokens": tokens},
    )


class TestDraftGenerator:
    """Tests for DraftGenerator."""

    @pytest.fixture
    def generator(self):
        """Create generator instance."""
        return DraftGenerator()

    @pytest.mark.asyncio
    async def test_generate_draft_referencing_invoices(
        self, generator, sample_generate_draft_request
    ):
        """Test draft generation references specific invoices."""
        sample_generate_draft_request.tone = "firm"

        # Mock LLM response containing invoice numbers
        mock_response = _make_llm_response(
            {
                "subject": "Overdue Invoices",
                "body": "Dear Customer, Please pay invoice INV-12345 immediately. INV-12346 is also overdue.",
            },
            tokens=150,
        )

        with patch(
            "src.engine.generator.llm_client.complete", new_callable=AsyncMock
        ) as mock_complete:
            mock_complete.return_value = mock_response

            result = await generator.generate(sample_generate_draft_request)

            assert isinstance(result, GenerateDraftResponse)
            assert result.tone_used == "firm"
            # Verify invoices are detected in the body
            assert "INV-12345" in result.invoices_referenced
            assert "INV-12346" in result.invoices_referenced

    @pytest.mark.asyncio
    async def test_generate_draft_different_tones(self, generator, sample_generate_draft_request):
        """Test draft generation with different tones."""
        tones = ["friendly_reminder", "professional", "urgent"]

        with patch(
            "src.engine.generator.llm_client.complete", new_callable=AsyncMock
        ) as mock_complete:
            for tone in tones:
                sample_generate_draft_request.tone = tone
                mock_complete.return_value = _make_llm_response(
                    {
                        "subject": f"{tone} subject",
                        "body": f"Body with {tone} tone.",
                    }
                )

                result = await generator.generate(sample_generate_draft_request)

                assert result.tone_used == tone
                assert result.body == f"Body with {tone} tone."

    @pytest.mark.asyncio
    async def test_generate_draft_no_invoices(self, generator, sample_generate_draft_request):
        """Test draft generation when LLM explicitly references no invoices."""
        # Clear obligations so the fallback path also returns empty
        sample_generate_draft_request.context.obligations = []
        mock_response = _make_llm_response(
            {
                "subject": "Payment Reminder",
                "body": "Dear Customer, Please contact us to discuss your account.",
                "invoices_referenced": [],
            }
        )

        with patch(
            "src.engine.generator.llm_client.complete", new_callable=AsyncMock
        ) as mock_complete:
            mock_complete.return_value = mock_response

            result = await generator.generate(sample_generate_draft_request)

            assert isinstance(result, GenerateDraftResponse)
            assert len(result.invoices_referenced) == 0

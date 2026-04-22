"""Unit tests for DraftGenerator."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from src.api.models.requests import GenerateDraftRequest
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


def test_generate_request_hydrates_sparse_lane_context(sample_generate_draft_request):
    """Sparse lane_context payloads should inherit required fields from context.lane."""
    payload = sample_generate_draft_request.model_dump(mode="python")
    payload["context"].update(
        {
            "collection_lane_id": "lane-123",
            "lane": {
                "collection_lane_id": "lane-123",
                "current_level": 2,
                "entry_level": 1,
                "scheduled_touch_index": 1,
                "max_touches_for_level": 3,
                "reminder_cadence_days_for_level": 7,
                "max_days_for_level": 21,
                "tone_ladder": ["professional", "firm"],
                "outstanding_amount": 1500.0,
            },
            "lane_contexts": [
                {
                    "collection_lane_id": "lane-123",
                    "lane_id": "lane-123",
                    "invoice_refs": ["INV-12345"],
                }
            ],
            "mode": "single_lane",
        }
    )

    with pytest.warns(
        DeprecationWarning,
        match="LaneContextInfo.invoice_refs is deprecated",
    ):
        request = GenerateDraftRequest.model_validate(payload)

    lane_context = request.context.lane_contexts[0]
    assert lane_context.lane_id == "lane-123"
    assert lane_context.current_level == 2
    assert lane_context.entry_level == 1
    assert lane_context.scheduled_touch_index == 1
    assert lane_context.max_touches_for_level == 3
    assert lane_context.reminder_cadence_days_for_level == 7
    assert lane_context.max_days_for_level == 21
    assert lane_context.tone_ladder == ["professional", "firm"]
    assert lane_context.__dict__["outstanding_amount"] == 1500.0

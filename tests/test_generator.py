"""Unit tests for DraftGenerator."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from src.api.models.requests import CaseContext, GenerateDraftRequest, ObligationInfo, PartyInfo
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

    @pytest.mark.asyncio
    async def test_current_context_uses_only_sendable_overdue_candidates(self, generator):
        """Current lake drafts should not chase not-yet-due or source-disputed invoices."""
        context = CaseContext(
            schema_version=4,
            party=PartyInfo(
                party_id="party-uuid-1",
                external_id="party-ext-1",
                provider_type="sage_200",
                customer_code="C001",
                name="Acme Ltd",
                source="sage_200",
            ),
            obligations=[
                ObligationInfo(
                    id="obl-1",
                    external_id="1",
                    provider_type="sage_200",
                    invoice_number="INV-1",
                    original_amount=100.0,
                    amount_due=100.0,
                    is_overdue=True,
                    days_overdue=10,
                    is_sendable=True,
                    is_chase_eligible=True,
                ),
                ObligationInfo(
                    id="obl-2",
                    external_id="2",
                    provider_type="sage_200",
                    invoice_number="INV-2",
                    original_amount=200.0,
                    amount_due=200.0,
                    is_overdue=False,
                    days_overdue=0,
                    is_sendable=True,
                    is_chase_eligible=True,
                ),
                ObligationInfo(
                    id="obl-3",
                    external_id="3",
                    provider_type="sage_200",
                    invoice_number="INV-3",
                    original_amount=300.0,
                    amount_due=300.0,
                    is_overdue=True,
                    days_overdue=20,
                    is_sendable=True,
                    is_chase_eligible=True,
                    is_source_disputed=True,
                    source_query_raw="Queried in Sage",
                ),
            ],
            debtor_contact={"email": "ap@example.com", "name": "AP Team"},
            source_sync_run_id="sync-1",
            application_run_id="app-1",
            core_snapshot_watermark="2026-05-01T00:00:00Z",
            application_snapshot_watermark="2026-05-01T00:10:00Z",
            application_decision_cutoff="2026-05-01T00:15:00Z",
            policy_snapshot_id="policy-1",
            draft_candidate_id="cand-1",
        )
        request = GenerateDraftRequest(context=context, tone="professional")

        mock_response = _make_llm_response(
            {
                "subject": "Invoice follow-up",
                "body": "<p>Please see the invoice below.</p><p>{INVOICE_TABLE}</p>",
                "invoices_referenced": [],
            }
        )

        with patch(
            "src.engine.generator.llm_client.complete", new_callable=AsyncMock
        ) as mock_complete:
            mock_complete.return_value = mock_response
            with patch("src.engine.generator.guardrail_pipeline") as mock_pipeline:
                mock_result = type(
                    "MockResult",
                    (),
                    {
                        "all_passed": True,
                        "results": [],
                        "blocking_guardrails": [],
                        "review_findings": [],
                        "total_token_usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        },
                    },
                )()
                mock_pipeline.validate.return_value = mock_result

                result = await generator.generate(request)

        user_prompt = mock_complete.await_args.kwargs["user_prompt"]
        guardrail_kwargs = mock_pipeline.validate.call_args.kwargs
        assert "- INV-1:" in user_prompt
        assert "- INV-2:" not in user_prompt
        assert "INV-3: excluded" in user_prompt
        assert guardrail_kwargs["candidate_invoice_refs"] == ["INV-1"]
        assert result.invoices_referenced == ["INV-1"]
        assert result.ai_audit is not None
        assert result.ai_audit.draft_candidate_id == "cand-1"


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

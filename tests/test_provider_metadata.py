"""Tests for provider metadata in API responses.

Validates that classify, generate, and gate evaluation responses
include provider/model/is_fallback fields.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from src.api.models.responses import EvaluateGatesResponse
from src.engine.gate_evaluator import GateEvaluator
from src.llm.base import LLMResponse


class TestProviderMetadata:
    """Test that all response types include provider metadata."""

    @pytest.mark.asyncio
    async def test_classify_response_includes_metadata(self, sample_classify_request):
        """Classify response should include provider/model/is_fallback."""
        from src.engine.classifier import EmailClassifier

        mock_response = LLMResponse(
            content=json.dumps(
                {
                    "classification": "HARDSHIP",
                    "confidence": 0.9,
                    "reasoning": "Customer mentions job loss",
                    "extracted_data": {
                        "promise_date": None,
                        "promise_amount": None,
                        "dispute_type": None,
                        "dispute_reason": None,
                        "redirect_contact": None,
                        "redirect_email": None,
                    },
                }
            ),
            model="gemini-2.5-flash",
            provider="vertex",
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )

        classifier = EmailClassifier()

        with patch("src.engine.classifier.llm_client") as mock_llm:
            mock_llm.complete = AsyncMock(return_value=mock_response)
            mock_llm.primary_provider_name = "vertex"

            with patch("src.engine.classifier.guardrail_pipeline") as mock_pipeline:
                mock_result = type(
                    "MockResult",
                    (),
                    {
                        "all_passed": True,
                        "results": [],
                        "blocking_guardrails": [],
                    },
                )()
                mock_pipeline.validate.return_value = mock_result

                result = await classifier.classify(sample_classify_request)

        assert result.provider == "vertex"
        assert result.model == "gemini-2.5-flash"
        assert result.is_fallback is False

    @pytest.mark.asyncio
    async def test_generate_response_includes_metadata(self, sample_generate_draft_request):
        """Generate response should include provider/model/is_fallback."""
        from src.engine.generator import DraftGenerator

        mock_response = LLMResponse(
            content=json.dumps(
                {
                    "subject": "Outstanding Balance",
                    "body": "<p>Dear Customer,</p><p>Please pay.</p>",
                }
            ),
            model="gpt-4o-mini",
            provider="openai",
            usage={"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300},
        )

        generator = DraftGenerator()

        with patch("src.engine.generator.llm_client") as mock_llm:
            mock_llm.complete = AsyncMock(return_value=mock_response)
            mock_llm.primary_provider_name = "vertex"

            with patch("src.engine.generator.guardrail_pipeline") as mock_pipeline:
                mock_result = type(
                    "MockResult",
                    (),
                    {
                        "all_passed": True,
                        "results": [],
                        "blocking_guardrails": [],
                        "total_token_usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        },
                    },
                )()
                mock_pipeline.validate.return_value = mock_result

                result = await generator.generate(sample_generate_draft_request)

        assert result.provider == "openai"
        assert result.model == "gpt-4o-mini"
        assert result.is_fallback is True

    @pytest.mark.asyncio
    async def test_gate_evaluation_returns_deterministic_metadata(
        self, sample_evaluate_gates_request
    ):
        """Gate evaluation should return provider=deterministic, model=rule_engine."""
        evaluator = GateEvaluator()

        sample_evaluate_gates_request.context.monthly_touch_count = 0
        sample_evaluate_gates_request.context.touch_cap = 10
        sample_evaluate_gates_request.context.active_dispute = False

        result = await evaluator.evaluate(sample_evaluate_gates_request)

        assert isinstance(result, EvaluateGatesResponse)
        assert result.provider == "deterministic"
        assert result.model == "rule_engine"
        assert result.is_fallback is False

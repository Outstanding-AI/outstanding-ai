"""Tests for Real-Time Evaluator."""

import pytest

from src.api.models.requests import (
    CaseContext,
    ClassifyRequest,
    EmailContent,
    ObligationInfo,
    PartyInfo,
)
from src.api.models.responses import ClassifyResponse, ExtractedData
from src.evals.realtime import RealTimeEvaluator
from src.guardrails.base import GuardrailPipelineResult, GuardrailResult, GuardrailSeverity


@pytest.fixture
def sample_context() -> CaseContext:
    """Create a sample context for testing."""
    return CaseContext(
        schema_version=2,
        party=PartyInfo(
            party_id="party-001",
            external_id="party-ext-001",
            provider_type="sage_200",
            customer_code="CUST001",
            name="Acme Corp",
            currency="GBP",
            source="sage_200",
        ),
        obligations=[
            ObligationInfo(
                id="obl-12345",
                external_id="12345",
                provider_type="sage_200",
                invoice_number="INV-12345",
                original_amount=1500.00,
                amount_due=1500.00,
                due_date="2024-01-01",
                days_past_due=30,
            ),
        ],
    )


@pytest.fixture
def sample_classify_request(sample_context) -> ClassifyRequest:
    """Create a sample classification request."""
    return ClassifyRequest(
        email=EmailContent(
            subject="Re: Invoice",
            body="I will pay £500 by Friday.",
            from_address="test@example.com",
        ),
        context=sample_context,
    )


class TestRealTimeEvaluator:
    """Tests for RealTimeEvaluator."""

    def test_evaluate_classification_success(self, sample_classify_request):
        """Test evaluation of successful classification."""
        evaluator = RealTimeEvaluator()

        response = ClassifyResponse(
            classification="PROMISE_TO_PAY",
            confidence=0.92,
            reasoning="Customer promises to pay",
            extracted_data=ExtractedData(
                promise_date=None,
                promise_amount=500.0,
            ),
        )

        # All guardrails passed
        guardrail_result = GuardrailPipelineResult(
            all_passed=True,
            should_block=False,
            results=[
                GuardrailResult(
                    passed=True,
                    guardrail_name="factual_grounding",
                    severity=GuardrailSeverity.CRITICAL,
                    message="Passed",
                )
            ],
        )

        metrics = evaluator.evaluate_classification(
            request=sample_classify_request,
            response=response,
            guardrail_result=guardrail_result,
            latency_ms=150.0,
            provider="openai",
            model="gpt-5-nano",
        )

        assert metrics.guardrails_passed
        assert metrics.factual_accuracy == 1.0
        assert metrics.classification == "PROMISE_TO_PAY"
        assert metrics.classification_confidence == 0.92
        assert metrics.promise_amount_extracted == 500.0
        assert metrics.latency_ms == 150.0
        assert metrics.provider_used == "openai"

    def test_evaluate_classification_with_failures(self, sample_classify_request):
        """Test evaluation when guardrails fail."""
        evaluator = RealTimeEvaluator()

        response = ClassifyResponse(
            classification="PROMISE_TO_PAY",
            confidence=0.85,
        )

        # One guardrail failed
        guardrail_result = GuardrailPipelineResult(
            all_passed=False,
            should_block=True,
            results=[
                GuardrailResult(
                    passed=True,
                    guardrail_name="factual_grounding",
                    severity=GuardrailSeverity.CRITICAL,
                    message="Passed",
                ),
                GuardrailResult(
                    passed=False,
                    guardrail_name="numerical_consistency",
                    severity=GuardrailSeverity.CRITICAL,
                    message="Amount mismatch",
                ),
            ],
            blocking_guardrails=["numerical_consistency"],
        )

        metrics = evaluator.evaluate_classification(
            request=sample_classify_request,
            response=response,
            guardrail_result=guardrail_result,
            latency_ms=200.0,
        )

        assert not metrics.guardrails_passed
        assert metrics.factual_accuracy == 0.5  # 1/2 passed
        assert "numerical_consistency" in metrics.guardrail_failures

    def test_metrics_to_dict(self, sample_classify_request):
        """Test metrics serialization."""
        evaluator = RealTimeEvaluator()

        response = ClassifyResponse(
            classification="COOPERATIVE",
            confidence=0.95,
        )

        metrics = evaluator.evaluate_classification(
            request=sample_classify_request,
            response=response,
        )

        metrics_dict = metrics.to_dict()

        assert "request_id" in metrics_dict
        assert "timestamp" in metrics_dict
        assert "guardrails_passed" in metrics_dict
        assert "classification" in metrics_dict
        assert metrics_dict["classification"] == "COOPERATIVE"

    def test_summary_stats(self, sample_classify_request):
        """Test summary statistics calculation."""
        evaluator = RealTimeEvaluator()

        # Add a few metrics
        for _ in range(5):
            response = ClassifyResponse(
                classification="COOPERATIVE",
                confidence=0.9,
                tokens_used=100,
            )
            evaluator.evaluate_classification(
                request=sample_classify_request,
                response=response,
                latency_ms=100.0,
            )

        stats = evaluator.get_summary_stats()

        assert stats["total_requests"] == 5
        assert stats["guardrail_pass_rate"] == 1.0
        assert stats["avg_latency_ms"] == 100.0
        assert stats["avg_tokens_used"] == 100

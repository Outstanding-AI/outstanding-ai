"""Tests for Guardrail Pipeline."""

import pytest

from src.api.models.requests import CaseContext, ObligationInfo, PartyInfo
from src.guardrails.base import GuardrailResult, GuardrailSeverity
from src.guardrails.pipeline import GuardrailPipeline


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
            ObligationInfo(
                id="obl-12346",
                external_id="12346",
                provider_type="sage_200",
                invoice_number="INV-12346",
                original_amount=2500.00,
                amount_due=2500.00,
                due_date="2024-01-05",
                days_past_due=26,
            ),
        ],
    )


class TestGuardrailPipeline:
    """Tests for GuardrailPipeline."""

    def test_valid_output_passes_all_guardrails(self, sample_context):
        """Test that valid output passes all guardrails."""
        pipeline = GuardrailPipeline()

        # Valid output with correct invoice and amounts
        output = """
        Dear Acme Corp,

        Your invoice INV-12345 for £1,500.00 is now 30 days overdue.
        Your total outstanding is £4,000.00.

        Please arrange payment at your earliest convenience.

        Best regards,
        Collections Team
        """

        result = pipeline.validate(output, sample_context)

        assert result.all_passed
        assert not result.should_block
        assert len(result.blocking_guardrails) == 0

    def test_fabricated_invoice_blocks(self, sample_context):
        """Test that fabricated invoice number blocks output."""
        pipeline = GuardrailPipeline()

        # Invalid invoice number
        output = "Your invoice INV-99999 for £1,500.00 is overdue."

        result = pipeline.validate(output, sample_context)

        assert not result.all_passed
        assert result.should_block
        assert "factual_grounding" in result.blocking_guardrails

    def test_fabricated_amount_blocks(self, sample_context):
        """Test that fabricated amount blocks output."""
        pipeline = GuardrailPipeline()

        # Invalid amount
        output = "Your outstanding balance is £10,000.00"

        result = pipeline.validate(output, sample_context)

        assert not result.all_passed
        assert result.should_block
        assert "factual_grounding" in result.blocking_guardrails

    def test_wrong_total_blocks(self, sample_context):
        """Test that incorrect total calculation blocks output."""
        pipeline = GuardrailPipeline()

        # Wrong total (should be £4,000)
        output = "Your total outstanding is £5,000.00"

        result = pipeline.validate(output, sample_context)

        assert not result.all_passed
        assert result.should_block

    def test_fail_fast_stops_on_critical(self, sample_context):
        """Test that fail_fast stops on first critical failure."""
        pipeline = GuardrailPipeline()

        # Multiple issues - should stop at first critical
        output = "Invoice INV-99999 for £99,999.99 is overdue."

        result = pipeline.validate(output, sample_context, fail_fast=True)

        assert not result.all_passed
        assert result.should_block
        assert result.retry_suggested

    def test_retry_prompt_addition(self, sample_context):
        """Test that retry prompt is generated for failures."""
        pipeline = GuardrailPipeline()

        # Fabricated invoice
        output = "Your invoice INV-99999 is overdue."

        result = pipeline.validate(output, sample_context)
        retry_prompt = pipeline.get_retry_prompt_addition(result)

        assert "VALIDATION REQUIREMENTS" in retry_prompt
        assert "invoice" in retry_prompt.lower()

    def test_to_dict_serialization(self, sample_context):
        """Test that results can be serialized to dict."""
        pipeline = GuardrailPipeline()

        output = "Your invoice INV-12345 for £1,500.00 is overdue."
        result = pipeline.validate(output, sample_context)

        result_dict = result.to_dict()

        assert "all_passed" in result_dict
        assert "should_block" in result_dict
        assert "results" in result_dict
        assert isinstance(result_dict["results"], list)
        assert "review_findings" in result_dict

    def test_review_findings_are_aggregated_without_blocking(self, sample_context):
        """Forbidden content review findings should not block output delivery."""

        class AlwaysFlagReview:
            name = "forbidden_content"
            severity = GuardrailSeverity.REVIEW

            def validate(self, output, context, **kwargs):
                return [
                    GuardrailResult(
                        passed=False,
                        guardrail_name=self.name,
                        severity=self.severity,
                        message="IBAN detected",
                        details={
                            "findings": [
                                {"category": "bank_payment_details", "excerpts": ["GB29..."]}
                            ]
                        },
                        is_review_finding=True,
                    )
                ]

        pipeline = GuardrailPipeline(guardrails=[AlwaysFlagReview()])
        result = pipeline.validate("Pay via GB29...", sample_context, parallel=False)

        assert result.all_passed is False
        assert result.should_block is False
        assert result.blocking_guardrails == []
        assert len(result.review_findings) == 1

    def test_multiple_guardrail_failures(self, sample_context):
        """Test output with multiple guardrail failures."""
        pipeline = GuardrailPipeline()

        # Multiple issues
        output = """
        Dear Wrong Company Name,

        Your invoice INV-99999 for £99,999.99 is 100 days overdue.
        Customer code: WRONG123

        Please pay immediately.
        """

        result = pipeline.validate(output, sample_context, fail_fast=False)

        assert not result.all_passed
        assert result.should_block
        # Should have multiple failures
        failed_results = [r for r in result.results if not r.passed]
        assert len(failed_results) >= 1

    def test_partial_name_match_passes(self, sample_context):
        """Draft without lane scope or unknown emails should pass deterministic identity checks."""
        pipeline = GuardrailPipeline()

        # Uses "Acme" which is part of "Acme Corp"
        output = """
        Dear Acme Team,

        Your invoice INV-12345 for £1,500.00 is overdue.
        """

        result = pipeline.validate(output, sample_context)

        assert result.all_passed
        assert not result.should_block

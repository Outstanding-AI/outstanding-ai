"""Tests for guardrail severity alignment with CLAUDE.md documentation.

Validates:
- Identity scope → HIGH (not CRITICAL)
- Temporal → MEDIUM (not HIGH) — failures don't block
- Contextual → LOW (not MEDIUM) — failures don't block
- Pipeline behavior with mixed severities
"""

from src.api.models.requests import CaseContext, ObligationInfo, PartyInfo
from src.guardrails.base import GuardrailResult, GuardrailSeverity
from src.guardrails.contextual import ContextualCoherenceGuardrail
from src.guardrails.forbidden_content import ForbiddenContentDetector
from src.guardrails.identity_scope import IdentityScopeGuardrail
from src.guardrails.pipeline import GuardrailPipeline
from src.guardrails.temporal import TemporalConsistencyGuardrail


class TestGuardrailSeverities:
    """Test that guardrail severities match documented values."""

    def test_identity_scope_severity_is_high(self):
        """Identity scope guardrail should be HIGH severity."""
        guardrail = IdentityScopeGuardrail()
        assert guardrail.severity == GuardrailSeverity.HIGH

    def test_temporal_severity_is_medium(self):
        """Temporal guardrail should be MEDIUM severity."""
        guardrail = TemporalConsistencyGuardrail()
        assert guardrail.severity == GuardrailSeverity.MEDIUM

    def test_contextual_severity_is_low(self):
        """Contextual guardrail should be LOW severity."""
        guardrail = ContextualCoherenceGuardrail()
        assert guardrail.severity == GuardrailSeverity.LOW

    def test_forbidden_content_severity_is_review(self):
        """Forbidden content findings should surface for review, not block."""
        guardrail = ForbiddenContentDetector()
        assert guardrail.severity == GuardrailSeverity.REVIEW

    def test_temporal_failure_does_not_block(self):
        """MEDIUM severity failure should not have should_block=True."""
        result = GuardrailResult(
            passed=False,
            guardrail_name="temporal_consistency",
            severity=GuardrailSeverity.MEDIUM,
            message="Date mismatch",
        )
        assert result.should_block is False

    def test_contextual_failure_does_not_block(self):
        """LOW severity failure should not have should_block=True."""
        result = GuardrailResult(
            passed=False,
            guardrail_name="contextual_coherence",
            severity=GuardrailSeverity.LOW,
            message="Tone mismatch",
        )
        assert result.should_block is False

    def test_review_finding_does_not_block(self):
        """REVIEW severity findings should not have should_block=True."""
        result = GuardrailResult(
            passed=False,
            guardrail_name="forbidden_content",
            severity=GuardrailSeverity.REVIEW,
            message="IBAN detected",
            is_review_finding=True,
        )
        assert result.should_block is False

    def test_pipeline_temporal_only_fails_not_blocked(self):
        """Pipeline with only temporal (MEDIUM) failure should not block."""

        # Create a mock guardrail that always fails with MEDIUM severity
        class AlwaysFailMedium:
            name = "temporal_consistency"
            severity = GuardrailSeverity.MEDIUM

            def validate(self, output, context, **kwargs):
                return [
                    GuardrailResult(
                        passed=False,
                        guardrail_name=self.name,
                        severity=self.severity,
                        message="Date mismatch",
                    )
                ]

        pipeline = GuardrailPipeline(guardrails=[AlwaysFailMedium()])
        context = CaseContext(
            party=PartyInfo(party_id="p1", customer_code="C1", name="Test Corp"),
            obligations=[
                ObligationInfo(
                    invoice_number="INV-1",
                    original_amount=100,
                    amount_due=100,
                    due_date="2024-01-01",
                    days_past_due=30,
                )
            ],
        )

        result = pipeline.validate("test output", context, parallel=False)

        assert result.all_passed is False
        assert result.should_block is False
        assert len(result.blocking_guardrails) == 0

    def test_pipeline_contextual_only_fails_not_blocked(self):
        """Pipeline with only contextual (LOW) failure should not block."""

        class AlwaysFailLow:
            name = "contextual_coherence"
            severity = GuardrailSeverity.LOW

            def validate(self, output, context, **kwargs):
                return [
                    GuardrailResult(
                        passed=False,
                        guardrail_name=self.name,
                        severity=self.severity,
                        message="Tone mismatch",
                    )
                ]

        pipeline = GuardrailPipeline(guardrails=[AlwaysFailLow()])
        context = CaseContext(
            party=PartyInfo(party_id="p1", customer_code="C1", name="Test Corp"),
            obligations=[
                ObligationInfo(
                    invoice_number="INV-1",
                    original_amount=100,
                    amount_due=100,
                    due_date="2024-01-01",
                    days_past_due=30,
                )
            ],
        )

        result = pipeline.validate("test output", context, parallel=False)

        assert result.all_passed is False
        assert result.should_block is False
        assert len(result.blocking_guardrails) == 0

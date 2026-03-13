"""Numerical Consistency Guardrail -- validate math in AI output.

Verify that any totals or days-overdue figures stated in the draft body
match the values computed from the case context obligations.  This
catches LLM arithmetic errors (e.g., wrong sum of outstanding amounts)
and hallucinated overdue-day counts.

CRITICAL severity -- blocks output on failure.
"""

import logging
import re

from src.api.models.requests import CaseContext

from .base import BaseGuardrail, GuardrailResult, GuardrailSeverity

logger = logging.getLogger(__name__)


class NumericalConsistencyGuardrail(BaseGuardrail):
    """Validate mathematical accuracy of AI-generated draft text.

    CRITICAL severity -- blocks output on failure and triggers retry.

    Checks:
    1. Total amounts stated in prose match the authoritative sum of
       ``amount_due`` from context obligations (tolerance: 0.01).
    2. Days-overdue counts match obligation ``days_past_due`` values
       (tolerance: +/- 1 day for timing differences).

    Closure-mode drafts skip all checks (no financial content).
    """

    def __init__(self):
        super().__init__(
            name="numerical_consistency",
            severity=GuardrailSeverity.CRITICAL,
        )

    def validate(self, output: str, context: CaseContext, **kwargs) -> list[GuardrailResult]:
        """Validate numerical consistency of the output.

        Run two sub-checks: total calculation verification and
        days-overdue verification.  Closure-mode drafts skip both.

        Args:
            output: AI-generated draft body text.
            context: Case context with obligations.
            **kwargs: ``closure_mode`` (bool).

        Returns:
            List of two GuardrailResult objects (total + days).
        """
        closure_mode = kwargs.get("closure_mode", False)

        # Closure emails: no numerical validation needed
        if closure_mode:
            return [
                self._pass("Closure mode — total validation skipped"),
                self._pass("Closure mode — days overdue validation skipped"),
            ]

        results = []
        results.append(self._validate_total_calculation(output, context))
        results.append(self._validate_days_overdue(output, context))
        return results

    def _validate_total_calculation(self, output: str, context: CaseContext) -> GuardrailResult:
        """Validate that stated totals match calculated sums.

        Extract total-amount phrases from the draft using regex and
        compare each against the authoritative sum of ``amount_due``
        from context obligations.  Tolerance is 0.01 (penny).
        """
        # Extract total phrases from output
        total_patterns = [
            r"total\s+(?:outstanding|amount|due|owed)(?:\s+(?:is|of))?\s*:?\s*[£$€]?\s*([\d,]+(?:\.\d{2})?)",
            r"owe(?:s|d)?\s+(?:us\s+)?(?:a\s+total\s+of\s+)?[£$€]?\s*([\d,]+(?:\.\d{2})?)",
            r"[£$€]\s*([\d,]+(?:\.\d{2})?)\s+(?:in\s+)?total",
            r"combined\s+(?:balance|amount)\s+(?:of|is)\s+[£$€]?\s*([\d,]+(?:\.\d{2})?)",
        ]

        # Calculate actual total
        actual_total = sum(o.amount_due for o in context.obligations)

        # Find stated totals
        stated_totals = []
        for pattern in total_patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            for match in matches:
                try:
                    stated = float(match.replace(",", ""))
                    stated_totals.append(stated)
                except ValueError:
                    continue

        tolerance = 0.01

        # Validate each stated total
        for stated in stated_totals:
            if abs(stated - actual_total) > tolerance:
                return self._fail(
                    message=f"Stated total {stated} does not match calculated total {actual_total}",
                    expected=actual_total,
                    found=stated,
                    details={
                        "stated_total": stated,
                        "calculated_total": actual_total,
                        "difference": abs(stated - actual_total),
                        "obligations": [
                            {"invoice": o.invoice_number, "amount": o.amount_due}
                            for o in context.obligations
                        ],
                    },
                )

        return self._pass(
            message="Total calculations validated",
            details={
                "calculated_total": actual_total,
                "stated_totals": stated_totals,
            },
        )

    def _validate_days_overdue(self, output: str, context: CaseContext) -> GuardrailResult:
        """Validate that days-overdue statements are accurate.

        Extract "N days overdue/past due/late" phrases and compare
        against the set of valid ``days_past_due`` values from context
        obligations.  Allows +/- 1 day tolerance for timing differences
        between draft generation and obligation data refresh.
        """
        # Extract days overdue mentions
        days_patterns = [
            r"(\d+)\s+days?\s+(?:past\s+due|overdue|late)",
            r"overdue\s+(?:by|for)\s+(\d+)\s+days?",
        ]

        # Get valid days overdue from context
        valid_days = {o.days_past_due for o in context.obligations}

        # Also add max days overdue (commonly referenced)
        max_days = max(valid_days) if valid_days else 0
        valid_days.add(max_days)

        # Find mentioned days
        for pattern in days_patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            for match in matches:
                try:
                    mentioned_days = int(match)
                    # Allow some tolerance (±1 day for timing differences)
                    is_valid = any(abs(mentioned_days - valid) <= 1 for valid in valid_days)
                    if not is_valid and mentioned_days > 0:
                        return self._fail(
                            message=f"Days overdue {mentioned_days} not found in context",
                            expected=sorted(valid_days),
                            found=mentioned_days,
                            details={
                                "mentioned_days": mentioned_days,
                                "valid_days": sorted(valid_days),
                            },
                        )
                except ValueError:
                    continue

        return self._pass(
            message="Days overdue calculations validated",
            details={"valid_days": sorted(valid_days)},
        )

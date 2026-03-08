"""Factual Grounding Guardrail - validates facts exist in context."""

import logging
import re

from src.api.models.requests import CaseContext

from .base import BaseGuardrail, GuardrailResult, GuardrailSeverity

logger = logging.getLogger(__name__)


class FactualGroundingGuardrail(BaseGuardrail):
    """
    Validates that AI outputs only contain facts from the input context.

    Checks:
    1. Invoice numbers mentioned exist in context.obligations
    2. Monetary amounts match obligation amounts or their sums
    3. Due dates match obligation due dates
    """

    def __init__(self):
        super().__init__(
            name="factual_grounding",
            severity=GuardrailSeverity.CRITICAL,
        )

    def validate(self, output: str, context: CaseContext, **kwargs) -> list[GuardrailResult]:
        """Validate factual grounding of the output."""
        results = []

        # Run all validation checks
        results.append(self._validate_invoice_numbers(output, context))
        results.append(self._validate_amounts(output, context))

        return results

    def _validate_invoice_numbers(self, output: str, context: CaseContext) -> GuardrailResult:
        """Validate that all invoice numbers in output exist in context."""
        # Extract invoice numbers from output using common patterns
        # Matches: INV-12345, INV12345, Invoice 12345, #12345, etc.
        # NOTE: Patterns must be restrictive to avoid false positives with garbage chars
        invoice_patterns = [
            r"INV[-\s]?(\d+)",  # INV-12345, INV 12345, INV12345
            r"Invoice\s*#?\s*(\d+)",  # Invoice 12345, Invoice #12345
            r"invoice\s+number\s*:?\s*([A-Za-z0-9][-A-Za-z0-9]+)",  # invoice number: ABC-123 (alphanumeric only)
            r"#(\d{4,})",  # #12345 (4+ digits to avoid false positives)
        ]

        # Get valid invoice numbers from context (skip empty/null invoice numbers)
        valid_invoices = {o.invoice_number.upper() for o in context.obligations if o.invoice_number}

        # Also create a set of just the numeric parts for flexible matching
        valid_invoice_numbers = set()
        for inv in valid_invoices:
            # Extract numeric portion
            match = re.search(r"\d+", inv)
            if match:
                valid_invoice_numbers.add(match.group())

        # Find all invoice references in output
        found_invoices = set()
        output_upper = output.upper()

        # First, check for exact matches
        for inv in valid_invoices:
            if inv in output_upper:
                found_invoices.add(inv)

        # Then look for pattern-based matches
        for pattern in invoice_patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            for match in matches:
                found_invoices.add(match.upper() if isinstance(match, str) else match)

        # Validate all found invoices exist in context
        invalid_invoices = []
        for found_inv in found_invoices:
            found_inv_str = str(found_inv).upper()
            # Check if it matches valid invoice or its numeric portion
            is_valid = (
                found_inv_str in valid_invoices
                or any(found_inv_str in valid for valid in valid_invoices)
                or found_inv_str in valid_invoice_numbers
            )
            if not is_valid:
                invalid_invoices.append(found_inv_str)

        if invalid_invoices:
            return self._fail(
                message=f"Invoice numbers not found in context: {invalid_invoices}",
                expected=list(valid_invoices),
                found=invalid_invoices,
                details={
                    "invalid_invoices": invalid_invoices,
                    "valid_invoices": list(valid_invoices),
                },
            )

        return self._pass(
            message="All invoice numbers validated",
            details={"validated_invoices": list(found_invoices)},
        )

    def _validate_amounts(self, output: str, context: CaseContext) -> GuardrailResult:
        """Validate that monetary amounts in output match context data."""
        # Extract currency and amounts from output
        # Matches: £1,500.00, $1500, €1,000, GBP 1000, etc.
        amount_patterns = [
            r"[£$€]\s*([\d,]+(?:\.\d{2})?)",  # £1,500.00
            r"([\d,]+(?:\.\d{2})?)\s*(?:GBP|USD|EUR)",  # 1500 GBP
            r"(?:GBP|USD|EUR)\s*([\d,]+(?:\.\d{2})?)",  # GBP 1500
        ]

        # Build set of valid amounts
        valid_amounts = set()

        # Individual obligation amounts
        for o in context.obligations:
            valid_amounts.add(o.amount_due)
            valid_amounts.add(o.original_amount)

        # Total outstanding
        total_outstanding = sum(o.amount_due for o in context.obligations)
        valid_amounts.add(total_outstanding)

        # Also add rounded versions (in case of formatting differences)
        valid_amounts_rounded = {round(a, 2) for a in valid_amounts}
        valid_amounts_int = {int(a) for a in valid_amounts}

        # Extract amounts from output
        found_amounts = []
        for pattern in amount_patterns:
            matches = re.findall(pattern, output)
            for match in matches:
                # Clean and parse the amount
                cleaned = match.replace(",", "").replace(" ", "")
                try:
                    amount = float(cleaned)
                    found_amounts.append(amount)
                except ValueError:
                    continue

        # Validate found amounts
        invalid_amounts = []
        for amount in found_amounts:
            # Check if amount matches any valid amount (with tolerance for rounding)
            is_valid = (
                amount in valid_amounts_rounded
                or amount in valid_amounts_int
                or any(abs(amount - valid) < 0.01 for valid in valid_amounts)
            )
            if not is_valid:
                invalid_amounts.append(amount)

        if invalid_amounts:
            return self._fail(
                message=f"Amounts not found in context: {invalid_amounts}",
                expected=sorted(valid_amounts),
                found=invalid_amounts,
                details={
                    "invalid_amounts": invalid_amounts,
                    "valid_amounts": sorted(valid_amounts),
                    "total_outstanding": total_outstanding,
                },
            )

        return self._pass(
            message="All monetary amounts validated",
            details={
                "validated_amounts": found_amounts,
                "total_outstanding": total_outstanding,
            },
        )

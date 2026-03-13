"""Factual Grounding Guardrail - validates facts exist in context."""

import logging
import re

from src.api.models.requests import CaseContext

from .base import BaseGuardrail, GuardrailResult, GuardrailSeverity

logger = logging.getLogger(__name__)

# Reusable amount extraction patterns
AMOUNT_PATTERNS = [
    r"[£$€]\s*([\d,]+(?:\.\d{2})?)",  # £1,500.00
    r"([\d,]+(?:\.\d{2})?)\s*(?:GBP|USD|EUR)",  # 1500 GBP
    r"(?:GBP|USD|EUR)\s*([\d,]+(?:\.\d{2})?)",  # GBP 1500
]


class FactualGroundingGuardrail(BaseGuardrail):
    """
    Validates that AI outputs only contain facts from the input context.

    Context-aware behaviour:
    - closure_mode=True: skip all validation (closure emails don't reference invoices)
    - skip_invoice_table=True: also treat amounts from recent_messages as valid
      (follow-up emails may echo amounts the debtor mentioned)
    - Standard drafts: strict validation against obligations only

    Checks:
    1. Invoice numbers mentioned exist in context.obligations
    2. Monetary amounts match obligation amounts or their sums
    """

    def __init__(self):
        super().__init__(
            name="factual_grounding",
            severity=GuardrailSeverity.CRITICAL,
        )

    def validate(self, output: str, context: CaseContext, **kwargs) -> list[GuardrailResult]:
        """Validate factual grounding of the output."""
        closure_mode = kwargs.get("closure_mode", False)

        # Closure emails: no invoice/amount validation needed
        if closure_mode:
            return [
                self._pass("Closure mode — invoice validation skipped"),
                self._pass("Closure mode — amount validation skipped"),
            ]

        results = []
        results.append(self._validate_invoice_numbers(output, context))
        results.append(self._validate_amounts(output, context, **kwargs))
        return results

    def _validate_invoice_numbers(self, output: str, context: CaseContext) -> GuardrailResult:
        """Validate that all invoice numbers in output exist in context."""
        # Extract invoice numbers from output using common patterns
        # Matches: INV-12345, INV12345, Invoice 12345, #12345, etc.
        # NOTE: Patterns must be restrictive to avoid false positives with garbage chars
        invoice_patterns = [
            r"INV[-\s]?(\d+)",  # INV-12345, INV 12345, INV12345
            r"Invoice\s*#?\s*(\d+)",  # Invoice 12345, Invoice #12345
            r"invoice\s+number\s*:?\s*([A-Za-z0-9][-A-Za-z0-9]+)",  # invoice number: ABC-123
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

    def _validate_amounts(self, output: str, context: CaseContext, **kwargs) -> GuardrailResult:
        """Validate that monetary amounts in output match context data."""
        skip_invoice_table = kwargs.get("skip_invoice_table", False)

        # Build set of valid amounts from obligations
        valid_amounts = set()
        for o in context.obligations:
            valid_amounts.add(o.amount_due)
            valid_amounts.add(o.original_amount)

        total_outstanding = sum(o.amount_due for o in context.obligations)
        valid_amounts.add(total_outstanding)

        # For follow-up drafts, also extract amounts from conversation history.
        # The debtor may have mentioned amounts (e.g., "We paid £10,000") that
        # the LLM legitimately echoes in the follow-up response.
        if skip_invoice_table and context.recent_messages:
            conversation_amounts = self._extract_conversation_amounts(context.recent_messages)
            valid_amounts.update(conversation_amounts)

        # Also add rounded versions (in case of formatting differences)
        valid_amounts_rounded = {round(a, 2) for a in valid_amounts}
        valid_amounts_int = {int(a) for a in valid_amounts}

        # Extract amounts from output
        found_amounts = []
        for pattern in AMOUNT_PATTERNS:
            matches = re.findall(pattern, output)
            for match in matches:
                cleaned = match.replace(",", "").replace(" ", "")
                try:
                    amount = float(cleaned)
                    found_amounts.append(amount)
                except ValueError:
                    continue

        if not found_amounts:
            return self._pass(
                message="No monetary amounts found in output",
                details={"total_outstanding": total_outstanding},
            )

        # Strict validation — every amount in prose must exist in context
        invalid_amounts = []
        for amount in found_amounts:
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

    @staticmethod
    def _extract_conversation_amounts(recent_messages: list) -> set:
        """Extract monetary amounts from conversation history.

        Sources (in priority order):
        1. Structured extracted fields (claimed_amount, disputed_amount, promise_amount)
           — most reliable, set by AI classifier
        2. Body snippet regex — fallback for amounts not captured by classifier
        """
        amounts = set()
        for msg in recent_messages:
            # Structured extracted amounts (from classifier)
            for field in ("promise_amount", "claimed_amount", "disputed_amount"):
                value = msg.get(field)
                if value is not None:
                    try:
                        amounts.add(float(value))
                    except (ValueError, TypeError):
                        pass

            # Regex fallback on body snippet
            snippet = msg.get("body_snippet", "") or ""
            for pattern in AMOUNT_PATTERNS:
                matches = re.findall(pattern, snippet)
                for match in matches:
                    cleaned = match.replace(",", "").replace(" ", "")
                    try:
                        amounts.add(float(cleaned))
                    except ValueError:
                        continue
        return amounts

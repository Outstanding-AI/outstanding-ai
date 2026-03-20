"""Factual Grounding Guardrail -- validate facts exist in context.

Ensure the LLM only references invoice numbers and monetary amounts
that actually exist in the case context.  This is the primary defense
against hallucinated financial data in collection emails.

CRITICAL severity -- blocks output on failure and triggers the
guardrail retry loop in the draft generator.
"""

import logging
import re

from src.api.models.requests import CaseContext

from .base import BaseGuardrail, GuardrailResult, GuardrailSeverity

logger = logging.getLogger(__name__)

# Reusable regex patterns to extract monetary amounts from prose.
# Covers symbol-prefixed (£1,500.00), code-suffixed (1500 GBP), and
# code-prefixed (GBP 1500) formats for GBP, USD, and EUR.
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
        """Validate factual grounding of the output.

        Run two sub-checks: invoice number validation and amount
        validation.  Closure-mode drafts skip both checks entirely
        since they contain no financial references.

        Args:
            output: AI-generated draft body text.
            context: Case context with obligations and recent messages.
            **kwargs: ``closure_mode`` (bool), ``skip_invoice_table``
                (bool).

        Returns:
            List of two GuardrailResult objects (invoice + amount).
        """
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
        """Validate that all invoice numbers in the output exist in context.

        Use both exact string matching and regex pattern matching to
        find invoice references.  Flexible matching allows numeric-only
        portions to match (e.g., "12345" matches "INV-12345").
        Obligations with null/empty invoice numbers are skipped.
        """
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
        """Validate that monetary amounts in the output match context data.

        Build a set of valid amounts from obligation ``amount_due`` and
        ``original_amount`` fields, plus the computed total.  For
        follow-up drafts (``skip_invoice_table=True``), also include
        amounts from conversation history so the LLM can legitimately
        echo debtor-mentioned figures.

        Every amount found in prose must exist in the valid set
        (with rounding tolerance of 0.01).
        """
        skip_invoice_table = kwargs.get("skip_invoice_table", False)

        # Build set of valid amounts from obligations
        # Normalize ALL to float to avoid Decimal/int/string comparison issues
        valid_amounts = set()
        for o in context.obligations:
            if o.amount_due is not None:
                valid_amounts.add(round(float(o.amount_due), 2))
            if o.original_amount is not None:
                valid_amounts.add(round(float(o.original_amount), 2))

        safe_dues = [float(o.amount_due) for o in context.obligations if o.amount_due is not None]
        total_outstanding = round(sum(safe_dues), 2)
        valid_amounts.add(total_outstanding)

        # Add per-party sub-totals (LLM often sums a subset of invoices)
        # and common rounding variants
        for a in list(valid_amounts):
            valid_amounts.add(round(a))  # integer rounding: 1179.34 → 1179
            valid_amounts.add(round(a, 0))  # same but explicit

        # For follow-up drafts, also extract amounts from conversation history.
        if skip_invoice_table and context.recent_messages:
            conversation_amounts = self._extract_conversation_amounts(context.recent_messages)
            valid_amounts.update(
                {round(float(a), 2) for a in conversation_amounts if a is not None}
            )

        # Deduplicated float set for comparison (22 == 22.0 == 22.00)
        valid_amounts_float = {round(float(a), 2) for a in valid_amounts if a is not None}
        valid_amounts_int = {int(a) for a in valid_amounts_float}

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

        # Validate — every amount in prose must exist in context (±1.00 tolerance)
        invalid_amounts = []
        for amount in found_amounts:
            rounded = round(amount, 2)
            is_valid = (
                rounded in valid_amounts_float
                or int(amount) in valid_amounts_int
                or any(abs(amount - valid) <= 1.00 for valid in valid_amounts_float)
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

        Use a two-tier extraction strategy:

        1. **Structured fields** (highest priority): Read
           ``claimed_amount``, ``disputed_amount``, ``promise_amount``
           from classifier-populated fields on each message.  These are
           the most reliable source.
        2. **Regex fallback**: Scan ``body_snippet`` text for currency
           patterns to catch amounts the classifier did not extract.

        Args:
            recent_messages: List of message dicts from
                ``context.recent_messages``.

        Returns:
            Set of float amounts found across all messages.
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

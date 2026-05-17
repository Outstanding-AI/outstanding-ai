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
from .lane_scope import _bare_digit_form, _normalize_invoice_ref

logger = logging.getLogger(__name__)

# Reusable regex patterns to extract monetary amounts from prose.
# Covers symbol-prefixed (£1,500.00), code-suffixed (1500 GBP), and
# code-prefixed (GBP 1500) formats for GBP, USD, and EUR.
AMOUNT_PATTERNS = [
    r"[£$€]\s*([\d,]+(?:\.\d{2})?)",  # £1,500.00
    r"([\d,]+(?:\.\d{2})?)\s*(?:GBP|USD|EUR)",  # 1500 GBP
    r"(?:GBP|USD|EUR)\s*([\d,]+(?:\.\d{2})?)",  # GBP 1500
]

CHASE_LANGUAGE_RE = re.compile(
    r"\b(pay|payment|settle|settlement|overdue|outstanding|owed|remit)\b|"
    r"\b(?:amount|balance|past)\s+due\b",
    re.IGNORECASE,
)


def _loose_alnum_pattern(normalized_ref: str) -> str:
    return r"[\W_]*".join(re.escape(ch) for ch in normalized_ref)


def _segment_mentions_invoice_ref(segment: str, invoice_ref: str) -> bool:
    normalized_ref = _normalize_invoice_ref(invoice_ref)
    if not normalized_ref:
        return False
    if re.search(
        rf"(?<![A-Z0-9]){_loose_alnum_pattern(normalized_ref)}(?![A-Z0-9])",
        segment,
        re.IGNORECASE,
    ):
        return True

    bare_digits = _bare_digit_form(invoice_ref)
    if bare_digits and re.search(rf"(?<!\d){re.escape(bare_digits)}(?!\d)", segment):
        return True
    return False


def _segments(text: str) -> list[str]:
    return [
        segment
        for segment in re.split(r"(?:<br\s*/?>|</p>|[.!?\n;])", text, flags=re.IGNORECASE)
        if segment
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
        results.append(self._validate_source_disputes_not_chased(output, context))
        results.append(self._validate_procurement_grounding(output, context))
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

        valid_invoice_refs = {_normalize_invoice_ref(inv) for inv in valid_invoices}
        valid_invoice_numbers = {
            _bare_digit_form(inv) for inv in valid_invoices if _bare_digit_form(inv)
        }

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
            normalized_found = _normalize_invoice_ref(found_inv_str)
            found_digits = _bare_digit_form(found_inv_str)
            is_valid = normalized_found in valid_invoice_refs or (
                bool(found_digits) and found_digits in valid_invoice_numbers
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
        (with rounding tolerance of 5.00).
        """
        skip_invoice_table = kwargs.get("skip_invoice_table", False)

        # Build set of valid amounts from obligations
        # Normalize ALL to float to avoid Decimal/int/string comparison issues
        valid_amounts = set()
        for o in context.obligations:
            if o.amount_due is not None:
                valid_amounts.add(round(float(o.amount_due), 2))
            if getattr(o, "amount_due_base", None) is not None:
                valid_amounts.add(round(float(o.amount_due_base), 2))
            if o.original_amount is not None:
                valid_amounts.add(round(float(o.original_amount), 2))
            if getattr(o, "original_amount_base", None) is not None:
                valid_amounts.add(round(float(o.original_amount_base), 2))

        safe_dues = [float(o.amount_due) for o in context.obligations if o.amount_due is not None]
        total_outstanding = round(sum(safe_dues), 2)
        valid_amounts.add(total_outstanding)

        # Add sum of original_amounts as valid (LLM may compute totals from either column)
        safe_originals = [
            float(o.original_amount) for o in context.obligations if o.original_amount is not None
        ]
        if safe_originals:
            valid_amounts.add(round(sum(safe_originals), 2))

        # Add per-party sub-totals (LLM often sums a subset of invoices)
        # and common rounding variants
        for a in list(valid_amounts):
            valid_amounts.add(round(a))  # integer rounding: 1179.34 → 1179
            valid_amounts.add(round(a, 0))  # same but explicit

        # For follow-up drafts, also extract amounts from conversation history.
        recent_messages = context.lane_recent_messages or context.recent_messages
        if skip_invoice_table and recent_messages:
            conversation_amounts = self._extract_conversation_amounts(recent_messages)
            valid_amounts.update(
                {round(float(a), 2) for a in conversation_amounts if a is not None}
            )

        # Manual touchpoints: amounts quoted by the AI from an operator's
        # phone-log notes ("you promised £500 by Friday") must not trigger a
        # hallucination flag. Apply on the same code path as conversation
        # history. Touches are typed Pydantic objects (TouchHistory) — pull
        # ``manual_notes`` directly.
        recent_touches = getattr(context, "recent_touches", None) or []
        manual_touch_amounts = self._extract_manual_touch_amounts(recent_touches)
        if manual_touch_amounts:
            valid_amounts.update(
                {round(float(a), 2) for a in manual_touch_amounts if a is not None}
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

        # Validate — every amount in prose must exist in context (±5.00 tolerance)
        invalid_amounts = []
        for amount in found_amounts:
            rounded = round(amount, 2)
            is_valid = (
                rounded in valid_amounts_float
                or int(amount) in valid_amounts_int
                or any(abs(amount - valid) <= 5.00 for valid in valid_amounts_float)
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

    def _validate_source_disputes_not_chased(
        self, output: str, context: CaseContext
    ) -> GuardrailResult:
        """Block payment asks against Sage-query/source-disputed obligations."""
        source_disputed = []
        for obligation in context.obligations:
            raw_query = str(getattr(obligation, "source_query_raw", None) or "").strip()
            if getattr(obligation, "is_source_disputed", False) or raw_query:
                source_disputed.append(obligation)

        if not source_disputed:
            return self._pass("No source-disputed obligations in context")

        if not CHASE_LANGUAGE_RE.search(output):
            return self._pass("No payment-chase language for source-disputed obligations")

        chased = []
        for obligation in source_disputed:
            invoice_ref = getattr(obligation, "invoice_number", None) or getattr(
                obligation, "document_no", None
            )
            if not invoice_ref:
                continue
            if self._chases_invoice_ref(output, str(invoice_ref)):
                chased.append(str(invoice_ref))

        if chased:
            return self._fail(
                message=f"Draft asks for payment on source-disputed obligations: {chased}",
                expected="Source-disputed invoices may be labelled excluded/disputed, not chased",
                found=chased,
                details={"source_disputed_invoice_refs": chased},
            )

        return self._pass("Source-disputed obligations were not chased")

    @staticmethod
    def _chases_invoice_ref(output: str, invoice_ref: str) -> bool:
        """Return True only when chase words occur in the same sentence as the ref."""
        for segment in _segments(output):
            if _segment_mentions_invoice_ref(segment, invoice_ref) and CHASE_LANGUAGE_RE.search(
                segment
            ):
                return True
        return False

    def _validate_procurement_grounding(self, output: str, context: CaseContext) -> GuardrailResult:
        """Allow PO/POD claims only when verified procurement evidence exists."""
        mentions_po = bool(
            re.search(
                r"\b(purchase order|po\s*(?:number|ref|reference|#))\b", output, re.IGNORECASE
            )
        )
        mentions_pod = bool(
            re.search(
                r"\b(proof of delivery|pod\b|delivery note|goods received)\b",
                output,
                re.IGNORECASE,
            )
        )
        if not mentions_po and not mentions_pod:
            return self._pass("No procurement claims found")

        has_verified_po = any(
            getattr(obligation, "has_verified_purchase_order", False)
            for obligation in context.obligations
        )
        has_verified_pod = any(
            getattr(obligation, "has_verified_pod", False) for obligation in context.obligations
        )

        failures = []
        if mentions_po and not has_verified_po:
            failures.append("purchase_order")
        if mentions_pod and not has_verified_pod:
            failures.append("proof_of_delivery")

        if failures:
            return self._fail(
                message="Draft claims unverified procurement evidence: " + ", ".join(failures),
                expected="PO/POD wording only when verified flags are true",
                found=failures,
                details={
                    "mentions_po": mentions_po,
                    "mentions_pod": mentions_pod,
                    "has_verified_purchase_order": has_verified_po,
                    "has_verified_pod": has_verified_pod,
                },
            )

        return self._pass("Procurement claims are grounded in verified context")

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

    @staticmethod
    def _extract_manual_touch_amounts(recent_touches: list) -> set:
        """Extract monetary amounts from operator-logged manual touchpoint notes.

        Mirrors ``_extract_conversation_amounts`` but operates on
        ``TouchHistory`` Pydantic objects rather than message dicts. Skips
        non-manual rows (email touches have no ``manual_notes``). Amounts
        found here join the validity set so the AI quoting a verbal
        commitment from a phone call ("you promised £500 by Friday") does
        not trigger a factual-grounding failure.
        """
        amounts = set()
        for touch in recent_touches:
            if getattr(touch, "touch_type", None) != "manual_log":
                continue
            notes = getattr(touch, "manual_notes", None) or ""
            if not notes:
                continue
            for pattern in AMOUNT_PATTERNS:
                matches = re.findall(pattern, notes)
                for match in matches:
                    cleaned = match.replace(",", "").replace(" ", "")
                    try:
                        amounts.add(float(cleaned))
                    except ValueError:
                        continue
        return amounts

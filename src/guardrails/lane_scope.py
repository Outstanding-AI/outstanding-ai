"""Candidate-scope guardrail for invoice references and candidate totals."""

import re
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from .base import BaseGuardrail, GuardrailSeverity

INVOICE_PATTERNS = [
    re.compile(r"INV[-\s]?(\d+)", re.IGNORECASE),
    re.compile(r"Invoice\s*#?\s*(\d+)", re.IGNORECASE),
]
TOTAL_PATTERNS = [
    re.compile(
        r"total\s+(?:outstanding|amount|due|owed)(?:\s+(?:is|of))?\s*:?\s*[£$€]?\s*([\d,]+(?:\.\d{2})?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"combined\s+(?:balance|amount)\s+(?:of|is)\s+[£$€]?\s*([\d,]+(?:\.\d{2})?)", re.IGNORECASE
    ),
    re.compile(r"subtotal\s+(?:of|is)\s+[£$€]?\s*([\d,]+(?:\.\d{2})?)", re.IGNORECASE),
]


def _q(value: Any) -> Decimal:
    return Decimal(str(value or 0)).quantize(Decimal("0.01"), ROUND_HALF_UP)


def _normalize_invoice_ref(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def _invoice_ref_variants(value: Any) -> set[str]:
    """Return the set of normalized variants for an invoice reference.

    Always includes the alphanumeric-normalized form. The bare-digit form is
    included only when the normalized form is itself digit-only — i.e. the
    invoice has no alpha prefix. For prefixed invoices (e.g. "INV-12345"), the
    bare-digit fallback is **not** put into the variant set: prefix collisions
    (e.g. "1234" vs. "12345") would otherwise survive set membership checks
    where another invoice's digits happened to be a prefix of this invoice's
    digits. Bare-digit body extractions still match prefixed cohort entries
    via the length-equal lookup in :class:`LaneScopeGuardrail`.
    """
    normalized = _normalize_invoice_ref(value)
    if not normalized:
        return set()
    variants = {normalized}
    digits = "".join(ch for ch in normalized if ch.isdigit())
    # Only add the bare-digit variant when the normalized form is itself
    # digit-only. Prefixed forms route through the length-equal bare-digit
    # bucket built by callers to avoid prefix collisions.
    if digits and digits == normalized:
        variants.add(digits)
    return variants


def _bare_digit_form(value: Any) -> str:
    """Return the digit-only form of a normalized invoice reference, or ''."""
    normalized = _normalize_invoice_ref(value)
    if not normalized:
        return ""
    digits = "".join(ch for ch in normalized if ch.isdigit())
    return digits


class LaneScopeGuardrail(BaseGuardrail):
    """Block drafts that escape the current candidate scope or total."""

    def __init__(self):
        super().__init__(name="lane_scope", severity=GuardrailSeverity.CRITICAL)

    def validate(self, output: str, context: Any, **kwargs) -> list:
        lane = kwargs.get("lane_context") or getattr(context, "lane", None) or {}
        candidate_refs = kwargs.get("candidate_invoice_refs") or []
        if not lane and not candidate_refs:
            return [self._pass("No lane context supplied")]

        scoped_refs = candidate_refs or lane.get("invoice_refs") or []
        candidate_invoices = set()
        # Bare-digit lookup keyed by length: digits string -> set of normalized forms.
        # A bare-digit body extraction matches a prefixed cohort entry only when the
        # body digit string is **exactly equal** to the cohort entry's bare digits;
        # set/dict key equality enforces both value- and length-equality, which
        # eliminates prefix collisions ("1234" cannot match "12345").
        cohort_bare_digits: dict[str, set[str]] = {}
        for ref in scoped_refs:
            candidate_invoices.update(_invoice_ref_variants(ref))
            normalized = _normalize_invoice_ref(ref)
            digits = _bare_digit_form(ref)
            if digits and digits != normalized:
                cohort_bare_digits.setdefault(digits, set()).add(normalized)
        if not candidate_invoices and not cohort_bare_digits:
            return [self._pass("Candidate scope has no invoice refs")]

        blocked_ids = {
            str(value) for value in (getattr(context, "blocked_obligation_ids", None) or [])
        }
        blocked_invoice_refs = set()
        blocked_bare_digits: set[str] = set()
        invoice_to_internal_id = {}
        bare_digit_to_internal_id: dict[str, str] = {}
        for obligation in getattr(context, "obligations", None) or []:
            obligation_id = str(getattr(obligation, "id", "") or "")
            invoice_ref = getattr(obligation, "invoice_number", "") or ""
            for variant in _invoice_ref_variants(invoice_ref):
                invoice_to_internal_id[variant] = obligation_id
            normalized = _normalize_invoice_ref(invoice_ref)
            digits = _bare_digit_form(invoice_ref)
            if digits and digits != normalized:
                bare_digit_to_internal_id[digits] = obligation_id
            source_query_raw = str(getattr(obligation, "source_query_raw", None) or "").strip()
            if (
                obligation_id in blocked_ids
                or getattr(obligation, "is_source_disputed", False)
                or source_query_raw
                or getattr(obligation, "is_sendable", None) is False
                or getattr(obligation, "is_chase_eligible", None) is False
            ):
                blocked_invoice_refs.update(_invoice_ref_variants(invoice_ref))
                if digits and digits != normalized:
                    blocked_bare_digits.add(digits)
        lane_total = _q(lane.get("outstanding_amount") or 0)

        for pattern in INVOICE_PATTERNS:
            for match in pattern.findall(output):
                invoice_ref = _normalize_invoice_ref(match)
                in_scope = invoice_ref in candidate_invoices or (
                    invoice_ref.isdigit() and invoice_ref in cohort_bare_digits
                )
                if not in_scope:
                    return [
                        self._fail(
                            f"Draft references invoice {invoice_ref} outside candidate scope"
                        )
                    ]
                if (
                    invoice_to_internal_id.get(invoice_ref) in blocked_ids
                    or bare_digit_to_internal_id.get(invoice_ref) in blocked_ids
                    or invoice_ref in blocked_invoice_refs
                    or (invoice_ref.isdigit() and invoice_ref in blocked_bare_digits)
                ):
                    return [self._fail(f"Draft references blocked obligation {invoice_ref}")]

        for pattern in TOTAL_PATTERNS:
            for match in pattern.findall(output):
                if _q(match.replace(",", "")) != lane_total:
                    return [
                        self._fail(f"Stated total {match} does not match lane total {lane_total}")
                    ]

        return [self._pass("Candidate scope validated")]

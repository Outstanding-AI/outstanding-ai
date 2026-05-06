"""Lane-cohort guardrail for invoice references and lane totals."""

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
    normalized = _normalize_invoice_ref(value)
    variants = {normalized} if normalized else set()
    digits = "".join(ch for ch in normalized if ch.isdigit())
    if digits:
        variants.add(digits)
    return variants


class LaneScopeGuardrail(BaseGuardrail):
    """Block drafts that escape the current lane cohort or total."""

    def __init__(self):
        super().__init__(name="lane_scope", severity=GuardrailSeverity.CRITICAL)

    def validate(self, output: str, context: Any, **kwargs) -> list:
        lane = kwargs.get("lane_context") or getattr(context, "lane", None) or {}
        candidate_refs = kwargs.get("candidate_invoice_refs") or []
        if not lane and not candidate_refs:
            return [self._pass("No lane context supplied")]

        scoped_refs = candidate_refs or lane.get("invoice_refs") or []
        cohort_invoices = set()
        for ref in scoped_refs:
            cohort_invoices.update(_invoice_ref_variants(ref))
        if not cohort_invoices:
            return [self._pass("Lane has no scoped invoice refs")]

        blocked_ids = {
            str(value) for value in (getattr(context, "blocked_obligation_ids", None) or [])
        }
        blocked_invoice_refs = set()
        invoice_to_internal_id = {}
        for obligation in getattr(context, "obligations", None) or []:
            obligation_id = str(getattr(obligation, "id", "") or "")
            invoice_ref = getattr(obligation, "invoice_number", "") or ""
            for variant in _invoice_ref_variants(invoice_ref):
                invoice_to_internal_id[variant] = obligation_id
            source_query_raw = str(getattr(obligation, "source_query_raw", None) or "").strip()
            if (
                obligation_id in blocked_ids
                or getattr(obligation, "is_source_disputed", False)
                or source_query_raw
                or getattr(obligation, "is_sendable", None) is False
                or getattr(obligation, "is_chase_eligible", None) is False
            ):
                blocked_invoice_refs.update(_invoice_ref_variants(invoice_ref))
        lane_total = _q(lane.get("outstanding_amount") or 0)

        for pattern in INVOICE_PATTERNS:
            for match in pattern.findall(output):
                invoice_ref = _normalize_invoice_ref(match)
                if invoice_ref not in cohort_invoices:
                    return [
                        self._fail(f"Draft references invoice {invoice_ref} outside lane cohort")
                    ]
                if (
                    invoice_to_internal_id.get(invoice_ref) in blocked_ids
                    or invoice_ref in blocked_invoice_refs
                ):
                    return [self._fail(f"Draft references blocked obligation {invoice_ref}")]

        for pattern in TOTAL_PATTERNS:
            for match in pattern.findall(output):
                if _q(match.replace(",", "")) != lane_total:
                    return [
                        self._fail(f"Stated total {match} does not match lane total {lane_total}")
                    ]

        return [self._pass("Lane scope validated")]

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


class LaneScopeGuardrail(BaseGuardrail):
    """Block drafts that escape the current lane cohort or total."""

    def __init__(self):
        super().__init__(name="lane_scope", severity=GuardrailSeverity.CRITICAL)

    def validate(self, output: str, context: Any, **kwargs) -> list:
        lane = kwargs.get("lane_context") or getattr(context, "lane", None) or {}
        if not lane:
            return [self._pass("No lane context supplied")]

        cohort_invoices = {str(ref).upper() for ref in (lane.get("invoice_refs") or [])}
        if not cohort_invoices:
            return [self._pass("Lane has no scoped invoice refs")]

        blocked_ids = {
            str(value) for value in (getattr(context, "blocked_obligation_ids", None) or [])
        }
        if getattr(context, "schema_version", 1) == 2:
            invoice_to_internal_id = {
                (getattr(obligation, "invoice_number", "") or "").upper(): str(
                    getattr(obligation, "id", "") or ""
                )
                for obligation in getattr(context, "obligations", None) or []
            }
        else:
            invoice_to_internal_id = {
                (getattr(obligation, "invoice_number", "") or "").upper(): str(
                    getattr(obligation, "sage_id", "") or ""
                )
                for obligation in getattr(context, "obligations", None) or []
            }
        lane_total = _q(lane.get("outstanding_amount") or 0)

        for pattern in INVOICE_PATTERNS:
            for match in pattern.findall(output):
                invoice_ref = str(match).upper()
                if invoice_ref not in cohort_invoices:
                    return [
                        self._fail(f"Draft references invoice {invoice_ref} outside lane cohort")
                    ]
                if invoice_to_internal_id.get(invoice_ref) in blocked_ids:
                    return [self._fail(f"Draft references blocked obligation {invoice_ref}")]

        for pattern in TOTAL_PATTERNS:
            for match in pattern.findall(output):
                if _q(match.replace(",", "")) != lane_total:
                    return [
                        self._fail(f"Stated total {match} does not match lane total {lane_total}")
                    ]

        return [self._pass("Lane scope validated")]

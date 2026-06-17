"""Promise-reply guardrail.

Blocks reply drafts that ask for a payment date/status when the debtor has
already provided a promise-to-pay date in the classified inbound reply.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from src.api.models.requests import CaseContext

from .base import BaseGuardrail, GuardrailResult, GuardrailSeverity

_TERMINAL_PROMISE_OUTCOMES = {
    "broken",
    "cancelled",
    "canceled",
    "clear",
    "cleared",
    "expired",
    "expired_unfulfilled",
    "fulfilled",
    "kept",
    "paid",
    "settled",
}

_MONTH_NAMES = {
    1: ("jan", "january"),
    2: ("feb", "february"),
    3: ("mar", "march"),
    4: ("apr", "april"),
    5: ("may",),
    6: ("jun", "june"),
    7: ("jul", "july"),
    8: ("aug", "august"),
    9: ("sep", "sept", "september"),
    10: ("oct", "october"),
    11: ("nov", "november"),
    12: ("dec", "december"),
}

_PAYMENT_TIMING_ASK_PATTERNS = (
    re.compile(
        r"\b(?:could|can|would)\s+you\b.{0,120}"
        r"\b(?:payment|settlement|funds?|remittance)\b.{0,80}"
        r"\b(?:date|timeline|timing|eta|status|update|expected|expect)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(?:provide|share|send|advise|confirm|give|let\s+(?:us|me)\s+know)\b.{0,120}"
        r"\b(?:payment|settlement|funds?|remittance)\b.{0,80}"
        r"\b(?:date|timeline|timing|eta|status|update|expected|expect)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(?:provide|share|send|advise|confirm|give|let\s+(?:us|me)\s+know)\b.{0,120}"
        r"\b(?:date|timeline|timing|eta|status|update)\b.{0,80}"
        r"\b(?:payment|settlement|funds?|remittance)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(?:when|what\s+date)\b.{0,80}"
        r"\b(?:payment|settlement|funds?|remittance)\b",
        re.IGNORECASE | re.DOTALL,
    ),
)


def _as_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return {}


def _parse_iso_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _compact_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip().lower()


class PromiseReplyGuardrail(BaseGuardrail):
    """Block incoherent promise-to-pay reply drafts.

    This guardrail is intentionally narrow: it only applies to reply drafts
    triggered by ``PROMISE_TO_PAY`` or reply-mode drafts with a current inbound
    promise fact. It does not interfere with broken-promise follow-ups where
    asking for an update may be appropriate.
    """

    def __init__(self):
        super().__init__(
            name="promise_reply",
            severity=GuardrailSeverity.HIGH,
        )

    def validate(self, output: str, context: CaseContext, **kwargs) -> list[GuardrailResult]:
        trigger = str(kwargs.get("trigger_classification") or "").upper()
        mail_mode = str(kwargs.get("mail_mode") or "").lower()
        promise_dates = self._extract_current_promise_dates(context)

        if not promise_dates:
            return [self._pass(message="No current promise date to validate")]

        is_reply_ack = trigger == "PROMISE_TO_PAY" or mail_mode in {
            "reply_ack",
            "handoff_reply",
            "resolution_follow_up",
        }
        if not is_reply_ack:
            return [self._pass(message="Promise date present outside reply-ack mode")]

        results: list[GuardrailResult] = []
        if self._asks_for_payment_timing(output):
            results.append(
                self._fail(
                    message=(
                        "Draft asks for payment date/status even though the debtor already "
                        "provided a promise-to-pay date"
                    ),
                    expected="Acknowledge the known promised payment date",
                    found=self._matched_timing_ask(output),
                    details={
                        "promise_dates": [d.isoformat() for d in promise_dates],
                        "trigger_classification": trigger,
                        "mail_mode": mail_mode,
                    },
                )
            )

        if not any(self._mentions_date(output, promised_date) for promised_date in promise_dates):
            results.append(
                self._fail(
                    message="Draft does not mention the known promised payment date",
                    expected=", ".join(d.isoformat() for d in promise_dates),
                    found="No supported date rendering found in output",
                    details={
                        "promise_dates": [d.isoformat() for d in promise_dates],
                        "trigger_classification": trigger,
                        "mail_mode": mail_mode,
                    },
                )
            )

        if results:
            return results
        return [
            self._pass(
                message="Promise reply acknowledges the debtor's known payment date",
                details={"promise_dates": [d.isoformat() for d in promise_dates]},
            )
        ]

    def _extract_current_promise_dates(self, context: CaseContext) -> list[date]:
        dates: list[date] = []
        seen: set[date] = set()

        def add(value: Any) -> None:
            parsed = _parse_iso_date(value)
            if parsed and parsed not in seen:
                seen.add(parsed)
                dates.append(parsed)

        for msg in (getattr(context, "lane_recent_messages", None) or []) + (
            getattr(context, "recent_messages", None) or []
        ):
            if not isinstance(msg, dict):
                continue
            direction = str(msg.get("direction") or "").lower()
            classification = str(msg.get("classification") or "").upper()
            if direction == "inbound" and (
                classification == "PROMISE_TO_PAY" or msg.get("promise_date")
            ):
                add(msg.get("promise_date"))

        for promise in getattr(context, "promises", []) or []:
            data = _as_dict(promise)
            outcome = str(data.get("outcome") or "pending").lower()
            if outcome not in _TERMINAL_PROMISE_OUTCOMES:
                add(data.get("promise_date"))

        return dates

    def _asks_for_payment_timing(self, output: str) -> bool:
        return self._matched_timing_ask(output) is not None

    def _matched_timing_ask(self, output: str) -> str | None:
        for pattern in _PAYMENT_TIMING_ASK_PATTERNS:
            match = pattern.search(output or "")
            if match:
                return _compact_text(match.group(0))[:180]
        return None

    def _mentions_date(self, output: str, promised_date: date) -> bool:
        text = _compact_text(output)
        if promised_date.isoformat() in text:
            return True

        day = promised_date.day
        month = promised_date.month
        year = promised_date.year
        numeric_forms = {
            f"{day}/{month}/{year}",
            f"{day:02d}/{month:02d}/{year}",
            f"{month}/{day}/{year}",
            f"{month:02d}/{day:02d}/{year}",
            f"{day}/{month}",
            f"{day:02d}/{month:02d}",
        }
        if any(form in text for form in numeric_forms):
            return True

        ordinal_suffix = "th"
        if day % 10 == 1 and day % 100 != 11:
            ordinal_suffix = "st"
        elif day % 10 == 2 and day % 100 != 12:
            ordinal_suffix = "nd"
        elif day % 10 == 3 and day % 100 != 13:
            ordinal_suffix = "rd"

        day_forms = {str(day), f"{day:02d}", f"{day}{ordinal_suffix}"}
        month_forms = _MONTH_NAMES[month]
        for month_text in month_forms:
            for day_text in day_forms:
                if re.search(rf"\b{re.escape(month_text)}\.?\s+{day_text}\b", text):
                    return True
                if re.search(rf"\b{day_text}\s+{re.escape(month_text)}\.?\b", text):
                    return True

        return False

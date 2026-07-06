"""Tone guardrail.

Validates that draft generation received an explicit runtime-selected tone.
The lane scheduler now chooses one exact tone slot per push; AI must honor
that concrete tone rather than infer a range.
"""

import logging
import re
from typing import Any

from .base import BaseGuardrail, GuardrailResult, GuardrailSeverity

logger = logging.getLogger(__name__)


class ToneClampingGuardrail(BaseGuardrail):
    """Validate that a concrete runtime-selected tone was provided and honored."""

    def __init__(self):
        super().__init__(name="tone_clamping", severity=GuardrailSeverity.HIGH)

    def validate(self, output: str, context: Any, **kwargs) -> list[GuardrailResult]:
        """Validate tone presence for the current draft request.

        Args:
            output: The AI-generated draft (not used by this guardrail)
            context: CaseContext
            **kwargs: Must include 'tone' and may include 'escalation_level'
        """
        tone = kwargs.get("tone", "professional")
        escalation_level = kwargs.get("escalation_level")

        if not tone:
            return [
                self._fail(
                    "Missing explicit runtime-selected tone for draft generation.",
                    expected="non-empty tone",
                    found=tone,
                    details={"tone": tone, "level": escalation_level},
                )
            ]

        tone_key = str(tone).strip().lower()
        body = _strip_quoted_reply_text(str(output or "")).lower()
        authorized_policies = (
            kwargs.get("authorized_policies") or getattr(context, "authorized_policies", None) or {}
        )
        legal_escalation_enabled = bool(authorized_policies.get("legal_escalation_enabled"))
        legal_pressure_phrases = [
            "legal action",
            "legal proceedings",
            "legal team",
            "legal referral",
            "refer this matter",
            "refer the matter",
            "account suspension",
            "final notice",
        ]
        if not legal_escalation_enabled:
            found_legal_pressure = [phrase for phrase in legal_pressure_phrases if phrase in body]
            if found_legal_pressure:
                return [
                    self._fail(
                        "Draft contains legal/escalation pressure without policy authorization.",
                        expected="operational follow-up wording unless legal escalation is authorized",
                        found=", ".join(found_legal_pressure),
                        details={
                            "tone": tone_key,
                            "level": escalation_level,
                            "legal_escalation_enabled": legal_escalation_enabled,
                        },
                    )
                ]
        if tone_key == "acknowledgement":
            pressure_phrases = [
                "please pay",
                "make payment",
                "payment is due",
                "pay immediately",
                "settle the balance",
                "settle this balance",
                "overdue balance",
                "final notice",
                "legal action",
                "legal proceedings",
                "account suspension",
                "within 7 days",
                "within seven days",
            ]
            found_pressure = [phrase for phrase in pressure_phrases if phrase in body]
            if found_pressure:
                return [
                    self._fail(
                        "Acknowledgement tone contains collection pressure.",
                        expected="receipt/thanks/confirmation without payment pressure",
                        found=", ".join(found_pressure),
                        details={"tone": tone_key, "level": escalation_level},
                    )
                ]

            acknowledgement_cues = [
                "thank",
                "thanks",
                "received",
                "receipt",
                "acknowledge",
                "confirm",
            ]
            if not any(cue in body for cue in acknowledgement_cues):
                return [
                    self._fail(
                        "Acknowledgement tone does not clearly acknowledge the debtor's message.",
                        expected="an acknowledgement cue such as thanks, received, or confirmed",
                        found=output[:160] if output else "",
                        details={"tone": tone_key, "level": escalation_level},
                    )
                ]

        if tone_key in {"friendly_reminder", "concerned_inquiry"}:
            pressure_phrases = [
                "legal action",
                "legal proceedings",
                "final notice",
                "account suspension",
            ]
            found_pressure = [phrase for phrase in pressure_phrases if phrase in body]
            if found_pressure:
                return [
                    self._fail(
                        f"{tone_key} tone includes escalation pressure.",
                        expected="soft reminder wording",
                        found=", ".join(found_pressure),
                        details={"tone": tone_key, "level": escalation_level},
                    )
                ]

        return [
            self._pass(
                f"Explicit tone '{tone}' supplied and honored for level {escalation_level}",
                details={
                    "tone": tone_key,
                    "level": escalation_level,
                },
            )
        ]


def _strip_quoted_reply_text(output: str) -> str:
    """Remove common quoted-reply blocks before tone phrase checks."""
    without_blockquotes = re.sub(
        r"<blockquote\b[^>]*>.*?</blockquote>",
        " ",
        output,
        flags=re.IGNORECASE | re.DOTALL,
    )
    kept_lines = [
        line for line in without_blockquotes.splitlines() if not line.lstrip().startswith(">")
    ]
    return "\n".join(kept_lines)

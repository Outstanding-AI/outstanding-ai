"""Tone clamping guardrail.

Validates that the tone used for draft generation matches the escalation
level's allowed tones from the v2 protocol. This is a safety net — the
backend clamp_tone() should have already clamped the tone, but this
catches mismatches.

Severity: HIGH — a tone mismatch means the email may be too aggressive
or too soft for the escalation level.
"""

import logging
from typing import Any

from .base import BaseGuardrail, GuardrailResult, GuardrailSeverity

logger = logging.getLogger(__name__)


class ToneClampingGuardrail(BaseGuardrail):
    """Validate tone matches escalation level's allowed_tones.

    This guardrail runs as part of the 7-guardrail pipeline. It checks
    that the tone parameter matches the protocol v2 allowed_tones list
    for the current escalation level.

    If allowed_tones is not provided (legacy v1 protocol or missing),
    the guardrail passes unconditionally.
    """

    def __init__(self):
        super().__init__(name="tone_clamping", severity=GuardrailSeverity.HIGH)

    def validate(self, output: str, context: Any, **kwargs) -> list[GuardrailResult]:
        """Validate tone against allowed_tones for the escalation level.

        Args:
            output: The AI-generated draft (not used by this guardrail)
            context: CaseContext
            **kwargs: Must include 'tone' and optionally 'allowed_tones', 'escalation_level'
        """
        allowed_tones = kwargs.get("allowed_tones")
        tone = kwargs.get("tone", "professional")
        escalation_level = kwargs.get("escalation_level")

        # No constraint = all tones allowed (v1 protocol or unconfigured)
        if not allowed_tones:
            return [
                self._pass("No tone constraints configured (v1 protocol or unconfigured level)")
            ]

        if tone in allowed_tones:
            return [
                self._pass(
                    f"Tone '{tone}' is within allowed tones {allowed_tones} for level {escalation_level}",
                    details={
                        "tone": tone,
                        "allowed_tones": allowed_tones,
                        "level": escalation_level,
                    },
                )
            ]

        return [
            self._fail(
                f"Tone '{tone}' is not in allowed tones {allowed_tones} for escalation level {escalation_level}. "
                f"Backend clamp_tone() should have caught this — possible bug.",
                expected=allowed_tones,
                found=tone,
                details={"tone": tone, "allowed_tones": allowed_tones, "level": escalation_level},
            )
        ]

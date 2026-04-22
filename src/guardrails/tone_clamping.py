"""Tone guardrail.

Validates that draft generation received an explicit runtime-selected tone.
The lane scheduler now chooses one exact tone slot per push; AI must honor
that concrete tone rather than infer a range.
"""

import logging
from typing import Any

from .base import BaseGuardrail, GuardrailResult, GuardrailSeverity

logger = logging.getLogger(__name__)


class ToneClampingGuardrail(BaseGuardrail):
    """Validate that a concrete runtime-selected tone was provided."""

    def __init__(self):
        super().__init__(name="tone_clamping", severity=GuardrailSeverity.MEDIUM)

    def validate(self, output: str, context: Any, **kwargs) -> list[GuardrailResult]:
        """Validate tone presence for the current draft request.

        Args:
            output: The AI-generated draft (not used by this guardrail)
            context: CaseContext
            **kwargs: Must include 'tone' and may include 'escalation_level'
        """
        tone = kwargs.get("tone", "professional")
        escalation_level = kwargs.get("escalation_level")

        if tone:
            return [
                self._pass(
                    f"Explicit tone '{tone}' supplied for level {escalation_level}",
                    details={
                        "tone": tone,
                        "level": escalation_level,
                    },
                )
            ]

        return [
            self._fail(
                "Missing explicit runtime-selected tone for draft generation.",
                expected="non-empty tone",
                found=tone,
                details={"tone": tone, "level": escalation_level},
            )
        ]

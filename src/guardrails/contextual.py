"""Contextual Coherence Guardrail -- validate situational awareness.

Check that the AI output is contextually appropriate given the party's
current state (active dispute, hardship indication, broken promise
history).  This guardrail catches tone-deaf outputs that ignore
important context signals.

LOW severity -- failures are logged but never block output, since
phrase-based detection is heuristic and may produce false positives.
"""

import logging

from src.api.models.requests import CaseContext

from .base import BaseGuardrail, GuardrailResult, GuardrailSeverity

logger = logging.getLogger(__name__)


class ContextualCoherenceGuardrail(BaseGuardrail):
    """Validate situational awareness of AI-generated draft text.

    LOW severity -- failures are logged but never block output, since
    detection uses phrase-matching heuristics that may produce false
    positives.

    Conditional checks (only run when the relevant flag is set):
    1. Active dispute: no payment demands without dispute acknowledgment.
    2. Hardship indicated: no harsh language without empathetic phrasing.
    3. Broken promises >= 2: history-referencing language expected.

    When no special conditions apply, a single pass result is returned.
    """

    def __init__(self):
        super().__init__(
            name="contextual_coherence",
            severity=GuardrailSeverity.LOW,  # Log only, don't block
        )

    def validate(self, output: str, context: CaseContext, **kwargs) -> list[GuardrailResult]:
        """Validate contextual coherence of the output.

        Conditionally run sub-checks based on context flags:
        - Active dispute: check for inappropriate payment demands.
        - Hardship indicated: check for harsh language without empathy.
        - Broken promises >= 2: check for history acknowledgment.

        If no special conditions apply, return a single pass result.

        Args:
            output: AI-generated draft body text.
            context: Case context with dispute/hardship/promise flags.
            **kwargs: Unused.

        Returns:
            List of GuardrailResult objects (one per applicable check).
        """
        results = []
        active_dispute = (
            context.lane_active_dispute
            if getattr(context, "lane_active_dispute", None) is not None
            else context.active_dispute
        )
        broken_promises_count = (
            context.lane_broken_promises_count
            if getattr(context, "lane_broken_promises_count", None) is not None
            else context.broken_promises_count
        )

        # Check dispute handling
        if active_dispute:
            results.append(self._validate_dispute_awareness(output, context))

        # Check hardship handling
        if context.hardship_indicated:
            results.append(self._validate_hardship_tone(output, context))

        # Check broken promise awareness
        if broken_promises_count > 0:
            results.append(self._validate_promise_awareness(output, context))

        # If no special conditions, just pass
        if not results:
            results.append(self._pass(message="No special context conditions to validate"))

        return results

    def _validate_dispute_awareness(self, output: str, context: CaseContext) -> GuardrailResult:
        """Validate that the output respects an active dispute.

        Fail if the draft contains payment-demand language without
        also acknowledging the dispute.  Fail if the dispute is not
        acknowledged at all.
        """
        output_lower = output.lower()

        # Phrases that suggest payment demand (inappropriate during dispute)
        demand_phrases = [
            "pay immediately",
            "pay now",
            "immediate payment",
            "pay in full",
            "demand payment",
            "must pay",
            "required to pay",
            "failure to pay will result",
            "legal action",
            "collection agency",
        ]

        # Phrases that acknowledge dispute (appropriate)
        dispute_phrases = [
            "dispute",
            "under review",
            "investigating",
            "looking into",
            "resolve",
            "concern",
            "issue",
        ]

        # Check for inappropriate demand language
        found_demands = [phrase for phrase in demand_phrases if phrase in output_lower]
        acknowledges_dispute = any(phrase in output_lower for phrase in dispute_phrases)

        if found_demands and not acknowledges_dispute:
            return self._fail(
                message="Output demands payment during active dispute without acknowledgment",
                expected="Acknowledge dispute, avoid payment demands",
                found=found_demands,
                details={
                    "demand_phrases_found": found_demands,
                    "dispute_acknowledged": acknowledges_dispute,
                    "active_dispute": True,
                },
            )

        if not acknowledges_dispute:
            return self._fail(
                message="Output does not acknowledge active dispute",
                expected="Reference to dispute or investigation",
                found="No dispute acknowledgment",
                details={"active_dispute": True, "dispute_acknowledged": False},
            )

        return self._pass(
            message="Output appropriately handles dispute context",
            details={"dispute_acknowledged": acknowledges_dispute},
        )

    def _validate_hardship_tone(self, output: str, context: CaseContext) -> GuardrailResult:
        """Validate that the output uses empathetic tone for hardship.

        Fail if harsh/demanding phrases appear without any empathetic
        language.  Also fail if no empathetic language is detected at
        all, even without harsh phrases.
        """
        output_lower = output.lower()

        # Harsh/demanding phrases (inappropriate for hardship)
        harsh_phrases = [
            "failure to pay",
            "will be forced",
            "no choice but",
            "legal consequences",
            "must pay immediately",
            "demand",
            "threaten",
        ]

        # Empathetic phrases (appropriate for hardship)
        empathetic_phrases = [
            "understand",
            "difficult",
            "challenging",
            "work with you",
            "payment plan",
            "options",
            "help",
            "support",
            "flexibility",
            "circumstances",
        ]

        found_harsh = [phrase for phrase in harsh_phrases if phrase in output_lower]
        found_empathetic = [phrase for phrase in empathetic_phrases if phrase in output_lower]

        # Fail if harsh without empathy
        if found_harsh and not found_empathetic:
            return self._fail(
                message="Output uses harsh tone for hardship case",
                expected="Empathetic language, payment options",
                found=found_harsh,
                details={
                    "harsh_phrases_found": found_harsh,
                    "empathetic_phrases_found": found_empathetic,
                    "hardship_indicated": True,
                },
            )

        # Warn if no empathetic language at all
        if not found_empathetic:
            return self._fail(
                message="Output lacks empathetic language for hardship case",
                expected="Understanding tone, payment options",
                found="No empathetic phrases detected",
                details={"hardship_indicated": True, "empathetic_count": 0},
            )

        return self._pass(
            message="Output uses appropriate tone for hardship case",
            details={"empathetic_phrases": found_empathetic},
        )

    def _validate_promise_awareness(self, output: str, context: CaseContext) -> GuardrailResult:
        """Validate that the output acknowledges broken promise history.

        Only triggers when ``broken_promises_count >= 2``.  Check for
        history-referencing phrases (e.g., "previous", "again",
        "commitment") in the draft text.
        """
        output_lower = output.lower()

        # If multiple broken promises, output should acknowledge history
        broken_promises_count = (
            context.lane_broken_promises_count
            if getattr(context, "lane_broken_promises_count", None) is not None
            else context.broken_promises_count
        )

        if broken_promises_count >= 2:
            history_phrases = [
                "previous",
                "history",
                "past",
                "again",
                "before",
                "commitment",
                "promise",
                "assured",
            ]

            acknowledges_history = any(phrase in output_lower for phrase in history_phrases)

            if not acknowledges_history:
                # This is a medium severity - just warn
                return self._fail(
                    message=f"Output doesn't reference {broken_promises_count} broken promises",
                    expected="Acknowledgment of payment history",
                    found="No history reference",
                    details={
                        "broken_promises_count": broken_promises_count,
                        "history_acknowledged": False,
                    },
                )

            return self._pass(
                message="Output acknowledges payment history",
                details={
                    "broken_promises_count": broken_promises_count,
                    "history_acknowledged": True,
                },
            )

        return self._pass(
            message="No significant promise history to reference",
            details={"broken_promises_count": broken_promises_count},
        )

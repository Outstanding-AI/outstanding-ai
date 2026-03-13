"""
Gate evaluation engine.

.. deprecated::
    This module is **DEPRECATED**.  Gate evaluation has been moved to the
    Django backend (``services/gate_checker.py``) which implements 9
    deterministic SQL/Python gates with access to the full data lake.

    The AI Engine endpoint (``POST /evaluate-gates``) still works for
    backward compatibility but is no longer called by the Django backend
    in production.  The 6 gates here are a simplified subset of the
    authoritative 9-gate implementation.

Evaluate 6 compliance gates before allowing collection actions.
Use deterministic Python logic instead of LLM calls for reliability
and speed.

Gates:
    touch_cap: Monthly touch count vs level cap
    cooling_off: Min days between touches + do_not_contact_until hold
    dispute_active: Block contact if dispute is pending
    hardship: Flag for sensitive tone (does not block)
    unsubscribe: Block contact if opted out
    escalation_appropriate: Validate tone escalation path with
        industry patience awareness
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from src.api.models.requests import EvaluateGatesRequest
from src.api.models.responses import EvaluateGatesResponse, GateResult

logger = logging.getLogger(__name__)

# Tone escalation order (for escalation_appropriate gate)
TONE_ESCALATION_ORDER = [
    "friendly_reminder",
    "professional",
    "concerned_inquiry",
    "firm",
    "final_notice",
]


class GateEvaluator:
    """Evaluate compliance gates using deterministic rules.

    .. deprecated::
        Gate evaluation is now handled by Django's
        ``services/gate_checker.py``.  This class remains for backward
        compatibility only.

    All gate checks are pure Python -- zero LLM calls.  The evaluate()
    method runs all 6 gates and returns an overall allow/block decision
    with per-gate results.
    """

    async def evaluate(self, request: EvaluateGatesRequest) -> EvaluateGatesResponse:
        """
        Evaluate gates for a proposed action using rule-based logic.

        Args:
            request: Gate evaluation request with context and proposed action

        Returns:
            Gate evaluation results with pass/fail for each gate
        """
        comm = request.context.communication
        context = request.context

        # Calculate days since last touch
        days_since_last_touch = 999  # Default to large number if never contacted
        if comm and comm.last_touch_at:
            last_touch = comm.last_touch_at
            if last_touch.tzinfo is None:
                last_touch = last_touch.replace(tzinfo=timezone.utc)
            delta = datetime.now(timezone.utc) - last_touch
            days_since_last_touch = delta.days

        # Check do_not_contact_until date
        do_not_contact_active = False
        if context.do_not_contact_until:
            try:
                hold_date = datetime.fromisoformat(context.do_not_contact_until)
                if hold_date.tzinfo is None:
                    hold_date = hold_date.replace(tzinfo=timezone.utc)
                do_not_contact_active = datetime.now(timezone.utc).date() < hold_date.date()
            except ValueError:
                logger.warning(f"Invalid do_not_contact_until date: {context.do_not_contact_until}")

        # Evaluate each gate
        gate_results = {}

        # 1. Touch Cap Gate
        gate_results["touch_cap"] = self._evaluate_touch_cap(
            monthly_count=context.monthly_touch_count,
            cap=context.touch_cap,
        )

        # 2. Cooling Off Gate
        gate_results["cooling_off"] = self._evaluate_cooling_off(
            days_since_last=days_since_last_touch,
            interval_days=context.touch_interval_days,
            do_not_contact_active=do_not_contact_active,
            do_not_contact_until=context.do_not_contact_until,
        )

        # 3. Dispute Active Gate
        gate_results["dispute_active"] = self._evaluate_dispute(
            active_dispute=context.active_dispute,
        )

        # 4. Hardship Gate
        gate_results["hardship"] = self._evaluate_hardship(
            hardship_indicated=context.hardship_indicated,
        )

        # 5. Unsubscribe Gate
        gate_results["unsubscribe"] = self._evaluate_unsubscribe(
            unsubscribe_requested=context.unsubscribe_requested,
        )

        # 6. Escalation Appropriate Gate
        gate_results["escalation_appropriate"] = self._evaluate_escalation(
            proposed_tone=request.proposed_tone,
            last_tone_used=comm.last_tone_used if comm else None,
            touch_count=comm.touch_count if comm else 0,
            broken_promises_count=context.broken_promises_count,
            case_state=context.case_state,
            industry=context.industry,
        )

        # Overall allowed if all gates pass
        all_passed = all(g.passed for g in gate_results.values())

        # Generate recommended action if blocked
        recommended_action = None
        if not all_passed:
            recommended_action = self._get_recommended_action(gate_results)

        logger.info(
            f"Evaluated gates for {context.party.customer_code}: "
            f"action={request.proposed_action}, allowed={all_passed}, "
            f"failed_gates={[k for k, v in gate_results.items() if not v.passed]}"
        )

        return EvaluateGatesResponse(
            allowed=all_passed,
            gate_results=gate_results,
            recommended_action=recommended_action,
            tokens_used=0,  # No LLM call
            provider="deterministic",
            model="rule_engine",
            is_fallback=False,
        )

    def _evaluate_touch_cap(self, monthly_count: int, cap: int) -> GateResult:
        """Check if monthly touch cap has been reached.

        Args:
            monthly_count: Number of touches this calendar month.
            cap: Maximum allowed touches per month.

        Returns:
            GateResult with pass if count < cap.
        """
        passed = monthly_count < cap
        return GateResult(
            passed=passed,
            reason=f"Monthly touches ({monthly_count}) {'below' if passed else 'at or exceeds'} cap ({cap})",
            current_value=monthly_count,
            threshold=cap,
        )

    def _evaluate_cooling_off(
        self,
        days_since_last: int,
        interval_days: int,
        do_not_contact_active: bool,
        do_not_contact_until: Optional[str],
    ) -> GateResult:
        """Check if cooling-off period has elapsed.

        A ``do_not_contact_until`` hold always takes precedence over
        the standard interval check.

        Args:
            days_since_last: Days since the most recent outbound touch.
            interval_days: Minimum required gap between touches.
            do_not_contact_active: Whether a DNC hold is in effect.
            do_not_contact_until: ISO date string of the hold expiry.

        Returns:
            GateResult with pass if interval is met and no DNC hold.
        """
        # Do not contact hold takes precedence
        if do_not_contact_active:
            return GateResult(
                passed=False,
                reason=f"Do not contact until {do_not_contact_until}",
                current_value=0,
                threshold=interval_days,
            )

        passed = days_since_last >= interval_days
        return GateResult(
            passed=passed,
            reason=f"Days since last touch ({days_since_last}) {'meets' if passed else 'below'} minimum interval ({interval_days})",
            current_value=days_since_last,
            threshold=interval_days,
        )

    def _evaluate_dispute(self, active_dispute: bool) -> GateResult:
        """Check if an active dispute blocks contact."""
        passed = not active_dispute
        return GateResult(
            passed=passed,
            reason="No active dispute" if passed else "Active dispute - contact blocked",
            current_value=active_dispute,
            threshold=False,
        )

    def _evaluate_hardship(self, hardship_indicated: bool) -> GateResult:
        """Check if hardship has been indicated.

        Hardship does not block contact but flags the need for a
        sensitive, empathetic tone in any outbound communication.
        """
        # Hardship doesn't block, but flags for special handling
        # For now, we pass but include warning in reason
        if hardship_indicated:
            return GateResult(
                passed=True,  # Allow but with special handling
                reason="Hardship indicated - use sensitive tone",
                current_value=hardship_indicated,
                threshold=None,
            )
        return GateResult(
            passed=True,
            reason="No hardship indicated",
            current_value=hardship_indicated,
            threshold=None,
        )

    def _evaluate_unsubscribe(self, unsubscribe_requested: bool) -> GateResult:
        """Check if the party has opted out of contact."""
        passed = not unsubscribe_requested
        return GateResult(
            passed=passed,
            reason="No unsubscribe request"
            if passed
            else "Unsubscribe requested - contact blocked",
            current_value=unsubscribe_requested,
            threshold=False,
        )

    def _evaluate_escalation(
        self,
        proposed_tone: Optional[str],
        last_tone_used: Optional[str],
        touch_count: int,
        broken_promises_count: int,
        case_state: Optional[str],
        industry=None,
    ) -> GateResult:
        """
        Check if proposed escalation is appropriate.

        Rules:
        - Can't jump to final_notice on first contact
        - Tone should follow escalation order (with some flexibility)
        - Broken promises justify faster escalation
        - Can't escalate if already at highest level
        - Industry escalation_patience affects allowed jump size:
          - patient: max 1 step (manufacturing, government)
          - standard: max 1 step, or 2 if broken promises
          - aggressive: max 2 steps (retail)
        """
        # Get industry escalation patience (affects allowed jump)
        escalation_patience = "standard"
        if industry and hasattr(industry, "escalation_patience"):
            escalation_patience = industry.escalation_patience
        if not proposed_tone:
            return GateResult(
                passed=True,
                reason="No specific tone proposed",
                current_value=None,
                threshold=None,
            )

        proposed_tone_lower = proposed_tone.lower()

        # Get positions in escalation order
        proposed_idx = (
            TONE_ESCALATION_ORDER.index(proposed_tone_lower)
            if proposed_tone_lower in TONE_ESCALATION_ORDER
            else -1
        )
        last_idx = (
            TONE_ESCALATION_ORDER.index(last_tone_used.lower())
            if last_tone_used and last_tone_used.lower() in TONE_ESCALATION_ORDER
            else -1
        )

        # Unknown tone - allow
        if proposed_idx == -1:
            return GateResult(
                passed=True,
                reason=f"Tone '{proposed_tone}' not in standard escalation path",
                current_value=proposed_tone,
                threshold=None,
            )

        # First contact rules
        if touch_count == 0 or last_idx == -1:
            # Can't start with firm or final_notice
            if proposed_idx >= 3:  # firm or final_notice
                return GateResult(
                    passed=False,
                    reason=f"Cannot start with '{proposed_tone}' on first contact",
                    current_value=proposed_tone,
                    threshold="friendly_reminder or professional",
                )
            return GateResult(
                passed=True,
                reason=f"'{proposed_tone}' appropriate for first contact",
                current_value=proposed_tone,
                threshold=None,
            )

        # Check escalation jump
        jump = proposed_idx - last_idx

        # Allow same or lower tone (de-escalation is fine)
        if jump <= 0:
            return GateResult(
                passed=True,
                reason=f"Tone '{proposed_tone}' same or lower than last '{last_tone_used}'",
                current_value=proposed_tone,
                threshold=last_tone_used,
            )

        # Determine max allowed escalation based on industry patience
        # patient (manufacturing, government): strict 1-step only
        # standard: 1-step, or 2-step if broken promises
        # aggressive (retail): 2-step allowed
        if escalation_patience == "aggressive":
            max_jump = 2
        elif escalation_patience == "patient":
            max_jump = 1
        else:  # standard
            max_jump = 2 if broken_promises_count > 0 else 1

        # Allow escalation within limits
        if jump <= max_jump:
            if jump == 1:
                return GateResult(
                    passed=True,
                    reason=f"Single-step escalation from '{last_tone_used}' to '{proposed_tone}'",
                    current_value=proposed_tone,
                    threshold=last_tone_used,
                )
            else:
                reason_suffix = ""
                if escalation_patience == "aggressive":
                    reason_suffix = f" (industry={escalation_patience})"
                elif broken_promises_count > 0:
                    reason_suffix = f" (justified by {broken_promises_count} broken promises)"
                return GateResult(
                    passed=True,
                    reason=f"Double-step escalation from '{last_tone_used}' to '{proposed_tone}'{reason_suffix}",
                    current_value=proposed_tone,
                    threshold=last_tone_used,
                )

        # Too aggressive escalation
        next_tone = (
            TONE_ESCALATION_ORDER[last_idx + 1]
            if last_idx + 1 < len(TONE_ESCALATION_ORDER)
            else "N/A"
        )
        patience_hint = f" (industry patience: {escalation_patience})" if industry else ""
        return GateResult(
            passed=False,
            reason=f"Escalation from '{last_tone_used}' to '{proposed_tone}' too aggressive (jump of {jump} levels){patience_hint}",
            current_value=proposed_tone,
            threshold=f"Max {max_jump} level escalation (try '{next_tone}')",
        )

    def _get_recommended_action(self, gate_results: dict[str, GateResult]) -> str:
        """Generate a human-readable recommended action based on failed gates.

        Returns the most actionable recommendation, prioritised by
        severity (unsubscribe > dispute > cooling_off > touch_cap >
        escalation).
        """
        failed = [k for k, v in gate_results.items() if not v.passed]

        if "unsubscribe" in failed:
            return "Remove from contact list - party has opted out"
        if "dispute_active" in failed:
            return "Wait for dispute resolution before contact"
        if "cooling_off" in failed:
            return "Wait until cooling off period ends"
        if "touch_cap" in failed:
            return "Monthly touch limit reached - wait until next month"
        if "escalation_appropriate" in failed:
            return "Use less aggressive tone or wait for more touchpoints"

        return "Review gate failures and adjust approach"


# Singleton instance used by the /evaluate-gates route handler.
gate_evaluator = GateEvaluator()

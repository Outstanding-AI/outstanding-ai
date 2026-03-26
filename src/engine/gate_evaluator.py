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

from .escalation_validator import evaluate_escalation, get_recommended_action

logger = logging.getLogger(__name__)


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
        gate_results["escalation_appropriate"] = evaluate_escalation(
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
            recommended_action = get_recommended_action(gate_results)

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


# Singleton instance used by the /evaluate-gates route handler.
gate_evaluator = GateEvaluator()

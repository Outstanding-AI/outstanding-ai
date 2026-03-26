"""
Escalation validation logic for the gate evaluator.

Determines whether a proposed tone escalation is appropriate based on
the escalation order, touch count, broken promises, case state, and
industry patience level.

Also provides the recommended action generator for failed gates.
"""

from typing import Optional

from src.api.models.responses import GateResult

# Tone escalation order (for escalation_appropriate gate)
TONE_ESCALATION_ORDER = [
    "friendly_reminder",
    "professional",
    "concerned_inquiry",
    "firm",
    "final_notice",
]


def evaluate_escalation(
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

    Args:
        proposed_tone: The tone being proposed for the next touch.
        last_tone_used: The tone used in the most recent touch.
        touch_count: Total touches sent to this party.
        broken_promises_count: Number of broken payment promises.
        case_state: Current case state (e.g. ACTIVE, DISPUTED).
        industry: Industry profile object (or None) with
            escalation_patience attribute.

    Returns:
        GateResult indicating whether the escalation is allowed.
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
        TONE_ESCALATION_ORDER[last_idx + 1] if last_idx + 1 < len(TONE_ESCALATION_ORDER) else "N/A"
    )
    patience_hint = f" (industry patience: {escalation_patience})" if industry else ""
    return GateResult(
        passed=False,
        reason=f"Escalation from '{last_tone_used}' to '{proposed_tone}' too aggressive (jump of {jump} levels){patience_hint}",
        current_value=proposed_tone,
        threshold=f"Max {max_jump} level escalation (try '{next_tone}')",
    )


def get_recommended_action(gate_results: dict[str, GateResult]) -> str:
    """Generate a human-readable recommended action based on failed gates.

    Returns the most actionable recommendation, prioritised by
    severity (unsubscribe > dispute > cooling_off > touch_cap >
    escalation).

    Args:
        gate_results: Dict mapping gate name to GateResult.

    Returns:
        Human-readable recommendation string.
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

"""Deterministic policy authorization guardrail."""

from __future__ import annotations

import re

from .base import BaseGuardrail, GuardrailSeverity

POLICY_TRIGGERS = {
    "legal_escalation_enabled": r"\b(legal\s+team|court|tribunal|solicitor|litigation|legal\s+referral)\b",
    "statutory_interest_enabled": r"\b(statutory\s+interest|Late\s+Payment.*Act|section\s+\d+)\b",
    "discount_allowed": r"\b(discount|reduction|\d+\s*%\s*off)\b",
    "settlement_allowed": r"\b(settle|settlement|compromise|accept\s+\d+\s*%)\b",
}


class PolicyGroundingGuardrail(BaseGuardrail):
    """Ensure commitment-shaped policy language is explicitly authorized.

    This guardrail is intentionally fail-closed: if the backend does not pass
    ``authorized_policies`` (or passes ``None``), we treat that as an empty
    authorization set and reject all policy-shaped claims.
    """

    def __init__(self):
        super().__init__(name="policy_grounding", severity=GuardrailSeverity.HIGH)

    def validate(self, output: str, context, **kwargs) -> list:
        authorized = (
            kwargs.get("authorized_policies") or getattr(context, "authorized_policies", None) or {}
        )
        for policy, pattern in POLICY_TRIGGERS.items():
            if re.search(pattern, output, re.IGNORECASE) and not authorized.get(policy, False):
                return [self._fail(f"Draft mentions {policy} content without authorization")]
        return [self._pass("Policy grounding validated")]

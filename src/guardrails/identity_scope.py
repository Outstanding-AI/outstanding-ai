"""Deterministic identity and email-scope validation for generated drafts."""

from __future__ import annotations

import re
from typing import Any

from .base import BaseGuardrail, GuardrailSeverity

GENERIC_NAMES = {
    "accounts",
    "team",
    "customer",
    "sir",
    "madam",
    "all",
}
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b")
GREETING_PATTERN = re.compile(r"\b(?:hey|hi|hello)\s+([A-Z][a-zA-Z'-]+)")
REPLY_TO_PATTERN = re.compile(r"reply\s+(?:to|at)\s+([\w.+-]+@[\w.-]+\.\w{2,})", re.IGNORECASE)


class IdentityScopeGuardrail(BaseGuardrail):
    """Validate greeting, sign-off, and embedded email identity deterministically."""

    def __init__(self):
        super().__init__(name="identity_scope", severity=GuardrailSeverity.HIGH)

    def validate(self, output: str, context: Any, **kwargs) -> list:
        recipient_name = (kwargs.get("recipient_name") or "").strip()
        recipient_first = recipient_name.split()[0].lower() if recipient_name else ""
        sender_first = ((kwargs.get("sender_name") or "").strip().split() or [""])[0].lower()

        greeting_match = GREETING_PATTERN.search(output)
        if greeting_match:
            mentioned = greeting_match.group(1).lower()
            if recipient_first and mentioned != recipient_first and mentioned not in GENERIC_NAMES:
                return [
                    self._fail(
                        f"Greeting {mentioned!r} does not match recipient {recipient_first!r}"
                    )
                ]

        signoff_fragment = output[-400:]
        signoff_matches = re.findall(
            r"(?:Regards|Thanks|Best|Kind regards|Sincerely),?\s*([^\n<]{2,40})", signoff_fragment
        )
        for signoff in signoff_matches:
            lowered = signoff.strip().lower()
            if "[sender_name]" in lowered or "{sender_name}" in lowered:
                continue
            token = lowered.split()[0]
            if sender_first and token and token != sender_first:
                return [self._fail(f"Sign-off {signoff!r} does not match sender {sender_first!r}")]

        authorized = set()
        for contact in getattr(context, "party_contacts", None) or []:
            email = contact.get("email") if isinstance(contact, dict) else None
            if email:
                authorized.add(email.lower())
        for field in ("sender_email", "reply_anchor_email"):
            value = kwargs.get(field)
            if value:
                authorized.add(str(value).lower())
        authorized.update(str(value).lower() for value in (kwargs.get("cc_emails") or []) if value)

        for email in EMAIL_PATTERN.findall(output):
            if email.lower() not in authorized:
                return [self._fail(f"Unknown email in draft body: {email}")]

        reply_to_match = REPLY_TO_PATTERN.search(output)
        if reply_to_match:
            expected = (
                kwargs.get("reply_anchor_email") or kwargs.get("sender_email") or ""
            ).lower()
            claimed = reply_to_match.group(1).lower()
            if expected and claimed != expected:
                return [
                    self._fail(f"Reply-to {claimed!r} does not match expected mailbox {expected!r}")
                ]

        return [self._pass("Identity scope validated")]

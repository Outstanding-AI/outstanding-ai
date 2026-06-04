"""Ensure overdue-scope drafts use overdue debtor-facing terminology."""

from __future__ import annotations

import re
from typing import Any

from .base import BaseGuardrail, GuardrailResult, GuardrailSeverity

OUTSTANDING_CHASE_LANGUAGE = re.compile(
    r"\boutstanding\s+"
    r"(?:invoice|invoices|balance|balances|amount|amounts|debt|debts|account|accounts|payment|payments)\b",
    re.IGNORECASE,
)


class OverdueTerminologyGuardrail(BaseGuardrail):
    """Block debtor-facing "outstanding" wording for overdue collection drafts."""

    def __init__(self):
        super().__init__("overdue_terminology", GuardrailSeverity.HIGH)

    def validate(self, output: str, context: Any, **kwargs) -> list[GuardrailResult]:
        if kwargs.get("closure_mode"):
            return [self._pass("Closure draft terminology check skipped")]

        basis = getattr(context, "chase_basis", None) or getattr(context, "collection_basis", None)
        schema_version = int(getattr(context, "schema_version", 0) or 0)
        if schema_version < 4 or basis != "overdue":
            return [self._pass("Terminology check not required for non-overdue scope")]

        subject = kwargs.get("subject") or ""
        combined = f"{subject}\n{output or ''}"
        matches = sorted(
            {match.group(0) for match in OUTSTANDING_CHASE_LANGUAGE.finditer(combined)}
        )
        if not matches:
            return [self._pass("Overdue terminology validated")]

        return [
            self._fail(
                "Use overdue terminology for overdue-scope collection drafts",
                expected="overdue invoice(s) / overdue balance",
                found=", ".join(matches),
                details={"matches": matches},
            )
        ]

"""Flag non-blocking forbidden content for operator review."""

from __future__ import annotations

import re

from .base import BaseGuardrail, GuardrailResult, GuardrailSeverity

PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "bank_payment_details": [
        re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
        re.compile(r"\b\d{2}[-\s]?\d{2}[-\s]?\d{2}\b"),
        re.compile(r"(?<!\d)\d{8}(?!\d)"),
        re.compile(r"\b[A-Z]{6}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b"),
    ],
    "legal_statute": [
        re.compile(r"\bLate\s+Payment.*Act\b", re.IGNORECASE),
        re.compile(r"\bsection\s+\d+\s*\(?\w?\)?", re.IGNORECASE),
        re.compile(r"\bstatutory\s+interest\b", re.IGNORECASE),
    ],
    "external_url": [
        re.compile(r"https?://[^\s<>\"]{5,}", re.IGNORECASE),
    ],
}


class ForbiddenContentDetector(BaseGuardrail):
    """Detect bank/legal/url content that should be reviewed by an operator."""

    def __init__(self):
        super().__init__("forbidden_content", GuardrailSeverity.REVIEW)

    def validate(self, output: str, context, **kwargs) -> list[GuardrailResult]:
        findings: list[dict] = []

        for category, patterns in PATTERNS.items():
            excerpts: list[str] = []
            for pattern in patterns:
                for match in pattern.finditer(output or ""):
                    excerpt = match.group(0)
                    if excerpt not in excerpts:
                        excerpts.append(excerpt[:200])
            if excerpts:
                findings.append({"category": category, "excerpts": excerpts})

        if not findings:
            return [self._pass("No forbidden content detected")]

        return [
            self._flag_for_review(
                "Draft contains operator-review content",
                details={"findings": findings},
            )
        ]

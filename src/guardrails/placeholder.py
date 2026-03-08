"""Placeholder validation guardrail — catches hallucinated/unreplaced placeholders."""

import re
from typing import Any

from .base import BaseGuardrail, GuardrailResult, GuardrailSeverity

# Allowed placeholders that the system handles programmatically
ALLOWED_PLACEHOLDERS = {
    "{INVOICE_TABLE}",  # Replaced by Django's invoice_table_builder
    "[SENDER_NAME]",  # Replaced by Django's _store_draft
    "[SENDER_TITLE]",  # Replaced by Django's _store_draft
    "[SENDER_COMPANY]",  # Replaced by Django's _store_draft
}

# Patterns to detect placeholder-style text in draft output
# Matches [WORD_WORD] and {WORD_WORD} patterns (all-caps with underscores/spaces)
BRACKET_PATTERN = re.compile(r"\[([A-Z][A-Z_\s]{2,})\]")
BRACE_PATTERN = re.compile(r"\{([A-Z][A-Z_\s]{2,})\}")


class PlaceholderValidationGuardrail(BaseGuardrail):
    """
    Detects hallucinated or unreplaced placeholders in AI-generated drafts.

    CRITICAL severity — blocks output if any non-whitelisted placeholder is found.

    This guardrail is deterministic (pure regex, no LLM calls) and runs first
    in the pipeline as the cheapest check.
    """

    def __init__(self):
        super().__init__(
            name="placeholder_validation",
            severity=GuardrailSeverity.CRITICAL,
        )

    def validate(self, output: str, context: Any, **kwargs) -> list[GuardrailResult]:
        """Scan draft for hallucinated placeholder patterns."""
        if not output:
            return [self._pass("No output to validate")]

        found_placeholders = set()

        # Find all [PLACEHOLDER] patterns
        for match in BRACKET_PATTERN.finditer(output):
            placeholder = f"[{match.group(1)}]"
            if placeholder not in ALLOWED_PLACEHOLDERS:
                found_placeholders.add(placeholder)

        # Find all {PLACEHOLDER} patterns
        for match in BRACE_PATTERN.finditer(output):
            placeholder = f"{{{match.group(1)}}}"
            if placeholder not in ALLOWED_PLACEHOLDERS:
                found_placeholders.add(placeholder)

        if found_placeholders:
            return [
                self._fail(
                    message=(
                        f"Draft contains {len(found_placeholders)} hallucinated placeholder(s): "
                        f"{', '.join(sorted(found_placeholders))}. "
                        f"Only allowed: {', '.join(sorted(ALLOWED_PLACEHOLDERS))}. "
                        f"Use actual values from context instead of inventing placeholders."
                    ),
                    expected="No hallucinated placeholders",
                    found=sorted(found_placeholders),
                    details={"hallucinated_placeholders": sorted(found_placeholders)},
                )
            ]

        return [self._pass("No hallucinated placeholders found")]

"""Placeholder validation guardrail -- catch hallucinated placeholders.

Detect ``[ALL_CAPS]`` and ``{ALL_CAPS}`` patterns in AI-generated drafts
that indicate the LLM invented placeholder tokens instead of using real
values from the context.

This is the cheapest guardrail (pure regex, zero LLM calls) and runs
first in the pipeline.  CRITICAL severity -- any non-whitelisted
placeholder blocks the draft.

Allowed placeholders (handled programmatically by Django post-generation):
    ``{INVOICE_TABLE}``  -- replaced by ``InvoiceTableBuilder``
    ``[SENDER_NAME]``    -- replaced by ``_store_draft()``
    ``[SENDER_TITLE]``   -- replaced by ``_store_draft()``
    ``[SENDER_COMPANY]`` -- replaced by ``_store_draft()``

Context-aware behaviour:
    - Standard drafts: ``{INVOICE_TABLE}`` is expected and allowed.
    - Follow-up (``skip_invoice_table=True``) / closure
      (``closure_mode=True``): ``{INVOICE_TABLE}`` becomes disallowed
      because the LLM was instructed not to use it.
"""

import re
from typing import Any

from src.config.constants import ALLOWED_PLACEHOLDERS

from .base import BaseGuardrail, GuardrailResult, GuardrailSeverity

# Patterns to detect placeholder-style text in draft output
# Matches uppercase, lowercase, and camelCase placeholders
BRACKET_PATTERNS = [
    re.compile(r"\[([A-Z][A-Z_\s]{2,})\]"),
    re.compile(r"\[([a-z][a-z_\s]{2,})\]"),
    re.compile(r"\[([a-z]+[A-Z][a-zA-Z]*)\]"),
]
BRACE_PATTERNS = [
    re.compile(r"\{([A-Z][A-Z_\s]{2,})\}"),
    re.compile(r"\{([a-z][a-z_\s]{2,})\}"),
]


class PlaceholderValidationGuardrail(BaseGuardrail):
    """
    Detects hallucinated or unreplaced placeholders in AI-generated drafts.

    CRITICAL severity — blocks output if any non-whitelisted placeholder is found.

    Context-aware behaviour:
    - skip_invoice_table=True or closure_mode=True: {INVOICE_TABLE} is DISALLOWED
      (LLM was told not to use it — if it appears, that's an error)
    - Standard drafts: {INVOICE_TABLE} is allowed (expected in output)

    This guardrail is deterministic (pure regex, no LLM calls) and runs first
    in the pipeline as the cheapest check.
    """

    def __init__(self):
        super().__init__(
            name="placeholder_validation",
            severity=GuardrailSeverity.CRITICAL,
        )

    def validate(self, output: str, context: Any, **kwargs) -> list[GuardrailResult]:
        """Scan the draft for hallucinated placeholder patterns.

        Args:
            output: AI-generated draft body text.
            context: Case context (unused, but required by interface).
            **kwargs: ``skip_invoice_table`` (bool),
                ``closure_mode`` (bool).

        Returns:
            Single-element list with pass or fail result.
        """
        if not output:
            return [self._pass("No output to validate")]

        skip_invoice_table = kwargs.get("skip_invoice_table", False)
        closure_mode = kwargs.get("closure_mode", False)

        # Build effective allowed set for this draft type
        # When skip_invoice_table or closure_mode, {INVOICE_TABLE} should NOT appear
        if skip_invoice_table or closure_mode:
            allowed = ALLOWED_PLACEHOLDERS - {"{INVOICE_TABLE}"}
        else:
            allowed = ALLOWED_PLACEHOLDERS

        found_placeholders = set()

        # Find all [PLACEHOLDER] patterns
        for pattern in BRACKET_PATTERNS:
            for match in pattern.finditer(output):
                placeholder = f"[{match.group(1)}]"
                if placeholder not in allowed:
                    found_placeholders.add(placeholder)

        # Find all {PLACEHOLDER} patterns
        for pattern in BRACE_PATTERNS:
            for match in pattern.finditer(output):
                placeholder = f"{{{match.group(1)}}}"
                if placeholder not in allowed:
                    found_placeholders.add(placeholder)

        if found_placeholders:
            # Provide context-specific error message
            if "{INVOICE_TABLE}" in found_placeholders and (skip_invoice_table or closure_mode):
                draft_type = "closure" if closure_mode else "follow-up"
                msg = (
                    f"Draft contains {{INVOICE_TABLE}} but this is a {draft_type} email "
                    f"where invoice table is suppressed. Remove it."
                )
            else:
                msg = (
                    f"Draft contains {len(found_placeholders)} hallucinated placeholder(s): "
                    f"{', '.join(sorted(found_placeholders))}. "
                    f"Only allowed: {', '.join(sorted(allowed))}. "
                    f"Use actual values from context instead of inventing placeholders."
                )

            return [
                self._fail(
                    message=msg,
                    expected="No hallucinated placeholders",
                    found=sorted(found_placeholders),
                    details={"hallucinated_placeholders": sorted(found_placeholders)},
                )
            ]

        return [self._pass("No hallucinated placeholders found")]

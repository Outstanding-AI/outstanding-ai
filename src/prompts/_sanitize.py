"""Prompt input sanitization helpers.

User-controlled strings (email bodies, draft custom instructions) are
interpolated into prompts that use XML-style delimiter tags such as
``<email_body>`` and ``<user_preferences>`` to demarcate untrusted
content. A crafted input containing a literal closing tag can escape
the delimiter and corrupt the prompt boundary, allowing injected text
to be treated as system-level instructions.

``sanitize_delimiter_tags`` neutralizes these specific tag names — both
open and close forms, case-insensitive — by replacing the angle brackets
with visibly-distinct lookalikes. The LLM still sees the text content
but the delimiter structure of the surrounding prompt stays intact.

This is a defense-in-depth measure; it does not replace the
prompt-injection phrase validation in ``src/guardrails/``.
"""

import re
from typing import Iterable

# Lookalike characters — visually similar to < > but not the same codepoint.
_NEUTRALIZED_OPEN = "\u2039"  # ‹
_NEUTRALIZED_CLOSE = "\u203a"  # ›

_DEFAULT_PROTECTED_TAGS: tuple[str, ...] = ("email_body", "user_preferences")


def _build_pattern(tags: Iterable[str]) -> re.Pattern[str]:
    joined = "|".join(re.escape(tag) for tag in tags)
    # Matches <tag>, </tag>, <TAG>, </TAG>, and variants with whitespace.
    return re.compile(rf"<\s*/?\s*(?:{joined})\s*>", re.IGNORECASE)


_DEFAULT_PATTERN = _build_pattern(_DEFAULT_PROTECTED_TAGS)


def sanitize_delimiter_tags(
    text: str | None,
    *,
    extra_tags: Iterable[str] | None = None,
) -> str:
    """Return ``text`` with protected delimiter tags neutralized.

    Replaces occurrences of ``<email_body>``, ``</email_body>``,
    ``<user_preferences>``, and ``</user_preferences>`` (case-insensitive,
    whitespace-tolerant) with a visually-similar but structurally-inert
    form. Non-string inputs pass through unchanged after ``str()``.

    Args:
        text: The untrusted string to sanitize. ``None`` and empty strings
            are returned as-is.
        extra_tags: Optional additional tag names to protect.
    """
    if not text:
        return text or ""

    if extra_tags:
        pattern = _build_pattern((*_DEFAULT_PROTECTED_TAGS, *extra_tags))
    else:
        pattern = _DEFAULT_PATTERN

    def _replace(match: re.Match[str]) -> str:
        return match.group(0).replace("<", _NEUTRALIZED_OPEN).replace(">", _NEUTRALIZED_CLOSE)

    return pattern.sub(_replace, text)

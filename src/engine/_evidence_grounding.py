"""Generic verbatim-evidence grounding primitives.

Extracted from ``manual_note_interpreter.py`` (the first classifier in this
codebase to enforce that every extracted date/amount/reference is backed by
an exact substring of the source text) so the same grounding contract can be
reused by other extractors — starting with the collection-email response
classifier — without duplicating the string-matching logic or letting the
two implementations drift apart.

Nothing here is note-specific, email-specific, or aware of any contract
model. Every function takes plain strings/values and returns plain
bool/str/tuple results.
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation


def normalize_ref(value: object) -> str:
    """Collapse a reference/invoice number to alnum-only uppercase for comparison."""
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def locate_evidence(source_text: str, evidence: object, *, field: str) -> tuple[int, int, str]:
    """Verify ``evidence`` is an exact, unique substring of ``source_text``.

    Raises ``ValueError`` (with a stable snake_case code as the message, used
    directly as a validation/remediation lookup key by callers) when the
    evidence is missing, not present verbatim, or not unique — a non-unique
    span means the model must extend the quote until it identifies one place
    in the text unambiguously.
    """
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError(f"{field}_evidence_text_required")
    start = source_text.find(evidence)
    if start < 0:
        raise ValueError(f"{field}_evidence_not_verbatim")
    if source_text.find(evidence, start + 1) >= 0:
        raise ValueError(f"{field}_evidence_not_unique")
    return start, start + len(evidence), evidence


def validate_optional_field_evidence(
    *,
    source_text: str,
    candidate: dict[str, object],
    field: str,
    assertion_evidence: str,
    evidence_key: str | None = None,
) -> str | None:
    """Return the verified evidence span for an optional field, or None if the field is null.

    ``evidence_key`` defaults to ``f"{field}_evidence_text"``; pass it
    explicitly when the caller's naming convention differs (manual notes use
    ``asserted_date`` -> ``date_evidence_text``, dropping the ``asserted_``
    prefix).
    """
    value = candidate.get(field)
    key = evidence_key or f"{field}_evidence_text"
    evidence = candidate.get(key)
    if value is None:
        if evidence is not None:
            raise ValueError(f"null_{field}_must_not_have_evidence")
        return None
    _, _, span = locate_evidence(source_text, evidence, field=field)
    if span not in assertion_evidence:
        raise ValueError(f"{field}_evidence_outside_assertion")
    return span


def date_is_explicit_in_span(value: object, span: str) -> bool:
    """Verify a model-extracted ISO date against its model-selected source span."""
    try:
        parsed = date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return False
    candidates = {
        parsed.isoformat(),
        parsed.strftime("%d/%m/%Y"),
        parsed.strftime("%d-%m-%Y"),
        parsed.strftime("%d.%m.%Y"),
        f"{parsed.day}/{parsed.month}/{parsed.year}",
        f"{parsed.day}-{parsed.month}-{parsed.year}",
        f"{parsed.day} {parsed.strftime('%B')} {parsed.year}",
        f"{parsed.strftime('%B')} {parsed.day}, {parsed.year}",
    }
    source = str(span or "").casefold()
    return any(candidate.casefold() in source for candidate in candidates)


def amount_is_explicit_in_span(value: object, span: str) -> bool:
    try:
        expected = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return False
    for raw in re.findall(r"(?<![A-Za-z0-9])\d[\d,]*(?:\.\d+)?", span):
        try:
            if Decimal(raw.replace(",", "")) == expected:
                return True
        except InvalidOperation:
            continue
    return False


def reference_is_explicit_in_span(value: object, span: str) -> bool:
    normalized_value = normalize_ref(value)
    return bool(normalized_value) and normalized_value in normalize_ref(span)


__all__ = [
    "amount_is_explicit_in_span",
    "date_is_explicit_in_span",
    "locate_evidence",
    "normalize_ref",
    "reference_is_explicit_in_span",
    "validate_optional_field_evidence",
]

"""Temporal Consistency Guardrail -- validate date references.

Check that dates mentioned in AI output are consistent with the case
context: promise dates should be in the future, and due dates should
match obligation records.

MEDIUM severity -- failures produce warnings but do not block output,
since date interpretation can be subjective (formatting differences,
timezone edge cases).
"""

import logging
import re
from datetime import date, datetime

from src.api.models.requests import CaseContext
from src.api.models.responses import ExtractedData

from .base import BaseGuardrail, GuardrailResult, GuardrailSeverity

logger = logging.getLogger(__name__)


class TemporalConsistencyGuardrail(BaseGuardrail):
    """Validate date references in AI output against case context.

    MEDIUM severity -- failures produce warnings but do not block,
    since date interpretation can be subjective (formatting, timezones).

    Checks:
    1. Promise dates (from ``extracted_data``) are in the future
       (today allowed).  Dates > 90 days out are flagged as unusual.
    2. Due dates mentioned in prose match obligation ``due_date``
       values (+/- 1 day tolerance).
    """

    def __init__(self):
        super().__init__(
            name="temporal_consistency",
            severity=GuardrailSeverity.MEDIUM,  # Medium - dates can be subjective
        )

    def validate(self, output: str, context: CaseContext, **kwargs) -> list[GuardrailResult]:
        """Validate temporal consistency of the output.

        Run two sub-checks:
        1. If ``extracted_data`` has a promise_date, verify it is in
           the future (or today) and not unreasonably distant (> 90d).
        2. Scan the output for due-date patterns and verify each
           against obligation due dates (+/- 1 day tolerance).

        Args:
            output: AI-generated text (draft body or reasoning).
            context: Case context with obligations and due dates.
            **kwargs: ``extracted_data`` (ExtractedData or None).

        Returns:
            List of GuardrailResult objects.
        """
        results = []

        # Check extracted promise dates
        extracted_data = kwargs.get("extracted_data")
        if extracted_data and hasattr(extracted_data, "promise_date"):
            results.append(self._validate_promise_date_is_future(extracted_data))

        # Check mentioned due dates
        results.append(self._validate_due_dates(output, context))

        return results

    def _validate_promise_date_is_future(self, extracted_data: ExtractedData) -> GuardrailResult:
        """Validate that the extracted promise date is in the future.

        Allow today as valid.  Flag dates > 90 days out as unusual
        (likely a parsing error or stalling tactic).
        """
        if not extracted_data.promise_date:
            return self._pass(message="No promise date to validate")

        today = date.today()
        promise_date = extracted_data.promise_date

        # Allow today as a valid promise date
        if promise_date < today:
            days_past = (today - promise_date).days
            return self._fail(
                message=f"Promise date {promise_date} is {days_past} days in the past",
                expected="Date in future or today",
                found=str(promise_date),
                details={
                    "promise_date": str(promise_date),
                    "today": str(today),
                    "days_past": days_past,
                },
            )

        # Warn if promise date is very far in the future (>90 days)
        days_future = (promise_date - today).days
        if days_future > 90:
            return self._fail(
                message=f"Promise date {promise_date} is {days_future} days in future (unusual)",
                expected="Date within 90 days",
                found=str(promise_date),
                details={
                    "promise_date": str(promise_date),
                    "days_future": days_future,
                    "note": "Unusually distant promise date",
                },
            )

        return self._pass(
            message="Promise date is valid",
            details={"promise_date": str(promise_date), "days_future": days_future},
        )

    def _validate_due_dates(self, output: str, context: CaseContext) -> GuardrailResult:
        """Validate that due dates mentioned in the output match obligations.

        Extract date patterns from the output text and compare against
        the set of valid due dates from context obligations.  Allow
        +/- 1 day tolerance for formatting / timezone differences.
        """
        # Extract date patterns from output
        date_patterns = [
            r"due\s+(?:on|by)\s+(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})",
            r"due\s+date[:\s]+(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})",
            r"(\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})",
        ]

        # Get valid due dates from context
        valid_due_dates = set()
        for o in context.obligations:
            try:
                if isinstance(o.due_date, str):
                    valid_due_dates.add(date.fromisoformat(o.due_date))
            except ValueError:
                continue

        # Find mentioned dates
        mentioned_dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            for match in matches:
                parsed = self._parse_date(match)
                if parsed:
                    mentioned_dates.append(parsed)

        # Validate mentioned dates
        for mentioned in mentioned_dates:
            if mentioned not in valid_due_dates:
                # Check if it's close to any valid date (±1 day tolerance)
                is_close = any(abs((mentioned - valid).days) <= 1 for valid in valid_due_dates)
                if not is_close:
                    return self._fail(
                        message=f"Due date {mentioned} not found in obligations",
                        expected=sorted([str(d) for d in valid_due_dates]),
                        found=str(mentioned),
                        details={
                            "mentioned_date": str(mentioned),
                            "valid_dates": sorted([str(d) for d in valid_due_dates]),
                        },
                    )

        return self._pass(
            message="Due dates validated",
            details={"valid_dates": sorted([str(d) for d in valid_due_dates])},
        )

    def _parse_date(self, date_str: str) -> date | None:
        """Try to parse a date string in various formats.

        Support DD/MM/YYYY, DD-MM-YYYY, MM/DD/YYYY, YYYY-MM-DD, and
        natural language dates like "15th January 2024".
        """
        formats = [
            "%d/%m/%Y",
            "%d-%m-%Y",
            "%d/%m/%y",
            "%d-%m-%y",
            "%m/%d/%Y",
            "%Y-%m-%d",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue

        # Try parsing natural dates like "15th January 2024"
        try:
            # Remove ordinal suffixes
            cleaned = re.sub(r"(\d+)(?:st|nd|rd|th)", r"\1", date_str)
            return datetime.strptime(cleaned, "%d %B %Y").date()
        except ValueError:
            pass

        return None

"""Base guardrail classes and types.

Define the abstract interface, result dataclasses, and severity enum
shared by all 7 guardrail implementations.  Every guardrail subclass
must extend ``BaseGuardrail`` and implement ``validate()``.

Severity hierarchy (highest to lowest):
    CRITICAL -- block output, must retry or escalate (e.g., placeholder,
        factual grounding, numerical consistency).
    HIGH -- block output, log for review (e.g., entity verification).
    MEDIUM -- warn, allow with flag (e.g., temporal consistency).
    LOW -- log only, allow (e.g., contextual coherence).

CRITICAL and HIGH failures are "blocking" -- they set ``should_block=True``
on the pipeline result, which the draft generator uses to decide whether
to retry or reject the draft.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class GuardrailSeverity(Enum):
    """Severity levels for guardrail failures.

    Determines whether a failure blocks output delivery or is
    treated as a non-blocking warning.  CRITICAL and HIGH both
    block; MEDIUM and LOW do not.
    """

    CRITICAL = "critical"  # Block output, must retry or escalate
    HIGH = "high"  # Block output, log for review
    MEDIUM = "medium"  # Warn, allow with flag
    LOW = "low"  # Log only, allow
    REVIEW = "review"  # Surface to operator review, do not block


@dataclass
class GuardrailResult:
    """Result of a single guardrail validation check.

    Each guardrail may return multiple results (one per sub-check).
    For example, ``FactualGroundingGuardrail`` returns separate results
    for invoice number validation and amount validation.

    Attributes:
        passed: Whether this check passed.
        guardrail_name: Name of the guardrail that produced this result.
        severity: Severity level inherited from the parent guardrail.
        message: Human-readable description of the outcome.
        details: Arbitrary dict of diagnostic data for logging/auditing.
        expected: What the guardrail expected (for failures).
        found: What the guardrail actually found (for failures).
        token_usage: LLM token usage dict (only populated by guardrails
            that make LLM calls, e.g. EntityVerificationGuardrail).
    """

    passed: bool
    guardrail_name: str
    severity: GuardrailSeverity
    message: str = ""
    details: dict = field(default_factory=dict)
    expected: Any = None
    found: Any = None
    token_usage: dict = field(default_factory=dict)
    is_review_finding: bool = False

    @property
    def should_block(self) -> bool:
        """Whether this failure should block the output."""
        return not self.passed and self.severity in [
            GuardrailSeverity.CRITICAL,
            GuardrailSeverity.HIGH,
        ]

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/API responses."""
        return {
            "passed": self.passed,
            "guardrail": self.guardrail_name,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "expected": str(self.expected) if self.expected else None,
            "found": str(self.found) if self.found else None,
            "is_review_finding": self.is_review_finding,
        }


@dataclass
class GuardrailPipelineResult:
    """Aggregate result of running all guardrails in the pipeline.

    Attributes:
        all_passed: True only if every individual check passed.
        should_block: True if any CRITICAL or HIGH check failed.
        results: Flat list of all individual GuardrailResult objects.
        retry_suggested: True when blocking failures exist but are
            few enough (<= 2) that a retry with feedback is likely
            to succeed.
        blocking_guardrails: Names of guardrails that produced
            blocking failures.
    """

    all_passed: bool
    should_block: bool
    results: list[GuardrailResult]
    retry_suggested: bool = False
    blocking_guardrails: list[str] = field(default_factory=list)
    review_findings: list[dict] = field(default_factory=list)

    @property
    def critical_failures(self) -> list[GuardrailResult]:
        """Get all critical failures."""
        return [
            r for r in self.results if not r.passed and r.severity == GuardrailSeverity.CRITICAL
        ]

    @property
    def high_failures(self) -> list[GuardrailResult]:
        """Get all high severity failures."""
        return [r for r in self.results if not r.passed and r.severity == GuardrailSeverity.HIGH]

    @property
    def total_token_usage(self) -> dict:
        """Aggregate token usage across all guardrail results."""
        total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for r in self.results:
            if r.token_usage:
                total["prompt_tokens"] += r.token_usage.get("prompt_tokens", 0)
                total["completion_tokens"] += r.token_usage.get("completion_tokens", 0)
                total["total_tokens"] += r.token_usage.get("total_tokens", 0)
        return total

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/API responses."""
        return {
            "all_passed": self.all_passed,
            "should_block": self.should_block,
            "retry_suggested": self.retry_suggested,
            "blocking_guardrails": self.blocking_guardrails,
            "review_findings": self.review_findings,
            "results": [r.to_dict() for r in self.results],
        }


class BaseGuardrail(ABC):
    """Abstract base class for all guardrails.

    Subclasses must implement ``validate()`` and call ``super().__init__()``
    with a unique name and severity level.  Helper methods ``_pass()`` and
    ``_fail()`` simplify result construction.
    """

    def __init__(self, name: str, severity: GuardrailSeverity):
        self.name = name
        self.severity = severity
        self.logger = logging.getLogger(f"{__name__}.{name}")

    @abstractmethod
    def validate(self, output: str, context: Any, **kwargs) -> list[GuardrailResult]:
        """
        Validate the AI output against this guardrail.

        Args:
            output: The AI-generated output to validate
            context: The input context (CaseContext, etc.)
            **kwargs: Additional context-specific arguments

        Returns:
            List of GuardrailResult objects (one per validation check)
        """

    def _pass(
        self,
        message: str = "",
        details: dict = None,
        token_usage: dict | None = None,
    ) -> GuardrailResult:
        """Create a passing GuardrailResult.

        ``token_usage`` is forwarded into ``GuardrailPipelineResult.total_token_usage``
        for the parent ``generate_draft`` call's per-draft cost rollup. Pass it
        from any guardrail that makes its own LLM call (policy_grounding,
        semantic_coherence, etc.) so per-draft attribution stays accurate.
        """
        return GuardrailResult(
            passed=True,
            guardrail_name=self.name,
            severity=self.severity,
            message=message or f"{self.name} validation passed",
            details=details or {},
            token_usage=token_usage or {},
        )

    def _fail(
        self,
        message: str,
        expected: Any = None,
        found: Any = None,
        details: dict = None,
        token_usage: dict | None = None,
    ) -> GuardrailResult:
        """Create a failing GuardrailResult and log a warning.

        See ``_pass`` for the ``token_usage`` contract.
        """
        self.logger.warning(f"Guardrail {self.name} failed: {message}")
        return GuardrailResult(
            passed=False,
            guardrail_name=self.name,
            severity=self.severity,
            message=message,
            details=details or {},
            expected=expected,
            found=found,
            token_usage=token_usage or {},
        )

    def _flag_for_review(
        self,
        message: str,
        *,
        details: dict | None = None,
    ) -> GuardrailResult:
        """Create a non-blocking review finding."""
        self.logger.warning("Guardrail %s flagged review: %s", self.name, message)
        return GuardrailResult(
            passed=False,
            guardrail_name=self.name,
            severity=self.severity,
            message=message,
            details=details or {},
            is_review_finding=True,
        )

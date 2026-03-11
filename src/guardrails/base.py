"""Base guardrail classes and types."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class GuardrailSeverity(Enum):
    """Severity levels for guardrail failures."""

    CRITICAL = "critical"  # Block output, must retry or escalate
    HIGH = "high"  # Block output, log for review
    MEDIUM = "medium"  # Warn, allow with flag
    LOW = "low"  # Log only, allow


@dataclass
class GuardrailResult:
    """Result of a guardrail validation check."""

    passed: bool
    guardrail_name: str
    severity: GuardrailSeverity
    message: str = ""
    details: dict = field(default_factory=dict)
    # For failed validations, what was expected vs found
    expected: Any = None
    found: Any = None
    # LLM token usage (for guardrails that make LLM calls)
    token_usage: dict = field(default_factory=dict)

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
        }


@dataclass
class GuardrailPipelineResult:
    """Result of running all guardrails in the pipeline."""

    all_passed: bool
    should_block: bool
    results: list[GuardrailResult]
    retry_suggested: bool = False
    blocking_guardrails: list[str] = field(default_factory=list)

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
            "results": [r.to_dict() for r in self.results],
        }


class BaseGuardrail(ABC):
    """Abstract base class for all guardrails."""

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

    def _pass(self, message: str = "", details: dict = None) -> GuardrailResult:
        """Helper to create a passing result."""
        return GuardrailResult(
            passed=True,
            guardrail_name=self.name,
            severity=self.severity,
            message=message or f"{self.name} validation passed",
            details=details or {},
        )

    def _fail(
        self,
        message: str,
        expected: Any = None,
        found: Any = None,
        details: dict = None,
    ) -> GuardrailResult:
        """Helper to create a failing result."""
        self.logger.warning(f"Guardrail {self.name} failed: {message}")
        return GuardrailResult(
            passed=False,
            guardrail_name=self.name,
            severity=self.severity,
            message=message,
            details=details or {},
            expected=expected,
            found=found,
        )

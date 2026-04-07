"""Evaluation metrics definitions."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class EvalMetric(Enum):
    """Evaluation metric types."""

    # Individual Interaction Metrics
    FACTUAL_ACCURACY = "factual_accuracy"
    CLASSIFICATION_ACCURACY = "classification_accuracy"
    CLASSIFICATION_CONFIDENCE = "classification_confidence"
    EXTRACTION_PRECISION = "extraction_precision"
    EXTRACTION_RECALL = "extraction_recall"
    TONE_APPROPRIATENESS = "tone_appropriateness"
    RESPONSE_LATENCY = "response_latency"
    TOKEN_USAGE = "token_usage"

    # Conversation/Case Metrics
    CONTEXT_RETENTION = "context_retention"
    ESCALATION_ACCURACY = "escalation_accuracy"
    PROMISE_TRACKING_ACCURACY = "promise_tracking_accuracy"
    TOUCH_EFFICIENCY = "touch_efficiency"

    # Portfolio/Aggregate Metrics
    COLLECTION_RATE = "collection_rate"
    PROMISE_TO_PAYMENT_CONVERSION = "promise_to_payment_conversion"
    DISPUTE_RESOLUTION_TIME = "dispute_resolution_time"
    AUTOMATION_RATE = "automation_rate"
    HUMAN_OVERRIDE_RATE = "human_override_rate"


@dataclass
class InteractionMetrics:
    """Metrics for a single AI interaction (classification or generation)."""

    request_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Core accuracy metrics
    guardrails_passed: bool = True
    guardrail_failures: list[str] = field(default_factory=list)
    factual_accuracy: float = 1.0  # 0.0-1.0, based on guardrail results

    # Classification specific
    classification: Optional[str] = None
    classification_confidence: float = 0.0
    classification_correct: Optional[bool] = None  # Set by human feedback

    # Extraction specific
    promise_date_extracted: Optional[str] = None
    promise_amount_extracted: Optional[float] = None
    dispute_type_extracted: Optional[str] = None
    extraction_correct: Optional[bool] = None  # Set by human feedback

    # Performance metrics
    latency_ms: float = 0.0
    tokens_used: int = 0
    provider_used: str = ""
    model_used: str = ""

    # Generation specific
    tone_requested: Optional[str] = None
    tone_appropriate: Optional[bool] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/storage."""
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "guardrails_passed": self.guardrails_passed,
            "guardrail_failures": self.guardrail_failures,
            "factual_accuracy": self.factual_accuracy,
            "classification": self.classification,
            "classification_confidence": self.classification_confidence,
            "classification_correct": self.classification_correct,
            "promise_date_extracted": self.promise_date_extracted,
            "promise_amount_extracted": self.promise_amount_extracted,
            "extraction_correct": self.extraction_correct,
            "latency_ms": self.latency_ms,
            "tokens_used": self.tokens_used,
            "provider_used": self.provider_used,
            "model_used": self.model_used,
            "tone_requested": self.tone_requested,
            "tone_appropriate": self.tone_appropriate,
        }


@dataclass
class ConversationMetrics:
    """Metrics for a conversation/case over time."""

    case_id: str
    party_id: str
    customer_code: str

    # Conversation tracking
    total_touches: int = 0
    ai_touches: int = 0
    human_touches: int = 0

    # Quality metrics
    context_retention_score: float = 1.0  # Did AI maintain consistency?
    escalation_decisions: int = 0
    escalation_correct: int = 0

    # Promise tracking
    promises_extracted: int = 0
    promises_kept: int = 0
    promises_broken: int = 0

    # Dispute handling
    disputes_identified: int = 0
    disputes_resolved: int = 0
    avg_dispute_resolution_days: Optional[float] = None

    # Outcome
    case_resolved: bool = False
    payment_received: bool = False
    payment_amount: float = 0.0
    days_to_resolution: Optional[int] = None

    @property
    def promise_keep_rate(self) -> float:
        """Calculate promise-to-payment conversion rate."""
        total = self.promises_kept + self.promises_broken
        if total == 0:
            return 0.0
        return self.promises_kept / total

    @property
    def escalation_accuracy(self) -> float:
        """Calculate escalation decision accuracy."""
        if self.escalation_decisions == 0:
            return 1.0
        return self.escalation_correct / self.escalation_decisions

    @property
    def automation_rate(self) -> float:
        """Calculate AI automation rate."""
        if self.total_touches == 0:
            return 0.0
        return self.ai_touches / self.total_touches

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/storage."""
        return {
            "case_id": self.case_id,
            "party_id": self.party_id,
            "customer_code": self.customer_code,
            "total_touches": self.total_touches,
            "ai_touches": self.ai_touches,
            "human_touches": self.human_touches,
            "context_retention_score": self.context_retention_score,
            "escalation_accuracy": self.escalation_accuracy,
            "promises_extracted": self.promises_extracted,
            "promise_keep_rate": self.promise_keep_rate,
            "automation_rate": self.automation_rate,
            "case_resolved": self.case_resolved,
            "payment_received": self.payment_received,
            "days_to_resolution": self.days_to_resolution,
        }


@dataclass
class PortfolioMetrics:
    """Aggregate metrics across all cases/conversations."""

    period_start: datetime
    period_end: datetime
    tenant_id: Optional[str] = None

    # Volume metrics
    total_cases: int = 0
    active_cases: int = 0
    resolved_cases: int = 0

    # Collection metrics
    total_outstanding_start: float = 0.0
    total_collected: float = 0.0
    collection_rate: float = 0.0

    # AI performance
    total_ai_interactions: int = 0
    guardrail_pass_rate: float = 1.0
    avg_classification_accuracy: float = 0.0
    avg_classification_confidence: float = 0.0

    # Efficiency
    avg_touches_to_resolution: float = 0.0
    automation_rate: float = 0.0
    human_override_rate: float = 0.0

    # DSO impact
    dso_start_of_period: float = 0.0
    dso_end_of_period: float = 0.0
    dso_change: float = 0.0

    def calculate_dso_impact(self) -> float:
        """Calculate DSO change for the period."""
        self.dso_change = self.dso_end_of_period - self.dso_start_of_period
        return self.dso_change

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/storage."""
        return {
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "tenant_id": self.tenant_id,
            "total_cases": self.total_cases,
            "collection_rate": self.collection_rate,
            "guardrail_pass_rate": self.guardrail_pass_rate,
            "avg_classification_accuracy": self.avg_classification_accuracy,
            "automation_rate": self.automation_rate,
            "human_override_rate": self.human_override_rate,
            "dso_start": self.dso_start_of_period,
            "dso_end": self.dso_end_of_period,
            "dso_change": self.dso_change,
        }

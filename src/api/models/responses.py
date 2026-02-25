from datetime import date
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ExtractedData(BaseModel):
    """Data extracted from email by AI."""

    # PROMISE_TO_PAY
    promise_date: Optional[date] = None
    promise_amount: Optional[float] = None
    # DISPUTE
    dispute_type: Optional[str] = None
    dispute_reason: Optional[str] = None
    invoice_refs: Optional[List[str]] = None
    disputed_amount: Optional[float] = None
    # ALREADY_PAID
    claimed_amount: Optional[float] = None
    claimed_date: Optional[date] = None
    claimed_reference: Optional[str] = None
    claimed_details: Optional[str] = None
    # INSOLVENCY
    insolvency_type: Optional[str] = None
    insolvency_details: Optional[str] = None
    administrator_name: Optional[str] = None
    administrator_email: Optional[str] = None
    reference_number: Optional[str] = None
    # OUT_OF_OFFICE
    return_date: Optional[date] = None
    # REDIRECT
    redirect_name: Optional[str] = None
    redirect_contact: Optional[str] = None
    redirect_email: Optional[str] = None


class GuardrailValidation(BaseModel):
    """Result of guardrail validation on AI output."""

    all_passed: bool = True
    guardrails_run: int = 0
    guardrails_passed: int = 0
    blocking_failures: List[str] = []
    warnings: List[str] = []
    factual_accuracy: float = Field(ge=0.0, le=1.0, default=1.0)


class ClassifyResponse(BaseModel):
    """Response from email classification."""

    classification: (
        str  # COOPERATIVE, PROMISE, DISPUTE, HOSTILE, QUERY, OUT_OF_OFFICE, UNSUBSCRIBE, OTHER
    )
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: Optional[str] = None
    extracted_data: Optional[ExtractedData] = None
    tokens_used: Optional[int] = None
    # Guardrail validation results
    guardrail_validation: Optional[GuardrailValidation] = None
    # Provider metadata
    provider: Optional[str] = None
    model: Optional[str] = None
    is_fallback: bool = False


class GenerateDraftResponse(BaseModel):
    """Response from draft generation."""

    subject: str
    body: str
    tone_used: str
    invoices_referenced: List[str] = []
    tokens_used: Optional[int] = None
    # Guardrail validation results
    guardrail_validation: Optional[GuardrailValidation] = None
    # Provider metadata
    provider: Optional[str] = None
    model: Optional[str] = None
    is_fallback: bool = False


class GateResult(BaseModel):
    """Result of a single gate evaluation."""

    passed: bool
    reason: str
    current_value: Optional[Any] = None
    threshold: Optional[Any] = None


class EvaluateGatesResponse(BaseModel):
    """Response from gate evaluation."""

    allowed: bool
    gate_results: Dict[str, GateResult]
    recommended_action: Optional[str] = None
    tokens_used: Optional[int] = None
    # Provider metadata
    provider: Optional[str] = None
    model: Optional[str] = None
    is_fallback: bool = False


class PartyGateResult(BaseModel):
    """Gate result for a single party in batch evaluation."""

    party_id: str
    customer_code: str
    allowed: bool
    gate_results: Dict[str, GateResult]
    recommended_action: Optional[str] = None
    blocking_gate: Optional[str] = None


class EvaluateGatesBatchResponse(BaseModel):
    """Response from batch gate evaluation."""

    total: int
    allowed_count: int
    blocked_count: int
    results: list[PartyGateResult]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str  # "healthy", "degraded", "unhealthy"
    version: str
    provider: str  # "gemini", "openai", etc.
    model: str
    fallback_provider: Optional[str] = None
    fallback_model: Optional[str] = None
    fallback_count: int = 0
    model_available: bool = True
    fallback_available: bool = False
    uptime_seconds: Optional[float] = None

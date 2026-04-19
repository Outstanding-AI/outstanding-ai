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
    # EMAIL_BOUNCE
    bounced_email: Optional[str] = None


class IntentDetail(BaseModel):
    """Per-intent extraction bundle from multi-intent classification (PR4).

    When a debtor email contains multiple intents (e.g. ALREADY_PAID for
    invoice A and PROMISE_TO_PAY for invoice B), each gets its own
    ``extracted_data`` block so downstream handlers receive exactly the
    fields that belong to their intent — no conflation of ``invoice_refs``
    or ``claimed_*`` / ``promise_*`` between different intents.

    Ordering contract: ``intent_details[0]`` is the primary intent and its
    ``intent`` field matches ``ClassifyResponse.classification``. Remaining
    entries correspond to ``secondary_intents`` in the same order.
    """

    intent: str
    extracted_data: Optional[ExtractedData] = None


class GuardrailValidation(BaseModel):
    """Result of guardrail validation on AI output."""

    all_passed: bool = True
    guardrails_run: int = 0
    guardrails_passed: int = 0
    blocking_failures: List[str] = []
    warnings: List[str] = []
    factual_accuracy: float = Field(ge=0.0, le=1.0, default=1.0)
    results: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Individual guardrail check results: pass/fail, severity, expected/found, messages",
    )


class ClassifyResponse(BaseModel):
    """Response from email classification."""

    classification: (
        str  # COOPERATIVE, PROMISE, DISPUTE, HOSTILE, QUERY, OUT_OF_OFFICE, UNSUBSCRIBE, OTHER
    )
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: Optional[str] = None
    secondary_intents: Optional[List[str]] = None
    # Flat extraction — kept for backward compat. Populated with the primary
    # intent's extraction so pre-PR4 consumers still work.
    extracted_data: Optional[ExtractedData] = None
    # PR4: per-intent extraction. When present, each handler picks its own
    # extracted_data here instead of the flat field above.
    intent_details: Optional[List[IntentDetail]] = None
    tokens_used: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    # Guardrail validation results
    guardrail_validation: Optional[GuardrailValidation] = None
    # Provider metadata
    provider: Optional[str] = None
    model: Optional[str] = None
    is_fallback: bool = False


class PersonaResult(BaseModel):
    """Generated persona for a single contact."""

    name: str
    level: int
    communication_style: Optional[str] = None
    formality_level: Optional[str] = None
    emphasis: Optional[str] = None


class GeneratePersonaResponse(BaseModel):
    """Response from persona generation."""

    personas: List[PersonaResult]
    tokens_used: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    is_fallback: bool = False


class RefinePersonaResponse(BaseModel):
    """Response from persona refinement."""

    communication_style: str
    formality_level: str
    emphasis: str
    reasoning: str
    tokens_used: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
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
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    # Guardrail validation results
    guardrail_validation: Optional[GuardrailValidation] = None
    # Provider metadata
    provider: Optional[str] = None
    model: Optional[str] = None
    is_fallback: bool = False
    # Structured reasoning from LLM
    reasoning: Optional[Dict[str, Any]] = None
    primary_cta: Optional[str] = None
    follow_up_days: Optional[int] = None


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
    provider: str  # "vertex", "openai", etc.
    model: str
    fallback_provider: Optional[str] = None
    fallback_model: Optional[str] = None
    fallback_count: int = 0
    model_available: bool = True
    fallback_available: bool = False
    uptime_seconds: Optional[float] = None

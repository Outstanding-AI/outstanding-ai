"""
Pydantic models for validating LLM responses.

These models ensure type safety when parsing LLM outputs and provide
clear error messages when the LLM returns malformed data.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from src.config.constants import CLASSIFICATION_CATEGORIES


class LLMExtractedData(BaseModel):
    """Data extracted from email content by the LLM."""

    # PROMISE_TO_PAY
    promise_date: Optional[str] = None  # String from LLM, parsed to date in engine
    promise_amount: Optional[float] = None
    # PROMISE_TO_PAY — strength of the commitment. Captured for future
    # use only (Codex P2, 2026-04-28). At the moment lane reply routing
    # treats every PROMISE_TO_PAY identically (full ``promise_suppressed``
    # for ``promise_date + promise_grace_days``); strength-aware
    # suppression (e.g. ``aspirational`` → next-touch defer instead of
    # full lane suppression) is Sprint B+ work.
    promise_strength: Optional[Literal["firm", "soft", "aspirational"]] = None
    # DISPUTE
    dispute_type: Optional[str] = None
    dispute_reason: Optional[str] = None
    invoice_refs: Optional[list[str]] = None
    disputed_amount: Optional[float] = None
    # ALREADY_PAID
    claimed_amount: Optional[float] = None
    claimed_date: Optional[str] = None
    claimed_reference: Optional[str] = None
    claimed_details: Optional[str] = None
    # INSOLVENCY
    insolvency_type: Optional[str] = None
    insolvency_details: Optional[str] = None
    administrator_name: Optional[str] = None
    administrator_email: Optional[str] = None
    reference_number: Optional[str] = None
    # OUT_OF_OFFICE
    return_date: Optional[str] = None
    # REDIRECT
    redirect_name: Optional[str] = None
    redirect_contact: Optional[str] = None  # Kept for backward compat
    redirect_email: Optional[str] = None
    # EMAIL_BOUNCE — AI fallback for the bounced recipient when the backend's
    # thread-based lookup can't resolve it. Classifier prompt already asks the
    # model to emit this; without the field here it was silently dropped.
    bounced_email: Optional[str] = None
    # SCOPE — set True only when the debtor uses explicit account-wide
    # language ("all invoices", "full balance", "everything outstanding",
    # "the whole account"). False (default) means scope must be resolved
    # from invoice_refs OR the message's tracked-thread lane. Without this
    # signal, downstream handlers that have no invoice_refs must NOT
    # default to all open obligations — they should restrict to the
    # message's lane (see ETL classification_service._resolve_promise_scope).
    account_wide: Optional[bool] = None


class IntentDetailLLM(BaseModel):
    """Per-intent extraction bundle from multi-intent classification.

    Before PR4, a multi-intent email (e.g. "paid invoice A, promising to pay
    invoice B next week") returned a single flat ``extracted_data`` shared
    across primary + ``secondary_intents``. That conflated per-intent fields —
    the ``claimed_reference`` belonged to ALREADY_PAID while ``promise_date``
    belonged to PROMISE_TO_PAY, but the consumer had no way to know which
    intent each field was for when multiple intents used overlapping fields
    (e.g. ``invoice_refs``).

    ``intent_details`` fixes that by giving each detected intent its own
    isolated ``extracted_data`` block. Backend routes each handler call to
    the matching entry.
    """

    intent: str = Field(
        ...,
        description="Classification category for this detail block (same vocabulary as top-level classification)",
    )
    extracted_data: Optional[LLMExtractedData] = Field(
        default=None,
        description="Fields extracted specifically for this intent",
    )


class ClassificationLLMResponse(BaseModel):
    """
    Expected response structure from classification LLM calls.

    The LLM must return JSON matching this schema.
    """

    classification: str = Field(
        ...,
        description="Email classification category",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1",
    )
    reasoning: Optional[str] = Field(
        None,
        description="Explanation for the classification",
    )
    secondary_intents: Optional[list[str]] = Field(
        default=None,
        description="Additional intents detected in multi-intent emails. "
        "Retained for backward compat — prefer intent_details when both are present.",
    )
    extracted_data: Optional[LLMExtractedData] = Field(
        None,
        description="Flat extraction (legacy). Keep populated for the primary intent so "
        "consumers that have not yet upgraded still work. When intent_details is provided, "
        "consumers should prefer per-intent extraction.",
    )
    intent_details: Optional[list[IntentDetailLLM]] = Field(
        default=None,
        description="Per-intent extraction (PR4). First entry must be the primary intent "
        "(must match the top-level classification field). Subsequent entries match "
        "secondary_intents in the same order. Optional — consumers fall back to the "
        "flat extracted_data when absent.",
    )
    forbidden_content_detected: list[dict] = Field(
        default_factory=list,
        description="Forbidden-content patterns detected in the inbound email body.",
    )

    @field_validator("classification")
    @classmethod
    def validate_classification(cls, v: str) -> str:
        upper_v = v.upper()
        if upper_v not in CLASSIFICATION_CATEGORIES:
            raise ValueError(
                f"Invalid classification '{v}'. Must be one of: {', '.join(sorted(CLASSIFICATION_CATEGORIES))}"
            )
        return upper_v


class DraftReasoningResponse(BaseModel):
    """Structured reasoning from the LLM about its draft generation decisions."""

    tone_rationale: str = Field(default="", description="Why this tone fits the debtor's situation")
    strategy: str = Field(default="", description="Approach given debtor behavior and history")
    key_factors: List[str] = Field(
        default_factory=list, description="Key factors that influenced the draft"
    )


class DraftGenerationLLMResponse(BaseModel):
    """
    Expected response structure from draft generation LLM calls.

    The LLM must return JSON matching this schema.
    """

    subject: str = Field(
        ...,
        min_length=1,
        description="Email subject line",
    )
    body: str = Field(
        ...,
        min_length=1,
        description="Email body content",
    )
    reasoning: Optional[DraftReasoningResponse] = Field(
        default=None,
        description="Structured reasoning about tone, strategy, and key factors",
    )
    primary_cta: Optional[str] = Field(
        default=None,
        description="Primary call-to-action type",
    )
    follow_up_days: Optional[int] = Field(
        default=None,
        description="Suggested follow-up period in days",
    )
    invoices_referenced: Optional[List[str]] = Field(
        default=None,
        description="Invoice numbers referenced in the email",
    )


class PersonaLLMResponse(BaseModel):
    """Expected response from persona generation LLM calls (cold start)."""

    communication_style: str = Field(
        ...,
        max_length=200,
        description="Voice direction, e.g. 'direct and authoritative'",
    )
    formality_level: str = Field(
        ...,
        description="Register: casual, conversational, professional, or formal",
    )
    emphasis: str = Field(
        ...,
        max_length=200,
        description="Focus area, e.g. 'building rapport and finding solutions'",
    )

    @field_validator("formality_level")
    @classmethod
    def validate_formality(cls, v: str) -> str:
        valid = {"casual", "conversational", "professional", "formal"}
        lower_v = v.lower()
        if lower_v not in valid:
            raise ValueError(f"formality_level must be one of: {', '.join(sorted(valid))}")
        return lower_v


class PersonaRefinementLLMResponse(BaseModel):
    """Expected response from persona refinement LLM calls."""

    communication_style: str = Field(
        ...,
        max_length=200,
        description="Updated voice direction",
    )
    formality_level: str = Field(
        ...,
        description="Updated register",
    )
    emphasis: str = Field(
        ...,
        max_length=200,
        description="Updated focus area",
    )
    reasoning: str = Field(
        ...,
        max_length=300,
        description="Brief explanation of what changed and why",
    )

    @field_validator("formality_level")
    @classmethod
    def validate_formality(cls, v: str) -> str:
        valid = {"casual", "conversational", "professional", "formal"}
        lower_v = v.lower()
        if lower_v not in valid:
            raise ValueError(f"formality_level must be one of: {', '.join(sorted(valid))}")
        return lower_v

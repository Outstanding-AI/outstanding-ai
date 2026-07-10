"""
Pydantic models for validating LLM responses.

These models ensure type safety when parsing LLM outputs and provide
clear error messages when the LLM returns malformed data.
"""

import re
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.config.constants import CLASSIFICATION_CATEGORIES

MATERIAL_SCOPE_INTENTS = frozenset(
    {
        "ALREADY_PAID",
        "AMOUNT_DISAGREEMENT",
        "DISPUTE",
        "DEBTOR_INTERNAL_PROCESSING_BLOCKER",
        "HARDSHIP",
        "PARTIAL_PAYMENT_NOTIFICATION",
        "PAYMENT_CONFIRMATION",
        "PAYMENT_TIMING_DISPUTE",
        "PLAN_REQUEST",
        "PROMISE_TO_PAY",
        "REMITTANCE_ADVICE",
        "RETENTION_CLAIM",
    }
)


def _normalize_invoice_ref(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def _dedupe_preserve_order(values: list[str] | None) -> list[str] | None:
    if not values:
        return values
    seen: set[str] = set()
    cleaned: list[str] = []
    for value in values:
        normalized = _normalize_invoice_ref(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(value)
    return cleaned


class LLMExtractedData(BaseModel):
    """Data extracted from email content by the LLM."""

    # PROMISE_TO_PAY
    promise_date: Optional[str] = None  # String from LLM, parsed to date in engine
    promise_amount: Optional[float] = None
    # PROMISE_TO_PAY — strength of the commitment. Captured for future
    # use only (2026-04-28). At the moment lane reply routing
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
    # PAYMENT_TIMING_DISPUTE
    claimed_due_date: Optional[str] = None
    claimed_payment_date: Optional[str] = None
    payment_timing_reason: Optional[str] = None
    # DEBTOR_INTERNAL_PROCESSING_BLOCKER
    internal_blocker_type: Optional[
        Literal[
            "goods_receipt_missing",
            "po_issue",
            "approval_pending",
            "payment_run_pending",
            "portal_processing",
            "internal_review",
            "other",
        ]
    ] = None
    internal_blocker_reason: Optional[str] = None
    internal_blocker_owner_hint: Optional[str] = None
    # Document / workflow references extracted from debtor-authored text or
    # debtor-provided internal forwards. These are evidence fields only; the
    # workflow resolver still validates them against same-tenant party data.
    po_refs: Optional[list[str]] = None
    grn_refs: Optional[list[str]] = None
    sales_order_refs: Optional[list[str]] = None
    purchase_order_refs: Optional[list[str]] = None
    document_refs: Optional[list[str]] = None
    approval_owner_hint: Optional[str] = None
    dependency_status: Optional[Literal["pending", "satisfied", "unknown"]] = None
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


class ForbiddenContentFinding(BaseModel):
    """Strict item schema for structured-output forbidden-content findings."""

    model_config = ConfigDict(extra="forbid")

    category: Literal[
        "bank_payment_details",
        "legal_statute_quotation",
        "unauthorized_offer_claim",
        "external_url",
        "prompt_injection_attempt",
    ]
    excerpt: str = Field(default="", max_length=200)

    @field_validator("excerpt", mode="before")
    @classmethod
    def truncate_excerpt(cls, value: object) -> str:
        return str(value or "")[:200]


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
    forbidden_content_detected: list[ForbiddenContentFinding] = Field(
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

    @model_validator(mode="after")
    def validate_intent_details_scope(self) -> "ClassificationLLMResponse":
        if not self.intent_details:
            return self

        primary_intent = str(self.classification or "").upper()
        first_intent = str(self.intent_details[0].intent or "").upper()
        if first_intent != primary_intent:
            raise ValueError("intent_details[0].intent must match classification")

        seen_invoice_refs: dict[str, str] = {}
        retained_details: list[IntentDetailLLM] = []
        retained_secondary_intents: list[str] = []
        for index, detail in enumerate(self.intent_details):
            intent = str(detail.intent or "").upper()
            detail.intent = intent
            if intent not in CLASSIFICATION_CATEGORIES:
                raise ValueError(f"Invalid intent_details[{index}].intent '{detail.intent}'")
            if index > 0 and intent in MATERIAL_SCOPE_INTENTS and detail.extracted_data is None:
                raise ValueError(
                    f"intent_details[{index}].extracted_data is required for material intent {intent}"
                )
            if not detail.extracted_data:
                retained_details.append(detail)
                if index > 0:
                    retained_secondary_intents.append(intent)
                continue

            detail.extracted_data.invoice_refs = _dedupe_preserve_order(
                detail.extracted_data.invoice_refs
            )
            retained_refs: list[str] = []
            dropped_duplicate_scope = False
            for raw_ref in detail.extracted_data.invoice_refs or []:
                invoice_ref = _normalize_invoice_ref(raw_ref)
                if not invoice_ref:
                    continue
                previous_intent = seen_invoice_refs.get(invoice_ref)
                if previous_intent and previous_intent != intent:
                    dropped_duplicate_scope = True
                    continue
                seen_invoice_refs[invoice_ref] = intent
                retained_refs.append(raw_ref)

            detail.extracted_data.invoice_refs = retained_refs or None
            if (
                index > 0
                and intent in MATERIAL_SCOPE_INTENTS
                and dropped_duplicate_scope
                and not retained_refs
                and not detail.extracted_data.account_wide
            ):
                # The same invoice cannot safely drive two material side-effects.
                # Keep the first/primary interpretation and drop the now-unscoped
                # secondary intent rather than rejecting the whole LLM response.
                continue

            retained_details.append(detail)
            if index > 0:
                retained_secondary_intents.append(intent)

        self.intent_details = retained_details
        if self.secondary_intents is not None:
            self.secondary_intents = [
                str(intent).upper()
                for intent in retained_secondary_intents
                if str(intent).upper() in CLASSIFICATION_CATEGORIES
            ]
        return self


class HistoricalThreadActionLLM(BaseModel):
    """Closed schema item for debtor-level thread adjudication.

    OpenAI strict structured outputs reject free-form mapping item schemas. Keep
    the provider-facing shape as a closed list while accepting the legacy dict
    form in validators below for tests and older model output.
    """

    model_config = ConfigDict(extra="forbid")

    conversation_id: str = Field(
        default="", description="Provider conversation id being adjudicated"
    )
    action: Literal["active", "superseded", "closed_history", "needs_review", "ignore"] = Field(
        default="needs_review",
        description="Recommended action for this candidate thread",
    )


class HistoricalIntentDetailLLM(BaseModel):
    """Closed schema item for historical per-intent facts.

    This is intentionally narrower than live debtor-reply extraction. Historical
    protocol classification is review evidence only; detailed invoice state is
    already carried by deterministic audit fields.
    """

    model_config = ConfigDict(extra="forbid")

    intent: str = Field(default="", max_length=100)
    invoice_refs: list[str] = Field(default_factory=list)
    evidence_message_ids: list[str] = Field(default_factory=list)
    summary: str = Field(default="", max_length=800)
    amount: Optional[float] = None
    date: Optional[str] = Field(default=None, max_length=40)
    status: Optional[str] = Field(default=None, max_length=100)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class HistoricalCollectionThreadLLMResponse(BaseModel):
    """Structured response for historical collection-thread evidence."""

    model_config = ConfigDict(extra="forbid")

    classification: Optional[str] = Field(
        default=None,
        description="Semantic message classification or debtor-thread recommendation label.",
    )
    protocol_touch_type: Optional[
        Literal[
            "initial_reminder",
            "same_level_reminder",
            "same_level_follow_up",
            "debtor_reply_response",
            "cross_contact_escalation",
            "same_contact_escalation",
            "promise_acknowledgement",
            "remittance_acknowledgement",
            "manual_off_protocol_touch",
            "non_collection_or_auto",
            "unknown",
        ]
    ] = None
    is_escalation: Optional[bool] = None
    escalation_kind: Optional[Literal["same_contact", "cross_contact", "none", "unclear"]] = None
    debtor_reply_response: Optional[bool] = None
    commitment_acknowledgement_type: Optional[
        Literal["promise_acknowledgement", "remittance_acknowledgement", "none", "unclear"]
    ] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = Field(default="", max_length=800)
    evidence_message_ids: list[str] = Field(default_factory=list)
    recommended_active_thread_id: Optional[str] = None
    thread_actions: list[HistoricalThreadActionLLM] = Field(default_factory=list)
    guardrail_warnings: list[str] = Field(default_factory=list)
    secondary_intents: list[str] = Field(default_factory=list)
    intent_details: list[HistoricalIntentDetailLLM] = Field(default_factory=list)
    relevance_label: Optional[Literal["collection_related", "non_collection", "uncertain"]] = None
    signal_codes: list[str] = Field(default_factory=list, max_length=20)
    evidence_message_ordinals: list[int] = Field(default_factory=list, max_length=50)
    abstention_reason: Optional[str] = Field(default=None, max_length=120)

    @field_validator("thread_actions", mode="before")
    @classmethod
    def coerce_thread_actions(cls, value: object) -> object:
        if isinstance(value, dict):
            return [
                {"conversation_id": str(key), "action": action} for key, action in value.items()
            ]
        return value

    @field_validator("intent_details", mode="before")
    @classmethod
    def coerce_intent_details(cls, value: object) -> object:
        if not isinstance(value, list):
            return value
        coerced: list[dict[str, object]] = []
        for item in value:
            if isinstance(item, dict):
                coerced.append(
                    {
                        "intent": str(item.get("intent") or item.get("classification") or ""),
                        "invoice_refs": item.get("invoice_refs") or [],
                        "evidence_message_ids": item.get("evidence_message_ids") or [],
                        "summary": str(
                            item.get("summary") or item.get("reason") or item.get("details") or ""
                        ),
                        "amount": item.get("amount"),
                        "date": item.get("date"),
                        "status": item.get("status"),
                        "confidence": item.get("confidence"),
                    }
                )
            else:
                coerced.append(item)
        return coerced

    def thread_actions_dict(self) -> dict[str, str]:
        return {
            item.conversation_id: item.action
            for item in self.thread_actions
            if item.conversation_id
        }

    def intent_details_payload(self) -> list[dict]:
        return [item.model_dump(exclude_none=True) for item in self.intent_details]


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

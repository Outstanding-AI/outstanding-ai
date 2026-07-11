from datetime import date
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ExtractedData(BaseModel):
    """Data extracted from email by AI."""

    # PROMISE_TO_PAY
    promise_date: Optional[date] = None
    promise_amount: Optional[float] = None
    # PROMISE_TO_PAY — strength of the commitment. ``firm`` (default) takes
    # full grace_days suppression downstream; ``soft`` takes half;
    # ``aspirational`` only defers next_touch_due_at without suppressing
    # the lane. See LLMExtractedData docstring for the full contract.
    promise_strength: Optional[Literal["firm", "soft", "aspirational"]] = None
    # DISPUTE
    dispute_type: Optional[str] = None
    dispute_reason: Optional[str] = None
    invoice_refs: Optional[List[str]] = None
    disputed_amount: Optional[float] = None
    # PAYMENT_TIMING_DISPUTE
    claimed_due_date: Optional[date] = None
    claimed_payment_date: Optional[date] = None
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
    # debtor-provided internal forwards.
    po_refs: Optional[List[str]] = None
    grn_refs: Optional[List[str]] = None
    sales_order_refs: Optional[List[str]] = None
    purchase_order_refs: Optional[List[str]] = None
    document_refs: Optional[List[str]] = None
    approval_owner_hint: Optional[str] = None
    dependency_status: Optional[Literal["pending", "satisfied", "unknown"]] = None
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
    # SCOPE — True iff debtor used explicit account-wide language
    # ("all invoices", "full balance", "everything outstanding"). Drives
    # the ETL scope resolver's fallback gate when invoice_refs is empty.
    account_wide: Optional[bool] = None


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
    review_findings: List[dict] = []
    factual_accuracy: float = Field(ge=0.0, le=1.0, default=1.0)
    results: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Individual guardrail check results: pass/fail, severity, expected/found, messages",
    )


class UsageBreakdownEntry(BaseModel):
    """Per-suboperation usage rollup for a single LLM call.

    Used inside ``UsageBreakdown.main_generation`` and per-guardrail
    entries under ``UsageBreakdown.guardrails``. The shape is a hard
    allowlist on the backend side (see
    ``services/ai_engine/_telemetry.py::_USAGE_SUBOP_ALLOWED_KEYS``);
    keep this Pydantic model aligned so additions land via the
    contracts bump cycle, not silent drift.
    """

    provider: Optional[str] = None
    model: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    cost_usd: Optional[float] = None
    passed: Optional[bool] = None
    blocking: Optional[bool] = None


class UsageBreakdown(BaseModel):
    """Per-suboperation usage breakdown attached to AI responses.

    ``main_generation`` covers the primary LLM call. ``guardrails`` is
    keyed by guardrail name and rolls up each guardrail's tokens /
    latency / pass-block state. Only guardrails that actually ran
    appear; deterministic guardrails with zero LLM cost still appear
    so the dashboard can attribute their latency contribution.
    """

    main_generation: Optional[UsageBreakdownEntry] = None
    guardrails: Optional[Dict[str, UsageBreakdownEntry]] = None


class AIAuditMetadata(BaseModel):
    """Prompt/model/lineage metadata for Silver Application audit tables."""

    ai_provider: Optional[str] = None
    ai_model: Optional[str] = None
    ai_region: Optional[str] = None
    prompt_template_id: Optional[str] = None
    prompt_template_version: Optional[str] = None
    system_prompt_hash: Optional[str] = None
    user_prompt_hash: Optional[str] = None
    prompt_input_hash: Optional[str] = None
    guardrail_pipeline_version: Optional[str] = None
    guardrail_result_ids: Optional[List[str]] = None
    input_silver_version_ids_json: Optional[str] = None
    input_sent_draft_analysis_event_ids_json: Optional[str] = None
    input_sent_draft_analysis_hashes_json: Optional[str] = None
    policy_snapshot_id: Optional[str] = None
    draft_candidate_id: Optional[str] = None
    draft_generation_run_id: Optional[str] = None
    source_sync_run_id: Optional[str] = None
    application_run_id: Optional[str] = None
    token_count: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    # Model invocation audit (May 2026) — EU AI Act Article 13 readiness.
    # ``model_invocation_config`` is the SANITIZED dict of explicit knobs we
    # passed to the SDK (per-provider allow-key list in src/llm/_invocation_audit.py).
    # NEVER includes prompt text, customer data, system_instruction, or raw
    # SDK config objects. The hash is over the sanitized dict only — SDK
    # version is NOT folded into the hash; it has its own dedicated field.
    model_invocation_config: Optional[Dict[str, Any]] = None
    model_invocation_config_hash: Optional[str] = None
    model_version_fingerprint: Optional[str] = (
        None  # vertex response.model_version / openai system_fingerprint
    )
    sdk_library: Optional[str] = None  # google-genai | langchain-openai | anthropic
    sdk_version: Optional[str] = None  # importlib.metadata version of the wrapper-of-record
    inference_profile: Optional[str] = (
        None  # draft_generation | classification | persona_gen | persona_refine | sent_scope_analysis
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
    forbidden_content_detected: List[dict] = []
    # Guardrail validation results
    guardrail_validation: Optional[GuardrailValidation] = None
    # Provider metadata
    provider: Optional[str] = None
    model: Optional[str] = None
    is_fallback: bool = False
    classification_evidence_only: bool = True
    ai_audit: Optional[AIAuditMetadata] = None


class HistoricalCollectionThreadResponse(BaseModel):
    """Response from historical collection-thread protocol, adjudication, or relevance classification."""

    classification: Optional[str] = None
    protocol_touch_type: Optional[str] = None
    is_escalation: Optional[bool] = None
    escalation_kind: Optional[str] = None
    debtor_reply_response: Optional[bool] = None
    commitment_acknowledgement_type: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    reason: Optional[str] = None
    evidence_message_ids: List[str] = []
    recommended_active_thread_id: Optional[str] = None
    thread_actions: Dict[str, str] = {}
    guardrail_warnings: List[str] = []
    secondary_intents: List[str] = []
    intent_details: List[dict] = []
    tokens_used: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    is_fallback: bool = False
    ai_audit: Optional[AIAuditMetadata] = None
    relevance_label: Optional[str] = None
    signal_codes: List[str] = []
    evidence_message_ordinals: List[int] = []
    abstention_reason: Optional[str] = None
    selected_candidate_key: Optional[str] = None
    selection_action: Optional[str] = None


class CollectionEmailEventResponse(BaseModel):
    relevance_status: str
    lifecycle_status: str
    semantic_classification: Optional[str] = None
    secondary_intents: List[str] = []
    intent_details: List[IntentDetail] = []
    invoice_assertions: List[str] = []
    amount_assertions: List[dict] = []
    date_assertions: List[dict] = []
    reason_codes: List[str] = []
    confidence: float = Field(ge=0.0, le=1.0)
    tokens_used: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    is_fallback: bool = False
    ai_audit: Optional[AIAuditMetadata] = None


class CollectionEmailFactExtractionResponse(BaseModel):
    invoice_assertions: List[str] = []
    amount_assertions: List[dict] = []
    date_assertions: List[dict] = []
    confidence: float = Field(ge=0.0, le=1.0)
    reason_codes: List[str] = []
    tokens_used: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    is_fallback: bool = False
    ai_audit: Optional[AIAuditMetadata] = None


class CollectionChainIdentificationResponse(BaseModel):
    collection_status: Literal["collection", "non_collection", "uncertain"]
    event_effect: Literal["new", "confirmed", "reopened", "closed", "no_change"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason_codes: List[str] = []
    evidence_message_ordinals: List[int] = []
    tokens_used: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    is_fallback: bool = False
    ai_audit: Optional[AIAuditMetadata] = None


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
    ai_audit: Optional[AIAuditMetadata] = None


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
    ai_audit: Optional[AIAuditMetadata] = None


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
    ai_audit: Optional[AIAuditMetadata] = None
    # Stage 3 (#8): per-suboperation usage rollup (main + per-guardrail).
    # Backend telemetry persists the dict into ``LLMRequestLog.metadata.
    # usage_breakdown`` once the AI engine emits it.
    usage_breakdown: Optional[UsageBreakdown] = None


class GenerateDraftFromManifestCandidateResult(BaseModel):
    """Draft generation result for one regional manifest candidate."""

    candidate_id: str
    party_id: str
    lane_id: str
    status: Literal["generated", "failed"]
    draft: Optional[GenerateDraftResponse] = None
    error: Optional[str] = None


class GenerateDraftFromManifestResponse(BaseModel):
    """Response from regional manifest-based draft generation."""

    tenant_id: str
    sync_run_id: str
    data_lake_region: str
    total: int
    generated_count: int
    failed_count: int
    status: Literal["completed", "partial_failed", "failed"]
    results: List[GenerateDraftFromManifestCandidateResult]


class SentDraftInvoiceScopeDecision(BaseModel):
    """One invoice decision from final sent-email scope analysis."""

    invoice_number: str
    obligation_id: Optional[str] = None
    status: Literal[
        "retained_generated_invoice",
        "operator_added_invoice",
        "removed_generated_invoice",
        "not_present",
        "ambiguous",
    ]
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    evidence: Optional[str] = None


class AnalyzeSentDraftScopeResponse(BaseModel):
    """Structured extraction of the invoice scope that was actually sent."""

    invoice_refs_sent: List[str] = []
    invoice_refs_retained: List[str] = []
    invoice_refs_operator_added: List[str] = []
    invoice_refs_removed: List[str] = []
    invoice_refs_ambiguous: List[str] = []
    invoice_scope_changed: bool = False
    scope_extraction_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    scope_extraction_status: Literal["succeeded", "review_required", "failed"] = "succeeded"
    review_recommended: bool = False
    review_reason_codes: List[str] = []
    decisions: List[SentDraftInvoiceScopeDecision] = []
    reasoning: Optional[str] = None
    tokens_used: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    scope_extraction_llm_request_id: Optional[str] = None
    is_fallback: bool = False
    ai_audit: Optional[AIAuditMetadata] = None


class GateResult(BaseModel):
    """Result of a single gate evaluation."""

    passed: bool
    reason: str
    current_value: Optional[Any] = None
    threshold: Optional[Any] = None


# EvaluateGatesResponse, PartyGateResult, EvaluateGatesBatchResponse removed
# 2026-04-26 alongside the /evaluate-gates route deletion. Gate evaluation
# moved to backend services/gate_checker.py (CLAUDE.md note #40); the AI-side
# response models had no remaining consumers. GateResult retained — it's a
# generic enough shape that future endpoints may want to surface gate-style
# outcomes.


class HealthResponse(BaseModel):
    """Health check response."""

    status: str  # "healthy", "degraded", "unhealthy"
    version: str
    provider: str  # "vertex", "openai", etc.
    model: str
    worker_class: str = "medium"
    fallback_provider: Optional[str] = None
    fallback_model: Optional[str] = None
    fallback_count: int = 0
    primary_failures_by_caller: Dict[str, int] = Field(default_factory=dict)
    model_available: bool = True
    fallback_available: bool = False
    uptime_seconds: Optional[float] = None

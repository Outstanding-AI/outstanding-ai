"""
Context models for AI operations.

Contains CaseContext, CommunicationInfo, ObligationInfo, IndustryInfo,
and related context models used across classification, generation,
and gate evaluation.
"""

import warnings
from datetime import datetime
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from solvix_contracts.ai.context.v2 import (
    CaseContextV2,
    CommunicationInfoV2,
)
from solvix_contracts.ai.context.v3 import (
    CollectionMailV3,
    DisputeEventV3,
    EvidenceV3,
    ObligationInfoV3,
    PaymentPlanInfoV3,
    StateTransitionV3,
    ThreadMessageV3,
    WorkflowEventV3,
)


def _coalesce_lane_value(*values):
    """Return the first non-empty lane value."""
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        return value
    return None


def _normalize_lane_context(
    raw_context: dict[str, Any],
    lane: dict[str, Any],
    collection_lane_id: str | None,
) -> dict[str, Any]:
    """Backfill sparse lane_context payloads from the canonical lane dict."""
    normalized = dict(raw_context)
    lane_id = _coalesce_lane_value(
        normalized.get("lane_id"),
        normalized.get("collection_lane_id"),
        collection_lane_id,
        lane.get("collection_lane_id"),
    )
    if lane_id is not None:
        normalized["lane_id"] = str(lane_id)

    current_level = _coalesce_lane_value(
        normalized.get("current_level"),
        lane.get("current_level"),
        lane.get("entry_level"),
        0,
    )
    if current_level is not None:
        normalized["current_level"] = current_level

    for field_name in (
        "entry_level",
        "stage_version",
        "scheduled_touch_index",
        "max_touches_for_level",
        "reminder_cadence_days_for_level",
        "max_days_for_level",
        "outstanding_amount",
        "protocol_due_at",
        "not_before_at",
        "is_forecast",
        "generation_policy_mode",
        "lookahead_window_start",
        "lookahead_window_end",
        "planned_send_at",
    ):
        if normalized.get(field_name) is None and lane.get(field_name) is not None:
            normalized[field_name] = lane.get(field_name)

    if not normalized.get("tone_ladder"):
        normalized["tone_ladder"] = lane.get("tone_ladder") or []
    if normalized.get("outstanding_amount") is None:
        normalized["outstanding_amount"] = lane.get("outstanding_amount") or 0.0

    return normalized


class ObligationInfo(ObligationInfoV3):
    """Single invoice/obligation.

    Extends the V3 contract: V3 is a strict superset of V2 (adds
    ``transaction_date``, ``collection_status`` + per-status sub-fields,
    ``is_sendable``, ``block_reasons``). V2 callers that don't populate
    those send a valid subset; V3 callers fill them in.
    """

    # Silver Core / Silver Application datalake context. These are additive
    # transition fields until the shared contract package ships the next schema.
    silver_version_id: Optional[str] = None
    document_no: Optional[str] = None
    sage_transaction_urn: Optional[str] = None
    document_currency_code: Optional[str] = None
    is_outstanding: Optional[bool] = None
    is_overdue: Optional[bool] = None
    days_overdue: Optional[int] = None
    effective_grace_days: Optional[int] = None
    is_chase_eligible: Optional[bool] = None
    source_query_raw: Optional[str] = None
    has_source_query_flag: Optional[bool] = None
    is_source_disputed: Optional[bool] = None
    source_dispute_type: Optional[str] = None
    source_dispute_observed_from: Optional[
        Literal["sales_posted_transactions", "sales_transaction_enquiry_views", "both"]
    ] = None
    has_verified_purchase_order: Optional[bool] = None
    has_verified_pod: Optional[bool] = None
    procurement_context_status: Optional[
        Literal[
            "verified",
            "candidate_reference",
            "missing",
            "rejected",
            "not_applicable",
            "manual",
        ]
    ] = None
    purchase_order_reference: Optional[str] = None
    pod_reference: Optional[str] = None
    allocated_credit_amount_native: Optional[float] = None
    allocated_credit_amount_base: Optional[float] = None
    credit_note_count: Optional[int] = None
    net_amount_due_after_credit_native: Optional[float] = None
    net_amount_due_after_credit_base: Optional[float] = None
    credit_adjustment_status: Optional[str] = None


class CreditPositionInfo(BaseModel):
    """Debtor/currency credit-note position supplied by the backend."""

    currency_code: str
    base_currency: Optional[str] = None
    unapplied_credit_amount_native: float = 0.0
    unapplied_credit_amount_base: Optional[float] = None
    financial_overdue_amount_native: float = 0.0
    financial_overdue_amount_base: Optional[float] = None
    recovery_eligible_overdue_amount_native: float = 0.0
    recovery_eligible_overdue_amount_base: Optional[float] = None
    net_recovery_eligible_overdue_native: float = 0.0
    net_recovery_eligible_overdue_base: Optional[float] = None
    credit_fully_covers_recovery_overdue: bool = False
    credit_partially_covers_recovery_overdue: bool = False
    requires_credit_review: bool = False
    credit_note_refs: List[dict[str, Any]] = []


class InvoiceCreditAdjustmentInfo(BaseModel):
    """Invoice-level allocated credit-note adjustment."""

    obligation_id: str
    invoice_number: Optional[str] = None
    currency_code: Optional[str] = None
    base_currency: Optional[str] = None
    invoice_amount_due_before_credit_native: Optional[float] = None
    invoice_amount_due_before_credit_base: Optional[float] = None
    allocated_credit_amount_native: float = 0.0
    allocated_credit_amount_base: Optional[float] = None
    invoice_amount_due_after_credit_native: Optional[float] = None
    invoice_amount_due_after_credit_base: Optional[float] = None
    credit_note_count: int = 0
    credit_note_refs: List[dict[str, Any]] = []


class CommunicationInfo(CommunicationInfoV2):
    """Communication history summary."""

    @model_validator(mode="after")
    def warn_deprecated_last_response_snippet(self) -> "CommunicationInfo":
        """Warn when legacy response snippets are still being passed."""
        legacy_snippet = self.__dict__.get("last_response_snippet")
        if legacy_snippet is not None:
            warnings.warn(
                "CommunicationInfo.last_response_snippet is deprecated; "
                "use CaseContext.recent_messages[0].body_snippet instead.",
                DeprecationWarning,
                stacklevel=3,
            )
        return self


class TouchHistory(BaseModel):
    """Single touch record.

    Covers both AI-generated email touches and operator-logged manual
    touchpoints (phone, SMS, letter, in-person, voicemail, other). The
    ``touch_type`` discriminator drives the prompt-template split — the AI
    sees email touches in the existing Conversation History section and
    manual touches in a separate Recent Manual Touchpoints section, never
    mixed.
    """

    sent_at: datetime
    tone: Optional[str] = None
    sender_level: Optional[int] = None
    sender_name: Optional[str] = None
    had_response: bool = False
    # Manual-communication fields (NULL on email rows). Both ``touch_type``
    # and ``logged_by_user_name`` are required by the prompt template:
    # generator_prompts.py filters ``recent_touches`` to
    # ``touch_type == 'manual_log'`` and renders ``"logged by <name>"``.
    # Backend hydrates the name from App DB; AI engine treats it as opaque.
    touch_type: Optional[str] = None  # 'ai_email' | 'manual_log' | 'system'
    channel: Optional[str] = (
        None  # 'email' | 'phone' | 'sms' | 'letter' | 'in_person' | 'voicemail' | 'other'
    )
    direction: Optional[str] = None  # 'inbound' | 'outbound'
    manual_notes: Optional[str] = None
    manual_purpose: Optional[str] = None  # 'general' | 'query' | 'chase'
    manual_obligations: list[dict[str, Any]] = Field(default_factory=list)
    logged_by_user_name: Optional[str] = None


class PromiseHistory(BaseModel):
    """Single promise record."""

    promise_date: Optional[str] = None
    promise_amount: Optional[float] = None
    outcome: Optional[str] = None  # kept, broken, pending


class RemittanceHistory(BaseModel):
    """Single remittance evidence record."""

    remittance_received_at: Optional[str] = None
    remittance_amount: Optional[float] = None
    bank_reference: Optional[str] = None
    outcome: Optional[str] = None  # pending, fulfilled, expired_unfulfilled, cancelled


class ActualSentScopeHistory(BaseModel):
    """What the debtor actually received after operator edits."""

    sent_draft_analysis_event_id: Optional[str] = None
    application_content_hash: Optional[str] = None
    draft_id: Optional[str] = None
    touch_id: Optional[str] = None
    provider_message_id: Optional[str] = None
    lane_id: Optional[str] = None
    sent_at: Optional[datetime] = None
    invoice_refs_generated: List[str] = []
    invoice_refs_sent: List[str] = []
    invoice_refs_added: List[str] = []
    invoice_refs_removed: List[str] = []
    invoice_scope_changed: bool = False
    edit_severity: Optional[str] = None
    payment_expectation_added: bool = False
    payment_expectation_kind: Optional[str] = None
    payment_expectation_date: Optional[str] = None
    payment_expectation_amount: Optional[float] = None
    review_reason_codes: List[str] = []


class CommunicationTrackingInfo(BaseModel):
    """Thread monitoring coverage for communication-aware generation."""

    tracking_status: Optional[str] = None
    tracking_reason: Optional[str] = None
    send_confirmation_state: Optional[str] = None
    sent_proof_type: Optional[str] = None
    reply_anchor_email: Optional[str] = None
    is_ai_tracked_thread: Optional[bool] = None


class LaneContextInfo(BaseModel):
    """Lane-aware collection context for one draft generation request."""

    lane_id: str = Field(..., max_length=100)
    role: str = Field("single", pattern=r"^(owner|guest|single)$")
    current_level: int
    entry_level: Optional[int] = None
    stage_version: Optional[int] = None
    level_started_at: Optional[datetime] = None
    scheduled_touch_index: int = 0
    max_touches_for_level: Optional[int] = None
    reminder_cadence_days_for_level: Optional[int] = None
    max_days_for_level: Optional[int] = None
    tone_ladder: List[str] = []
    invoice_refs: List[str] = Field(
        default_factory=list,
        deprecated="Use CaseContext.lane.invoice_refs instead.",
    )
    outstanding_amount: float = Field(
        default=0.0,
        deprecated="Use CaseContext.lane.outstanding_amount instead.",
    )
    prior_touch_dates: List[str] = []
    is_newly_joined: bool = False
    action: Optional[str] = None
    obligation_ids: List[str] = []
    open_obligation_ids: List[str] = []
    overdue_obligation_ids: List[str] = []
    blocked_obligation_ids: List[str] = []
    protocol_slot_key: Optional[str] = None
    protocol_selected_day: Optional[int] = None
    protocol_selected_level: Optional[int] = None
    protocol_actual_sender_level: Optional[int] = None
    protocol_due_at: Optional[str] = None
    not_before_at: Optional[str] = None
    is_forecast: Optional[bool] = None
    generation_policy_mode: Optional[str] = None
    lookahead_window_start: Optional[str] = None
    lookahead_window_end: Optional[str] = None
    planned_send_at: Optional[str] = None


class IndustryInfo(BaseModel):
    """Industry-specific context for AI operations.

    Provides industry benchmarks and AI context that affects:
    - Draft tone and escalation speed
    - Dispute classification and handling
    - Hardship detection and response
    """

    code: str = Field(..., max_length=100)  # Industry identifier
    name: str = Field(..., max_length=200)  # Display name
    typical_dso_days: int  # Normal payment cycle for this industry
    alarm_dso_days: int  # DSO that signals concern
    payment_cycle: str = Field(..., max_length=20)  # immediate, net15, net30, net45, net60, net90
    escalation_patience: str = "standard"  # patient, standard, aggressive
    common_dispute_types: List[str] = []  # Expected dispute types
    hardship_indicators: List[str] = []  # Industry-specific hardship signals
    preferred_tone: str = "professional"  # formal, professional, casual
    ai_context_notes: str = Field("", max_length=2000)  # Free-form context for prompts
    seasonal_patterns: dict = {}  # Q1, Q2, Q3, Q4 patterns
    dispute_handling_notes: str = Field("", max_length=2000)  # How to handle disputes
    hardship_handling_notes: str = Field("", max_length=2000)  # How to handle hardship
    communication_notes: str = Field("", max_length=2000)  # Industry communication conventions


class CaseContext(CaseContextV2):
    """Full case context for AI operations.

    Defaults to the V4/current datalake payload. Legacy classification-only
    compatibility remains available when callers explicitly send
    ``schema_version`` 2 or 3. Draft-generation routes require V4 and reject
    older payloads before a model call can be made.
    """

    model_config = ConfigDict(extra="ignore")

    schema_version: Literal[2, 3, 4] = 4
    party: "PartyInfo"  # Forward ref resolved at module level
    behavior: Optional["BehaviorInfo"] = None  # Forward ref resolved at module level
    obligations: List[ObligationInfo] = Field(default_factory=list)
    communication: Optional[CommunicationInfo] = None
    communication_tracking: Optional[CommunicationTrackingInfo] = None
    recent_touches: List[TouchHistory] = []
    promises: List[PromiseHistory] = []
    remittances: List[RemittanceHistory] = []
    actual_sent_scope_history: List[ActualSentScopeHistory] = []
    party_credit_position_by_currency: List[CreditPositionInfo] = []
    invoice_credit_adjustments: List[InvoiceCreditAdjustmentInfo] = []
    credit_review_flags: List[str] = []
    net_recovery_eligible_by_currency: dict[str, float] = Field(default_factory=dict)

    # ------------------------------------------------------------------
    # V3-only top-level fields (Optional so V2 callers omit them safely).
    # ------------------------------------------------------------------
    thread_messages: List[ThreadMessageV3] = Field(default_factory=list)
    dispute_history: List[DisputeEventV3] = Field(default_factory=list)
    payment_plans: List[PaymentPlanInfoV3] = Field(default_factory=list)
    evidence: List[EvidenceV3] = Field(default_factory=list)
    case_state_history: List[StateTransitionV3] = Field(default_factory=list)
    recent_workflow_events: List[WorkflowEventV3] = Field(default_factory=list)
    collection_mails: List[CollectionMailV3] = Field(default_factory=list)
    case_state_changed_at: Optional[datetime] = None
    overrides_applied: dict[str, Any] = Field(default_factory=dict)

    # Case state
    case_state: Optional[str] = None
    base_currency: str = "GBP"
    total_outstanding_base: Optional[float] = None
    days_in_state: Optional[int] = None
    broken_promises_count: int = 0
    active_dispute: bool = False
    hardship_indicated: bool = False

    # Tenant settings (effective values after override resolution by Django)
    # These are the EFFECTIVE values: party.X_override OR tenant.X
    brand_tone: str = Field(
        "professional", max_length=50
    )  # Effective: party.tone_override OR tenant.brand_tone
    touch_cap: int = 10  # Effective: party.touch_cap_override OR tenant.touch_cap
    touch_interval_days: int = 3
    grace_days: int = 0  # Effective: party.grace_days_override OR tenant.grace_days

    # Promise verification settings
    promise_grace_days: int = 0

    # Debtor-specific context (NEW - for gate evaluation and draft generation)
    do_not_contact_until: Optional[str] = None  # ISO date if set (from party)
    monthly_touch_count: int = 0  # Current month's touch count (from party)
    relationship_tier: str = "standard"  # From party (vip, standard, high_risk)
    unsubscribe_requested: bool = False  # True if debtor opted out of communications

    # Industry context
    industry: Optional[IndustryInfo] = None

    # Extended tenant settings (passed through from Django)
    tenant_settings: Optional[dict] = None

    # Debtor contact details
    debtor_contact: Optional[dict] = None
    party_contacts: Optional[List[dict]] = Field(default_factory=list)

    # Sender context (R&R, style)
    sender_context: Optional[dict] = None
    authorized_policies: Optional[dict] = Field(default_factory=dict)

    # Per-obligation collection statuses
    obligation_statuses: Optional[list] = None

    # Obligation snapshot for staleness detection
    obligation_snapshot: Optional[list] = None

    # Recent message excerpts for reply context
    recent_messages: Optional[list] = None

    # Escalation history (all prior senders for handoff narrative)
    escalation_history: Optional[list] = None

    # Currency symbol for invoice table formatting
    currency_symbol: Optional[str] = None

    # Lane-only pilot runtime context
    collection_lane_id: Optional[str] = None
    lane: Optional[dict] = None
    lane_history: Optional[list] = None
    lane_mail_mode: Optional[str] = None
    sendable_obligation_ids: Optional[List[str]] = None
    blocked_obligation_ids: Optional[List[str]] = None
    blocked_reasons_by_obligation_id: Optional[dict] = None
    lane_recent_messages: Optional[list] = None
    lane_active_dispute: Optional[bool] = None
    lane_broken_promises_count: Optional[int] = None
    lane_last_tone_used: Optional[str] = None
    lane_contexts: List[LaneContextInfo] = []
    protocol_due_at: Optional[str] = None
    not_before_at: Optional[str] = None
    is_scheduled_prep: Optional[bool] = None
    planned_send_at: Optional[str] = None
    mode: Optional[str] = Field(
        default=None,
        pattern=r"^(single_lane|multi_lane)$",
    )

    # Silver Application / Gold current-context lineage. These fields are
    # optional during transition, but schema_version=4 requests must provide
    # them before production draft generation.
    context_version: Optional[str] = None
    source_sync_run_id: Optional[str] = None
    application_run_id: Optional[str] = None
    core_snapshot_watermark: Optional[datetime | str] = None
    application_snapshot_watermark: Optional[datetime | str] = None
    application_decision_cutoff: Optional[datetime] = None
    input_silver_version_ids_json: Optional[str] = None
    input_silver_version_ids: Optional[List[str]] = Field(default_factory=list)
    policy_snapshot_id: Optional[str] = None
    draft_candidate_id: Optional[str] = None
    draft_generation_run_id: Optional[str] = None
    collection_basis: Optional[Literal["overdue", "outstanding", "invoice_date"]] = None
    chase_basis: Optional[Literal["overdue", "outstanding", "invoice_date"]] = None
    total_outstanding_amount: Optional[float] = None
    total_overdue_amount: Optional[float] = None
    outstanding_invoice_count: Optional[int] = None
    overdue_invoice_count: Optional[int] = None

    # Current projections supplied by the backend/context builder. Kept as
    # dict/list payloads here so the AI repo stays stateless and additive.
    party_communication_state_current: Optional[dict[str, Any]] = None
    party_collection_state_current: Optional[dict[str, Any]] = None
    party_behavior_profile_current: Optional[dict[str, Any]] = None
    party_verification_state_current: Optional[dict[str, Any]] = None
    obligation_collection_status_current: Optional[list[dict[str, Any]]] = None
    silver_app_verification_tasks_current: Optional[list[dict[str, Any]]] = None
    silver_app_payment_verifications_current: Optional[list[dict[str, Any]]] = None
    payment_verification_obligations_current: Optional[list[dict[str, Any]]] = None
    silver_app_promise_history_current: Optional[list[dict[str, Any]]] = None
    promise_obligations_current: Optional[list[dict[str, Any]]] = None
    silver_app_dispute_history_current: Optional[list[dict[str, Any]]] = None
    dispute_obligations_current: Optional[list[dict[str, Any]]] = None
    insolvency_history_current: Optional[list[dict[str, Any]]] = None
    sender_selection_events_current: Optional[list[dict[str, Any]]] = None
    recipient_selection_events_current: Optional[list[dict[str, Any]]] = None
    sender_performance_current: Optional[dict[str, Any]] = None
    excluded_source_disputed_obligations: Optional[list[dict[str, Any]]] = None

    @model_validator(mode="before")
    @classmethod
    def hydrate_sparse_lane_contexts(cls, data: Any) -> Any:
        """Accept sparse lane bundles from older backend producers.

        Some backend flows only pass invoice refs + obligation ids in
        `lane_contexts` while the full lane metadata is already present in
        `context.lane`. Backfill the required lane fields from that canonical
        lane snapshot so request validation remains backward-compatible.
        """
        if not isinstance(data, dict):
            return data

        lane = data.get("lane") if isinstance(data.get("lane"), dict) else {}
        collection_lane_id = _coalesce_lane_value(
            data.get("collection_lane_id"), lane.get("collection_lane_id")
        )
        raw_lane_contexts = data.get("lane_contexts") or []

        if isinstance(raw_lane_contexts, list):
            for raw_context in raw_lane_contexts:
                if not isinstance(raw_context, dict):
                    continue
                if "invoice_refs" in raw_context:
                    warnings.warn(
                        "LaneContextInfo.invoice_refs is deprecated; "
                        "use CaseContext.lane.invoice_refs instead.",
                        DeprecationWarning,
                        stacklevel=3,
                    )
                if "outstanding_amount" in raw_context:
                    warnings.warn(
                        "LaneContextInfo.outstanding_amount is deprecated; "
                        "use CaseContext.lane.outstanding_amount instead.",
                        DeprecationWarning,
                        stacklevel=3,
                    )

        if raw_lane_contexts:
            normalized_contexts = [
                _normalize_lane_context(raw_context, lane, collection_lane_id)
                if isinstance(raw_context, dict)
                else raw_context
                for raw_context in raw_lane_contexts
            ]
            hydrated = dict(data)
            hydrated["lane_contexts"] = normalized_contexts
            return hydrated

        if lane and collection_lane_id:
            hydrated = dict(data)
            hydrated["lane_contexts"] = [
                _normalize_lane_context({}, lane, collection_lane_id),
            ]
            return hydrated

        return data

    @model_validator(mode="after")
    def validate_schema_version_fields(self) -> "CaseContext":
        """Require canonical identity fields for all case-context payloads."""
        if not getattr(self.party, "external_id", None):
            raise ValueError("party.external_id is required")
        if not getattr(self.party, "provider_type", None):
            raise ValueError("party.provider_type is required")

        for obligation in self.obligations:
            if not obligation.id:
                raise ValueError("obligations[].id is required")
            if not obligation.external_id:
                raise ValueError("obligations[].external_id is required")
            if not obligation.provider_type:
                raise ValueError("obligations[].provider_type is required")

        if self.schema_version == 4 and not (self.collection_basis or self.chase_basis):
            self.collection_basis = "overdue"

        return self

    def uses_current_datalake_contract(self) -> bool:
        """Return True when the payload is using the current lake context."""
        return self.schema_version == 4


# Import here to resolve forward references after all models are defined
from .party import BehaviorInfo, PartyInfo  # noqa: E402

CaseContext.model_rebuild()

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
    ObligationInfoV2,
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
    ):
        if normalized.get(field_name) is None and lane.get(field_name) is not None:
            normalized[field_name] = lane.get(field_name)

    if not normalized.get("tone_ladder"):
        normalized["tone_ladder"] = lane.get("tone_ladder") or []
    if normalized.get("outstanding_amount") is None:
        normalized["outstanding_amount"] = lane.get("outstanding_amount") or 0.0

    return normalized


class ObligationInfo(ObligationInfoV2):
    """Single invoice/obligation using the shared CaseContext v2 contract."""


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
    """Single touch record."""

    sent_at: datetime
    tone: Optional[str] = None
    sender_level: Optional[int] = None
    sender_name: Optional[str] = None
    had_response: bool = False


class PromiseHistory(BaseModel):
    """Single promise record."""

    promise_date: Optional[str] = None
    promise_amount: Optional[float] = None
    outcome: Optional[str] = None  # kept, broken, pending


class CommunicationTrackingInfo(BaseModel):
    """Thread monitoring coverage for communication-aware generation."""

    tracking_status: Optional[str] = None
    tracking_reason: Optional[str] = None
    send_confirmation_state: Optional[str] = None
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
    """Full case context for AI operations."""

    model_config = ConfigDict(extra="ignore")

    schema_version: Literal[2] = 2
    party: "PartyInfo"  # Forward ref resolved at module level
    behavior: Optional["BehaviorInfo"] = None  # Forward ref resolved at module level
    obligations: List[ObligationInfo] = Field(default_factory=list)
    communication: Optional[CommunicationInfo] = None
    communication_tracking: Optional[CommunicationTrackingInfo] = None
    recent_touches: List[TouchHistory] = []
    promises: List[PromiseHistory] = []

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
    grace_days: int = 14  # Effective: party.grace_days_override OR tenant.grace_days

    # Promise verification settings
    promise_grace_days: int = 3

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
    mode: Optional[str] = Field(
        default=None,
        pattern=r"^(single_lane)$",
    )

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

        return self


# Import here to resolve forward references after all models are defined
from .party import BehaviorInfo, PartyInfo  # noqa: E402

CaseContext.model_rebuild()

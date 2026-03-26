"""
Context models for AI operations.

Contains CaseContext, CommunicationInfo, ObligationInfo, IndustryInfo,
and related context models used across classification, generation,
and gate evaluation.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class ObligationInfo(BaseModel):
    """Single invoice/obligation."""

    invoice_number: str = Field(..., max_length=100)
    original_amount: float
    amount_due: float
    due_date: Optional[str] = Field(None, max_length=30)
    days_past_due: int = 0
    state: str = Field("open", max_length=30)


class CommunicationInfo(BaseModel):
    """Communication history summary."""

    touch_count: int = 0
    last_touch_at: Optional[datetime] = None
    last_touch_channel: Optional[str] = None
    last_sender_level: Optional[int] = None
    last_sender_name: Optional[str] = None
    last_sender_title: Optional[str] = None
    last_tone_used: Optional[str] = None
    last_response_at: Optional[datetime] = None
    last_response_type: Optional[str] = None
    last_response_subject: Optional[str] = None
    last_response_snippet: Optional[str] = None
    last_outbound_subject: Optional[str] = None


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


class CaseContext(BaseModel):
    """Full case context for AI operations."""

    party: "PartyInfo"  # Forward ref resolved at module level
    behavior: Optional["BehaviorInfo"] = None  # Forward ref resolved at module level
    obligations: List[ObligationInfo] = []
    communication: Optional[CommunicationInfo] = None
    recent_touches: List[TouchHistory] = []
    promises: List[PromiseHistory] = []

    # Case state
    case_state: Optional[str] = None
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

    # Sender context (R&R, style)
    sender_context: Optional[dict] = None

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


# Import here to resolve forward references after all models are defined
from .party import BehaviorInfo, PartyInfo  # noqa: E402

CaseContext.model_rebuild()

"""
Request models for Solvix AI Engine API.

Security:
- All string fields have max_length constraints to prevent memory exhaustion
- custom_instructions has prompt injection detection
- party_id/customer_code have flexible validation (external IDs from accounting software)
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class EmailContent(BaseModel):
    """Email content for classification."""

    subject: str = Field(..., min_length=1, max_length=500)
    body: str = Field(..., min_length=1, max_length=50000)  # 50KB max for email body
    from_address: str = Field(..., min_length=1, max_length=320)  # RFC 5321 max email length
    from_name: Optional[str] = Field(None, max_length=200)
    received_at: Optional[datetime] = None


class PartyInfo(BaseModel):
    """Party (debtor) information."""

    # Flexible validation for external IDs (come from accounting software like Sage)
    party_id: str = Field(..., min_length=1, max_length=100)
    customer_code: str = Field(..., min_length=1, max_length=100)
    name: str = Field(..., min_length=1, max_length=500)
    country_code: Optional[str] = None
    currency: str = "GBP"
    credit_limit: Optional[float] = None
    on_hold: bool = False

    # Debtor-level override fields (NEW)
    relationship_tier: str = "standard"  # vip, standard, high_risk
    tone_override: Optional[str] = None  # friendly, professional, firm (overrides brand_tone)
    grace_days_override: Optional[int] = None  # Overrides tenant grace_days
    touch_cap_override: Optional[int] = None  # Overrides tenant touch_cap
    do_not_contact_until: Optional[str] = None  # ISO date YYYY-MM-DD
    monthly_touch_count: int = 0  # Touches this month (for monthly cap reset)
    is_verified: bool = True  # False for placeholder parties from unknown emails
    source: str = "sage"  # sage, email_inbound, manual


class BehaviorInfo(BaseModel):
    """Historical payment behavior."""

    lifetime_value: Optional[float] = None
    avg_days_to_pay: Optional[float] = None
    on_time_rate: Optional[float] = None
    partial_payment_rate: Optional[float] = None
    segment: Optional[str] = None
    # Enhanced behaviour context
    behaviour_profile: Optional[dict] = None
    behaviour_segment: Optional[str] = None


class ObligationInfo(BaseModel):
    """Single invoice/obligation."""

    invoice_number: str
    original_amount: float
    amount_due: float
    due_date: Optional[str] = None
    days_past_due: int = 0
    state: str = "open"


class CommunicationInfo(BaseModel):
    """Communication history summary."""

    touch_count: int = 0
    last_touch_at: Optional[datetime] = None
    last_touch_channel: Optional[str] = None
    last_sender_level: Optional[int] = None
    last_tone_used: Optional[str] = None
    last_response_at: Optional[datetime] = None
    last_response_type: Optional[str] = None
    last_response_subject: Optional[str] = None
    last_response_snippet: Optional[str] = None


class TouchHistory(BaseModel):
    """Single touch record."""

    sent_at: datetime
    tone: Optional[str] = None
    sender_level: Optional[int] = None
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

    code: str  # Industry identifier: retail, manufacturing, b2b_services, etc.
    name: str  # Display name
    typical_dso_days: int  # Normal payment cycle for this industry
    alarm_dso_days: int  # DSO that signals concern
    payment_cycle: str  # immediate, net15, net30, net45, net60, net90
    escalation_patience: str = "standard"  # patient, standard, aggressive
    common_dispute_types: List[str] = []  # Expected dispute types
    hardship_indicators: List[str] = []  # Industry-specific hardship signals
    preferred_tone: str = "professional"  # formal, professional, casual
    ai_context_notes: str = ""  # Free-form context for prompts
    seasonal_patterns: dict = {}  # Q1, Q2, Q3, Q4 patterns
    dispute_handling_notes: str = ""  # How to handle disputes
    hardship_handling_notes: str = ""  # How to handle hardship
    communication_notes: str = ""  # Industry communication conventions


class CaseContext(BaseModel):
    """Full case context for AI operations."""

    party: PartyInfo
    behavior: Optional[BehaviorInfo] = None
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
    brand_tone: str = "professional"  # Effective: party.tone_override OR tenant.brand_tone
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

    # Currency symbol for invoice table formatting
    currency_symbol: Optional[str] = None


# Dangerous patterns that indicate potential prompt injection
PROMPT_INJECTION_PATTERNS = [
    "ignore previous",
    "ignore above",
    "disregard",
    "system prompt",
    "forget your instructions",
    "new instructions",
    "you are now",
    "act as",
    "pretend to be",
    "override",
    "bypass",
]


# =============================================================================
# PERSONA MODELS
# =============================================================================


class SenderPersona(BaseModel):
    """Sender persona for draft generation (injected into prompt)."""

    name: str = Field(..., max_length=255)
    title: str = Field("", max_length=100)
    communication_style: Optional[str] = Field(None, max_length=200)
    formality_level: Optional[str] = Field(None, max_length=20)
    emphasis: Optional[str] = Field(None, max_length=200)


class PersonaContact(BaseModel):
    """Contact info for persona generation."""

    name: str = Field(..., max_length=255)
    title: str = Field("", max_length=100)
    level: int = Field(..., ge=1, le=4)
    style_description: str = Field("", max_length=2000)
    style_examples: List[str] = Field(default_factory=list)


class GeneratePersonaRequest(BaseModel):
    """Request to generate personas for escalation contacts (cold start)."""

    contacts: List[PersonaContact] = Field(..., max_length=10)
    total_levels: int = Field(default=4, ge=1, le=4)


class SenderPerformanceStats(BaseModel):
    """Sender performance stats for persona refinement."""

    total_touches: int = 0
    total_unique_parties: int = 0
    responses_received: int = 0
    response_rate: Optional[float] = None
    avg_response_days: Optional[float] = None
    cooperative_count: int = 0
    hostile_count: int = 0
    promise_count: int = 0
    dispute_count: int = 0
    no_response_count: int = 0
    cases_resolved_pif: int = 0
    amount_collected_after: Optional[float] = None
    avg_days_to_payment: Optional[float] = None
    promises_elicited: int = 0
    promises_kept: int = 0
    promises_broken: int = 0
    promise_fulfillment_rate: Optional[float] = None
    disputes_raised_after: int = 0
    disputes_resolved: int = 0
    early_state_pct: Optional[float] = None
    escalated_state_pct: Optional[float] = None
    tone_distribution: Optional[dict] = None
    segment_distribution: Optional[dict] = None
    avg_days_between_touches: Optional[float] = None


class RefinePersonaRequest(BaseModel):
    """Request to refine a sender persona based on performance data."""

    name: str = Field(..., max_length=255)
    title: str = Field("", max_length=100)
    level: int = Field(..., ge=1, le=4)
    current_persona: SenderPersona
    performance: SenderPerformanceStats
    persona_version: int = Field(default=0, ge=0)
    style_description: Optional[str] = Field(None, max_length=2000)
    style_examples: Optional[List[str]] = Field(default_factory=list)


# =============================================================================
# CLASSIFICATION & GENERATION MODELS
# =============================================================================


class ClassifyRequest(BaseModel):
    """Request to classify an inbound email."""

    email: EmailContent
    context: CaseContext


class SenderContext(BaseModel):
    """Extended sender context for style-aware draft generation."""

    roles_responsibilities: Optional[str] = None
    style_description: Optional[str] = None
    style_examples: Optional[List[str]] = None


class GenerateDraftRequest(BaseModel):
    """Request to generate a collection email draft."""

    context: CaseContext
    sender_persona: Optional[SenderPersona] = None
    sender_name: Optional[str] = Field(None, max_length=255)
    sender_title: Optional[str] = Field(None, max_length=100)
    sender_company: Optional[str] = Field(None, max_length=255)
    sender_context: Optional[SenderContext] = None
    tone: str = Field(
        default="professional",
        pattern=r"^(friendly_reminder|professional|firm|final_notice|concerned_inquiry)$",
    )
    objective: Optional[str] = Field(
        default=None,
        pattern=r"^(follow_up|promise_reminder|escalation|initial_contact)$",
    )
    closure_mode: bool = False
    skip_invoice_table: bool = False
    trigger_classification: Optional[str] = None
    tone_preference: Optional[str] = Field(None, pattern=r"^(diplomatic|professional|direct)$")
    # SECURITY: Limited to 1000 chars with prompt injection detection
    custom_instructions: Optional[str] = Field(default=None, max_length=1000)

    @field_validator("custom_instructions")
    @classmethod
    def sanitize_custom_instructions(cls, v: Optional[str]) -> Optional[str]:
        """
        Validate custom_instructions for potential prompt injection attacks.

        Checks for common patterns used to manipulate LLM behavior.
        """
        if v is None:
            return v

        v_lower = v.lower()
        for pattern in PROMPT_INJECTION_PATTERNS:
            if pattern in v_lower:
                raise ValueError("Invalid instructions: contains potentially unsafe pattern")
        return v


class EvaluateGatesRequest(BaseModel):
    """Request to evaluate gates before taking action."""

    context: CaseContext
    proposed_action: str = Field(
        ...,
        pattern=r"^(send_email|create_case|escalate|close_case)$",
    )
    proposed_tone: Optional[str] = Field(
        default=None,
        pattern=r"^(friendly_reminder|professional|firm|final_notice|concerned_inquiry)$",
    )


class EvaluateGatesBatchRequest(BaseModel):
    """Batch request to evaluate gates for multiple parties at once.

    Used to efficiently evaluate gates for many parties before parallel
    draft generation. Since gate evaluation is deterministic (no LLM),
    this reduces HTTP overhead significantly.
    """

    contexts: List[CaseContext] = Field(..., max_length=100)  # Max 100 parties per batch
    proposed_action: str = Field(
        ...,
        pattern=r"^(send_email|create_case|escalate|close_case)$",
    )
    proposed_tone: Optional[str] = Field(
        default=None,
        pattern=r"^(friendly_reminder|professional|firm|final_notice|concerned_inquiry)$",
    )

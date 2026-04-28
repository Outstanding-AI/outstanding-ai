"""
Request validation models for AI Engine API endpoints.

Contains the main request classes used by the AI service.

Security:
- All string fields have max_length constraints to prevent memory exhaustion
- custom_instructions has prompt injection detection
- party_id/customer_code have flexible validation (external IDs from accounting software)
"""

import warnings
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from src.config.constants import OBJECTIVE_REGEX, TONE_PREFERENCE_REGEX, TONE_REGEX

from .context import CaseContext
from .party import EmailContent
from .persona import SenderContext, SenderPersona

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


class ClassifyRequest(BaseModel):
    """Request to classify an inbound email."""

    email: EmailContent
    context: CaseContext


class FollowUpContext(BaseModel):
    """Verification claim/match context for queued follow-up drafts.

    Sprint A item #3 follow-up (Codex P1, 2026-04-28): supplies the AI
    prompt with the debtor's ORIGINAL claim (amount/date/reference) and
    what we did/didn't find on Sage when the verifier ran. Without this
    block, ``payment_not_found`` and ``partial_payment_ack`` templates
    can only reference generic placeholders or invent numbers.

    All fields optional — the verifier may not have every field for every
    claim (e.g. claim with no reference, partial without a precise
    matched amount).
    """

    trigger_classification: Optional[str] = Field(None, max_length=50)
    verification_id: Optional[str] = Field(None, max_length=64)
    claimed_amount: Optional[float] = None
    claimed_date: Optional[str] = Field(None, max_length=20)
    claimed_reference: Optional[str] = Field(None, max_length=100)
    matched_amount: Optional[float] = None
    residual_amount: Optional[float] = None
    obligation_ids: Optional[List[str]] = Field(default_factory=list)


class GenerateDraftRequest(BaseModel):
    """Request to generate a collection email draft."""

    context: CaseContext
    sender_persona: Optional[SenderPersona] = None
    sender_name: Optional[str] = Field(None, max_length=255)
    sender_title: Optional[str] = Field(None, max_length=100)
    sender_company: Optional[str] = Field(None, max_length=255)
    sender_email: Optional[str] = Field(None, max_length=320)
    cc_emails: Optional[List[str]] = Field(default_factory=list)
    sender_context: Optional[SenderContext] = None
    tone: str = Field(
        default="professional",
        pattern=TONE_REGEX,
    )
    objective: Optional[str] = Field(
        default=None,
        pattern=OBJECTIVE_REGEX,
    )
    closure_mode: bool = False
    skip_invoice_table: bool = False
    trigger_classification: Optional[str] = Field(None, max_length=50)
    escalation_level: Optional[int] = Field(None, description="Current escalation level (0-4)")
    tone_preference: Optional[str] = Field(None, pattern=TONE_PREFERENCE_REGEX)
    # SECURITY: Limited to 1000 chars with prompt injection detection
    custom_instructions: Optional[str] = Field(default=None, max_length=1000)
    follow_up_context: Optional[FollowUpContext] = None

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

    @model_validator(mode="after")
    def normalize_deprecated_sender_persona_fields(self) -> "GenerateDraftRequest":
        """Backfill canonical sender identity fields from legacy sender_persona fields."""
        if not self.sender_persona:
            return self

        persona_name = (self.sender_persona.name or "").strip()
        persona_title = (self.sender_persona.title or "").strip()

        if persona_name:
            warnings.warn(
                "GenerateDraftRequest.sender_persona.name is deprecated; use sender_name instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            if self.sender_name and self.sender_name != persona_name:
                raise ValueError(
                    "sender_persona.name must match sender_name when both are provided"
                )
            if not self.sender_name:
                self.sender_name = persona_name

        if persona_title:
            warnings.warn(
                "GenerateDraftRequest.sender_persona.title is deprecated; use sender_title instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            if self.sender_title and self.sender_title != persona_title:
                raise ValueError(
                    "sender_persona.title must match sender_title when both are provided"
                )
            if not self.sender_title:
                self.sender_title = persona_title

        return self


# EvaluateGatesRequest + EvaluateGatesBatchRequest removed 2026-04-26 alongside
# the /evaluate-gates route deletion. Gate evaluation moved to backend
# services/gate_checker.py (CLAUDE.md note #40); the AI-side request models
# had no remaining consumers.

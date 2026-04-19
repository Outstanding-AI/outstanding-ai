"""
Request validation models for AI Engine API endpoints.

Contains the main request classes used by the AI service.

Security:
- All string fields have max_length constraints to prevent memory exhaustion
- custom_instructions has prompt injection detection
- party_id/customer_code have flexible validation (external IDs from accounting software)
"""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

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
        pattern=r"^(friendly_reminder|friendly_escalating|professional|professional_escalating|firm|firm_escalating|final_notice|legal_pre_action|acknowledgement|concerned_inquiry)$",
    )
    objective: Optional[str] = Field(
        default=None,
        pattern=r"^(follow_up|promise_reminder|escalation|initial_contact)$",
    )
    closure_mode: bool = False
    skip_invoice_table: bool = False
    trigger_classification: Optional[str] = Field(None, max_length=50)
    escalation_level: Optional[int] = Field(None, description="Current escalation level (0-4)")
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
        pattern=r"^(friendly_reminder|friendly_escalating|professional|professional_escalating|firm|firm_escalating|final_notice|legal_pre_action|acknowledgement|concerned_inquiry)$",
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
        pattern=r"^(friendly_reminder|friendly_escalating|professional|professional_escalating|firm|firm_escalating|final_notice|legal_pre_action|acknowledgement|concerned_inquiry)$",
    )

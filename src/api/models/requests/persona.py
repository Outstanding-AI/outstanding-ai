"""
Persona models for sender persona generation and refinement.

Contains SenderPersona, PersonaContact, GeneratePersonaRequest,
RefinePersonaRequest, SenderPerformanceStats, and SenderContext.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class SenderPersona(BaseModel):
    """Sender persona for draft generation (injected into prompt)."""

    name: str = Field(..., max_length=255)
    title: str = Field("", max_length=100)
    communication_style: Optional[str] = Field(None, max_length=200)
    formality_level: Optional[str] = Field(None, max_length=20)
    emphasis: Optional[str] = Field(None, max_length=200)
    level: Optional[int] = Field(None, ge=1, le=4)


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


class SenderContext(BaseModel):
    """Extended sender context for style-aware draft generation."""

    roles_responsibilities: Optional[str] = Field(None, max_length=2000)
    style_description: Optional[str] = Field(None, max_length=2000)
    style_examples: Optional[List[str]] = None

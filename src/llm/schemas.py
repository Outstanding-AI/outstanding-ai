"""
Pydantic models for validating LLM responses.

These models ensure type safety when parsing LLM outputs and provide
clear error messages when the LLM returns malformed data.
"""

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class LLMExtractedData(BaseModel):
    """Data extracted from email content by the LLM."""

    # PROMISE_TO_PAY
    promise_date: Optional[str] = None  # String from LLM, parsed to date in engine
    promise_amount: Optional[float] = None
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
    extracted_data: Optional[LLMExtractedData] = Field(
        None,
        description="Data extracted from the email",
    )

    @field_validator("classification")
    @classmethod
    def validate_classification(cls, v: str) -> str:
        valid_classifications = {
            "INSOLVENCY",
            "DISPUTE",
            "ALREADY_PAID",
            "UNSUBSCRIBE",
            "HOSTILE",
            "PROMISE_TO_PAY",
            "HARDSHIP",
            "PLAN_REQUEST",
            "REDIRECT",
            "REQUEST_INFO",
            "OUT_OF_OFFICE",
            "COOPERATIVE",
            "UNCLEAR",
        }
        upper_v = v.upper()
        if upper_v not in valid_classifications:
            raise ValueError(
                f"Invalid classification '{v}'. Must be one of: {', '.join(sorted(valid_classifications))}"
            )
        return upper_v


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

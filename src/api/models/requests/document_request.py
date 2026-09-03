"""Typed, fail-closed input for the additive document-request interpreter."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DocumentRequestMessage(BaseModel):
    """Provider-normalized message evidence required for admission."""

    model_config = ConfigDict(extra="forbid")

    message_key: str = Field(min_length=1, max_length=128)
    timestamp: datetime
    direction: Literal["inbound", "outbound"]
    message_class: Literal[
        "debtor_authored", "operator_authored", "automated", "deleted", "unavailable"
    ]
    subject: str = Field(default="", max_length=300)
    authored_body: str = Field(default="", max_length=12_000)
    body_source: Literal["unique_body", "deterministic_sanitized_authored_body"]
    is_automated: bool
    is_deleted: bool
    is_available: bool
    debtor_verified: bool
    # Deterministic server output used for safe deictic adoption.  Model text
    # never creates or expands this antecedent set.
    causal_reference_set: list[str] = Field(default_factory=list, max_length=30)
    causal_reference_set_complete: bool = False

    @field_validator("timestamp")
    @classmethod
    def timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("message timestamp must be timezone-aware")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def admission_flags(self) -> "DocumentRequestMessage":
        if self.message_class == "automated" and not self.is_automated:
            raise ValueError("automated message class requires is_automated")
        if self.message_class == "deleted" and not self.is_deleted:
            raise ValueError("deleted message class requires is_deleted")
        if self.message_class == "unavailable" and self.is_available:
            raise ValueError("unavailable message class requires is_available=false")
        if not self.is_available:
            raise ValueError("unavailable message cannot carry an authored body source")
        return self


class InvoiceDocumentRequestInterpretationRequest(BaseModel):
    """Current typed message and at most six strictly earlier causal messages."""

    model_config = ConfigDict(extra="forbid")

    current_message: DocumentRequestMessage
    prior_messages: list[DocumentRequestMessage] = Field(default_factory=list, max_length=6)
    # Server-computed admission result. False is fail-closed.
    admission_eligible: bool = False

    @model_validator(mode="after")
    def causal_order(self) -> "InvoiceDocumentRequestInterpretationRequest":
        messages = [*self.prior_messages, self.current_message]
        if any(item.timestamp >= self.current_message.timestamp for item in self.prior_messages):
            raise ValueError(
                "prior message timestamps must be strictly earlier than current message"
            )
        if self.prior_messages != sorted(self.prior_messages, key=lambda item: item.timestamp):
            raise ValueError("prior messages must be in deterministic chronological order")
        if len({item.message_key for item in messages}) != len(messages):
            raise ValueError("message keys must be unique")
        if len({item.timestamp for item in messages}) != len(messages):
            raise ValueError("message timestamps must be unique")
        if self.current_message.direction != "inbound":
            raise ValueError("current document request message must be inbound")
        total_chars = sum(len(item.subject) + len(item.authored_body) for item in messages)
        if total_chars > 12_000:
            raise ValueError("causal message context exceeds 12000 characters")
        return self


DocumentRequestInterpretationRequest = InvoiceDocumentRequestInterpretationRequest
CurrentDocumentRequestMessage = DocumentRequestMessage

__all__ = [
    "DocumentRequestMessage",
    "CurrentDocumentRequestMessage",
    "InvoiceDocumentRequestInterpretationRequest",
    "DocumentRequestInterpretationRequest",
]

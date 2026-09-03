"""Strict response models for additive invoice-document request evidence."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DocumentType = Literal[
    "original_invoice_pdf",
    "corrected_invoice",
    "credit_note",
    "statement",
    "pod_or_tracking",
    "other",
]
RequestAction = Literal["copy", "resend", "correct", "reissue", "other"]
RequestStrength = Literal["explicit", "inferred", "ambiguous"]
RequestState = Literal["active_request", "no_request", "uncertain"]
OtherDocumentFamily = Literal[
    "remittance_advice",
    "purchase_order",
    "sales_order",
    "delivery_note",
    "goods_receipt",
    "tax_certificate",
    "supporting_document",
    "document",
    "file",
    "attachment",
    "copy",
]


class DocumentRequestItem(BaseModel):
    """One document assertion with independent invoice/order reference scope."""

    model_config = ConfigDict(extra="forbid")

    request_ordinal: int = Field(ge=1, le=20)
    document_type: DocumentType
    other_document_family: OtherDocumentFamily | None = None
    invoice_refs: list[str] = Field(default_factory=list, max_length=30)
    order_refs: list[str] = Field(default_factory=list, max_length=30)
    request_action: RequestAction = "other"
    request_strength: RequestStrength = "ambiguous"
    request_evidence_text: str = Field(min_length=1, max_length=500)
    evidence_message_key: str = Field(min_length=1, max_length=128)
    confidence: float = Field(ge=0.0, le=1.0)
    reference_source: Literal["current_message", "causal_context", "unresolved"] | None = None

    @model_validator(mode="after")
    def unique_references(self) -> "DocumentRequestItem":
        if len(self.invoice_refs) != len(set(self.invoice_refs)):
            raise ValueError("document request invoice references must be unique")
        if len(self.order_refs) != len(set(self.order_refs)):
            raise ValueError("document request order references must be unique")
        if set(self.invoice_refs) & set(self.order_refs):
            raise ValueError("invoice and order reference arrays must not overlap")
        if self.document_type == "other" and self.other_document_family is None:
            raise ValueError("other document requests must name a document family")
        if self.document_type != "other" and self.other_document_family is not None:
            raise ValueError("other document family is only valid for other document requests")
        return self


class InvoiceDocumentRequestInterpretationResponse(BaseModel):
    """Evidence response; no debtor, artifact, availability or control authority."""

    model_config = ConfigDict(extra="forbid")

    request_state: RequestState
    request_retracted: bool = False
    document_requests: list[DocumentRequestItem] = Field(default_factory=list, max_length=20)
    reason_codes: list[str] = Field(default_factory=list, max_length=20)
    confidence: float = Field(ge=0.0, le=1.0)
    automated_response: bool = False
    disposition: Literal[
        "invoice_pdf_request",
        "corrected_invoice_request",
        "other_document_request",
        "mixed_document_request",
        "no_request",
        "uncertain",
    ] = "no_request"
    scope_status: Literal[
        "none",
        "exact",
        "partial",
        "reference_exact_invoice_unresolved",
        "account_wide",
        "ambiguous",
    ] = "none"
    requested_invoice_refs: list[str] = Field(default_factory=list, max_length=100)
    requested_order_refs: list[str] = Field(default_factory=list, max_length=100)
    deterministic_override_reason: str | None = Field(default=None, max_length=120)
    model_request_state_before_admission_gate: RequestState | None = None
    model_document_request_count_before_admission_gate: int | None = Field(default=None, ge=0)
    provider: str | None = None
    model: str | None = None
    is_fallback: bool = False
    tokens_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ai_audit: dict[str, Any] | None = None

    @model_validator(mode="after")
    def request_invariants(self) -> "InvoiceDocumentRequestInterpretationResponse":
        ordinals = [item.request_ordinal for item in self.document_requests]
        if ordinals != list(range(1, len(ordinals) + 1)):
            raise ValueError("document request ordinals must be contiguous and unique")
        if self.request_state == "no_request" and self.document_requests:
            raise ValueError("no_request response must not contain document items")
        if self.request_state == "active_request" and not self.document_requests:
            raise ValueError("active_request response requires document items")
        if self.request_retracted and self.request_state != "no_request":
            raise ValueError("retracted request must have no_request state")
        invoice_refs = list(
            dict.fromkeys(ref for item in self.document_requests for ref in item.invoice_refs)
        )
        order_refs = list(
            dict.fromkeys(ref for item in self.document_requests for ref in item.order_refs)
        )
        if invoice_refs != self.requested_invoice_refs or order_refs != self.requested_order_refs:
            raise ValueError("requested reference summary does not match document items")
        if self.request_state in {"no_request", "uncertain"} and (
            self.requested_invoice_refs or self.requested_order_refs
        ):
            raise ValueError("non-active request must have empty reference summary")
        if self.request_state == "uncertain" and self.disposition != "uncertain":
            raise ValueError("uncertain request must have uncertain disposition")
        if self.request_state == "no_request" and self.disposition != "no_request":
            raise ValueError("no_request must have no_request disposition")
        if self.request_state == "active_request":
            types = {item.document_type for item in self.document_requests}
            expected_disposition = (
                "mixed_document_request"
                if len(types) > 1
                else "invoice_pdf_request"
                if types == {"original_invoice_pdf"}
                else "corrected_invoice_request"
                if types == {"corrected_invoice"}
                else "other_document_request"
            )
            if self.disposition != expected_disposition:
                raise ValueError("document request disposition summary is inconsistent")
            if invoice_refs and all(item.invoice_refs for item in self.document_requests):
                expected_scope = "exact"
            elif invoice_refs:
                expected_scope = "partial"
            elif order_refs:
                expected_scope = "reference_exact_invoice_unresolved"
            elif all(item.document_type == "statement" for item in self.document_requests):
                expected_scope = "account_wide"
            else:
                expected_scope = "ambiguous"
            if self.scope_status != expected_scope:
                raise ValueError("document request scope summary is inconsistent")
        if self.automated_response:
            if self.request_state != "no_request" or self.disposition != "no_request":
                raise ValueError("automated response must be no_request")
            if self.deterministic_override_reason != "automated_response_fail_closed":
                raise ValueError("automated response override summary is inconsistent")
        if self.deterministic_override_reason and self.request_state != "no_request":
            raise ValueError("override summary requires no_request state")
        return self


DocumentRequestInterpretationResponse = InvoiceDocumentRequestInterpretationResponse
DocumentRequest = DocumentRequestItem

__all__ = [
    "DocumentRequest",
    "DocumentRequestItem",
    "InvoiceDocumentRequestInterpretationResponse",
    "DocumentRequestInterpretationResponse",
]

"""ASGI boundary checks for the protected document-request route."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from src.api.models.document_request_responses import (
    DocumentRequestItem,
    InvoiceDocumentRequestInterpretationResponse,
)
from src.api.models.requests.document_request import (
    InvoiceDocumentRequestInterpretationRequest,
)
from src.api.routes import interpret_invoice_document_request as route_module
from src.main import app


def _payload(*, admission_eligible: bool = True) -> dict[str, object]:
    return {
        "current_message": {
            "message_key": "message-1",
            "timestamp": "2026-09-04T10:00:00Z",
            "direction": "inbound",
            "message_class": "debtor_authored",
            "subject": "Invoice copy please",
            "authored_body": "Please send invoice INV-100.",
            "body_source": "unique_body",
            "is_automated": False,
            "is_deleted": False,
            "is_available": True,
            "debtor_verified": True,
        },
        "prior_messages": [],
        "admission_eligible": admission_eligible,
    }


class _FakeInterpreter:
    def __init__(self, response: InvoiceDocumentRequestInterpretationResponse) -> None:
        self.response = response
        self.calls: list[InvoiceDocumentRequestInterpretationRequest] = []

    async def interpret(self, request: InvoiceDocumentRequestInterpretationRequest):
        self.calls.append(request)
        return self.response


class _NoLLM:
    complete = AsyncMock(side_effect=AssertionError("admission-gated request reached the LLM"))


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_document_request_route_requires_service_auth(client: TestClient) -> None:
    response = client.post("/interpret-invoice-document-request", json=_payload())
    assert response.status_code == 401


def test_authorized_tenant_scoped_request_uses_fake_interpreter(
    monkeypatch, client: TestClient
) -> None:
    fake = _FakeInterpreter(
        InvoiceDocumentRequestInterpretationResponse(
            request_state="active_request",
            document_requests=[
                DocumentRequestItem(
                    request_ordinal=1,
                    document_type="original_invoice_pdf",
                    invoice_refs=["INV-100"],
                    request_action="copy",
                    request_strength="explicit",
                    request_evidence_text="Please send invoice INV-100.",
                    evidence_message_key="message-1",
                    confidence=1.0,
                )
            ],
            confidence=1.0,
            disposition="invoice_pdf_request",
            scope_status="exact",
            requested_invoice_refs=["INV-100"],
        )
    )
    monkeypatch.setattr(route_module, "document_request_interpreter", fake)

    response = client.post(
        "/interpret-invoice-document-request",
        json=_payload(),
        headers={"Authorization": "Bearer test-secret-token", "X-Tenant-ID": "tenant-a"},
    )

    assert response.status_code == 200
    assert response.json()["request_state"] == "active_request"
    assert response.json()["requested_invoice_refs"] == ["INV-100"]
    assert fake.calls and fake.calls[0].current_message.message_key == "message-1"


def test_admission_gate_skips_llm_and_returns_strict_no_request(
    monkeypatch, client: TestClient
) -> None:
    from src.engine.document_request_interpreter import DocumentRequestInterpreter

    interpreter = DocumentRequestInterpreter(client=_NoLLM())
    monkeypatch.setattr(route_module, "document_request_interpreter", interpreter)

    response = client.post(
        "/interpret-invoice-document-request",
        json=_payload(admission_eligible=False),
        headers={"Authorization": "Bearer test-secret-token", "X-Tenant-ID": "tenant-a"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["request_state"] == "no_request"
    assert data["document_requests"] == []
    assert data["requested_invoice_refs"] == []
    assert data["deterministic_override_reason"] == "ineligible_debtor_context_fail_closed"


def test_request_validation_remains_strict_after_auth(client: TestClient) -> None:
    response = client.post(
        "/interpret-invoice-document-request",
        json={"current_message": {"message_key": "only-key"}},
        headers={"Authorization": "Bearer test-secret-token", "X-Tenant-ID": "tenant-a"},
    )
    assert response.status_code == 422

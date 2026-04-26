"""Shared test fixtures for Outstanding AI Engine tests."""

# Force a deterministic auth token BEFORE any src imports (settings singleton
# is created at import time and the middleware caches its token at init).
# This ensures tests work identically on local dev and CI.
import os

os.environ.setdefault("SERVICE_AUTH_TOKEN", "test-secret-token")

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.api.models.requests import (
    BehaviorInfo,
    CaseContext,
    ClassifyRequest,
    CommunicationInfo,
    EmailContent,
    GenerateDraftRequest,
    ObligationInfo,
    PartyInfo,
    TouchHistory,
)


@pytest.fixture
def sample_email_content() -> EmailContent:
    """Sample inbound email for classification."""
    return EmailContent(
        subject="Re: Invoice #12345",
        body="I cannot pay right now. I lost my job last month. Can we work out a payment plan?",
        from_address="customer@example.com",
        received_at="2024-01-15T10:30:00Z",
    )


@pytest.fixture
def sample_party_info() -> PartyInfo:
    """Sample party/customer info."""
    return PartyInfo(
        party_id="party-123",
        external_id="party-ext-123",
        provider_type="sage_200",
        customer_code="CUST001",
        name="Acme Corp",
        country_code="GB",
        currency="GBP",
        source="sage_200",
    )


@pytest.fixture
def sample_behavior_info() -> BehaviorInfo:
    """Sample payment behavior metrics."""
    return BehaviorInfo(
        lifetime_value=50000.0,
        avg_days_to_pay=35.5,
        on_time_rate=0.65,
        behaviour_segment="reliable_late_payer",
    )


@pytest.fixture
def sample_obligations() -> list[ObligationInfo]:
    """Sample outstanding invoices."""
    return [
        ObligationInfo(
            id="obl-12345",
            external_id="12345",
            provider_type="sage_200",
            invoice_number="INV-12345",
            original_amount=1500.0,
            amount_due=1500.0,
            due_date="2024-01-01",
            days_past_due=14,
            state="open",
        ),
        ObligationInfo(
            id="obl-12346",
            external_id="12346",
            provider_type="sage_200",
            invoice_number="INV-12346",
            original_amount=2500.0,
            amount_due=2500.0,
            due_date="2024-01-05",
            days_past_due=10,
            state="open",
        ),
    ]


@pytest.fixture
def sample_communication_info() -> CommunicationInfo:
    """Sample communication history summary."""
    return CommunicationInfo(
        touch_count=3,
        last_touch_at="2024-01-10T09:00:00Z",
        last_touch_channel="email",
        last_sender_level=1,
        last_tone_used="friendly_reminder",
    )


@pytest.fixture
def sample_case_context(
    sample_party_info,
    sample_behavior_info,
    sample_obligations,
    sample_communication_info,
) -> CaseContext:
    """Complete case context for AI operations."""
    return CaseContext(
        schema_version=2,
        party=sample_party_info,
        behavior=sample_behavior_info,
        obligations=sample_obligations,
        communication=sample_communication_info,
        recent_touches=[
            TouchHistory(
                sent_at="2024-01-10T09:00:00Z",
                tone="friendly_reminder",
                sender_level=1,
                had_response=False,
            )
        ],
        case_state="ACTIVE",
        days_in_state=30,
        broken_promises_count=1,
        active_dispute=False,
        hardship_indicated=False,
        brand_tone="professional",
        touch_cap=10,
        touch_interval_days=3,
    )


@pytest.fixture
def sample_classify_request(sample_email_content, sample_case_context) -> ClassifyRequest:
    """Complete classification request."""
    return ClassifyRequest(
        email=sample_email_content,
        context=sample_case_context,
    )


@pytest.fixture
def sample_generate_draft_request(sample_case_context) -> GenerateDraftRequest:
    """Complete draft generation request."""
    return GenerateDraftRequest(
        context=sample_case_context,
        tone="concerned_inquiry",
        objective="follow_up",
    )


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock = MagicMock()
    mock.chat = MagicMock()
    mock.chat.completions = MagicMock()
    mock.chat.completions.create = AsyncMock()
    return mock


@pytest.fixture
def authed_client():
    """Authenticated test client.

    Uses the deterministic token set via os.environ at the top of conftest.py.
    The middleware was initialized with this token at app startup.
    """
    from fastapi.testclient import TestClient

    from src.main import app

    return TestClient(app, headers={"Authorization": "Bearer test-secret-token"})

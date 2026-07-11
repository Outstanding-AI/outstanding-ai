from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from src.api.models.requests import (
    CollectionChainIdentificationRequest,
    CollectionEmailFactExtractionRequest,
)
from src.llm.base import LLMResponse
from src.llm.schemas import (
    CollectionChainIdentificationLLMResponse,
    CollectionEmailFactExtractionLLMResponse,
)


def test_fact_extraction_contract_contains_no_relevance_or_route_fields():
    request = CollectionEmailFactExtractionRequest(
        current_message={"body": "Invoice INV-1 is overdue"}
    )
    assert request.prior_messages == []
    parsed = CollectionEmailFactExtractionLLMResponse(
        invoice_assertions=["INV-1"],
        amount_assertions=[],
        date_assertions=[],
        confidence=0.9,
        reason_codes=["explicit_invoice"],
    )
    assert parsed.invoice_assertions == ["INV-1"]


def test_chain_identifier_contract_is_bounded_and_strict():
    request = CollectionChainIdentificationRequest(
        current_message={"body": "Please settle the invoice"},
        prior_messages=[{"ordinal": index} for index in range(6)],
        reconciled_scope=[{"mapping_status": "exact"}],
    )
    assert len(request.prior_messages) == 6
    parsed = CollectionChainIdentificationLLMResponse(
        collection_status="collection",
        event_effect="confirmed",
        confidence=0.9,
        reason_codes=["payment_request"],
        evidence_message_ordinals=[1],
    )
    assert parsed.collection_status == "collection"
    with pytest.raises(Exception):
        CollectionChainIdentificationLLMResponse(
            collection_status="collection",
            event_effect="route_to_this_thread",
            confidence=0.9,
        )


@pytest.mark.asyncio
async def test_fact_extractor_uses_the_closed_provider_schema():
    from src.engine.collection_email_fact_extractor import CollectionEmailFactExtractor

    extractor = CollectionEmailFactExtractor()
    extractor._client.complete = AsyncMock(
        return_value=LLMResponse(
            content=json.dumps(
                {
                    "invoice_assertions": [],
                    "amount_assertions": [],
                    "date_assertions": [],
                    "confidence": 0.0,
                    "reason_codes": ["no_explicit_invoice_fact"],
                }
            ),
            provider="vertex",
            model="gemini-2.5-flash",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )
    )

    await extractor.extract(
        CollectionEmailFactExtractionRequest(current_message={"body": "synthetic"})
    )

    assert (
        extractor._client.complete.await_args.kwargs["response_schema"]
        is CollectionEmailFactExtractionLLMResponse
    )
    assert "json_mode" not in extractor._client.complete.await_args.kwargs


@pytest.mark.asyncio
async def test_chain_identifier_uses_the_closed_provider_schema():
    from src.engine.collection_chain_identifier import CollectionChainIdentifier

    identifier = CollectionChainIdentifier()
    identifier._client.complete = AsyncMock(
        return_value=LLMResponse(
            content=json.dumps(
                {
                    "collection_status": "uncertain",
                    "event_effect": "no_change",
                    "confidence": 0.0,
                    "reason_codes": ["insufficient_email_evidence"],
                    "evidence_message_ordinals": [],
                }
            ),
            provider="vertex",
            model="gemini-2.5-flash",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )
    )

    await identifier.identify(
        CollectionChainIdentificationRequest(current_message={"body": "synthetic"})
    )

    assert (
        identifier._client.complete.await_args.kwargs["response_schema"]
        is CollectionChainIdentificationLLMResponse
    )
    assert "json_mode" not in identifier._client.complete.await_args.kwargs

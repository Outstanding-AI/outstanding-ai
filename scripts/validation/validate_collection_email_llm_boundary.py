#!/usr/bin/env python3
"""Bounded collection-email schema and real-provider release validation.

The live modes use synthetic content only and emit aggregate telemetry. They
never print prompts, model output, credentials, email addresses, or identifiers.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from solvix_contracts import __version__ as contracts_version
from solvix_contracts.datalake.v2 import load_manifest_v2

from src.api.models.requests import (
    CollectionEmailEventRequest,
    CollectionEmailFactExtractionRequest,
)
from src.api.models.responses import (
    CollectionEmailEventResponse,
    CollectionEmailFactExtractionResponse,
)
from src.engine.collection_email_event_classifier import CollectionEmailEventClassifier
from src.engine.collection_email_fact_extractor import CollectionEmailFactExtractor
from src.llm.base import LLMProviderUnavailableError
from src.llm.schemas import (
    CollectionEmailEventLLMResponse,
    CollectionEmailFactExtractionLLMResponse,
)


class _TransientVertexFailure:
    provider_name = "vertex"

    async def complete(self, *_args: Any, **_kwargs: Any):
        raise LLMProviderUnavailableError("controlled_transient_primary_failure")


def _assert_closed_schema(model: type, nested_fields: tuple[str, ...]) -> None:
    schema = model.model_json_schema()
    assert schema.get("additionalProperties") is False
    for field in nested_fields:
        item = schema["properties"][field]["items"]
        definition = schema["$defs"][item["$ref"].rsplit("/", 1)[-1]]
        assert definition.get("additionalProperties") is False


def validate_schema() -> dict[str, Any]:
    _assert_closed_schema(
        CollectionEmailEventLLMResponse,
        ("amount_assertions", "date_assertions"),
    )
    _assert_closed_schema(
        CollectionEmailEventResponse,
        ("amount_assertions", "date_assertions"),
    )
    _assert_closed_schema(
        CollectionEmailFactExtractionLLMResponse,
        ("amount_assertions", "date_assertions"),
    )
    _assert_closed_schema(
        CollectionEmailFactExtractionResponse,
        ("amount_assertions", "date_assertions"),
    )

    manifest = load_manifest_v2()
    required_tables = {
        "collection_email_message_evidence",
        "collection_email_chain_identifier_evidence",
        "collection_email_invoice_assertion_states",
        "collection_email_chain_invoice_states",
        "collection_email_chain_statuses",
    }
    available = set(manifest.silver_application)
    missing = sorted(required_tables - available)
    assert not missing, f"contracts manifest is missing collection tables: {missing}"

    return {
        "schema_valid": True,
        "contracts_version": contracts_version,
        "collection_contract_tables_checked": len(required_tables),
    }


async def validate_live(*, force_fallback: bool) -> dict[str, Any]:
    extractor = CollectionEmailFactExtractor()
    if force_fallback:
        extractor._client._primary = _TransientVertexFailure()

    response = await extractor.extract(
        CollectionEmailFactExtractionRequest(
            current_message={
                "direction": "inbound",
                "body": "We will pay invoice TEST-100 on 2026-07-20.",
                "quote_removal_status": "complete",
            },
            prior_chain_invoice_context={
                "invoice_candidates": [{"invoice_ref": "TEST-100"}],
                "candidate_count": 1,
                "is_truncated": False,
            },
        )
    )
    expected_provider = "openai" if force_fallback else "vertex"
    assert response.provider == expected_provider
    assert response.is_fallback is force_fallback
    assert int(response.prompt_tokens or 0) > 0
    assert int(response.completion_tokens or 0) > 0
    assert int(response.tokens_used or 0) >= int(response.prompt_tokens or 0) + int(
        response.completion_tokens or 0
    )
    assert response.ai_audit is not None
    assert response.ai_audit.ai_provider == expected_provider
    assert response.ai_audit.prompt_input_hash
    assert response.ai_audit.system_prompt_hash
    assert response.ai_audit.user_prompt_hash

    return {
        "live_call_valid": True,
        "provider": response.provider,
        "model": response.model,
        "is_fallback": response.is_fallback,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "tokens_used": response.tokens_used,
        "audit_hashes_present": True,
    }


def _promise_detail(response: CollectionEmailEventResponse) -> dict[str, Any]:
    assert response.semantic_classification == "PROMISE_TO_PAY"
    detail = next(item for item in response.intent_details if item.intent == "PROMISE_TO_PAY")
    assert detail.extracted_data is not None
    return detail.extracted_data.model_dump(mode="json")


async def validate_promise_amount_provenance(*, force_fallback: bool) -> dict[str, Any]:
    """Exercise full-balance and explicit-partial promises against a real provider.

    Synthetic invoice references keep the release check customer-data-free.
    Only aggregate provider telemetry is returned to stdout.
    """

    classifier = CollectionEmailEventClassifier()
    if force_fallback:
        classifier._client._primary = _TransientVertexFailure()

    common = {
        "mode": "known_collection_inbound",
        "prior_messages": [],
        "prior_evidence": [
            {
                "chain_invoice_context": {
                    "invoice_candidates": [{"invoice_ref": "TEST-100"}],
                    "candidate_count": 1,
                    "is_truncated": False,
                }
            }
        ],
        "chain_status": {"status": "active"},
    }
    amount_less = await classifier.classify(
        CollectionEmailEventRequest(
            **common,
            current_message={
                "direction": "inbound",
                "body": "We will pay invoice TEST-100 on 2026-08-14.",
                "quote_removal_status": "complete",
            },
        )
    )
    full_balance = _promise_detail(amount_less)
    assert full_balance["promise_amount"] is None
    assert full_balance["full_current_balance"] is True
    assert not any(
        item.amount is not None and item.assertion_type == "promised_payment"
        for item in amount_less.amount_assertions
    )

    explicit_partial = await classifier.classify(
        CollectionEmailEventRequest(
            **common,
            current_message={
                "direction": "inbound",
                "body": "We will pay GBP 125.00 against invoice TEST-100 on 2026-08-14.",
                "quote_removal_status": "complete",
            },
        )
    )
    partial = _promise_detail(explicit_partial)
    assert partial["promise_amount"] == 125.0
    assert partial["full_current_balance"] is False
    assert partial["promise_amount_evidence_text"]

    expected_provider = "openai" if force_fallback else "vertex"
    for response in (amount_less, explicit_partial):
        assert response.provider == expected_provider
        assert response.is_fallback is force_fallback
        assert int(response.prompt_tokens or 0) > 0
        assert int(response.completion_tokens or 0) > 0
        assert response.ai_audit is not None
        assert response.ai_audit.prompt_input_hash

    return {
        "promise_amount_provenance_valid": True,
        "provider": expected_provider,
        "calls": 2,
        "full_balance_default_valid": True,
        "explicit_partial_grounding_valid": True,
        "prompt_tokens": sum(
            int(item.prompt_tokens or 0) for item in (amount_less, explicit_partial)
        ),
        "completion_tokens": sum(
            int(item.completion_tokens or 0) for item in (amount_less, explicit_partial)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(
            "schema",
            "vertex",
            "openai-fallback",
            "promise-vertex",
            "promise-openai-fallback",
        ),
        default="schema",
    )
    args = parser.parse_args()

    result = validate_schema()
    if args.mode in {"vertex", "openai-fallback"}:
        result.update(asyncio.run(validate_live(force_fallback=args.mode == "openai-fallback")))
    elif args.mode in {"promise-vertex", "promise-openai-fallback"}:
        result.update(
            asyncio.run(
                validate_promise_amount_provenance(
                    force_fallback=args.mode == "promise-openai-fallback"
                )
            )
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

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

from src.api.models.requests import CollectionEmailFactExtractionRequest
from src.api.models.responses import (
    CollectionEmailEventResponse,
    CollectionEmailFactExtractionResponse,
)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("schema", "vertex", "openai-fallback"),
        default="schema",
    )
    args = parser.parse_args()

    result = validate_schema()
    if args.mode != "schema":
        result.update(asyncio.run(validate_live(force_fallback=args.mode == "openai-fallback")))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

"""Run an authenticated, local-only V3 semantic fixture server.

This is an integration harness, not a provider adapter.  It injects a
deterministic synthetic provider into the real ASGI route so Backend can prove
its typed HTTP, telemetry, and lane-completion boundaries without OpenRouter,
mailbox data, or a deployment.
"""

from __future__ import annotations

import argparse
import json
import os

# These values must be present before importing the AI settings singleton.
os.environ.setdefault("ENVIRONMENT", "local")
os.environ.setdefault("SERVICE_AUTH_TOKEN", "semantic-mail-local-test-token")

import uvicorn

from src.api.routes import interpret_collection_email_v3
from src.config.settings import settings
from src.engine.mail_semantic_evidence_v3 import MailSemanticEvidenceInterpreterV3
from src.llm.base import LLMResponse
from src.main import app


class SyntheticLocalProvider:
    """Return one source-grounded query from an in-memory synthetic packet."""

    provider_name = "synthetic-local"
    model_name = "semantic-mail-v3-fixture"

    async def complete(
        self, system_prompt: str, user_prompt: str, **_kwargs: object
    ) -> LLMResponse:
        del system_prompt
        request = json.loads(user_prompt)
        segments = list(((request.get("current_message") or {}).get("source_segments") or []))
        segment = next(
            (
                item
                for item in segments
                if item.get("source_kind") == "authored_body" and str(item.get("text") or "")
            ),
            None,
        )
        if segment is None:
            raise ValueError("fixture_requires_authored_body")
        span = {
            "source_id": segment["source_id"],
            "evidence_text": segment["text"],
            "supports": ["relevance", "operational_state"],
        }
        response = {
            "relevance": "collection",
            "semantic_events": [
                {
                    "event_id": "fixture-query-1",
                    "family": "query",
                    "transition": "requested",
                    "polarity": "affirmed",
                    "temporal_orientation": "current",
                    "invoice_refs": [],
                    "account_wide": False,
                    "amount": None,
                    "currency": None,
                    "asserted_date": None,
                    "reference": None,
                    "evidence": [span],
                    "confidence": 1.0,
                    "reason_codes": ["synthetic_local_fixture"],
                }
            ],
            "response_evidence": [span],
            "disposition": "accepted",
            "adjudication_status": "not_required",
            "confidence": 1.0,
            "reason_codes": ["synthetic_local_fixture"],
            "ambiguity_reason_codes": [],
        }
        return LLMResponse(
            content=json.dumps(response, sort_keys=True),
            provider=self.provider_name,
            model=self.model_name,
            usage={"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8011)
    args = parser.parse_args()
    if settings.environment == "production":
        raise RuntimeError("local_fixture_server_refuses_production")
    settings.enable_mail_semantic_evidence_v3 = True
    interpret_collection_email_v3.mail_semantic_evidence_interpreter_v3 = (
        MailSemanticEvidenceInterpreterV3(primary_provider=SyntheticLocalProvider())
    )
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

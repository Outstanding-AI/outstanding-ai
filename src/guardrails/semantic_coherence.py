"""Narrow LLM guardrail for follow-up coherence and tone alignment."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from src.llm.factory import llm_client

from .base import BaseGuardrail, GuardrailSeverity

logger = logging.getLogger(__name__)


class SemanticCoherenceResult(BaseModel):
    coherent: bool = Field(..., description="Whether the reply is coherent with the prior inbound")
    tone_aligned: bool = Field(
        ..., description="Whether the chosen tone matches the request context"
    )
    reasoning: str = Field("", description="Short rationale")


PROMPT = """You are validating a debt-collection draft for follow-up coherence.

Requested tone: {tone}
Last inbound message:
{last_inbound}

Draft:
{draft}

Judge only two things:
1. Is the draft a coherent response to the last inbound message?
2. Is the tone aligned with the requested tone?

Return JSON only."""


class SemanticCoherenceGuardrail(BaseGuardrail):
    """LLM-backed coherence check for follow-up drafts only."""

    def __init__(self):
        super().__init__(name="semantic_coherence", severity=GuardrailSeverity.MEDIUM)

    def validate(self, output: str, context: Any, **kwargs) -> list:
        if kwargs.get("mail_mode") == "initial":
            return [self._pass("Skipped semantic coherence for initial mail mode")]

        last_inbound = next(
            (
                msg
                for msg in (getattr(context, "lane_recent_messages", None) or [])
                if isinstance(msg, dict) and msg.get("direction") == "inbound"
            ),
            None,
        )
        if not last_inbound:
            return [self._pass("Skipped semantic coherence: no prior inbound on lane")]

        prompt = PROMPT.format(
            tone=kwargs.get("tone") or "professional",
            last_inbound=last_inbound.get("body_snippet") or last_inbound.get("subject") or "",
            draft=output,
        )
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response = loop.run_until_complete(
                    llm_client.complete(
                        system_prompt="You are a precise validation assistant.",
                        user_prompt=prompt,
                        temperature=0,
                        response_schema=SemanticCoherenceResult,
                        caller="semantic_coherence",
                    )
                )
            finally:
                asyncio.set_event_loop(None)
                loop.close()
            result = json.loads(response.content)
        except Exception as exc:
            logger.warning("Semantic coherence guardrail failed open: %s", exc)
            return [
                self._pass("Semantic coherence unavailable; skipped", details={"error": str(exc)})
            ]

        # ``token_usage`` surfaces this guardrail's token spend into the
        # parent ``generate_draft`` call's ``LLMRequestLog`` row via the
        # ``guardrail_result.total_token_usage`` aggregation in
        # ``generator.py``. Without this, the call lands in CloudWatch
        # only and per-draft cost reporting under-counts the true spend.
        usage = dict(getattr(response, "usage", {}) or {})
        details = {"reasoning": result.get("reasoning", "")}
        if not result.get("coherent", True) or not result.get("tone_aligned", True):
            return [
                self._fail(
                    "Draft is not semantically coherent with the lane reply context",
                    details=details,
                    token_usage=usage,
                )
            ]
        return [self._pass("Semantic coherence validated", details=details, token_usage=usage)]

"""Attribution semantics for ``GenerateDraftResponse.usage_breakdown``.

The response's top-level ``tokens_used / prompt_tokens / completion_tokens``
fields aggregate the full per-draft cost (main LLM + guardrail LLM).
``usage_breakdown.main_generation`` must report ONLY the primary draft
LLM call(s); ``usage_breakdown.guardrails.<name>`` must report ONLY
the per-guardrail LLM cost. Without the split, attribution is silently
wrong — ``main_generation`` was inflated by guardrail tokens before the
2026-05-09 fix.
"""

from __future__ import annotations

from src.engine.generator import _build_usage_breakdown, _TokenTotals
from src.guardrails.base import (
    GuardrailPipelineResult,
    GuardrailResult,
    GuardrailSeverity,
)


def _make_pipeline_result(token_usage: dict) -> GuardrailPipelineResult:
    """One guardrail with the given token_usage; entity_verification is the
    canonical LLM-using guardrail in the production pipeline.
    """
    r = GuardrailResult(
        passed=True,
        guardrail_name="entity_verification",
        severity=GuardrailSeverity.HIGH,
        token_usage=token_usage,
    )
    return GuardrailPipelineResult(
        all_passed=True,
        should_block=False,
        results=[r],
        per_guardrail_latency_ms={"entity_verification": 45.0},
    )


def test_main_100_guardrail_20_split_correctly():
    """Main 100 + guardrail 20 must split, not pile onto main_generation."""
    tokens = _TokenTotals()

    # Simulate a single main LLM call returning 100 tokens (60 prompt + 40 completion).
    tokens.total += 100
    tokens.prompt += 60
    tokens.completion += 40
    tokens.main_total += 100
    tokens.main_prompt += 60
    tokens.main_completion += 40
    tokens.main_attempts += 1

    # Then guardrails add 20 tokens to the aggregate accumulator only.
    tokens.total += 20
    tokens.prompt += 12
    tokens.completion += 8

    pipeline = _make_pipeline_result(
        {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}
    )

    breakdown = _build_usage_breakdown(
        main_provider="vertex",
        main_model="gemini-2.5-flash",
        main_prompt_tokens=tokens.main_prompt,
        main_completion_tokens=tokens.main_completion,
        main_total_tokens=tokens.main_total,
        main_latency_ms=2300.0,
        guardrail_result=pipeline,
    )

    # The aggregate accumulator is what feeds the response top-level tokens.
    assert tokens.total == 120
    # Main bucket is main-only.
    assert tokens.main_total == 100
    assert breakdown.main_generation.total_tokens == 100
    assert breakdown.main_generation.prompt_tokens == 60
    assert breakdown.main_generation.completion_tokens == 40
    # Guardrail bucket is guardrail-only and keyed by name.
    assert breakdown.guardrails is not None
    assert "entity_verification" in breakdown.guardrails
    gr = breakdown.guardrails["entity_verification"]
    assert gr.total_tokens == 20
    assert gr.prompt_tokens == 12
    assert gr.completion_tokens == 8


def test_retries_sum_into_main_generation():
    """Two main attempts must aggregate into main_generation, not just the last one."""
    tokens = _TokenTotals()

    # Attempt 1 -- failed guardrails, so a retry happens. 80 tokens main.
    tokens.total += 80
    tokens.prompt += 50
    tokens.completion += 30
    tokens.main_total += 80
    tokens.main_prompt += 50
    tokens.main_completion += 30
    tokens.main_attempts += 1

    # Attempt 2 -- guardrails clean, 90 tokens main.
    tokens.total += 90
    tokens.prompt += 55
    tokens.completion += 35
    tokens.main_total += 90
    tokens.main_prompt += 55
    tokens.main_completion += 35
    tokens.main_attempts += 1

    # Guardrail LLM call only on the final pass (15 tokens).
    tokens.total += 15
    tokens.prompt += 9
    tokens.completion += 6

    pipeline = _make_pipeline_result(
        {"prompt_tokens": 9, "completion_tokens": 6, "total_tokens": 15}
    )

    breakdown = _build_usage_breakdown(
        main_provider="vertex",
        main_model="gemini-2.5-flash",
        main_prompt_tokens=tokens.main_prompt,
        main_completion_tokens=tokens.main_completion,
        main_total_tokens=tokens.main_total,
        main_latency_ms=4100.0,
        guardrail_result=pipeline,
    )

    # Aggregate: 80 + 90 + 15 = 185.
    assert tokens.total == 185
    # Main bucket: 80 + 90 = 170 (sums across retries, not last only).
    assert tokens.main_total == 170
    assert tokens.main_attempts == 2
    assert breakdown.main_generation.total_tokens == 170
    assert breakdown.main_generation.prompt_tokens == 105
    assert breakdown.main_generation.completion_tokens == 65
    # Guardrail bucket reports only its own 15 tokens.
    assert breakdown.guardrails["entity_verification"].total_tokens == 15


def test_no_prompt_or_body_content_in_breakdown():
    """``usage_breakdown`` must not carry prompt / body / customer text.

    The Pydantic models only declare typed numeric / boolean / provider
    fields -- a regression that adds free-text fields would fail this.
    """
    pipeline = _make_pipeline_result({})
    breakdown = _build_usage_breakdown(
        main_provider="vertex",
        main_model="gemini-2.5-flash",
        main_prompt_tokens=10,
        main_completion_tokens=5,
        main_total_tokens=15,
        main_latency_ms=100.0,
        guardrail_result=pipeline,
    )
    dumped = breakdown.model_dump()
    forbidden = {
        "prompt",
        "user_prompt",
        "system_prompt",
        "body",
        "body_plain",
        "body_html",
        "subject",
        "customer_name",
        "to_email",
        "from_email",
    }
    assert not (set(dumped.get("main_generation") or {}) & forbidden)
    for entry in (dumped.get("guardrails") or {}).values():
        assert not (set(entry) & forbidden)

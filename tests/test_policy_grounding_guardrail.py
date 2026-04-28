"""Regression tests for the LLM-backed PolicyGroundingGuardrail.

Background — ESWL activation 2026-04-28: the legacy regex
``\\b(settle|settlement|compromise|accept \\d+%)\\b`` blocked 100% of
OpenAI-fallback drafts because routine collection language ("please
settle this account") tripped a bare verb match. The 3-attempt regen
loop always exhausted, the entire 30-draft batch failed, push never ran.

The replacement is an LLM judge (this guardrail). These tests pin:

1. The LLM is the authoritative judge: when LLM says ``violation=True``
   under an unauthorised policy → block; otherwise → pass.
2. Authorised policies short-circuit (no LLM call when nothing to check).
3. The strict regex fallback only triggers when the LLM raises, and
   only blocks on unambiguous commitment phrases (NOT the historical
   loose-regex false positives).
4. The fail-closed contract is preserved end-to-end: an unauthorised
   policy committed to in the draft text DOES block, whether decided
   by the LLM or the strict fallback.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.guardrails.policy_grounding import (
    POLICY_CATEGORIES,
    STRICT_FALLBACK_TRIGGERS,
    PolicyGroundingGuardrail,
    PolicyGroundingResult,
)


@pytest.fixture
def guardrail():
    return PolicyGroundingGuardrail()


@pytest.fixture
def empty_context():
    """Stand-in CaseContext with no ``authorized_policies`` attribute."""

    class _Ctx:
        pass

    return _Ctx()


def _mock_llm_response(
    *, violation: bool, policy: str | None, reasoning: str = "test", usage: dict | None = None
):
    """Build a fake LLMResponse-like object whose ``.content`` parses to
    the requested verdict."""
    response = MagicMock()
    response.content = PolicyGroundingResult(
        violation=violation, policy=policy, reasoning=reasoning
    ).model_dump_json()
    response.usage = (
        usage
        if usage is not None
        else {
            "prompt_tokens": 250,
            "completion_tokens": 40,
            "total_tokens": 290,
        }
    )
    return response


# Phrases that previously tripped the loose regex but are routine
# collection language. The LLM should classify each as not-a-violation.
HISTORICAL_FALSE_POSITIVES = [
    "Please settle this overdue invoice at your earliest convenience.",
    "To settle the matter, please remit the outstanding amount.",
    "We would appreciate prompt settlement of the outstanding balance.",
    "Please contact us to discuss settlement options.",
    "Hi John, just a friendly reminder about invoice INV-001.",
    "Please make payment to settle your account by 30 April.",
    "We are seeking settlement of the outstanding £5,432.",
]


# Real commitment-shaped phrases the LLM should classify as violations.
COMMITMENT_PHRASES = [
    ("settlement_allowed", "We can settle for 80% of the balance to close this out."),
    (
        "settlement_allowed",
        "We are willing to negotiate the amount.",
    ),  # paraphrase the regex would miss
    ("settlement_allowed", "Let me see what reduction I can secure for you."),  # indirect
    ("discount_allowed", "We can offer a 10% discount if paid by Friday."),
    ("discount_allowed", "We will waive the late fee this once."),
    ("legal_escalation_enabled", "We will refer this matter to our solicitor."),
    ("legal_escalation_enabled", "Court action will be taken if payment is not received."),
    ("statutory_interest_enabled", "We will charge statutory interest at 8% above base rate."),
]


class TestShortCircuitWhenAllAuthorised:
    """If every category is authorised, no LLM call should fire — pure
    cost-saver and avoids unnecessary latency."""

    def test_all_authorised_passes_without_llm(self, guardrail, empty_context):
        all_on = {p: True for p in POLICY_CATEGORIES}
        with patch("src.guardrails.policy_grounding.llm_client.complete") as mock_complete:
            results = guardrail.validate(
                "We can settle for 80% of the balance.",
                empty_context,
                authorized_policies=all_on,
            )
        assert all(r.passed for r in results)
        mock_complete.assert_not_called()


class TestLlmJudgeAuthoritative:
    """When the LLM is reachable, its verdict is binding (subject to the
    policy being in the unauthorised set)."""

    def test_llm_violation_blocks(self, guardrail, empty_context):
        with patch(
            "src.guardrails.policy_grounding.llm_client.complete",
            new=AsyncMock(
                return_value=_mock_llm_response(
                    violation=True,
                    policy="settlement_allowed",
                    reasoning="explicit settlement offer",
                )
            ),
        ):
            results = guardrail.validate(
                "We can settle for 80%.", empty_context, authorized_policies={}
            )
        failures = [r for r in results if not r.passed]
        assert len(failures) == 1
        assert "settlement_allowed" in failures[0].message
        assert failures[0].details.get("judge") == "llm"

    def test_llm_pass_passes(self, guardrail, empty_context):
        with patch(
            "src.guardrails.policy_grounding.llm_client.complete",
            new=AsyncMock(
                return_value=_mock_llm_response(
                    violation=False, policy=None, reasoning="just asks for payment"
                )
            ),
        ):
            results = guardrail.validate(
                "Please settle this account.", empty_context, authorized_policies={}
            )
        assert all(r.passed for r in results)
        assert results[0].details.get("judge") == "llm"

    def test_llm_violation_on_authorised_policy_passes(self, guardrail, empty_context):
        """If the LLM flags ``settlement_allowed`` but the tenant has
        authorised it, the block is suppressed — authorisation is a
        hard override of the LLM verdict."""
        with patch(
            "src.guardrails.policy_grounding.llm_client.complete",
            new=AsyncMock(
                return_value=_mock_llm_response(
                    violation=True, policy="settlement_allowed", reasoning="real settlement offer"
                )
            ),
        ):
            results = guardrail.validate(
                "We can settle for 80%.",
                empty_context,
                authorized_policies={"settlement_allowed": True},
            )
        # settlement_allowed is in the authorised set → unauthorised_set
        # excludes it → guardrail short-circuits before calling the LLM.
        # (Actually the LLM also won't be reached since unauthorised_set
        # would still contain the OTHER 3 policies; but the verdict
        # ``policy=settlement_allowed`` would be an authorised category
        # and wouldn't trigger a block.)
        assert all(r.passed for r in results)


class TestHistoricalFalsePositivesDoNotRegress:
    """The whole point of the LLM rewrite: the phrases that broke ESWL
    activation must not block when the LLM correctly classifies them."""

    @pytest.mark.parametrize("phrase", HISTORICAL_FALSE_POSITIVES)
    def test_routine_collection_language_passes(self, guardrail, empty_context, phrase):
        with patch(
            "src.guardrails.policy_grounding.llm_client.complete",
            new=AsyncMock(
                return_value=_mock_llm_response(
                    violation=False, policy=None, reasoning="routine collection ask"
                )
            ),
        ):
            results = guardrail.validate(phrase, empty_context, authorized_policies={})
        assert all(r.passed for r in results), (
            f"Routine phrase incorrectly blocked: {phrase!r}\n"
            f"Messages: {[r.message for r in results if not r.passed]}"
        )


class TestCommitmentPhrasesBlock:
    """Commitment-shaped phrasing must block when the LLM correctly flags
    it. Mock the LLM to return the expected violation per phrase."""

    @pytest.mark.parametrize(("policy", "phrase"), COMMITMENT_PHRASES)
    def test_real_commitment_blocks(self, guardrail, empty_context, policy, phrase):
        with patch(
            "src.guardrails.policy_grounding.llm_client.complete",
            new=AsyncMock(
                return_value=_mock_llm_response(
                    violation=True, policy=policy, reasoning=f"detected {policy}"
                )
            ),
        ):
            results = guardrail.validate(phrase, empty_context, authorized_policies={})
        assert any(not r.passed for r in results), (
            f"Commitment phrase incorrectly passed: {phrase!r}"
        )
        failure_msg = next(r.message for r in results if not r.passed)
        assert policy in failure_msg


class TestStrictRegexFallback:
    """When the LLM raises, the guardrail falls back to a strict regex
    that only matches unambiguous commitment phrases. Looser signals pass."""

    def test_llm_failure_falls_back_with_strict_block(self, guardrail, empty_context):
        with patch(
            "src.guardrails.policy_grounding.llm_client.complete",
            new=AsyncMock(side_effect=RuntimeError("vertex unavailable")),
        ):
            results = guardrail.validate(
                "We can settle for £500.", empty_context, authorized_policies={}
            )
        failures = [r for r in results if not r.passed]
        assert len(failures) == 1
        assert "settlement_allowed" in failures[0].message
        assert failures[0].details.get("judge") == "regex_fallback"
        assert "vertex unavailable" in failures[0].details.get("llm_error", "")

    def test_llm_failure_loose_phrase_passes_under_fallback(self, guardrail, empty_context):
        """The fallback is intentionally narrow — a routine collection
        phrase must NOT block even when the LLM is down. Otherwise an
        LLM outage would reproduce the historical 100%-block failure."""
        with patch(
            "src.guardrails.policy_grounding.llm_client.complete",
            new=AsyncMock(side_effect=RuntimeError("vertex unavailable")),
        ):
            results = guardrail.validate(
                "Please settle this account at your earliest convenience.",
                empty_context,
                authorized_policies={},
            )
        assert all(r.passed for r in results), (
            "Strict fallback must NOT block routine collection language; "
            "otherwise an LLM outage replays the historical 100%-block bug"
        )

    def test_llm_failure_legal_threat_blocks(self, guardrail, empty_context):
        with patch(
            "src.guardrails.policy_grounding.llm_client.complete",
            new=AsyncMock(side_effect=RuntimeError("vertex unavailable")),
        ):
            results = guardrail.validate(
                "Court action will be taken if payment is not received.",
                empty_context,
                authorized_policies={},
            )
        assert any(not r.passed for r in results)
        failure_msg = next(r.message for r in results if not r.passed)
        assert "legal_escalation_enabled" in failure_msg

    def test_llm_failure_authorised_category_does_not_block_via_fallback(
        self, guardrail, empty_context
    ):
        """Fallback must respect the authorisation set: a strict-pattern
        match on an authorised category should not produce a block."""
        with patch(
            "src.guardrails.policy_grounding.llm_client.complete",
            new=AsyncMock(side_effect=RuntimeError("down")),
        ):
            results = guardrail.validate(
                "We can settle for £500.",
                empty_context,
                authorized_policies={"settlement_allowed": True},
            )
        # settlement_allowed authorised → still in unauthorised_set are
        # the OTHER 3 policies. Strict fallback for settlement does not
        # fire because settlement is no longer in the unauthorised set.
        assert all(r.passed for r in results)

    def test_llm_parse_failure_falls_back(self, guardrail, empty_context):
        """If the LLM returns garbage that fails Pydantic validation,
        treat it as an LLM failure and fall back."""
        bad_response = MagicMock()
        bad_response.content = "this is not json"
        with patch(
            "src.guardrails.policy_grounding.llm_client.complete",
            new=AsyncMock(return_value=bad_response),
        ):
            results = guardrail.validate(
                "We can settle for £500.", empty_context, authorized_policies={}
            )
        # Fallback path engaged → strict pattern matched → block.
        failures = [r for r in results if not r.passed]
        assert len(failures) == 1
        assert failures[0].details.get("judge") == "regex_fallback"


class TestStructuralInvariants:
    def test_policy_categories_match_backend_keys(self):
        """If anyone renames a key in
        ``Solvix/services/config_hierarchy.py::resolve_authorized_policies``,
        this test calls out the backend side too — both repos must move
        together."""
        assert set(POLICY_CATEGORIES) == {
            "legal_escalation_enabled",
            "statutory_interest_enabled",
            "discount_allowed",
            "settlement_allowed",
        }

    def test_strict_fallback_triggers_cover_every_category(self):
        assert set(STRICT_FALLBACK_TRIGGERS.keys()) == set(POLICY_CATEGORIES)

    def test_strict_fallback_patterns_compile(self):
        import re as _re

        for policy, pattern in STRICT_FALLBACK_TRIGGERS.items():
            try:
                _re.compile(pattern, _re.IGNORECASE)
            except _re.error as exc:
                pytest.fail(f"Strict fallback pattern {policy!r} does not compile: {exc}")

    def test_severity_is_high(self, guardrail):
        from src.guardrails.base import GuardrailSeverity

        assert guardrail.severity == GuardrailSeverity.HIGH


class TestTokenAttribution:
    """Per-call LLM cost lands in the parent ``generate_draft`` row of
    ``LLMRequestLog`` only when each guardrail surfaces ``token_usage``
    on its result. The aggregator in ``base.py::total_token_usage``
    reads ``r.token_usage``, NOT ``r.details``. These tests pin the
    contract so a future refactor can't silently break per-draft cost
    attribution again."""

    def test_pass_result_carries_usage(self, guardrail, empty_context):
        with patch(
            "src.guardrails.policy_grounding.llm_client.complete",
            new=AsyncMock(
                return_value=_mock_llm_response(
                    violation=False,
                    policy=None,
                    usage={"prompt_tokens": 250, "completion_tokens": 40, "total_tokens": 290},
                )
            ),
        ):
            results = guardrail.validate(
                "Please settle this account.", empty_context, authorized_policies={}
            )
        assert results[0].passed
        assert results[0].token_usage == {
            "prompt_tokens": 250,
            "completion_tokens": 40,
            "total_tokens": 290,
        }

    def test_fail_result_carries_usage(self, guardrail, empty_context):
        with patch(
            "src.guardrails.policy_grounding.llm_client.complete",
            new=AsyncMock(
                return_value=_mock_llm_response(
                    violation=True,
                    policy="settlement_allowed",
                    usage={"prompt_tokens": 250, "completion_tokens": 40, "total_tokens": 290},
                )
            ),
        ):
            results = guardrail.validate(
                "We can settle for 80%.", empty_context, authorized_policies={}
            )
        assert not results[0].passed
        assert results[0].token_usage["total_tokens"] == 290

    def test_aggregation_includes_guardrail_usage(self, guardrail, empty_context):
        """Concrete proof that ``GuardrailPipelineResult.total_token_usage``
        rolls up our reported usage. This is the path that flows into
        the parent ``generate_draft`` LLMRequestLog row."""
        from src.guardrails.base import GuardrailPipelineResult

        with patch(
            "src.guardrails.policy_grounding.llm_client.complete",
            new=AsyncMock(
                return_value=_mock_llm_response(
                    violation=False,
                    policy=None,
                    usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                )
            ),
        ):
            results = guardrail.validate(
                "Please settle this account.", empty_context, authorized_policies={}
            )

        pipeline_result = GuardrailPipelineResult(
            all_passed=True,
            should_block=False,
            results=results,
        )
        assert pipeline_result.total_token_usage == {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        }

    def test_short_circuit_no_usage(self, guardrail, empty_context):
        """When all policies are authorised the LLM is not called →
        no usage to attribute. Pin this so the short-circuit doesn't
        accidentally start fabricating zero-token rows."""
        with patch("src.guardrails.policy_grounding.llm_client.complete") as mock_complete:
            results = guardrail.validate(
                "Anything goes.",
                empty_context,
                authorized_policies={p: True for p in POLICY_CATEGORIES},
            )
        mock_complete.assert_not_called()
        assert results[0].passed
        assert results[0].token_usage == {}

    def test_fallback_path_no_usage(self, guardrail, empty_context):
        """When the LLM raises and we fall back to the strict regex,
        there's no LLM usage to attribute — fallback results carry an
        empty ``token_usage``."""
        with patch(
            "src.guardrails.policy_grounding.llm_client.complete",
            new=AsyncMock(side_effect=RuntimeError("down")),
        ):
            results = guardrail.validate(
                "We can settle for £500.", empty_context, authorized_policies={}
            )
        assert results[0].token_usage == {}

    def test_name_is_policy_grounding(self, guardrail):
        assert guardrail.name == "policy_grounding"

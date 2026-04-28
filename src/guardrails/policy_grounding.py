"""LLM-backed policy authorisation guardrail.

Replaces the legacy keyword regex (deterministic but with structural
holes — see incident note below) with a small LLM judge that reads the
draft body in context and decides whether it makes any commitment under
a policy category the tenant has not authorised.

Why LLM, not regex
------------------
The bare-keyword regex blocked 100% of OpenAI-fallback drafts during the
ESWL activation 2026-04-28: the pattern ``\\bsettle\\b`` matched routine
collection language ("please settle this account" — a euphemism for
"please pay") and the 3-attempt regen loop always exhausted. Tightening
the patterns reduced false positives but left semantic holes:
paraphrases like "we'd be open to negotiating" / "let me see what I can
do for you" are real settlement offers the regex would never catch.

LLM judgement handles:
- Synonym paraphrasing (negotiate ≈ settle, haircut ≈ partial settlement)
- Conditional framing ("if X then we can do Y") vs definite commitment
- Past-tense factual reference vs future-tense offer
- Implicit numeric reductions ("if you can pay £500 we can close this")
- Distinguishing "settle the matter" (a euphemism) from "settle for £500"
  (a real settlement offer)

Fallback contract
-----------------
On LLM unavailability or parse failure, the guardrail falls back to a
**strict** regex matcher (much narrower than the historic loose regex).
Only unambiguous commitment phrases ("settle for £X", "in full and final
settlement", "court action") trigger blocks via the fallback path. We
accept that a subtle violation could slip through during an LLM outage —
the alternative is the historical 100%-block failure mode, which kills
entire sync runs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from src.llm.factory import llm_client

from .base import BaseGuardrail, GuardrailResult, GuardrailSeverity

logger = logging.getLogger(__name__)


# Policy keys MUST match the keys produced by
# ``Solvix/services/config_hierarchy.py::resolve_authorized_policies``.
# Keep this list in sync with the backend resolver.
POLICY_CATEGORIES: tuple[str, ...] = (
    "legal_escalation_enabled",
    "statutory_interest_enabled",
    "discount_allowed",
    "settlement_allowed",
)


class PolicyGroundingResult(BaseModel):
    """Structured LLM verdict.

    ``violation`` is True if the draft makes a hard commitment under any
    UNAUTHORISED category. The LLM is told which categories are
    authorised so it knows which ones to enforce.
    """

    violation: bool = Field(..., description="Whether the draft commits to an unauthorised policy")
    policy: str | None = Field(
        None,
        description="The unauthorised policy that the draft commits to, or null when no violation",
    )
    reasoning: str = Field("", description="One short sentence rationale")


SYSTEM_PROMPT = "You are a precise debt-collection policy validator. Reply in JSON only."

PROMPT_TEMPLATE = """You are validating a debt-collection draft email for policy compliance.

Four policy categories exist:
- legal_escalation_enabled: hard threats or commitments to refer to legal counsel, instruct solicitors, take court action, file at small claims, commence legal proceedings, escalate to a tribunal, pursue litigation
- statutory_interest_enabled: applying or threatening to charge statutory interest under the UK Late Payment of Commercial Debts (Interest) Act, or referencing the Act in the context of charging interest
- discount_allowed: offering a price reduction, percentage discount, waiver of fees or charges, or other reduction of the amount owed
- settlement_allowed: offering to accept LESS than the full balance — partial settlement, full and final settlement, compromise, accepting a percentage instead of full amount, or any synonym for settling for less (negotiating the amount, agreeing a haircut, exploring a reduced figure, etc.)

The tenant has AUTHORISED these categories: {authorised_list}
The tenant has NOT AUTHORISED these categories: {unauthorised_list}

Your task: decide whether the draft makes any HARD COMMITMENT or EXPLICIT OFFER under an UNAUTHORISED category.

DO flag (commitment-shaped):
- Concrete offers ("we can settle for £500", "we can offer a 10% discount", "we'll waive the late fee")
- Explicit threats ("we will refer to our solicitor", "court action will be taken")
- Affirmative willingness to depart from full payment ("we are willing to negotiate the amount", "happy to discuss a partial payment", "let me see what reduction I can secure")
- Implicit numeric reductions ("if you can pay £500 by Friday we can close this matter" when the actual balance is higher)

DO NOT flag (routine, non-commitment):
- Asking for payment ("please settle this account", "settle the matter promptly", "settle the outstanding balance" — these are euphemisms for "pay")
- Past-tense factual references ("a discount of £100 was previously applied")
- Mentioning the law factually without invoking interest ("under the Late Payment Act, payment is due within 30 days")
- Bare contact invitations ("please call to discuss", "if you have questions, please get in touch")
- Conditional discussion of options ("if you contact us, we can review your account")
- Internal escalation ("I will escalate this internally")

Draft body:
---
{draft}
---

Return JSON ONLY: {{"violation": <bool>, "policy": <one of [{policy_keys}] or null>, "reasoning": "<one short sentence>"}}"""


# Strict fallback regex used ONLY when the LLM is unavailable. These
# patterns deliberately match only the most unambiguous commitment-shaped
# phrases — narrower than the pre-LLM loose regex so we don't reintroduce
# the false-positive cascade. If the LLM is healthy, this code path is
# unreachable.
STRICT_FALLBACK_TRIGGERS: dict[str, str] = {
    "legal_escalation_enabled": (
        r"(?:"
        r"\bcourt\s+action\b"
        r"|\blegal\s+proceedings\b"
        r"|\bsmall\s+claims\b"
        r"|\binstruct\w*\s+(?:our\s+|a\s+)?solicitor"
        r"|\brefer\w*\s+(?:this|the\s+matter|you)\s+to\s+(?:our\s+|a\s+)?(?:solicitor|legal\s+counsel)"
        r"|\bcommenc\w+\s+legal\s+proceedings"
        r")"
    ),
    "statutory_interest_enabled": (
        r"(?:"
        r"\bstatutory\s+interest\b"
        r"|\bcharging\s+(?:statutory\s+)?interest\s+(?:at|of)\s+\d"
        r")"
    ),
    "discount_allowed": (
        r"(?:"
        r"\b(?:offer|give|grant)\s+(?:you\s+)?(?:a\s+)?\d+\s*%\s*discount"
        r"|\b\d+\s*%\s+(?:discount|off)\b"
        r"|\bdiscount\s+of\s+(?:£|\$|€)\s*\d"
        r")"
    ),
    "settlement_allowed": (
        r"(?:"
        r"\bsettle\s+(?:for|at)\s+(?:£|\$|€)?\s*\d"
        r"|\baccept\s+(?:£|\$|€)\s*\d"
        r"|\baccept\s+\d+\s*%"
        r"|\bin\s+full\s+and\s+final\s+settlement\b"
        r"|\bfull\s+and\s+final\s+settlement\b"
        r"|\bpartial\s+settlement\b"
        r"|\bsettlement\s+offer\b"
        r")"
    ),
}


class PolicyGroundingGuardrail(BaseGuardrail):
    """LLM-backed authorisation-policy guardrail.

    Behaviour
    ---------
    1. Resolve ``authorized_policies`` from kwargs (preferred) or from
       ``context.authorized_policies`` (legacy).
    2. If every policy is authorised, short-circuit pass — nothing to
       check.
    3. Otherwise, ask the LLM to judge whether the draft commits to any
       UNAUTHORISED policy. Block on positive verdict.
    4. On LLM exception or parse failure, fall through to the strict
       regex matcher. Only unambiguous commitment phrases block via the
       fallback; everything else passes (fail-open for soft signals).

    HIGH severity — blocking failures trigger the guardrail retry loop in
    ``generator.py::_run_llm_with_guardrails``.
    """

    def __init__(self):
        super().__init__(name="policy_grounding", severity=GuardrailSeverity.HIGH)

    def validate(self, output: str, context: Any, **kwargs) -> list:
        authorised = (
            kwargs.get("authorized_policies") or getattr(context, "authorized_policies", None) or {}
        )

        authorised_set = {p for p in POLICY_CATEGORIES if authorised.get(p, False)}
        unauthorised_set = set(POLICY_CATEGORIES) - authorised_set

        if not unauthorised_set:
            return [self._pass("All policy categories authorised")]

        try:
            verdict = self._llm_judge(output, authorised_set, unauthorised_set)
        except Exception as exc:
            logger.warning(
                "policy_grounding LLM unavailable (%s); falling back to strict regex",
                exc,
            )
            return self._strict_regex_fallback(output, unauthorised_set, exc)

        if verdict.violation and verdict.policy in unauthorised_set:
            return [
                self._fail(
                    f"Draft commits to {verdict.policy} content without authorisation",
                    details={"reasoning": verdict.reasoning, "judge": "llm"},
                )
            ]
        return [
            self._pass(
                "Policy grounding validated",
                details={"reasoning": verdict.reasoning, "judge": "llm"},
            )
        ]

    def _llm_judge(
        self,
        output: str,
        authorised_set: set[str],
        unauthorised_set: set[str],
    ) -> PolicyGroundingResult:
        """Run the LLM judge synchronously.

        The guardrail pipeline is sync; we wrap an async LLM call in a
        fresh event loop and tear it down on every invocation. This
        matches the pattern in ``semantic_coherence.py`` and avoids
        cross-loop primitive issues seen in the entity guardrail prior
        to the April 2026 fix.
        """
        prompt = PROMPT_TEMPLATE.format(
            authorised_list=", ".join(sorted(authorised_set)) or "none",
            unauthorised_list=", ".join(sorted(unauthorised_set)),
            policy_keys=", ".join(POLICY_CATEGORIES),
            draft=output,
        )
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(
                llm_client.complete(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=prompt,
                    temperature=0,
                    max_tokens=300,
                    response_schema=PolicyGroundingResult,
                    caller="policy_grounding",
                )
            )
        finally:
            asyncio.set_event_loop(None)
            loop.close()

        return PolicyGroundingResult.model_validate(json.loads(response.content))

    def _strict_regex_fallback(
        self,
        output: str,
        unauthorised_set: set[str],
        exc: Exception,
    ) -> list[GuardrailResult]:
        """Conservative deterministic fallback for the LLM-unavailable case.

        Only unambiguous commitment phrases trigger here. We deliberately
        do NOT replicate the loose pre-LLM regex — the goal is to keep
        the worst-case fallback narrow enough that an LLM outage does
        not produce a wave of false-positive blocks.
        """
        for policy, pattern in STRICT_FALLBACK_TRIGGERS.items():
            if policy not in unauthorised_set:
                continue
            if re.search(pattern, output, re.IGNORECASE):
                return [
                    self._fail(
                        f"Draft commits to {policy} content without authorisation",
                        details={
                            "judge": "regex_fallback",
                            "reason": "LLM judge unavailable; matched strict commitment phrase",
                            "llm_error": str(exc),
                        },
                    )
                ]
        return [
            self._pass(
                "Policy grounding validated (strict regex fallback)",
                details={"judge": "regex_fallback", "llm_error": str(exc)},
            )
        ]

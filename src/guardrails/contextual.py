"""Contextual Coherence Guardrail -- validate situational awareness.

Check that the AI output is contextually appropriate given the party's
current state (active dispute, hardship indication, broken promise
history) AND that the invoice references in the prose match the
structured ``CaseContext.obligations`` data.

Sprint C item #11 (2026-04-28): the original guardrail relied entirely
on phrase-matching heuristics on the output text. Phrase-matching is
unavoidable for tone signals (dispute / hardship / history language),
but the strongest contextual checks are STRUCTURAL: are the invoice
numbers / collection_statuses in the prose consistent with what the
backend told us is true? Those checks moved into ``_validate_invoice_references``
and ``_validate_no_paid_invoice_chase`` — deterministic, no
phrase-matching, leverages the structured ``CaseContext.obligations``
field directly.

Severity stays LOW for the legacy phrase checks (heuristic). The new
structural checks use the same severity for consistency: a hallucinated
invoice number is a real bug, but blocking on it would surface as a
hard generation failure for the operator. Logging-only is the right
default; promotion to HIGH is a future call once we see the
false-positive rate in production.
"""

import logging
import re

from src.api.models.requests import CaseContext

from .base import BaseGuardrail, GuardrailResult, GuardrailSeverity

logger = logging.getLogger(__name__)

# Sprint C item #11 (2026-04-28): regex for invoice numbers cited in
# the AI output prose. Common patterns from real Sage debtor data:
#   INV-12345, Invoice #12345, Inv 12345, INV12345
# Conservative: requires a recognisable prefix so we don't over-extract
# bare numbers (which could be amounts / dates / receipt refs).
# ``\binv(?:oice)?\b`` requires a word boundary so the prefix is its
# own token — this lets us detect double-prefix forms ("Your invoice
# INV-12345") and use the inner ``INV-12345`` for comparison.
_INVOICE_REF_PATTERN = re.compile(
    r"\binv(?:oice)?\b[\s\-#]*([A-Z0-9][A-Z0-9\-]{2,})\b",
    re.IGNORECASE,
)

# Statuses where an obligation is no longer collectible. If the AI's
# prose demands payment for an invoice with one of these statuses, the
# contextual guardrail flags it.
_NON_COLLECTIBLE_STATUSES = frozenset({"paid", "credited", "written_off"})


class ContextualCoherenceGuardrail(BaseGuardrail):
    """Validate situational awareness of AI-generated draft text.

    LOW severity -- failures are logged but never block output, since
    detection uses phrase-matching heuristics for tone signals AND
    structured cross-checks for invoice references. False positives
    on the phrase side are common; the structured checks are tighter
    but still best-effort (regex-based extraction).

    Conditional checks (only run when the relevant flag is set):
    1. Active dispute: no payment demands without dispute acknowledgment.
    2. Hardship indicated: no harsh language without empathetic phrasing.
    3. Broken promises >= 2: history-referencing language expected.

    Always-on structural checks (Sprint C #11, 2026-04-28):
    4. Invoice references: every invoice number in the prose must
       appear in ``context.obligations``. Hallucinated references fail.
    5. Paid-invoice chase: if the prose demands payment for an invoice
       whose ``collection_status`` is paid / credited / written_off,
       fail.

    When no checks apply, a single pass result is returned.
    """

    def __init__(self):
        super().__init__(
            name="contextual_coherence",
            severity=GuardrailSeverity.LOW,  # Log only, don't block
        )

    def validate(self, output: str, context: CaseContext, **kwargs) -> list[GuardrailResult]:
        """Validate contextual coherence of the output.

        Conditionally run sub-checks based on context flags:
        - Active dispute: check for inappropriate payment demands.
        - Hardship indicated: check for harsh language without empathy.
        - Broken promises >= 2: check for history acknowledgment.

        If no special conditions apply, return a single pass result.

        Args:
            output: AI-generated draft body text.
            context: Case context with dispute/hardship/promise flags.
            **kwargs: Unused.

        Returns:
            List of GuardrailResult objects (one per applicable check).
        """
        results = []
        active_dispute = (
            context.lane_active_dispute
            if getattr(context, "lane_active_dispute", None) is not None
            else context.active_dispute
        )
        broken_promises_count = (
            context.lane_broken_promises_count
            if getattr(context, "lane_broken_promises_count", None) is not None
            else context.broken_promises_count
        )

        # Check dispute handling
        if active_dispute:
            results.append(self._validate_dispute_awareness(output, context))

        # Check hardship handling
        if context.hardship_indicated:
            results.append(self._validate_hardship_tone(output, context))

        # Check broken promise awareness
        if broken_promises_count > 0:
            results.append(self._validate_promise_awareness(output, context))

        # Sprint C item #11 (2026-04-28): structural cross-checks against
        # ``context.obligations``. These run unconditionally because they
        # don't depend on context flags — a hallucinated invoice
        # reference is always wrong, regardless of dispute / hardship /
        # promise state.
        obligations = list(getattr(context, "obligations", None) or [])
        if obligations:
            results.append(self._validate_invoice_references(output, obligations))
            results.append(self._validate_no_paid_invoice_chase(output, obligations))

        # If no checks ran (no flags set, no obligations), pass-through.
        if not results:
            results.append(self._pass(message="No context conditions or obligations to validate"))

        return results

    def _validate_dispute_awareness(self, output: str, context: CaseContext) -> GuardrailResult:
        """Validate that the output respects an active dispute.

        Fail if the draft contains payment-demand language without
        also acknowledging the dispute.  Fail if the dispute is not
        acknowledged at all.
        """
        output_lower = output.lower()

        # Phrases that suggest payment demand (inappropriate during dispute)
        demand_phrases = [
            "pay immediately",
            "pay now",
            "immediate payment",
            "pay in full",
            "demand payment",
            "must pay",
            "required to pay",
            "failure to pay will result",
            "legal action",
            "collection agency",
        ]

        # Phrases that acknowledge dispute (appropriate)
        dispute_phrases = [
            "dispute",
            "under review",
            "investigating",
            "looking into",
            "resolve",
            "concern",
            "issue",
        ]

        # Check for inappropriate demand language
        found_demands = [phrase for phrase in demand_phrases if phrase in output_lower]
        acknowledges_dispute = any(phrase in output_lower for phrase in dispute_phrases)

        if found_demands and not acknowledges_dispute:
            return self._fail(
                message="Output demands payment during active dispute without acknowledgment",
                expected="Acknowledge dispute, avoid payment demands",
                found=found_demands,
                details={
                    "demand_phrases_found": found_demands,
                    "dispute_acknowledged": acknowledges_dispute,
                    "active_dispute": True,
                },
            )

        if not acknowledges_dispute:
            return self._fail(
                message="Output does not acknowledge active dispute",
                expected="Reference to dispute or investigation",
                found="No dispute acknowledgment",
                details={"active_dispute": True, "dispute_acknowledged": False},
            )

        return self._pass(
            message="Output appropriately handles dispute context",
            details={"dispute_acknowledged": acknowledges_dispute},
        )

    def _validate_hardship_tone(self, output: str, context: CaseContext) -> GuardrailResult:
        """Validate that the output uses empathetic tone for hardship.

        Fail if harsh/demanding phrases appear without any empathetic
        language.  Also fail if no empathetic language is detected at
        all, even without harsh phrases.
        """
        output_lower = output.lower()

        # Harsh/demanding phrases (inappropriate for hardship)
        harsh_phrases = [
            "failure to pay",
            "will be forced",
            "no choice but",
            "legal consequences",
            "must pay immediately",
            "demand",
            "threaten",
        ]

        # Empathetic phrases (appropriate for hardship)
        empathetic_phrases = [
            "understand",
            "difficult",
            "challenging",
            "work with you",
            "payment plan",
            "options",
            "help",
            "support",
            "flexibility",
            "circumstances",
        ]

        found_harsh = [phrase for phrase in harsh_phrases if phrase in output_lower]
        found_empathetic = [phrase for phrase in empathetic_phrases if phrase in output_lower]

        # Fail if harsh without empathy
        if found_harsh and not found_empathetic:
            return self._fail(
                message="Output uses harsh tone for hardship case",
                expected="Empathetic language, payment options",
                found=found_harsh,
                details={
                    "harsh_phrases_found": found_harsh,
                    "empathetic_phrases_found": found_empathetic,
                    "hardship_indicated": True,
                },
            )

        # Warn if no empathetic language at all
        if not found_empathetic:
            return self._fail(
                message="Output lacks empathetic language for hardship case",
                expected="Understanding tone, payment options",
                found="No empathetic phrases detected",
                details={"hardship_indicated": True, "empathetic_count": 0},
            )

        return self._pass(
            message="Output uses appropriate tone for hardship case",
            details={"empathetic_phrases": found_empathetic},
        )

    def _validate_promise_awareness(self, output: str, context: CaseContext) -> GuardrailResult:
        """Validate that the output acknowledges broken promise history.

        Only triggers when ``broken_promises_count >= 2``.  Check for
        history-referencing phrases (e.g., "previous", "again",
        "commitment") in the draft text.
        """
        output_lower = output.lower()

        # If multiple broken promises, output should acknowledge history
        broken_promises_count = (
            context.lane_broken_promises_count
            if getattr(context, "lane_broken_promises_count", None) is not None
            else context.broken_promises_count
        )

        if broken_promises_count >= 2:
            history_phrases = [
                "previous",
                "history",
                "past",
                "again",
                "before",
                "commitment",
                "promise",
                "assured",
            ]

            acknowledges_history = any(phrase in output_lower for phrase in history_phrases)

            if not acknowledges_history:
                # This is a medium severity - just warn
                return self._fail(
                    message=f"Output doesn't reference {broken_promises_count} broken promises",
                    expected="Acknowledgment of payment history",
                    found="No history reference",
                    details={
                        "broken_promises_count": broken_promises_count,
                        "history_acknowledged": False,
                    },
                )

            return self._pass(
                message="Output acknowledges payment history",
                details={
                    "broken_promises_count": broken_promises_count,
                    "history_acknowledged": True,
                },
            )

        return self._pass(
            message="No significant promise history to reference",
            details={"broken_promises_count": broken_promises_count},
        )

    # =========================================================================
    # Sprint C item #11 (2026-04-28): structural cross-checks
    # =========================================================================

    def _validate_invoice_references(self, output: str, obligations: list) -> GuardrailResult:
        """Pin invoice numbers cited in the prose to ``context.obligations``.

        Pre-fix: the AI could hallucinate an invoice reference (e.g.
        cite ``INV-99999`` when only ``INV-001/002/003`` are open) and
        no guardrail caught it. The placeholder guardrail validates
        ``{INVOICE_TABLE}`` integrity but doesn't scan free-text prose
        for invoice patterns.

        Post-fix: extract all ``INV-...`` / ``Invoice #...`` patterns
        from the output and check each against the structured
        obligations list. Anything not found is flagged.

        ``severity=LOW`` so this never blocks generation; it surfaces
        in ``guardrail_validation`` for review.
        """
        cited_refs = self._extract_invoice_refs(output)
        if not cited_refs:
            return self._pass(
                message="No invoice references in prose to validate",
                details={"prose_invoice_refs": []},
            )

        known_refs = {
            self._normalise_invoice_ref(getattr(ob, "invoice_number", "") or "")
            for ob in obligations
            if getattr(ob, "invoice_number", None)
        }
        known_refs.discard("")

        unknown = [ref for ref in cited_refs if self._normalise_invoice_ref(ref) not in known_refs]
        if unknown:
            return self._fail(
                message=f"AI cited {len(unknown)} invoice reference(s) not found in context.obligations",
                expected="All invoice references in prose must match context.obligations[*].invoice_number",
                found=unknown,
                details={
                    "hallucinated_refs": unknown,
                    "known_refs": sorted(known_refs),
                },
            )

        return self._pass(
            message=f"All {len(cited_refs)} invoice reference(s) in prose match context.obligations",
            details={"validated_refs": sorted(cited_refs)},
        )

    def _validate_no_paid_invoice_chase(self, output: str, obligations: list) -> GuardrailResult:
        """Fail if the prose demands payment for a non-collectible
        invoice (paid / credited / written_off in
        ``obligation.collection_status``).

        Pre-fix: nothing checked this — the AI could be told the lane
        had INV-001 (paid) and INV-002 (open) and produce prose that
        demanded payment for both. Operators only catch this in review.

        Post-fix: any obligation with non-collectible status whose
        invoice number appears in the prose AND co-occurs with demand
        language fails the guardrail.
        """
        cited_refs = self._extract_invoice_refs(output)
        if not cited_refs:
            return self._pass(
                message="No invoice references in prose; nothing to cross-check",
                details={},
            )

        non_collectible_by_ref = {}
        for ob in obligations:
            inv_num = getattr(ob, "invoice_number", None)
            status = getattr(ob, "collection_status", None) or ""
            if not inv_num:
                continue
            if status.lower() in _NON_COLLECTIBLE_STATUSES:
                non_collectible_by_ref[self._normalise_invoice_ref(inv_num)] = status.lower()

        if not non_collectible_by_ref:
            return self._pass(
                message="No paid / credited / written-off obligations to cross-check",
                details={},
            )

        cited_normalised = {self._normalise_invoice_ref(ref) for ref in cited_refs}
        offending = {
            ref: status for ref, status in non_collectible_by_ref.items() if ref in cited_normalised
        }
        if not offending:
            return self._pass(
                message="No prose references to non-collectible invoices",
                details={"non_collectible_count": len(non_collectible_by_ref)},
            )

        # Demand-language check — only fail if the prose ALSO demands
        # payment. Pure references to a paid invoice (e.g. "thank you
        # for paying INV-001") are fine.
        demand_phrases = (
            "pay",
            "payment",
            "settle",
            "outstanding",
            "owed",
            "balance",
            "due",
        )
        output_lower = output.lower()
        if not any(phrase in output_lower for phrase in demand_phrases):
            return self._pass(
                message="Prose references non-collectible invoices but uses no demand language",
                details={"referenced_non_collectible": offending},
            )

        return self._fail(
            message=f"AI demands payment for {len(offending)} non-collectible invoice(s)",
            expected="Do not chase paid / credited / written-off invoices",
            found=offending,
            details={"offending_refs": offending},
        )

    @staticmethod
    def _extract_invoice_refs(output: str) -> list[str]:
        """Extract invoice references from prose. Order-preserving,
        deduplicated.

        Handles the "double prefix" form ("Your invoice INV-12345") by
        detecting when the captured body itself starts with ``INV`` —
        in that case we return only the inner ``INV-12345``, not the
        outer ``invoice INV-12345``, so it normalises to the same key
        as ``ObligationInfo.invoice_number``.
        """
        seen: set[str] = set()
        out: list[str] = []
        for match in _INVOICE_REF_PATTERN.finditer(output or ""):
            body = match.group(1).strip()
            if body.upper().startswith("INV"):
                # Double-prefix form — body IS a self-contained ref.
                full_ref = body
            else:
                full_ref = match.group(0).strip()
            key = ContextualCoherenceGuardrail._normalise_invoice_ref(full_ref)
            if key and key not in seen:
                seen.add(key)
                out.append(full_ref)
        return out

    @staticmethod
    def _normalise_invoice_ref(ref: str) -> str:
        """Normalise to lower-case alphanumeric for comparison, with
        the leading invoice prefix stripped so prose forms ("Invoice
        #98765") and context forms ("98765" / "INV-98765") collapse
        to the same key.

        Examples:
            "INV-12345" → "inv12345" → "12345"
            "Invoice #98765" → "invoice98765" → "98765"
            "98765" → "98765" → "98765"
        """
        normalised = re.sub(r"[^a-z0-9]", "", (ref or "").lower())
        if normalised.startswith("invoice"):
            normalised = normalised[len("invoice") :]
        elif normalised.startswith("inv"):
            normalised = normalised[len("inv") :]
        return normalised

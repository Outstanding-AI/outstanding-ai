"""Manual-communication context-and-prompt regression tests.

Covers four behaviors specific to PR-4 (manual touchpoints feeding the AI):

1. The generator prompt grows a ``Recent Manual Touchpoints`` section when
   ``recent_touches`` carries any ``touch_type='manual_log'`` row, and the
   operator notes are rendered verbatim (so the LLM can quote a verbal
   payment commitment in the next draft).
2. Email-only touch contexts (the existing happy path) never emit that
   section — there is no behavior change for tenants that have not enabled
   the manual-comm feature flag.
3. The ``FactualGroundingGuardrail`` accepts amounts that appear in
   ``manual_notes`` so the AI quoting "£500 by Friday" from a phone log
   does not trigger a hallucination flag.
4. The redaction-split invariant: filtering out redacted manual rows
   happens at the *backend* AI-context query layer, but the AI engine's
   contract on the data it does see is "if it's a manual_log touch, render
   it". A test here pins that contract: redacted-shaped rows (notes=None,
   manual_status='redacted') should never appear in ``recent_touches``
   when produced by the backend's ``_get_recent_touches`` — and the
   engine MUST surface manual rows that DO arrive (we don't second-guess
   the backend filter).
"""

from datetime import datetime, timezone

from src.api.models.requests import (
    CaseContext,
    GenerateDraftRequest,
    ObligationInfo,
    PartyInfo,
)
from src.api.models.requests.context import TouchHistory
from src.engine.generator_prompts import build_extra_sections
from src.guardrails.factual_grounding import FactualGroundingGuardrail


def _make_party() -> PartyInfo:
    return PartyInfo(
        party_id="party-uuid-1",
        external_id="party-ext-1",
        provider_type="sage_200",
        customer_code="C001",
        name="Acme Ltd",
        source="sage_200",
    )


def _make_obligation(
    obligation_id: str = "obl-1",
    invoice_number: str = "INV-100",
    amount_due: float = 500.0,
) -> ObligationInfo:
    return ObligationInfo(
        id=obligation_id,
        external_id=f"{obligation_id}-ext",
        provider_type="sage_200",
        invoice_number=invoice_number,
        original_amount=amount_due,
        amount_due=amount_due,
        currency="GBP",
    )


def _make_request(recent_touches: list[TouchHistory]) -> GenerateDraftRequest:
    context = CaseContext(
        schema_version=2,
        party=_make_party(),
        obligations=[_make_obligation()],
        recent_touches=recent_touches,
        case_state="ACTIVE",
        brand_tone="professional",
        touch_cap=10,
        touch_interval_days=3,
    )
    return GenerateDraftRequest(
        context=context,
        tone="professional",
        objective="follow_up",
    )


def _phone_touch(notes: str, *, sent_at: datetime | None = None) -> TouchHistory:
    return TouchHistory(
        sent_at=sent_at or datetime(2026, 5, 15, 10, 30, tzinfo=timezone.utc),
        touch_type="manual_log",
        channel="phone",
        direction="outbound",
        manual_notes=notes,
        logged_by_user_name="Sarah Operator",
    )


def _email_touch() -> TouchHistory:
    return TouchHistory(
        sent_at=datetime(2026, 5, 14, 9, 0, tzinfo=timezone.utc),
        touch_type="ai_email",
        tone="firm",
        sender_level=2,
        sender_name="Operator A",
    )


class TestManualTouchpointsPromptSection:
    def test_section_appears_when_manual_log_touch_present(self):
        request = _make_request(
            recent_touches=[
                _phone_touch("Spoke to John; he promised payment by 2026-05-20 for £500."),
            ],
        )
        sections = build_extra_sections(request, None)
        joined = "".join(sections)
        assert "**Recent Manual Touchpoints:**" in joined, (
            "Expected the Recent Manual Touchpoints section to render when "
            "a manual_log touch is present; without it the AI cannot "
            "acknowledge operator phone calls in follow-up drafts."
        )
        # The operator notes are rendered verbatim (key requirement for
        # quoting verbal commitments back to the debtor).
        assert "promised payment by 2026-05-20" in joined
        assert "logged by Sarah Operator" in joined.lower() or "Sarah Operator" in joined

    def test_section_omitted_when_only_email_touches(self):
        request = _make_request(recent_touches=[_email_touch()])
        sections = build_extra_sections(request, None)
        joined = "".join(sections)
        assert "**Recent Manual Touchpoints:**" not in joined, (
            "Email-only contexts must not grow a manual-touchpoints section "
            "— tenants without the feature flag should see no behavior change."
        )

    def test_section_omitted_when_no_touches(self):
        request = _make_request(recent_touches=[])
        sections = build_extra_sections(request, None)
        joined = "".join(sections)
        assert "**Recent Manual Touchpoints:**" not in joined

    def test_section_renders_oldest_first(self):
        """Chronological order so the most recent touch is the last thing
        the model reads before the directive (priming for relevance)."""
        request = _make_request(
            recent_touches=[
                _phone_touch(
                    "MOST RECENT call",
                    sent_at=datetime(2026, 5, 16, 10, 0, tzinfo=timezone.utc),
                ),
                _phone_touch(
                    "OLDER call",
                    sent_at=datetime(2026, 5, 10, 10, 0, tzinfo=timezone.utc),
                ),
            ],
        )
        sections = build_extra_sections(request, None)
        joined = "".join(sections)
        older_pos = joined.find("OLDER call")
        recent_pos = joined.find("MOST RECENT call")
        assert older_pos != -1 and recent_pos != -1
        assert older_pos < recent_pos, (
            "Manual touchpoints must render oldest-first so the most recent "
            "is closest to the directive — got recent before older."
        )


class TestFactualGroundingAcceptsManualNotes:
    def test_amount_in_manual_notes_passes_grounding(self):
        """If the AI quotes "£500 by Friday" from a phone-log note, the
        factual grounding guard must accept that amount rather than flag
        it as hallucination. Without this, follow-up drafts referencing
        verbal commitments would fail the guardrail pipeline."""
        guard = FactualGroundingGuardrail()
        # No obligations matching the £500 amount — only the manual touch
        # supplies it. This isolates the new code path.
        context = CaseContext(
            schema_version=2,
            party=_make_party(),
            obligations=[
                _make_obligation(obligation_id="obl-2", invoice_number="INV-200", amount_due=999.0),
            ],
            recent_touches=[
                _phone_touch("You promised £500 by Friday during yesterday's call."),
            ],
            case_state="ACTIVE",
        )
        output = "Thanks for your call — as discussed, please confirm the £500 payment."
        result = guard._validate_amounts(  # noqa: SLF001 — exercise the inner check directly
            output, context, skip_invoice_table=True
        )
        assert result.passed, (
            f"Expected grounding to accept £500 from manual_notes, got: {result.message!r}"
        )

    def test_grounding_still_rejects_unsourced_amount(self):
        """Sanity: the new code path must not over-broaden the validity
        set. An amount that appears in neither obligations nor any touch
        notes still fails."""
        guard = FactualGroundingGuardrail()
        context = CaseContext(
            schema_version=2,
            party=_make_party(),
            obligations=[
                _make_obligation(obligation_id="obl-3", invoice_number="INV-300", amount_due=100.0),
            ],
            recent_touches=[_phone_touch("Talked briefly, no figures mentioned.")],
            case_state="ACTIVE",
        )
        output = "Please remit the outstanding £7,432.18 today."
        result = guard._validate_amounts(  # noqa: SLF001
            output, context, skip_invoice_table=True
        )
        assert not result.passed, (
            "Amount £7,432.18 is in neither obligations nor manual notes — "
            "grounding must still flag it as ungrounded."
        )


class TestRedactionSplitInvariant:
    """The AI engine MUST surface manual-log rows it receives but never
    second-guesses the backend's redaction filter.

    The contract: the backend's AI-context SQL drops rows where
    ``manual_status = 'redacted'``. So if the engine sees a manual row in
    ``recent_touches``, it can assume the operator considers it visible
    to the AI. Conversely, the engine must NOT add its own redaction
    filter — that would silently hide rows whose redaction status the
    backend has authoritative knowledge of and we don't.
    """

    def test_manual_row_with_notes_renders_in_prompt(self):
        """Active manual rows (not redacted) MUST show up in the prompt
        section, including their notes verbatim."""
        request = _make_request(
            recent_touches=[
                _phone_touch("Operator phone-log; £500 commitment by 2026-05-20."),
            ],
        )
        sections = build_extra_sections(request, None)
        joined = "".join(sections)
        assert "**Recent Manual Touchpoints:**" in joined
        assert "£500 commitment" in joined

    def test_engine_does_not_filter_by_manual_status_field(self):
        """If the backend somehow lets a redacted row through (drift /
        bug), the engine still passes it through to the prompt — the
        engine is not the gate. The backend integration test
        ``test_redaction_split_invariant`` (PR-2b verification) pins the
        complementary half: ``_current`` view + AI-context SQL drop
        redacted rows before they reach the engine.
        """
        # Construct a TouchHistory with manual_log + notes but no
        # explicit redaction filter — model the case where the backend
        # accidentally let it through.
        touch = TouchHistory(
            sent_at=datetime(2026, 5, 15, 10, 30, tzinfo=timezone.utc),
            touch_type="manual_log",
            channel="phone",
            direction="outbound",
            manual_notes="leaked-through note",
            logged_by_user_name="Op",
        )
        request = _make_request(recent_touches=[touch])
        sections = build_extra_sections(request, None)
        joined = "".join(sections)
        # The engine surfaces it — confirming there is no defense-in-depth
        # filter inside generator_prompts. The redaction gate lives on
        # the backend, and the cross-cutting backend test (PR-2b §verify)
        # is what proves redacted rows never arrive here.
        assert "leaked-through note" in joined, (
            "Engine must surface every manual_log row it receives. "
            "If this assertion ever fails because we added a "
            "manual_status check inside generator_prompts, ROLL BACK — "
            "the backend is the source of truth for redaction."
        )

"""Unit tests for GateEvaluator (deterministic rule-based logic).

16 scenarios covering all 6 gates:
- Touch Cap (1 parametrized test, 4 cases)
- Cooling Off (1 parametrized test, 5 cases)
- Dispute/Hardship/Unsubscribe (1 parametrized test, 6 cases)
- Escalation Appropriate (9 tests)
- Combined Scenarios (3 tests)
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.api.models.responses import EvaluateGatesResponse
from src.engine.gate_evaluator import GateEvaluator


class TestGateEvaluator:
    """Tests for GateEvaluator deterministic evaluation."""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator instance."""
        return GateEvaluator()

    # =========================================================================
    # Touch Cap Gate (parametrized)
    # =========================================================================

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "count,cap,expected_passed,expected_allowed",
        [
            (0, 10, True, None),
            (9, 10, True, None),
            (10, 10, False, False),
            (15, 10, False, False),
        ],
        ids=["zero_count", "just_under", "at_limit", "over_limit"],
    )
    async def test_touch_cap(
        self,
        evaluator,
        sample_evaluate_gates_request,
        count,
        cap,
        expected_passed,
        expected_allowed,
    ):
        """Test touch cap gate with various count/cap combinations."""
        sample_evaluate_gates_request.context.monthly_touch_count = count
        sample_evaluate_gates_request.context.touch_cap = cap

        result = await evaluator.evaluate(sample_evaluate_gates_request)

        assert result.gate_results["touch_cap"].passed is expected_passed
        if expected_allowed is not None:
            assert result.allowed is expected_allowed

    # =========================================================================
    # Cooling Off Gate (parametrized)
    # =========================================================================

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "last_touch_days,interval,dnc_offset_days,expected_passed",
        [
            (None, 3, None, True),  # no last touch -> first contact
            (5, 3, None, True),  # 5 days > 3 interval
            (1, 3, None, False),  # 1 day < 3 interval
            (5, 3, 7, False),  # DNC in future
            (5, 3, -7, True),  # DNC in past
        ],
        ids=["no_last_touch", "sufficient_gap", "insufficient_gap", "dnc_future", "dnc_past"],
    )
    async def test_cooling_off(
        self,
        evaluator,
        sample_evaluate_gates_request,
        last_touch_days,
        interval,
        dnc_offset_days,
        expected_passed,
    ):
        """Test cooling off gate with various timing scenarios."""
        if last_touch_days is not None:
            sample_evaluate_gates_request.context.communication.last_touch_at = datetime.now(
                timezone.utc
            ) - timedelta(days=last_touch_days)
        else:
            sample_evaluate_gates_request.context.communication.last_touch_at = None

        sample_evaluate_gates_request.context.touch_interval_days = interval

        if dnc_offset_days is not None:
            dnc_date = (datetime.now(timezone.utc) + timedelta(days=dnc_offset_days)).strftime(
                "%Y-%m-%d"
            )
            sample_evaluate_gates_request.context.do_not_contact_until = dnc_date
        else:
            sample_evaluate_gates_request.context.do_not_contact_until = None

        result = await evaluator.evaluate(sample_evaluate_gates_request)

        assert result.gate_results["cooling_off"].passed is expected_passed

    # =========================================================================
    # Dispute / Hardship / Unsubscribe Gates (parametrized)
    # =========================================================================

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "gate_name,attr,value,expected_passed,check_reason",
        [
            ("dispute_active", "active_dispute", False, True, None),
            ("dispute_active", "active_dispute", True, False, None),
            ("hardship", "hardship_indicated", False, True, None),
            ("hardship", "hardship_indicated", True, True, "sensitive tone"),
            ("unsubscribe", "unsubscribe_requested", False, True, None),
            ("unsubscribe", "unsubscribe_requested", True, False, None),
        ],
        ids=[
            "dispute_inactive",
            "dispute_active",
            "hardship_not_indicated",
            "hardship_indicated_warning",
            "unsubscribe_not_requested",
            "unsubscribe_requested",
        ],
    )
    async def test_binary_state_gates(
        self,
        evaluator,
        sample_evaluate_gates_request,
        gate_name,
        attr,
        value,
        expected_passed,
        check_reason,
    ):
        """Test binary state gates (dispute, hardship, unsubscribe)."""
        setattr(sample_evaluate_gates_request.context, attr, value)

        result = await evaluator.evaluate(sample_evaluate_gates_request)

        assert result.gate_results[gate_name].passed is expected_passed
        if not expected_passed and gate_name != "hardship":
            assert result.allowed is False
        if check_reason:
            assert check_reason in result.gate_results[gate_name].reason.lower()

    # =========================================================================
    # Escalation Gate (9 tests — kept individual due to complex setup)
    # =========================================================================

    @pytest.mark.asyncio
    async def test_escalation_first_contact_friendly(
        self, evaluator, sample_evaluate_gates_request
    ):
        """Test first contact with friendly_reminder passes."""
        sample_evaluate_gates_request.context.communication.touch_count = 0
        sample_evaluate_gates_request.context.communication.last_tone_used = None
        sample_evaluate_gates_request.proposed_tone = "friendly_reminder"

        result = await evaluator.evaluate(sample_evaluate_gates_request)

        assert result.gate_results["escalation_appropriate"].passed is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "proposed_tone",
        ["firm", "final_notice"],
        ids=["firm", "final_notice"],
    )
    async def test_escalation_first_contact_aggressive_fails(
        self, evaluator, sample_evaluate_gates_request, proposed_tone
    ):
        """Test first contact with aggressive tone fails."""
        sample_evaluate_gates_request.context.communication.touch_count = 0
        sample_evaluate_gates_request.context.communication.last_tone_used = None
        sample_evaluate_gates_request.proposed_tone = proposed_tone

        result = await evaluator.evaluate(sample_evaluate_gates_request)

        assert result.gate_results["escalation_appropriate"].passed is False

    @pytest.mark.asyncio
    async def test_escalation_single_step_standard(self, evaluator, sample_evaluate_gates_request):
        """Test single-step escalation professional->firm passes (standard industry)."""
        sample_evaluate_gates_request.context.communication.touch_count = 3
        sample_evaluate_gates_request.context.communication.last_tone_used = "professional"
        sample_evaluate_gates_request.proposed_tone = "firm"

        result = await evaluator.evaluate(sample_evaluate_gates_request)

        assert result.gate_results["escalation_appropriate"].passed is True

    @pytest.mark.asyncio
    async def test_escalation_double_step_standard_no_broken_promises_fails(
        self, evaluator, sample_evaluate_gates_request
    ):
        """Test double-step escalation professional->firm fails (standard, no broken promises).

        Tone order: friendly(0), professional(1), concerned(2), firm(3), final(4).
        professional(1)->firm(3) = jump of 2. Standard allows max 1 (no broken promises).
        """
        sample_evaluate_gates_request.context.communication.touch_count = 3
        sample_evaluate_gates_request.context.communication.last_tone_used = "professional"
        sample_evaluate_gates_request.context.broken_promises_count = 0
        sample_evaluate_gates_request.context.industry = None
        sample_evaluate_gates_request.proposed_tone = "firm"

        result = await evaluator.evaluate(sample_evaluate_gates_request)

        assert result.gate_results["escalation_appropriate"].passed is False

    @pytest.mark.asyncio
    async def test_escalation_double_step_aggressive_industry(
        self, evaluator, sample_evaluate_gates_request
    ):
        """Test double-step escalation passes with aggressive industry (professional->firm, jump=2).

        Tone order: friendly(0), professional(1), concerned(2), firm(3), final(4).
        professional(1)->firm(3) = jump of 2. Aggressive allows max 2.
        """
        from src.api.models.requests import IndustryInfo

        sample_evaluate_gates_request.context.communication.touch_count = 3
        sample_evaluate_gates_request.context.communication.last_tone_used = "professional"
        sample_evaluate_gates_request.context.industry = IndustryInfo(
            code="retail",
            name="Retail",
            typical_dso_days=30,
            alarm_dso_days=45,
            payment_cycle="net30",
            escalation_patience="aggressive",
        )
        sample_evaluate_gates_request.proposed_tone = "firm"

        result = await evaluator.evaluate(sample_evaluate_gates_request)

        assert result.gate_results["escalation_appropriate"].passed is True

    @pytest.mark.asyncio
    async def test_escalation_single_step_patient_industry(
        self, evaluator, sample_evaluate_gates_request
    ):
        """Test single-step escalation firm->final_notice passes with patient industry."""
        from src.api.models.requests import IndustryInfo

        sample_evaluate_gates_request.context.communication.touch_count = 5
        sample_evaluate_gates_request.context.communication.last_tone_used = "firm"
        sample_evaluate_gates_request.context.industry = IndustryInfo(
            code="manufacturing",
            name="Manufacturing",
            typical_dso_days=60,
            alarm_dso_days=90,
            payment_cycle="net60",
            escalation_patience="patient",
        )
        sample_evaluate_gates_request.proposed_tone = "final_notice"

        result = await evaluator.evaluate(sample_evaluate_gates_request)

        assert result.gate_results["escalation_appropriate"].passed is True

    @pytest.mark.asyncio
    async def test_escalation_de_escalation_allowed(self, evaluator, sample_evaluate_gates_request):
        """Test de-escalation firm->friendly_reminder always passes."""
        sample_evaluate_gates_request.context.communication.touch_count = 3
        sample_evaluate_gates_request.context.communication.last_tone_used = "firm"
        sample_evaluate_gates_request.proposed_tone = "friendly_reminder"

        result = await evaluator.evaluate(sample_evaluate_gates_request)

        assert result.gate_results["escalation_appropriate"].passed is True

    @pytest.mark.asyncio
    async def test_escalation_double_step_broken_promises_standard(
        self, evaluator, sample_evaluate_gates_request
    ):
        """Test double-step escalation passes with broken promises (professional->firm, jump=2).

        Tone order: friendly(0), professional(1), concerned(2), firm(3), final(4).
        professional(1)->firm(3) = jump of 2. Standard allows 2 with broken promises.
        """
        sample_evaluate_gates_request.context.communication.touch_count = 3
        sample_evaluate_gates_request.context.communication.last_tone_used = "professional"
        sample_evaluate_gates_request.context.broken_promises_count = 2
        sample_evaluate_gates_request.context.industry = None
        sample_evaluate_gates_request.proposed_tone = "firm"

        result = await evaluator.evaluate(sample_evaluate_gates_request)

        assert result.gate_results["escalation_appropriate"].passed is True

    # =========================================================================
    # Combined Scenarios (3 tests)
    # =========================================================================

    @pytest.mark.asyncio
    async def test_combined_all_pass(self, evaluator, sample_evaluate_gates_request):
        """Test all gates pass -> allowed=True."""
        sample_evaluate_gates_request.context.monthly_touch_count = 0
        sample_evaluate_gates_request.context.touch_cap = 10
        sample_evaluate_gates_request.context.active_dispute = False
        sample_evaluate_gates_request.context.hardship_indicated = False
        sample_evaluate_gates_request.context.unsubscribe_requested = False
        sample_evaluate_gates_request.context.do_not_contact_until = None
        sample_evaluate_gates_request.context.communication.touch_count = 3
        sample_evaluate_gates_request.context.communication.last_tone_used = "friendly_reminder"
        sample_evaluate_gates_request.proposed_tone = "professional"

        result = await evaluator.evaluate(sample_evaluate_gates_request)

        assert isinstance(result, EvaluateGatesResponse)
        assert result.allowed is True
        assert all(g.passed for g in result.gate_results.values())

    @pytest.mark.asyncio
    async def test_combined_multiple_failures(self, evaluator, sample_evaluate_gates_request):
        """Test multiple gate failures -> allowed=False with all failures in results."""
        sample_evaluate_gates_request.context.monthly_touch_count = 10
        sample_evaluate_gates_request.context.touch_cap = 10
        sample_evaluate_gates_request.context.active_dispute = True
        sample_evaluate_gates_request.context.unsubscribe_requested = True

        result = await evaluator.evaluate(sample_evaluate_gates_request)

        assert isinstance(result, EvaluateGatesResponse)
        assert result.allowed is False
        assert result.gate_results["touch_cap"].passed is False
        assert result.gate_results["dispute_active"].passed is False
        assert result.gate_results["unsubscribe"].passed is False

    @pytest.mark.asyncio
    async def test_combined_hardship_warning_only(self, evaluator, sample_evaluate_gates_request):
        """Test only hardship warning -> allowed=True (hardship doesn't block)."""
        sample_evaluate_gates_request.context.monthly_touch_count = 0
        sample_evaluate_gates_request.context.touch_cap = 10
        sample_evaluate_gates_request.context.active_dispute = False
        sample_evaluate_gates_request.context.hardship_indicated = True
        sample_evaluate_gates_request.context.unsubscribe_requested = False
        sample_evaluate_gates_request.context.do_not_contact_until = None
        sample_evaluate_gates_request.context.communication.touch_count = 3
        sample_evaluate_gates_request.context.communication.last_tone_used = "friendly_reminder"
        sample_evaluate_gates_request.proposed_tone = "professional"

        result = await evaluator.evaluate(sample_evaluate_gates_request)

        assert isinstance(result, EvaluateGatesResponse)
        assert result.allowed is True
        assert result.gate_results["hardship"].passed is True

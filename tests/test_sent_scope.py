from src.engine.sent_scope import SentDraftScopeAnalyzer, _LLMScopeDecision


class _Candidate:
    def __init__(self, invoice_number: str, obligation_id: str):
        self.invoice_number = invoice_number
        self.obligation_id = obligation_id


def test_sent_scope_validation_reclassifies_operator_added_and_retained():
    analyzer = SentDraftScopeAnalyzer()
    decisions = analyzer._validated_decisions(
        [
            _LLMScopeDecision(
                invoice_number="inv-1001",
                obligation_id="obl-1",
                status="operator_added_invoice",
                confidence=0.91,
            ),
            _LLMScopeDecision(
                invoice_number="INV-2002",
                obligation_id="obl-2",
                status="retained_generated_invoice",
                confidence=0.88,
            ),
        ],
        {
            "INV-1001": _Candidate("INV-1001", "obl-1"),
            "INV-2002": _Candidate("INV-2002", "obl-2"),
        },
        {"INV-1001"},
    )

    by_invoice = {decision.invoice_number: decision for decision in decisions}
    assert by_invoice["INV-1001"].status == "retained_generated_invoice"
    assert by_invoice["INV-2002"].status == "operator_added_invoice"


def test_sent_scope_validation_rejects_hallucinated_invoice_refs():
    analyzer = SentDraftScopeAnalyzer()
    decisions = analyzer._validated_decisions(
        [
            _LLMScopeDecision(
                invoice_number="INV-9999",
                obligation_id="obl-9999",
                status="retained_generated_invoice",
                confidence=0.99,
            )
        ],
        {"INV-1001": _Candidate("INV-1001", "obl-1")},
        {"INV-1001"},
    )

    assert len(decisions) == 1
    assert decisions[0].invoice_number == "INV-1001"
    assert decisions[0].status == "not_present"

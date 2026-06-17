from src.guardrails.promise_reply import PromiseReplyGuardrail


def _promise_context(sample_case_context):
    sample_case_context.recent_messages = [
        {
            "direction": "inbound",
            "classification": "PROMISE_TO_PAY",
            "body_snippet": "Invoice approved and funds will be issued on July 7 th.",
            "promise_date": "2026-07-07",
            "invoice_refs": ["0000007324"],
        }
    ]
    sample_case_context.lane_mail_mode = "reply_ack"
    return sample_case_context


def test_blocks_reply_that_asks_for_date_when_promise_date_is_known(sample_case_context):
    guardrail = PromiseReplyGuardrail()
    context = _promise_context(sample_case_context)

    results = guardrail.validate(
        (
            "Hello Subsea7 GoM Accounts Payable,\n\n"
            "Thanks for your recent reply. Could you please provide an update "
            "on the status of this payment from your end?"
        ),
        context,
        trigger_classification="PROMISE_TO_PAY",
        mail_mode="reply_ack",
    )

    assert any(not result.passed and result.should_block for result in results)
    assert any("already provided" in result.message for result in results)


def test_blocks_reply_that_omits_known_promise_date(sample_case_context):
    guardrail = PromiseReplyGuardrail()
    context = _promise_context(sample_case_context)

    results = guardrail.validate(
        (
            "Hello Subsea7 GoM Accounts Payable,\n\n"
            "Thanks for your reply. We will look out for the payment and reconcile "
            "it once received."
        ),
        context,
        trigger_classification="PROMISE_TO_PAY",
        mail_mode="reply_ack",
    )

    assert any(not result.passed and result.should_block for result in results)
    assert any("does not mention" in result.message for result in results)


def test_allows_reply_acknowledging_known_promise_date(sample_case_context):
    guardrail = PromiseReplyGuardrail()
    context = _promise_context(sample_case_context)

    results = guardrail.validate(
        (
            "Hello Subsea7 GoM Accounts Payable,\n\n"
            "Thanks for confirming that funds will be issued on July 7. "
            "We will look out for the payment and reconcile it once received."
        ),
        context,
        trigger_classification="PROMISE_TO_PAY",
        mail_mode="reply_ack",
    )

    assert all(result.passed for result in results)

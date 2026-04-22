"""
Guardrail retry feedback generation.

Builds context-aware retry prompts from guardrail failures so the LLM
can self-correct on the next draft generation attempt.
"""

from .base import GuardrailPipelineResult


def get_retry_prompt_addition(pipeline_result: GuardrailPipelineResult, **kwargs) -> str:
    """Generate context-aware retry prompt based on guardrail failures.

    Build specific remediation instructions for each failed guardrail
    so the LLM can self-correct on the next attempt.  Instructions
    are adapted to the draft type (standard, follow-up, closure).

    Args:
        pipeline_result: Results from the guardrail pipeline.
        **kwargs: Draft context flags -- ``skip_invoice_table`` and
            ``closure_mode``.

    Returns:
        Prompt addition string, or empty string if all passed.
    """
    if pipeline_result.all_passed:
        return ""

    skip_invoice_table = kwargs.get("skip_invoice_table", False)
    closure_mode = kwargs.get("closure_mode", False)

    additions = [
        "\n\n**IMPORTANT VALIDATION REQUIREMENTS:**",
        "The previous response had validation errors. Please ensure:",
    ]

    for result in pipeline_result.results:
        if not result.passed:
            if result.guardrail_name == "placeholder_validation":
                if skip_invoice_table or closure_mode:
                    additions.append(
                        "- Do NOT use {INVOICE_TABLE} — this is a "
                        f"{'closure' if closure_mode else 'follow-up'} email. "
                        "Do NOT invent other placeholders either."
                    )
                else:
                    additions.append(
                        "- Do NOT invent placeholders like [SOMETHING] or {SOMETHING}. "
                        "The ONLY allowed placeholder is {INVOICE_TABLE}. "
                        "Use actual values from the context provided."
                    )
                if result.found:
                    additions.append(f"- Remove these placeholders: {result.found}")
            elif result.guardrail_name == "factual_grounding":
                if closure_mode:
                    additions.append(
                        "- This is a CLOSURE email. Remove all invoice references "
                        "and monetary amounts."
                    )
                elif skip_invoice_table:
                    additions.append(
                        "- This is a follow-up email. Only reference amounts "
                        "the debtor mentioned in conversation. Do NOT use "
                        "{INVOICE_TABLE}."
                    )
                else:
                    additions.append(
                        "- Do NOT write monetary amounts in the email body — use the "
                        "{INVOICE_TABLE} placeholder for all invoice details. "
                        "Remove any amounts you wrote in the prose."
                    )
            elif result.guardrail_name == "numerical_consistency":
                if closure_mode:
                    additions.append(
                        "- This is a CLOSURE email. Remove all totals and numerical references."
                    )
                else:
                    additions.append(
                        "- Do NOT state totals or amounts in prose — the "
                        "{INVOICE_TABLE} handles all figures. "
                        "Remove any stated totals from the email body."
                    )
            elif result.guardrail_name == "identity_scope":
                additions.append(
                    "- Use only the recipient, sender, reply-to, and contact emails provided in context."
                )
            elif result.guardrail_name == "lane_scope":
                additions.append("- Only mention invoices and totals from the current lane cohort.")
            elif result.guardrail_name == "policy_grounding":
                additions.append(
                    "- Remove any discount, settlement, statutory-interest, or legal-escalation language unless authorized."
                )

    return "\n".join(additions)

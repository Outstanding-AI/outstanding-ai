"""
Draft generation engine.

Generates collection email drafts with 5 tones based on ai_logic.md:
friendly_reminder, professional, firm, final_notice, concerned_inquiry

Includes guardrail retry mechanism: if guardrails fail, the generator
will retry with feedback about what went wrong, giving the LLM a chance
to correct its output.
"""

import json
import logging
import time

from pydantic import ValidationError

from src.api.errors import LLMResponseInvalidError
from src.api.models.requests import GenerateDraftRequest
from src.api.models.responses import GenerateDraftResponse, GuardrailValidation
from src.guardrails.base import GuardrailPipelineResult, GuardrailSeverity
from src.guardrails.pipeline import guardrail_pipeline
from src.llm.factory import llm_client
from src.llm.schemas import DraftGenerationLLMResponse
from src.prompts import GENERATE_DRAFT_SYSTEM, GENERATE_DRAFT_USER

logger = logging.getLogger(__name__)

# Maximum retries when guardrails fail
MAX_GUARDRAIL_RETRIES = 2


class DraftGenerator:
    """Generates collection email drafts with guardrail retry mechanism."""

    async def generate(self, request: GenerateDraftRequest) -> GenerateDraftResponse:
        """
        Generate a collection email draft with automatic retry on guardrail failures.

        Args:
            request: Generation request with context and parameters

        Returns:
            Generated draft with subject, body, and guardrail validation

        The generator will retry up to MAX_GUARDRAIL_RETRIES times if guardrails
        fail, passing the failure reasons back to the LLM to help it correct
        the output.
        """
        # Calculate derived values
        total_outstanding = sum(o.amount_due for o in request.context.obligations)

        # Build invoices list (top 10 by days overdue)
        sorted_obligations = sorted(
            request.context.obligations, key=lambda o: o.days_past_due, reverse=True
        )[:10]

        invoices_list = (
            "\n".join(
                [
                    f"- {o.invoice_number}: {request.context.party.currency} {o.amount_due:,.2f} "
                    f"({o.days_past_due} days overdue)"
                    for o in sorted_obligations
                ]
            )
            if sorted_obligations
            else "No specific invoices provided"
        )

        # Get communication info
        comm = request.context.communication

        # Calculate days since last touch
        days_since_last_touch = request.context.days_in_state or 0
        if comm and comm.last_touch_at:
            from datetime import datetime, timezone

            delta = datetime.now(timezone.utc) - comm.last_touch_at
            days_since_last_touch = delta.days

        # Get behavior info
        behavior = request.context.behavior

        # Build industry context section
        industry_context = self._format_industry_context(request.context.industry)

        # Build sender persona context section
        sender_persona_context = self._format_sender_persona(request)

        # Build base user prompt
        base_user_prompt = GENERATE_DRAFT_USER.format(
            party_name=request.context.party.name,
            customer_code=request.context.party.customer_code,
            currency=request.context.party.currency,
            total_outstanding=total_outstanding,
            relationship_tier=request.context.relationship_tier,
            is_verified=request.context.party.is_verified,
            invoices_list=invoices_list,
            monthly_touch_count=request.context.monthly_touch_count,
            touch_count=comm.touch_count if comm else 0,
            last_touch_at=comm.last_touch_at.strftime("%Y-%m-%d")
            if comm and comm.last_touch_at
            else "Never",
            last_tone_used=comm.last_tone_used if comm else "None",
            last_response_type=comm.last_response_type if comm else "No response",
            case_state=request.context.case_state or "ACTIVE",
            days_since_last_touch=days_since_last_touch,
            broken_promises_count=request.context.broken_promises_count,
            active_dispute=request.context.active_dispute,
            hardship_indicated=request.context.hardship_indicated,
            segment=behavior.segment if behavior else "standard",
            on_time_rate=f"{behavior.on_time_rate:.0%}"
            if behavior and behavior.on_time_rate
            else "Unknown",
            avg_days_to_pay=behavior.avg_days_to_pay if behavior else "Unknown",
            industry_context=industry_context,
            sender_persona_context=sender_persona_context,
            tone=request.tone,
            objective=request.objective or "collect payment",
            brand_tone=request.context.brand_tone,
            custom_instructions=f"\nAdditional: {request.custom_instructions}"
            if request.custom_instructions
            else "",
        )

        # Retry loop for guardrail failures
        guardrail_feedback = None
        total_tokens_used = 0
        result = None
        guardrail_result = None
        generation_start_time = time.perf_counter()
        llm_latencies = []
        guardrail_latencies = []

        for attempt in range(MAX_GUARDRAIL_RETRIES + 1):
            # Build prompt with any guardrail feedback from previous attempt
            user_prompt = base_user_prompt
            if guardrail_feedback:
                user_prompt += guardrail_feedback
                logger.info(
                    f"Retrying draft generation (attempt {attempt + 1}) with guardrail feedback"
                )

            # Call LLM with higher temperature for creative generation
            # Use response_schema for guaranteed valid JSON (no markdown wrapping)
            llm_start = time.perf_counter()
            response = await llm_client.complete(
                system_prompt=GENERATE_DRAFT_SYSTEM,
                user_prompt=user_prompt,
                temperature=0.7,
                response_schema=DraftGenerationLLMResponse,
            )
            llm_latencies.append((time.perf_counter() - llm_start) * 1000)

            # Track total tokens across retries
            total_tokens_used += response.usage.get("total_tokens", 0)

            # Parse JSON response - structured output guarantees valid JSON
            raw_result = json.loads(response.content)

            # Validate LLM response using Pydantic schema
            try:
                result = DraftGenerationLLMResponse(**raw_result)
            except ValidationError as e:
                logger.error(f"LLM response validation failed: {e}")
                raise LLMResponseInvalidError(
                    message="LLM returned invalid draft generation response",
                    details={"validation_errors": e.errors(), "raw_response": raw_result},
                )

            # Run guardrails on generated draft body (critical for factual accuracy)
            guardrail_start = time.perf_counter()
            guardrail_result = guardrail_pipeline.validate(
                output=result.body,
                context=request.context,
            )
            guardrail_latencies.append((time.perf_counter() - guardrail_start) * 1000)

            # If all guardrails passed, we're done
            if guardrail_result.all_passed:
                if attempt > 0:
                    logger.info(
                        f"Guardrails passed on retry attempt {attempt + 1} for "
                        f"{request.context.party.customer_code}"
                    )
                break

            # If this was the last attempt, exit loop with failed guardrails
            if attempt >= MAX_GUARDRAIL_RETRIES:
                logger.warning(
                    f"Guardrails still failing after {MAX_GUARDRAIL_RETRIES + 1} attempts for "
                    f"{request.context.party.customer_code}: {guardrail_result.blocking_guardrails}"
                )
                break

            # Build feedback for next attempt
            guardrail_feedback = self._build_guardrail_feedback(guardrail_result)

        # Extract referenced invoices from generated body
        invoices_referenced = [
            o.invoice_number for o in request.context.obligations if o.invoice_number in result.body
        ]

        # Calculate factual accuracy
        total_checks = len(guardrail_result.results)
        passed_checks = sum(1 for r in guardrail_result.results if r.passed)
        factual_accuracy = passed_checks / total_checks if total_checks > 0 else 1.0

        # Separate warnings from blocking failures
        warnings = [
            r.guardrail_name
            for r in guardrail_result.results
            if not r.passed and r.severity in (GuardrailSeverity.MEDIUM, GuardrailSeverity.LOW)
        ]

        guardrail_validation = GuardrailValidation(
            all_passed=guardrail_result.all_passed,
            guardrails_run=total_checks,
            guardrails_passed=passed_checks,
            blocking_failures=guardrail_result.blocking_guardrails,
            warnings=warnings,
            factual_accuracy=factual_accuracy,
        )

        if not guardrail_result.all_passed:
            logger.warning(
                f"Guardrails failed for draft {request.context.party.customer_code}: "
                f"blocking={guardrail_result.blocking_guardrails}, warnings={warnings}"
            )

        # Calculate end-to-end timing
        total_latency_ms = (time.perf_counter() - generation_start_time) * 1000
        retry_count = len(llm_latencies) - 1  # First attempt is not a retry

        logger.info(
            "Draft generation completed",
            extra={
                "metric_type": "draft_generation_completed",
                "customer_code": request.context.party.customer_code,
                "tone": request.tone,
                "latency_ms": round(total_latency_ms, 2),
                "llm_latency_ms": round(sum(llm_latencies), 2),
                "guardrail_latency_ms": round(sum(guardrail_latencies), 2),
                "retry_count": retry_count,
                "total_tokens": total_tokens_used,
                "guardrails_passed": guardrail_result.all_passed,
                "invoices_referenced": len(invoices_referenced),
                "blocking_failures": guardrail_result.blocking_guardrails,
            },
        )

        return GenerateDraftResponse(
            subject=result.subject,
            body=result.body,
            tone_used=request.tone,
            invoices_referenced=invoices_referenced,
            tokens_used=total_tokens_used,
            guardrail_validation=guardrail_validation,
            provider=response.provider,
            model=response.model,
            is_fallback=(response.provider != llm_client.primary_provider_name),
        )

    def _format_sender_persona(self, request: GenerateDraftRequest) -> str:
        """Format sender persona context for prompt inclusion."""
        persona = request.sender_persona
        if not persona or not persona.communication_style:
            # No persona — use name/title if available
            name = request.sender_name or "[SENDER_NAME]"
            title = request.sender_title or "[SENDER_TITLE]"
            return f"Name: {name}, Title: {title} (no persona profile — use neutral professional voice)"

        lines = [
            f"- Name: {persona.name}",
            f"- Title: {persona.title or 'Team Member'}",
            f"- Communication Style: {persona.communication_style}",
            f"- Formality Level: {persona.formality_level}",
            f"- Emphasis: {persona.emphasis}",
        ]
        return "\n".join(lines)

    def _format_industry_context(self, industry) -> str:
        """Format industry context for prompt inclusion."""
        if not industry:
            return "Not specified (general B2B collection)"

        lines = [
            f"- Industry: {industry.name} ({industry.code})",
            f"- Payment Norm: {industry.payment_cycle} (typical DSO: {industry.typical_dso_days} days)",
            f"- Escalation Approach: {industry.escalation_patience}",
            f"- Communication Style: {industry.preferred_tone}",
        ]

        if industry.common_dispute_types:
            lines.append(f"- Common Disputes: {', '.join(industry.common_dispute_types)}")

        if industry.ai_context_notes:
            lines.append(f"- Industry Notes: {industry.ai_context_notes}")

        if industry.seasonal_patterns:
            # Get current quarter
            from datetime import datetime

            quarter = f"Q{(datetime.now().month - 1) // 3 + 1}"
            if quarter in industry.seasonal_patterns:
                lines.append(f"- Current Season ({quarter}): {industry.seasonal_patterns[quarter]}")

        return "\n".join(lines)

    def _build_guardrail_feedback(self, guardrail_result: GuardrailPipelineResult) -> str:
        """
        Build feedback prompt addition from guardrail failures.

        This feedback is appended to the user prompt on retry attempts
        to help the LLM correct its output.
        """
        failures = [r for r in guardrail_result.results if not r.passed]
        if not failures:
            return ""

        feedback_lines = [
            "\n\n**CRITICAL: Your previous draft had validation errors. Fix these issues:**\n"
        ]

        for failure in failures:
            feedback_lines.append(f"- {failure.guardrail_name}: {failure.message}")
            if failure.expected:
                feedback_lines.append(f"  Expected: {failure.expected}")
            if failure.found:
                feedback_lines.append(f"  Found: {failure.found}")

        feedback_lines.append(
            "\nEnsure the new draft addresses ALL validation issues listed above."
        )

        return "\n".join(feedback_lines)


# Singleton instance
generator = DraftGenerator()

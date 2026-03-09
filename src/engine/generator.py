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
        max_days_overdue = max((o.days_past_due for o in request.context.obligations), default=0)
        obligation_count = len(request.context.obligations)

        # Build invoices list (top 10 by days overdue)
        sorted_obligations = sorted(
            request.context.obligations, key=lambda o: o.days_past_due, reverse=True
        )[:10]

        invoices_list = (
            "\n".join(
                [
                    f"- {o.invoice_number or '(no invoice number)'}: "
                    f"{request.context.party.currency} {o.amount_due:,.2f} "
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

            last_touch = comm.last_touch_at
            if last_touch.tzinfo is None:
                last_touch = last_touch.replace(tzinfo=timezone.utc)
            delta = datetime.now(timezone.utc) - last_touch
            days_since_last_touch = delta.days

        # Get behavior info
        behavior = request.context.behavior

        # Build industry context section
        industry_context = self._format_industry_context(request.context.industry)

        # Build sender persona context section
        sender_persona_context = self._format_sender_persona(request)

        # Build extended context sections
        extra_sections = self._build_extra_sections(request, behavior)

        # Determine if this is a follow-up
        has_conversation = bool(request.context.recent_messages)
        has_response = (
            comm
            and comm.last_response_type
            and comm.last_response_type not in ("No response", "None", None)
        )
        is_follow_up = (
            "YES — debtor has responded, see conversation history below"
            if (has_conversation or has_response)
            else "No — first contact"
        )

        # Get contact person name from debtor_contact context
        contact_name = ""
        if request.context.debtor_contact and request.context.debtor_contact.get("name"):
            contact_name = request.context.debtor_contact["name"]

        # Build base user prompt
        base_user_prompt = GENERATE_DRAFT_USER.format(
            party_name=request.context.party.name,
            contact_name=contact_name or "(not available)",
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
            segment=behavior.behaviour_segment or behavior.segment if behavior else "standard",
            on_time_rate=f"{behavior.on_time_rate:.0%}"
            if behavior and behavior.on_time_rate
            else "Unknown",
            avg_days_to_pay=behavior.avg_days_to_pay if behavior else "Unknown",
            max_days_overdue=max_days_overdue,
            obligation_count=obligation_count,
            industry_context=industry_context,
            sender_persona_context=sender_persona_context,
            tone=request.tone,
            objective=request.objective or "collect payment",
            brand_tone=request.context.brand_tone,
            is_follow_up=is_follow_up,
            custom_instructions=f"\nAdditional: {request.custom_instructions}"
            if request.custom_instructions
            else "",
        )

        # Append extended sections
        base_user_prompt += extra_sections

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
            o.invoice_number
            for o in request.context.obligations
            if o.invoice_number and o.invoice_number in result.body
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

        # Build reasoning dict for Django consumption
        reasoning_dict = None
        if result.reasoning:
            reasoning_dict = result.reasoning.model_dump()

        # Use LLM-provided invoices_referenced if available, fall back to body scan,
        # then fall back to all context obligations (since {INVOICE_TABLE} includes them all)
        # Filter out empty strings (obligations with no invoice number in source data)
        llm_refs = [r for r in (result.invoices_referenced or []) if r]
        final_invoices = llm_refs or invoices_referenced
        if not final_invoices:
            final_invoices = [
                o.invoice_number for o in request.context.obligations if o.invoice_number
            ]

        return GenerateDraftResponse(
            subject=result.subject,
            body=result.body,
            tone_used=request.tone,
            invoices_referenced=final_invoices,
            tokens_used=total_tokens_used,
            guardrail_validation=guardrail_validation,
            provider=response.provider,
            model=response.model,
            is_fallback=(response.provider != llm_client.primary_provider_name),
            reasoning=reasoning_dict,
            primary_cta=result.primary_cta,
            follow_up_days=result.follow_up_days,
        )

    def _format_sender_persona(self, request: GenerateDraftRequest) -> str:
        """Format sender persona context for prompt inclusion."""
        company = request.sender_company or ""
        persona = request.sender_persona
        if not persona or not persona.communication_style:
            # No persona — use name/title/company if available
            name = request.sender_name or "[SENDER_NAME]"
            title = request.sender_title or "[SENDER_TITLE]"
            company_line = f", Company: {company}" if company else ""
            return f"Name: {name}, Title: {title}{company_line} (no persona profile — use neutral professional voice)"

        lines = [
            f"- Name: {persona.name}",
            f"- Title: {persona.title or 'Team Member'}",
        ]
        if company:
            lines.append(f"- Company: {company}")
        lines.extend(
            [
                f"- Communication Style: {persona.communication_style}",
                f"- Formality Level: {persona.formality_level}",
                f"- Emphasis: {persona.emphasis}",
            ]
        )
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

    def _build_extra_sections(self, request, behavior) -> str:
        """Build extended prompt sections for new context layers."""
        sections = []

        # Behaviour segment
        if behavior and behavior.behaviour_segment:
            sections.append(f"\n\n**Behaviour Segment:** {behavior.behaviour_segment}")
            if behavior.behaviour_profile and isinstance(behavior.behaviour_profile, dict):
                profile = behavior.behaviour_profile
                profile_lines = []
                for k in (
                    "responsiveness_trend",
                    "promise_fulfilment_rate",
                    "dispute_frequency",
                    "avg_response_time",
                ):
                    if k in profile:
                        profile_lines.append(f"- {k.replace('_', ' ').title()}: {profile[k]}")
                if profile_lines:
                    sections.append("\n".join(profile_lines))

        # Sender style context
        if request.sender_context:
            sc = request.sender_context
            style_lines = []
            if sc.roles_responsibilities:
                style_lines.append(f"- Level R&R: {sc.roles_responsibilities}")
            if sc.style_description:
                style_lines.append(f"- Writing Style: {sc.style_description}")
            if sc.style_examples:
                style_lines.append("- Style Examples:")
                for i, ex in enumerate(sc.style_examples[:2], 1):
                    snippet = ex[:300] if len(ex) > 300 else ex
                    style_lines.append(f"  Example {i}: {snippet}")
            if style_lines:
                sections.append("\n\n**Sender Style:**\n" + "\n".join(style_lines))

        # Conversation history (recent messages for follow-up context)
        recent_msgs = request.context.recent_messages
        if recent_msgs:
            msg_lines = []
            for msg in reversed(recent_msgs):  # chronological order
                direction = msg.get("direction", "unknown")
                label = "DEBTOR REPLIED" if direction == "inbound" else "OUR EMAIL"
                classification = msg.get("classification")
                subject = msg.get("subject", "")
                body = msg.get("body_snippet", "")
                sent_at = msg.get("sent_at", "")
                line = f"- [{label}] ({sent_at})"
                if classification:
                    line += f" Classification: {classification}"
                if subject:
                    line += f"\n  Subject: {subject}"
                if body:
                    line += f"\n  Content: {body}"
                msg_lines.append(line)
            if msg_lines:
                sections.append(
                    "\n\n**Recent Conversation History (IMPORTANT — reference this in your reply):**\n"
                    + "\n".join(msg_lines)
                    + "\n\nThis is a FOLLOW-UP email. You MUST acknowledge the debtor's most recent "
                    "response and build on it. Do NOT write a generic first-contact collection email."
                )

        # Last response snippet (fallback if no recent_messages)
        if not recent_msgs and request.context.communication:
            comm = request.context.communication
            if comm.last_response_snippet:
                sections.append(
                    f"\n\n**Debtor's Last Response:**\n"
                    f"- Type: {comm.last_response_type or 'Unknown'}\n"
                    f"- Subject: {comm.last_response_subject or 'N/A'}\n"
                    f"- Content: {comm.last_response_snippet}\n\n"
                    "This is a FOLLOW-UP email. Acknowledge the debtor's response and build on it."
                )

        # Tone preference
        if request.tone_preference:
            sections.append(f"\n\n**Tone Preference:** {request.tone_preference}")

        # Closure mode
        if request.closure_mode:
            sections.append(
                "\n\n**CLOSURE EMAIL MODE**: This is a closure/thank-you email. "
                "The debtor has paid in full or the case is resolved. "
                "Use a grateful, relationship-preserving tone. "
                "Do NOT include any collection language, payment demands, "
                "or references to other invoices. Keep it brief and positive."
            )
        else:
            # Invoice table instruction (non-closure only)
            sections.append(
                "\n\nIMPORTANT: Do NOT write invoice numbers, amounts, or dates "
                "in the email body. Instead, include the exact placeholder "
                "{INVOICE_TABLE} where invoice details should appear. "
                "The system will replace this with a programmatic table. "
                "You may reference 'the invoices listed below' or "
                "'the outstanding items' in your prose."
            )

        return "".join(sections)

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

"""
Persona generation and refinement engine.

- Cold start: generates initial persona from name + title + level
- Refinement: LLM-driven update based on sender performance stats
"""

import json
import logging

from pydantic import ValidationError

from src.api.errors import LLMResponseInvalidError
from src.config.constants import (
    LEVEL_DESCRIPTIONS,
    PERSONA_GENERATION_SYSTEM,
    PERSONA_GENERATION_USER,
    PERSONA_REFINEMENT_SYSTEM,
    PERSONA_REFINEMENT_USER,
)
from src.llm.factory import llm_client
from src.llm.schemas import PersonaLLMResponse, PersonaRefinementLLMResponse

logger = logging.getLogger(__name__)


class PersonaGenerator:
    """Generates and refines sender personas using LLM."""

    async def generate_personas(self, contacts: list, total_levels: int = 4) -> dict:
        """
        Generate initial personas for a list of contacts (cold start).

        Args:
            contacts: List of dicts with name, title, level
            total_levels: Total number of escalation levels

        Returns:
            Dict with personas list and aggregate token/provider metadata
        """
        results = []
        total_tokens = 0
        last_provider = None
        last_model = None
        is_fallback = False
        for contact in contacts:
            try:
                persona, response_meta = await self._generate_single(contact, total_levels)
                results.append(persona)
                total_tokens += response_meta.get("tokens_used", 0)
                last_provider = response_meta.get("provider", last_provider)
                last_model = response_meta.get("model", last_model)
                if response_meta.get("is_fallback"):
                    is_fallback = True
            except Exception as e:
                logger.warning(
                    "Failed to generate persona for %s (level %d): %s",
                    contact.get("name", "unknown"),
                    contact.get("level", 0),
                    e,
                )
                # Return empty persona on failure — non-fatal
                results.append(
                    {
                        "name": contact.get("name", ""),
                        "level": contact.get("level", 1),
                        "communication_style": None,
                        "formality_level": None,
                        "emphasis": None,
                    }
                )
        return {
            "personas": results,
            "tokens_used": total_tokens,
            "provider": last_provider,
            "model": last_model,
            "is_fallback": is_fallback,
        }

    async def _generate_single(self, contact: dict, total_levels: int) -> tuple:
        """Generate persona for a single contact. Returns (persona_dict, response_meta)."""
        level = contact.get("level", 1)
        level_description = LEVEL_DESCRIPTIONS.get(level, LEVEL_DESCRIPTIONS[1])

        # Build style section from user-provided inputs
        style_section = ""
        if contact.get("style_description"):
            style_section += f"\nUser-provided style guidance: {contact['style_description']}"
        if contact.get("style_examples"):
            style_section += "\nExample emails from this person:"
            for i, ex in enumerate(contact["style_examples"][:3], 1):
                style_section += f"\n  Example {i}: {ex[:500]}"
        if not style_section:
            style_section = "\n(No style guidance provided — infer from name, title, and role)"

        user_prompt = PERSONA_GENERATION_USER.format(
            name=contact.get("name", "Unknown"),
            title=contact.get("title", "") or "Team Member",
            level=level,
            total_levels=total_levels,
            level_description=level_description,
            style_section=style_section,
        )

        response = await llm_client.complete(
            system_prompt=PERSONA_GENERATION_SYSTEM,
            user_prompt=user_prompt,
            temperature=0.7,
            response_schema=PersonaLLMResponse,
        )

        raw_result = json.loads(response.content)

        try:
            parsed = PersonaLLMResponse(**raw_result)
        except ValidationError as e:
            logger.error("Persona LLM response validation failed: %s", e)
            raise LLMResponseInvalidError(
                message="LLM returned invalid persona response",
                details={"validation_errors": e.errors()},
            )

        persona = {
            "name": contact.get("name", ""),
            "level": level,
            "communication_style": parsed.communication_style,
            "formality_level": parsed.formality_level,
            "emphasis": parsed.emphasis,
        }
        response_meta = {
            "tokens_used": response.usage.get("total_tokens", 0),
            "provider": response.provider,
            "model": response.model,
            "is_fallback": getattr(response, "is_fallback", False),
        }
        return persona, response_meta

    async def refine_persona(
        self,
        contact: dict,
        current_persona: dict,
        performance: dict,
        persona_version: int = 0,
        style_description: str = None,
        style_examples: list = None,
    ) -> dict:
        """
        Refine a persona based on sender performance stats (LLM-driven).

        Args:
            contact: dict with name, title, level
            current_persona: dict with communication_style, formality_level, emphasis, persona_version
            performance: dict with all sender_performance stats

        Returns:
            Updated persona dict
        """
        total_touches = performance.get("total_touches", 0)

        # Format percentages for prompt readability
        def pct(count, total):
            if total == 0:
                return "N/A"
            return f"{count / total:.0%}"

        def fmt(val, suffix=""):
            if val is None:
                return "N/A"
            return f"{val}{suffix}"

        # Build style anchor section
        style_section = ""
        if style_description:
            style_section += (
                f"\n## User-Provided Style Anchor\n- Style Guidance: {style_description}"
            )
        if style_examples:
            style_section += "\n- Example emails:"
            for i, ex in enumerate(style_examples[:3], 1):
                style_section += f"\n  Example {i}: {ex[:500]}"
        if not style_section:
            style_section = ""

        user_prompt = PERSONA_REFINEMENT_USER.format(
            name=contact.get("name", "Unknown"),
            title=contact.get("title", "") or "Team Member",
            level=contact.get("level", 1),
            current_communication_style=current_persona.get("communication_style", "Not set"),
            current_formality_level=current_persona.get("formality_level", "Not set"),
            current_emphasis=current_persona.get("emphasis", "Not set"),
            persona_version=persona_version,
            style_section=style_section,
            total_touches=total_touches,
            total_unique_parties=performance.get("total_unique_parties", 0),
            response_rate=fmt(performance.get("response_rate"), ""),
            avg_response_days=fmt(performance.get("avg_response_days"), " days"),
            no_response_count=performance.get("no_response_count", 0),
            cooperative_count=performance.get("cooperative_count", 0),
            cooperative_pct=pct(performance.get("cooperative_count", 0), total_touches),
            hostile_count=performance.get("hostile_count", 0),
            hostile_pct=pct(performance.get("hostile_count", 0), total_touches),
            promise_count=performance.get("promise_count", 0),
            promise_pct=pct(performance.get("promise_count", 0), total_touches),
            dispute_count=performance.get("dispute_count", 0),
            dispute_pct=pct(performance.get("dispute_count", 0), total_touches),
            cases_resolved_pif=performance.get("cases_resolved_pif", 0),
            amount_collected_after=fmt(performance.get("amount_collected_after")),
            avg_days_to_payment=fmt(performance.get("avg_days_to_payment"), " days"),
            promises_elicited=performance.get("promises_elicited", 0),
            promises_kept=performance.get("promises_kept", 0),
            promises_broken=performance.get("promises_broken", 0),
            promise_fulfillment_rate=fmt(performance.get("promise_fulfillment_rate")),
            early_state_pct=fmt(performance.get("early_state_pct")),
            escalated_state_pct=fmt(performance.get("escalated_state_pct")),
            tone_distribution=json.dumps(performance.get("tone_distribution") or {}),
            segment_distribution=json.dumps(performance.get("segment_distribution") or {}),
            avg_days_between_touches=fmt(performance.get("avg_days_between_touches"), " days"),
        )

        response = await llm_client.complete(
            system_prompt=PERSONA_REFINEMENT_SYSTEM,
            user_prompt=user_prompt,
            temperature=0.5,  # Lower temp for more consistent refinement
            response_schema=PersonaRefinementLLMResponse,
        )

        raw_result = json.loads(response.content)

        try:
            parsed = PersonaRefinementLLMResponse(**raw_result)
        except ValidationError as e:
            logger.error("Persona refinement LLM response validation failed: %s", e)
            raise LLMResponseInvalidError(
                message="LLM returned invalid persona refinement response",
                details={"validation_errors": e.errors()},
            )

        logger.info(
            "Persona refined for %s (level %d): %s",
            contact.get("name", "unknown"),
            contact.get("level", 0),
            parsed.reasoning,
        )

        return {
            "communication_style": parsed.communication_style,
            "formality_level": parsed.formality_level,
            "emphasis": parsed.emphasis,
            "reasoning": parsed.reasoning,
            "tokens_used": response.usage.get("total_tokens", 0),
            "provider": response.provider,
            "model": response.model,
            "is_fallback": getattr(response, "is_fallback", False),
        }


# Singleton instance
persona_generator = PersonaGenerator()

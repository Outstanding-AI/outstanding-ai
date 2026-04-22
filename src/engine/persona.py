"""
Persona generation and refinement engine.

Manage the 4-layer persona pipeline for escalation contacts:

1. **Tone** -- selected by Django's ``_select_tone_from_context()``
   (behaviour segment, touch count, amount percentile).
2. **Persona** -- generated here (communication_style, formality_level,
   emphasis) from the contact's name, title, escalation level, and
   optional style guidance.
3. **Style Examples** -- user-provided email samples anchoring the
   persona's voice across refinement cycles.
4. **Case Context** -- injected at draft-generation time.

Two operations:

- **Cold start** (``generate_personas``): Called when the admin saves the
  escalation hierarchy.  Produces an initial persona for each contact
  using a single LLM call per contact.
- **Refinement** (``refine_persona``): Called during the sync cycle
  (``gold.refine_sender_personas``) for senders with >= 10 accumulated
  touches. The LLM receives the current persona alongside aggregated
  performance stats (response rate, cooperative count, hostile count,
  promise fulfillment, normalized dispute/promise outcomes, cadence,
  and tone distribution) and outputs an updated persona with reasoning.
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
from src.config.settings import settings
from src.llm.factory import llm_client
from src.llm.schemas import PersonaLLMResponse, PersonaRefinementLLMResponse

logger = logging.getLogger(__name__)


class PersonaGenerator:
    """Generate and refine sender personas using LLM.

    A singleton instance (``persona_generator``) is exported at module
    level for use by the FastAPI route handlers.
    """

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
        total_prompt_tokens = 0
        total_completion_tokens = 0
        last_provider = None
        last_model = None
        is_fallback = False
        for contact in contacts:
            # Skip persona generation for generic/shared mailboxes
            if getattr(contact, "is_generic_mailbox", False) or (
                isinstance(contact, dict) and contact.get("is_generic_mailbox")
            ):
                results.append(
                    {
                        "name": contact.name
                        if hasattr(contact, "name")
                        else contact.get("name", ""),
                        "level": contact.level
                        if hasattr(contact, "level")
                        else contact.get("level", 1),
                        "communication_style": "team-oriented professional",
                        "formality_level": "professional",
                        "emphasis": "clear and efficient communication",
                        "skipped": True,
                        "skip_reason": "generic_mailbox",
                    }
                )
                continue
            try:
                persona, response_meta = await self._generate_single(contact, total_levels)
                results.append(persona)
                total_tokens += response_meta.get("tokens_used", 0)
                total_prompt_tokens += response_meta.get("prompt_tokens", 0)
                total_completion_tokens += response_meta.get("completion_tokens", 0)
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
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "provider": last_provider,
            "model": last_model,
            "is_fallback": is_fallback,
        }

    async def _generate_single(self, contact: dict, total_levels: int) -> tuple:
        """Generate persona for a single contact.

        Build a prompt incorporating the contact's name, title,
        escalation level description (from ``LEVEL_DESCRIPTIONS``),
        and any user-provided style guidance / example emails.

        Args:
            contact: Dict with keys ``name``, ``title``, ``level``,
                and optionally ``style_description``, ``style_examples``.
            total_levels: Total escalation levels in the hierarchy
                (typically 4).

        Returns:
            Tuple of (persona_dict, response_meta) where persona_dict
            contains ``name``, ``level``, ``communication_style``,
            ``formality_level``, ``emphasis``; and response_meta
            contains token usage and provider info.
        """
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
            temperature=settings.persona_gen_temperature,
            response_schema=PersonaLLMResponse,
            caller="persona_generation",
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
            "prompt_tokens": response.usage.get("prompt_tokens", 0),
            "completion_tokens": response.usage.get("completion_tokens", 0),
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
        """Refine a persona based on sender performance stats.

        The LLM receives the current persona profile alongside
        aggregated performance metrics and returns an updated persona
        with a ``reasoning`` field explaining the changes.

        Style anchors (``style_description``, ``style_examples``) are
        included when available to prevent persona drift away from the
        user's intended voice.

        Args:
            contact: Dict with ``name``, ``title``, ``level``.
            current_persona: Dict with ``communication_style``,
                ``formality_level``, ``emphasis``.
            performance: Dict with the refiner-safe subset of
                ``sender_performance`` stats (responded_touches,
                response_rate, cooperative_count, hostile_count,
                promises_elicited, disputes_raised_after,
                promise_fulfillment_rate, cadence, etc.).
            persona_version: Current persona version number for
                tracking refinement iterations.
            style_description: Optional user-provided style guidance.
            style_examples: Optional list of example emails from this
                sender.

        Returns:
            Dict with updated persona fields (communication_style,
            formality_level, emphasis, reasoning) and token/provider
            metadata.
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
            responded_touches=performance.get("responded_touches", 0),
            response_rate=fmt(performance.get("response_rate"), ""),
            avg_response_days=fmt(performance.get("avg_response_days"), " days"),
            cooperative_count=performance.get("cooperative_count", 0),
            cooperative_pct=pct(performance.get("cooperative_count", 0), total_touches),
            hostile_count=performance.get("hostile_count", 0),
            hostile_pct=pct(performance.get("hostile_count", 0), total_touches),
            cases_resolved_pif=performance.get("cases_resolved_pif", 0),
            amount_collected_after=fmt(performance.get("amount_collected_after")),
            avg_days_to_payment=fmt(performance.get("avg_days_to_payment"), " days"),
            promises_elicited=performance.get("promises_elicited", 0),
            promises_kept=performance.get("promises_kept", 0),
            promise_fulfillment_rate=fmt(performance.get("promise_fulfillment_rate")),
            disputes_raised_after=performance.get("disputes_raised_after", 0),
            disputes_resolved=performance.get("disputes_resolved", 0),
            avg_days_between_touches=fmt(performance.get("avg_days_between_touches"), " days"),
        )

        response = await llm_client.complete(
            system_prompt=PERSONA_REFINEMENT_SYSTEM,
            user_prompt=user_prompt,
            temperature=settings.persona_refine_temperature,
            response_schema=PersonaRefinementLLMResponse,
            caller="persona_refinement",
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
            "prompt_tokens": response.usage.get("prompt_tokens", 0),
            "completion_tokens": response.usage.get("completion_tokens", 0),
            "provider": response.provider,
            "model": response.model,
            "is_fallback": getattr(response, "is_fallback", False),
        }


# Singleton instance used by the /generate-persona and /refine-persona
# route handlers.
persona_generator = PersonaGenerator()

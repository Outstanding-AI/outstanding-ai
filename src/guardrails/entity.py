"""Entity Verification Guardrail - LLM-based validation of customer/party identifiers."""

import asyncio
import json
import logging
import re
import time
from typing import Any, List

from pydantic import BaseModel, Field

from src.api.models.requests import CaseContext
from src.llm.factory import llm_client

from .base import BaseGuardrail, GuardrailResult, GuardrailSeverity

logger = logging.getLogger(__name__)

# Retry configuration for LLM validation
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 1.0


class EntityValidationResult(BaseModel):
    """Structured output schema for entity validation.

    Using Pydantic model with with_structured_output() ensures the LLM
    returns valid JSON matching this exact schema - no markdown, no parsing errors.
    """

    customer_code_valid: bool = Field(
        default=True, description="True if customer code is correct or not mentioned in draft"
    )
    customer_code_reason: str = Field(
        default="Not evaluated", description="Brief explanation of customer code validation result"
    )
    party_name_valid: bool = Field(
        default=True, description="True if party name matches or is a reasonable variation"
    )
    party_name_reason: str = Field(
        default="Not evaluated", description="Brief explanation of party name validation result"
    )
    issues_found: List[str] = Field(
        default_factory=list, description="List of specific issues found, empty if none"
    )
    passed: bool = Field(
        default=True, description="True if overall validation passed (no mismatches found)"
    )


# Validation prompt for LLM-based entity verification
ENTITY_VALIDATION_PROMPT = """Validate the following draft email for entity accuracy.

EXPECTED ENTITIES:
- Customer Code: {customer_code}
- Party/Company Name: {party_name}

DRAFT TO VALIDATE:
{draft}

Your task:
1. Check if the draft correctly references the customer code (if mentioned at all)
2. Check if the draft addresses the correct party/company name
3. Identify any hallucinated, fabricated, or mismatched identifiers

IMPORTANT: The draft does NOT need to explicitly mention the customer code. Only flag it as invalid if it mentions a DIFFERENT code than expected.

For party name validation:
- Accept reasonable variations (e.g., "Acme Corp" vs "ACME Corporation Ltd")
- Accept generic greetings like "Dear Customer", "Dear Accounts Team", "Dear Sir/Madam"
- Only flag as invalid if the draft clearly addresses a DIFFERENT company

Set "passed" to false only if there are actual mismatches or hallucinated identifiers."""


class EntityVerificationGuardrail(BaseGuardrail):
    """
    LLM-based entity verification guardrail.

    Uses the LLM to validate that:
    1. Customer code is correct (if mentioned)
    2. Party name is accurate (or acceptably generic)
    3. No hallucinated identifiers exist

    This uses LLM-as-judge pattern because:
    - Regex can't understand semantic variations ("Acme Corp" vs "ACME Corporation")
    - Only LLM can judge if "Dear Accounts Team" is acceptable for "Compton Packaging"
    - Scales across different industries and naming conventions
    """

    def __init__(self):
        super().__init__(
            name="entity_verification",
            severity=GuardrailSeverity.HIGH,
        )

    def validate(self, output: str, context: CaseContext, **kwargs) -> list[GuardrailResult]:
        """
        Validate entity identifiers using LLM-based verification with retry.

        Runs the LLM synchronously using asyncio.run() since guardrails
        execute in a thread pool. Retries on failure with exponential backoff.
        """
        results = []

        # Run LLM-based entity validation with retry
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                llm_result = self._validate_entities_with_llm(output, context)
                results.extend(llm_result)
                last_error = None
                break
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    backoff = INITIAL_BACKOFF_SECONDS * (2**attempt)
                    logger.warning(
                        "Entity validation attempt %d failed: %s. Retrying in %.1fs...",
                        attempt + 1,
                        e,
                        backoff,
                    )
                    time.sleep(backoff)
                else:
                    logger.error("Entity validation failed after %d attempts: %s", MAX_RETRIES, e)

        # If all retries failed, fail the guardrail (don't silently pass)
        if last_error is not None:
            results.append(
                self._fail(
                    message=f"Entity validation failed: {last_error}",
                    expected="Valid LLM response",
                    found=str(last_error),
                    details={"error": str(last_error), "retries": MAX_RETRIES},
                )
            )

        # Only validate emails if extracted_data is provided (keep this deterministic)
        extracted_data = kwargs.get("extracted_data")
        if extracted_data:
            results.append(self._validate_emails(output, context, extracted_data))

        return results

    def _validate_entities_with_llm(
        self, output: str, context: CaseContext
    ) -> list[GuardrailResult]:
        """
        Use LLM to validate entity accuracy with structured output.

        Uses response_schema parameter to ensure the LLM returns valid JSON
        matching EntityValidationResult schema. This is more reliable than
        json_mode alone, which can still return markdown-wrapped JSON.

        Returns list of GuardrailResults for customer code and party name.
        """
        prompt = ENTITY_VALIDATION_PROMPT.format(
            customer_code=context.party.customer_code,
            party_name=context.party.name,
            draft=output,
        )

        # Run async LLM call in sync context (guardrails run in thread pool)
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response = loop.run_until_complete(
                    llm_client.complete(
                        system_prompt="You are a validation assistant.",
                        user_prompt=prompt,
                        temperature=0,  # Deterministic for validation
                        max_tokens=2048,  # High for reasoning models that consume tokens for "thinking"
                        # Use structured output for guaranteed valid JSON
                        response_schema=EntityValidationResult,
                    )
                )
            finally:
                loop.close()
        except Exception as e:
            logger.error("LLM call failed in entity verification: %s", e)
            raise

        # Parse the response - should be clean JSON from structured output
        result = json.loads(response.content)

        # Extract token usage for cost tracking
        token_usage = {
            "prompt_tokens": response.usage.get("prompt_tokens", 0),
            "completion_tokens": response.usage.get("completion_tokens", 0),
            "total_tokens": response.usage.get("total_tokens", 0),
        }

        results = []

        # Customer code validation result
        if result.get("customer_code_valid", True):
            results.append(
                self._pass(
                    message=result.get("customer_code_reason", "Customer code validated"),
                    details={"customer_code": context.party.customer_code},
                )
            )
        else:
            results.append(
                self._fail(
                    message=result.get(
                        "customer_code_reason",
                        f"Customer code validation failed for {context.party.customer_code}",
                    ),
                    expected=context.party.customer_code,
                    found=None,
                    details={"issues": result.get("issues_found", [])},
                )
            )

        # Party name validation result
        if result.get("party_name_valid", True):
            results.append(
                self._pass(
                    message=result.get("party_name_reason", "Party name validated"),
                    details={"party_name": context.party.name},
                )
            )
        else:
            results.append(
                self._fail(
                    message=result.get(
                        "party_name_reason",
                        f"Party name validation failed for {context.party.name}",
                    ),
                    expected=context.party.name,
                    found=None,
                    details={"issues": result.get("issues_found", [])},
                )
            )

        # Attach token usage to the first result for pipeline aggregation
        if results:
            results[0].token_usage = token_usage

        logger.info(
            "Entity verification completed: customer_code_valid=%s, party_name_valid=%s, passed=%s",
            result.get("customer_code_valid"),
            result.get("party_name_valid"),
            result.get("passed"),
        )

        return results

    def _validate_emails(
        self, output: str, _context: CaseContext, extracted_data: Any
    ) -> GuardrailResult:
        """Validate that email addresses are not fabricated."""
        # Extract email addresses from output
        email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        found_emails = set(re.findall(email_pattern, output))

        if not found_emails:
            return self._pass(message="No email addresses to validate")

        # Valid emails could come from:
        # 1. The email being classified (from_address)
        # 2. Extracted redirect email
        valid_emails = set()

        # This would need to be passed in context - for now, check extracted_data
        if hasattr(extracted_data, "redirect_email") and extracted_data.redirect_email:
            valid_emails.add(extracted_data.redirect_email.lower())

        # If no valid emails defined, any email could be suspicious
        if not valid_emails and found_emails:
            return self._fail(
                message=f"Fabricated email addresses found: {found_emails}",
                expected="No emails or extracted redirect email",
                found=list(found_emails),
                details={
                    "found_emails": list(found_emails),
                    "note": "Emails should come from extracted_data only",
                },
            )

        # Check for invalid emails
        found_emails_lower = {e.lower() for e in found_emails}
        invalid_emails = found_emails_lower - valid_emails
        if invalid_emails:
            return self._fail(
                message=f"Unverified email addresses found: {invalid_emails}",
                expected=list(valid_emails),
                found=list(invalid_emails),
            )

        return self._pass(
            message="Email addresses validated",
            details={"validated_emails": list(found_emails)},
        )

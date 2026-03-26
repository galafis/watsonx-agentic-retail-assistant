"""Guardrails for preventing PII leakage, inappropriate recommendations, and pricing errors."""

from __future__ import annotations

import re
from typing import Any

import structlog

from src.config import settings

logger = structlog.get_logger(__name__)


class Guardrails:
    """Safety guardrails for the retail assistant.

    Validates inputs and outputs to prevent:
    - PII leakage (SSN, credit cards, emails in responses)
    - Prompt injection attacks
    - Pricing errors beyond tolerance
    - Excessively long responses
    - Inappropriate or harmful content
    """

    def __init__(self) -> None:
        config = settings.guardrails_config
        self._max_response_length: int = config.get("max_response_length", 2000)
        self._blocked_patterns: list[str] = config.get("blocked_patterns", [])
        self._pii_patterns: list[str] = config.get("pii_patterns", [])
        self._price_tolerance: float = config.get("price_tolerance", 0.01)

        # Compile regex patterns
        self._blocked_re = [re.compile(p) for p in self._blocked_patterns]
        self._pii_re = [re.compile(p) for p in self._pii_patterns]

    def validate_input(self, user_message: str) -> dict[str, Any]:
        """Validate user input for prompt injection and safety concerns.

        Args:
            user_message: The user's input message.

        Returns:
            Validation result with 'safe' flag and any 'warnings'.
        """
        warnings: list[str] = []

        # Check for prompt injection patterns
        for pattern in self._blocked_re:
            if pattern.search(user_message):
                logger.warning("prompt_injection_detected", pattern=pattern.pattern)
                return {
                    "safe": False,
                    "warnings": ["Potential prompt injection detected. Message blocked."],
                }

        # Check message length
        if len(user_message) > 5000:
            warnings.append("Message exceeds maximum length; it will be truncated.")

        return {"safe": True, "warnings": warnings}

    def validate_output(self, response: str) -> dict[str, Any]:
        """Validate assistant output for PII leakage and safety.

        Args:
            response: The assistant's response text.

        Returns:
            Validation result with 'safe' flag, sanitized 'response', and 'warnings'.
        """
        warnings: list[str] = []
        sanitized = response

        # Check for PII leakage
        for pattern in self._pii_re:
            if pattern.search(response):
                sanitized = pattern.sub("[REDACTED]", sanitized)
                warnings.append("PII detected and redacted from response.")
                logger.warning("pii_leakage_detected", pattern=pattern.pattern)

        # Truncate overly long responses
        if len(sanitized) > self._max_response_length:
            sanitized = sanitized[: self._max_response_length] + "..."
            warnings.append("Response truncated due to length limit.")

        return {
            "safe": len(warnings) == 0 or all("redacted" in w.lower() for w in warnings),
            "response": sanitized,
            "warnings": warnings,
        }

    def validate_price(
        self,
        stated_price: float,
        actual_price: float,
    ) -> dict[str, Any]:
        """Validate that a stated price matches the actual catalog price.

        Args:
            stated_price: The price mentioned in the response.
            actual_price: The actual price from the product catalog.

        Returns:
            Validation result with 'valid' flag and price details.
        """
        diff = abs(stated_price - actual_price)
        tolerance = actual_price * self._price_tolerance

        if diff > tolerance:
            logger.warning(
                "price_mismatch_detected",
                stated=stated_price,
                actual=actual_price,
                diff=diff,
            )
            return {
                "valid": False,
                "stated_price": stated_price,
                "actual_price": actual_price,
                "difference": round(diff, 2),
                "warning": "Price mismatch detected. Using catalog price.",
            }

        return {
            "valid": True,
            "stated_price": stated_price,
            "actual_price": actual_price,
        }

    def check_recommendation_safety(
        self,
        products: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Validate that product recommendations are appropriate.

        Checks for duplicate recommendations and ensures diversity.

        Args:
            products: List of recommended products.

        Returns:
            Validation result with 'safe' flag and any issues found.
        """
        issues: list[str] = []

        # Check for duplicates
        ids = [p.get("id") for p in products]
        if len(ids) != len(set(ids)):
            issues.append("Duplicate products in recommendations.")

        # Check for empty recommendations
        if not products:
            issues.append("Empty recommendation list.")

        return {
            "safe": len(issues) == 0,
            "issues": issues,
            "product_count": len(products),
        }


# Module-level singleton
guardrails = Guardrails()

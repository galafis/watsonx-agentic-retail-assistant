"""Support agent for FAQ retrieval with human escalation triggers."""

from __future__ import annotations

from typing import Any

import structlog

from src.governance.audit_logger import audit_logger
from src.tools.faq_retriever import faq_retriever
from src.tools.sentiment import sentiment_analyzer

logger = structlog.get_logger(__name__)


class SupportAgent:
    """Specialized agent for customer support queries.

    Handles FAQ retrieval, general support questions, and detects
    when a conversation should be escalated to a human agent.
    """

    AGENT_NAME = "support_agent"

    def handle(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Handle a customer support query.

        Args:
            query: Customer's support question.
            context: Optional context with category or prior conversation state.

        Returns:
            Agent response with FAQ results, escalation flag, and tools used.
        """
        context = context or {}

        # Analyze sentiment for escalation detection
        sentiment = sentiment_analyzer.analyze(query)
        audit_logger.log_tool_call(
            agent_name=self.AGENT_NAME,
            tool_name="sentiment_analyzer",
            tool_input={"query": query},
            tool_output=sentiment,
        )

        # Check if escalation is needed
        if sentiment["needs_escalation"]:
            return self._handle_escalation(query, sentiment)

        # Retrieve FAQ entries
        category = context.get("category")
        faq_results = faq_retriever.retrieve(query, category=category)
        audit_logger.log_tool_call(
            agent_name=self.AGENT_NAME,
            tool_name="faq_retriever",
            tool_input={"query": query, "category": category},
            tool_output={"results_count": len(faq_results)},
        )

        if faq_results:
            return self._format_faq_response(faq_results, query)
        else:
            return self._handle_no_faq_match(query)

    def _format_faq_response(
        self,
        faq_results: list[dict[str, Any]],
        query: str,
    ) -> dict[str, Any]:
        """Format FAQ results into a response."""
        top_faq = faq_results[0]
        response_text = top_faq["answer"]

        if len(faq_results) > 1:
            response_text += "\n\nYou might also find these helpful:\n"
            for faq in faq_results[1:]:
                response_text += f"- {faq['question']}\n"

        return {
            "response": response_text,
            "faq_results": faq_results,
            "needs_escalation": False,
            "tools_used": ["faq_retriever", "sentiment_analyzer"],
            "agent": self.AGENT_NAME,
        }

    def _handle_escalation(
        self,
        query: str,
        sentiment: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle cases requiring human escalation."""
        response_text = (
            "I understand your frustration, and I want to make sure you get the best help. "
            "I'm connecting you with a human support specialist who can assist you further. "
            "A representative will be with you shortly."
        )

        logger.warning(
            "escalation_triggered",
            sentiment_score=sentiment["score"],
            sentiment_label=sentiment["label"],
        )

        return {
            "response": response_text,
            "faq_results": [],
            "needs_escalation": True,
            "sentiment": sentiment,
            "tools_used": ["sentiment_analyzer"],
            "agent": self.AGENT_NAME,
        }

    def _handle_no_faq_match(self, query: str) -> dict[str, Any]:
        """Handle queries that don't match any FAQ entries."""
        response_text = (
            "I don't have a specific answer for that in our FAQ. "
            "Let me connect you with our support team for a more detailed response. "
            "In the meantime, you can also check our help center for common topics."
        )

        return {
            "response": response_text,
            "faq_results": [],
            "needs_escalation": False,
            "tools_used": ["faq_retriever", "sentiment_analyzer"],
            "agent": self.AGENT_NAME,
        }

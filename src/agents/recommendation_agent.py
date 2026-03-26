"""Recommendation agent for product suggestions based on preferences and history."""

from __future__ import annotations

from typing import Any

import structlog

from src.governance.audit_logger import audit_logger
from src.tools.recommendation_engine import recommendation_engine

logger = structlog.get_logger(__name__)


class RecommendationAgent:
    """Specialized agent for product recommendation queries.

    Provides content-based recommendations (similar products) and
    collaborative-style recommendations (based on customer history).
    """

    AGENT_NAME = "recommendation_agent"

    def handle(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Handle a recommendation query.

        Args:
            query: User's recommendation request.
            context: Optional context with product_id, customer_id, purchase_history.

        Returns:
            Agent response with recommended products and reasoning.
        """
        context = context or {}

        product_id = context.get("product_id")
        customer_id = context.get("customer_id")
        purchase_history = context.get("purchase_history", [])

        if product_id:
            return self._recommend_similar(product_id, query)
        elif customer_id:
            return self._recommend_for_customer(customer_id, purchase_history, query)
        else:
            return self._recommend_general(query)

    def _recommend_similar(self, product_id: str, query: str) -> dict[str, Any]:
        """Recommend products similar to a given product."""
        products = recommendation_engine.recommend_similar(product_id)
        audit_logger.log_tool_call(
            agent_name=self.AGENT_NAME,
            tool_name="recommendation_engine.similar",
            tool_input={"product_id": product_id},
            tool_output={"count": len(products)},
        )

        if not products:
            return {
                "response": "I couldn't find similar products at the moment. "
                "Would you like to browse our categories instead?",
                "products": [],
                "tools_used": ["recommendation_engine"],
                "agent": self.AGENT_NAME,
            }

        lines = ["Based on your interest, here are some similar products:\n"]
        for p in products:
            lines.append(f"- **{p['name']}** (${p['price']:.2f}) - {p.get('category', '')}")

        return {
            "response": "\n".join(lines),
            "products": products,
            "tools_used": ["recommendation_engine"],
            "agent": self.AGENT_NAME,
        }

    def _recommend_for_customer(
        self,
        customer_id: str,
        purchase_history: list[str],
        query: str,
    ) -> dict[str, Any]:
        """Recommend products based on customer purchase history."""
        products = recommendation_engine.recommend_for_customer(
            customer_id=customer_id,
            purchase_history=purchase_history,
        )
        audit_logger.log_tool_call(
            agent_name=self.AGENT_NAME,
            tool_name="recommendation_engine.for_customer",
            tool_input={"customer_id": customer_id, "history_size": len(purchase_history)},
            tool_output={"count": len(products)},
        )

        if not products:
            return {
                "response": "I don't have enough data to make personalized recommendations yet. "
                "Here are some of our popular items!",
                "products": [],
                "tools_used": ["recommendation_engine"],
                "agent": self.AGENT_NAME,
            }

        lines = ["Based on your purchase history, you might like:\n"]
        for p in products:
            lines.append(f"- **{p['name']}** (${p['price']:.2f}) - {p.get('category', '')}")

        return {
            "response": "\n".join(lines),
            "products": products,
            "tools_used": ["recommendation_engine"],
            "agent": self.AGENT_NAME,
        }

    def _recommend_general(self, query: str) -> dict[str, Any]:
        """Provide general recommendations when no specific context is available."""
        products = recommendation_engine._popular_products(5)
        audit_logger.log_tool_call(
            agent_name=self.AGENT_NAME,
            tool_name="recommendation_engine.popular",
            tool_input={"query": query},
            tool_output={"count": len(products)},
        )

        if not products:
            return {
                "response": "Our product catalog is currently being updated. "
                "Please check back shortly!",
                "products": [],
                "tools_used": ["recommendation_engine"],
                "agent": self.AGENT_NAME,
            }

        lines = ["Here are some popular products you might enjoy:\n"]
        for p in products:
            lines.append(f"- **{p['name']}** (${p['price']:.2f}) - {p.get('category', '')}")

        return {
            "response": "\n".join(lines),
            "products": products,
            "tools_used": ["recommendation_engine"],
            "agent": self.AGENT_NAME,
        }

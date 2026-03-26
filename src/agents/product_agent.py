"""Product agent for handling product search, comparison, and details."""

from __future__ import annotations

from typing import Any

import structlog

from src.governance.audit_logger import audit_logger
from src.tools.catalog_search import catalog_search_tool

logger = structlog.get_logger(__name__)


class ProductAgent:
    """Specialized agent for product-related queries.

    Handles product search, product details, product comparison,
    and category browsing using the catalog search tool.
    """

    AGENT_NAME = "product_agent"

    def handle(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Handle a product-related query.

        Args:
            query: User's product query.
            context: Optional context from the orchestrator (e.g., extracted entities).

        Returns:
            Agent response with products found, response text, and tools used.
        """
        context = context or {}
        query_lower = query.lower()

        # Determine sub-intent
        if any(kw in query_lower for kw in ["compare", "vs", "versus", "difference"]):
            return self._handle_comparison(query, context)
        elif any(kw in query_lower for kw in ["detail", "info", "about", "tell me"]):
            return self._handle_details(query, context)
        else:
            return self._handle_search(query, context)

    def _handle_search(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        """Handle product search queries."""
        category = context.get("category")
        min_price = context.get("min_price")
        max_price = context.get("max_price")

        products = catalog_search_tool.search(
            query=query,
            category=category,
            min_price=min_price,
            max_price=max_price,
        )

        audit_logger.log_tool_call(
            agent_name=self.AGENT_NAME,
            tool_name="catalog_search",
            tool_input={"query": query, "category": category},
            tool_output={"results_count": len(products)},
        )

        if not products:
            response_text = (
                f"I couldn't find any products matching '{query}'. "
                "Could you try different search terms or browse our categories?"
            )
        else:
            product_lines = []
            for p in products:
                product_lines.append(
                    f"- **{p['name']}** (${p['price']:.2f}) - {p.get('description', '')[:80]}"
                )
            response_text = f"I found {len(products)} products:\n" + "\n".join(product_lines)

        return {
            "response": response_text,
            "products": products,
            "tools_used": ["catalog_search"],
            "agent": self.AGENT_NAME,
        }

    def _handle_details(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        """Handle product detail queries."""
        product_id = context.get("product_id")

        if product_id:
            product = catalog_search_tool.get_product_details(product_id)
            audit_logger.log_tool_call(
                agent_name=self.AGENT_NAME,
                tool_name="catalog_search.get_details",
                tool_input={"product_id": product_id},
                tool_output={"found": product is not None},
            )

            if product:
                response_text = (
                    f"**{product['name']}**\n"
                    f"- Category: {product.get('category', 'N/A')}\n"
                    f"- Price: ${product['price']:.2f}\n"
                    f"- Description: {product.get('description', 'N/A')}\n"
                    f"- Tags: {', '.join(product.get('tags', []))}"
                )
                return {
                    "response": response_text,
                    "products": [product],
                    "tools_used": ["catalog_search"],
                    "agent": self.AGENT_NAME,
                }

        # Fall back to search if no product_id
        return self._handle_search(query, context)

    def _handle_comparison(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        """Handle product comparison queries."""
        product_ids = context.get("product_ids", [])

        if len(product_ids) >= 2:
            products = catalog_search_tool.compare_products(product_ids)
            audit_logger.log_tool_call(
                agent_name=self.AGENT_NAME,
                tool_name="catalog_search.compare",
                tool_input={"product_ids": product_ids},
                tool_output={"compared_count": len(products)},
            )

            if products:
                lines = ["Here's a comparison of the products:\n"]
                for p in products:
                    lines.append(
                        f"**{p['name']}** - ${p['price']:.2f}\n"
                        f"  Category: {p.get('category', 'N/A')} | "
                        f"Tags: {', '.join(p.get('tags', []))}\n"
                    )
                return {
                    "response": "\n".join(lines),
                    "products": products,
                    "tools_used": ["catalog_search"],
                    "agent": self.AGENT_NAME,
                }

        # Fall back to search
        return self._handle_search(query, context)

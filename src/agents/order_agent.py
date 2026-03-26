"""Order agent for handling order status, returns, and delivery tracking."""

from __future__ import annotations

from typing import Any

import structlog

from src.governance.audit_logger import audit_logger
from src.tools.order_management import order_management_tool

logger = structlog.get_logger(__name__)


class OrderAgent:
    """Specialized agent for order-related queries.

    Handles order status lookups, return initiation, delivery tracking,
    and order history retrieval.
    """

    AGENT_NAME = "order_agent"

    def handle(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Handle an order-related query.

        Args:
            query: User's order query.
            context: Optional context with extracted entities (order_id, customer_id).

        Returns:
            Agent response with order info, response text, and tools used.
        """
        context = context or {}
        query_lower = query.lower()

        if any(kw in query_lower for kw in ["return", "refund", "send back"]):
            return self._handle_return(query, context)
        elif any(kw in query_lower for kw in ["track", "delivery", "where", "shipping"]):
            return self._handle_tracking(query, context)
        elif any(kw in query_lower for kw in ["status", "order", "placed"]):
            return self._handle_status(query, context)
        else:
            return self._handle_status(query, context)

    def _handle_status(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        """Handle order status queries."""
        order_id = context.get("order_id")

        if not order_id:
            return {
                "response": "Could you please provide your order ID? "
                "It usually starts with 'ORD-' followed by numbers.",
                "data": None,
                "tools_used": [],
                "agent": self.AGENT_NAME,
            }

        status = order_management_tool.get_order_status(order_id)
        audit_logger.log_tool_call(
            agent_name=self.AGENT_NAME,
            tool_name="order_management.get_status",
            tool_input={"order_id": order_id},
            tool_output={"found": "error" not in status},
        )

        if "error" in status:
            return {
                "response": f"I couldn't find order '{order_id}'. "
                "Please double-check the order ID and try again.",
                "data": None,
                "tools_used": ["order_management"],
                "agent": self.AGENT_NAME,
            }

        items_list = ", ".join(
            f"{item['name']} (x{item.get('quantity', 1)})" for item in status.get("items", [])
        )

        response_text = (
            f"**Order {status['order_id']}**\n"
            f"- Status: {status['status'].replace('_', ' ').title()}\n"
            f"- Items: {items_list}\n"
            f"- Total: ${status.get('total', 0):.2f}\n"
        )

        if status.get("estimated_delivery"):
            response_text += f"- Estimated Delivery: {status['estimated_delivery']}\n"
        if status.get("tracking_number"):
            response_text += f"- Tracking: {status['tracking_number']}\n"

        return {
            "response": response_text,
            "data": status,
            "tools_used": ["order_management"],
            "agent": self.AGENT_NAME,
        }

    def _handle_return(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        """Handle return/refund requests."""
        order_id = context.get("order_id")
        reason = context.get("return_reason", "Customer requested return")

        if not order_id:
            return {
                "response": "To process a return, I need your order ID. "
                "Could you please provide it?",
                "data": None,
                "tools_used": [],
                "agent": self.AGENT_NAME,
            }

        result = order_management_tool.initiate_return(order_id, reason)
        audit_logger.log_tool_call(
            agent_name=self.AGENT_NAME,
            tool_name="order_management.initiate_return",
            tool_input={"order_id": order_id, "reason": reason},
            tool_output=result,
        )

        if result.get("success"):
            response_text = (
                f"Your return has been initiated!\n"
                f"- Return ID: {result['return_id']}\n"
                f"- Order: {result['order_id']}\n"
                f"- Instructions: {result['instructions']}"
            )
        else:
            response_text = (
                f"Unable to process the return: {result.get('error', 'Unknown error')}. "
                "Please contact our support team for assistance."
            )

        return {
            "response": response_text,
            "data": result,
            "tools_used": ["order_management"],
            "agent": self.AGENT_NAME,
        }

    def _handle_tracking(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        """Handle delivery tracking queries."""
        order_id = context.get("order_id")

        if not order_id:
            return {
                "response": "I need your order ID to track your delivery. "
                "Could you share it with me?",
                "data": None,
                "tools_used": [],
                "agent": self.AGENT_NAME,
            }

        tracking = order_management_tool.track_delivery(order_id)
        audit_logger.log_tool_call(
            agent_name=self.AGENT_NAME,
            tool_name="order_management.track_delivery",
            tool_input={"order_id": order_id},
            tool_output={"found": "error" not in tracking},
        )

        if "error" in tracking:
            return {
                "response": f"I couldn't find tracking info for order '{order_id}'. "
                "Please verify the order ID.",
                "data": None,
                "tools_used": ["order_management"],
                "agent": self.AGENT_NAME,
            }

        response_text = (
            f"**Delivery Tracking for {tracking['order_id']}**\n"
            f"- Carrier: {tracking.get('carrier', 'N/A')}\n"
            f"- Status: {tracking['status'].replace('_', ' ').title()}\n"
        )

        if tracking.get("tracking_number"):
            response_text += f"- Tracking Number: {tracking['tracking_number']}\n"
        if tracking.get("estimated_delivery"):
            response_text += f"- Estimated Delivery: {tracking['estimated_delivery']}\n"

        history = tracking.get("tracking_history", [])
        if history:
            response_text += "\n**Tracking History:**\n"
            for event in history:
                response_text += f"  - {event.get('date', 'N/A')}: {event.get('status', 'N/A')}\n"

        return {
            "response": response_text,
            "data": tracking,
            "tools_used": ["order_management"],
            "agent": self.AGENT_NAME,
        }

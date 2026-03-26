"""Order management tool for order lifecycle operations."""

from __future__ import annotations

from typing import Any

import structlog

from src.data.order_store import order_store

logger = structlog.get_logger(__name__)


class OrderManagementTool:
    """Handles order status queries, return initiation, and delivery tracking.

    Wraps the order store with logging, validation, and structured responses
    for use by the order agent.
    """

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        """Get current status and details for an order.

        Args:
            order_id: The order identifier.

        Returns:
            Order status details or error dict.
        """
        logger.info("order_status_query", order_id=order_id)

        status = order_store.get_order_status(order_id)
        if not status:
            logger.warning("order_not_found", order_id=order_id)
            return {"error": f"Order '{order_id}' not found", "order_id": order_id}

        logger.info("order_status_retrieved", order_id=order_id, status=status["status"])
        return status

    def initiate_return(self, order_id: str, reason: str) -> dict[str, Any]:
        """Initiate a return for a delivered or shipped order.

        Args:
            order_id: The order to return.
            reason: Customer-provided reason for the return.

        Returns:
            Return confirmation or error details.
        """
        logger.info("return_initiated", order_id=order_id, reason=reason)

        result = order_store.initiate_return(order_id, reason)
        if not result:
            logger.warning("return_order_not_found", order_id=order_id)
            return {"success": False, "error": f"Order '{order_id}' not found"}

        if result.get("success"):
            logger.info("return_created", return_id=result["return_id"])
        else:
            logger.warning("return_failed", order_id=order_id, error=result.get("error"))

        return result

    def track_delivery(self, order_id: str) -> dict[str, Any]:
        """Get delivery tracking information for an order.

        Args:
            order_id: The order identifier.

        Returns:
            Tracking details or error dict.
        """
        logger.info("delivery_tracking_query", order_id=order_id)

        tracking = order_store.track_delivery(order_id)
        if not tracking:
            logger.warning("tracking_order_not_found", order_id=order_id)
            return {"error": f"Order '{order_id}' not found", "order_id": order_id}

        logger.info("delivery_tracking_retrieved", order_id=order_id, status=tracking["status"])
        return tracking

    def get_customer_orders(self, customer_id: str) -> list[dict[str, Any]]:
        """Retrieve all orders for a given customer.

        Args:
            customer_id: The customer identifier.

        Returns:
            List of orders belonging to the customer.
        """
        logger.info("customer_orders_query", customer_id=customer_id)
        orders = order_store.get_orders_by_customer(customer_id)
        logger.info("customer_orders_retrieved", customer_id=customer_id, count=len(orders))
        return orders


# Module-level singleton
order_management_tool = OrderManagementTool()

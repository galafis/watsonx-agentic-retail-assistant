"""In-memory order store with sample data for order management operations."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class OrderStore:
    """In-memory order store backed by a JSON file.

    Provides order lookup, status tracking, return initiation, and delivery info.
    In production, this would be backed by a PostgreSQL database.
    """

    def __init__(self, data_path: str | None = None) -> None:
        self._data_path = Path(
            data_path or Path(__file__).parent.parent.parent / "data" / "sample_orders" / "orders.json"
        )
        self._orders: list[dict[str, Any]] = []
        self._load_orders()

    def _load_orders(self) -> None:
        """Load orders from JSON file."""
        if self._data_path.exists():
            with open(self._data_path) as f:
                self._orders = json.load(f)
            logger.info("orders_loaded", count=len(self._orders))
        else:
            logger.warning("order_store_not_found", path=str(self._data_path))

    def get_order(self, order_id: str) -> dict[str, Any] | None:
        """Retrieve a single order by ID."""
        for order in self._orders:
            if order["id"] == order_id:
                return order
        return None

    def get_orders_by_customer(self, customer_id: str) -> list[dict[str, Any]]:
        """Retrieve all orders for a customer."""
        return [o for o in self._orders if o.get("customer_id") == customer_id]

    def get_order_status(self, order_id: str) -> dict[str, Any] | None:
        """Get the current status and tracking info for an order."""
        order = self.get_order(order_id)
        if not order:
            return None
        return {
            "order_id": order["id"],
            "status": order["status"],
            "tracking_number": order.get("tracking_number"),
            "estimated_delivery": order.get("estimated_delivery"),
            "items": order.get("items", []),
            "total": order.get("total"),
        }

    def initiate_return(self, order_id: str, reason: str) -> dict[str, Any] | None:
        """Initiate a return for an order.

        Args:
            order_id: The order to return.
            reason: Customer-provided return reason.

        Returns:
            Return confirmation details or None if order not found.
        """
        order = self.get_order(order_id)
        if not order:
            return None

        if order["status"] not in ("delivered", "shipped"):
            return {
                "success": False,
                "error": f"Cannot return order with status '{order['status']}'",
                "order_id": order_id,
            }

        return_id = f"RET-{order_id}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')}"
        return {
            "success": True,
            "return_id": return_id,
            "order_id": order_id,
            "reason": reason,
            "status": "return_initiated",
            "instructions": "Please ship the item back within 14 days using the prepaid label.",
        }

    def track_delivery(self, order_id: str) -> dict[str, Any] | None:
        """Get delivery tracking details for an order."""
        order = self.get_order(order_id)
        if not order:
            return None

        tracking = order.get("tracking", [])
        return {
            "order_id": order["id"],
            "tracking_number": order.get("tracking_number"),
            "carrier": order.get("carrier", "Standard Shipping"),
            "status": order["status"],
            "estimated_delivery": order.get("estimated_delivery"),
            "tracking_history": tracking,
        }


# Module-level singleton
order_store = OrderStore()

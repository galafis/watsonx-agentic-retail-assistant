"""Tests for ProductAgent search, comparison, and details."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agents.product_agent import ProductAgent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def agent() -> ProductAgent:
    """Create a ProductAgent with audit logger mocked."""
    with patch("src.agents.product_agent.audit_logger"):
        return ProductAgent()


# ---------------------------------------------------------------------------
# Sub-intent routing via handle()
# ---------------------------------------------------------------------------

class TestProductAgentRouting:
    """Verify handle() dispatches to the correct sub-handler based on keywords."""

    @patch("src.agents.product_agent.catalog_search_tool")
    @patch("src.agents.product_agent.audit_logger")
    def test_compare_keyword_routes_to_comparison(
        self, mock_audit: MagicMock, mock_catalog: MagicMock,
    ) -> None:
        mock_catalog.compare_products.return_value = [
            {"id": "PROD-1", "name": "Item A", "price": 29.99, "category": "Electronics", "tags": []},
            {"id": "PROD-2", "name": "Item B", "price": 39.99, "category": "Electronics", "tags": []},
        ]
        agent = ProductAgent()
        result = agent.handle(
            "Compare PROD-1 vs PROD-2",
            context={"product_ids": ["PROD-1", "PROD-2"]},
        )
        assert result["agent"] == "product_agent"
        assert len(result["products"]) == 2

    @patch("src.agents.product_agent.catalog_search_tool")
    @patch("src.agents.product_agent.audit_logger")
    def test_versus_keyword_routes_to_comparison(
        self, mock_audit: MagicMock, mock_catalog: MagicMock,
    ) -> None:
        mock_catalog.compare_products.return_value = [
            {"id": "P1", "name": "A", "price": 10.0, "category": "C", "tags": []},
            {"id": "P2", "name": "B", "price": 20.0, "category": "C", "tags": []},
        ]
        agent = ProductAgent()
        result = agent.handle(
            "Show me difference between P1 and P2",
            context={"product_ids": ["P1", "P2"]},
        )
        assert result["agent"] == "product_agent"

    @patch("src.agents.product_agent.catalog_search_tool")
    @patch("src.agents.product_agent.audit_logger")
    def test_detail_keyword_routes_to_details(
        self, mock_audit: MagicMock, mock_catalog: MagicMock,
    ) -> None:
        mock_catalog.get_product_details.return_value = {
            "id": "PROD-1", "name": "Widget", "price": 19.99,
            "category": "Gadgets", "description": "A useful widget",
            "tags": ["gadget", "tool"],
        }
        agent = ProductAgent()
        result = agent.handle(
            "Tell me about PROD-1",
            context={"product_id": "PROD-1"},
        )
        assert "Widget" in result["response"]
        assert result["agent"] == "product_agent"

    @patch("src.agents.product_agent.catalog_search_tool")
    @patch("src.agents.product_agent.audit_logger")
    def test_generic_query_routes_to_search(
        self, mock_audit: MagicMock, mock_catalog: MagicMock,
    ) -> None:
        mock_catalog.search.return_value = [
            {"id": "P10", "name": "Laptop Stand", "price": 45.00, "description": "Adjustable stand"},
        ]
        agent = ProductAgent()
        result = agent.handle("I need a laptop stand")
        assert "found" in result["response"].lower()
        assert result["tools_used"] == ["catalog_search"]


# ---------------------------------------------------------------------------
# Search handler (_handle_search)
# ---------------------------------------------------------------------------

class TestProductSearch:
    """Test _handle_search() calls catalog_search_tool and formats output."""

    @patch("src.agents.product_agent.catalog_search_tool")
    @patch("src.agents.product_agent.audit_logger")
    def test_search_returns_formatted_results(
        self, mock_audit: MagicMock, mock_catalog: MagicMock,
    ) -> None:
        mock_catalog.search.return_value = [
            {"id": "P1", "name": "Running Shoes", "price": 89.99, "description": "Comfortable running shoes"},
            {"id": "P2", "name": "Trail Shoes", "price": 109.99, "description": "Durable trail shoes"},
        ]
        agent = ProductAgent()
        result = agent.handle("Find running shoes")

        assert "2 products" in result["response"]
        assert "Running Shoes" in result["response"]
        assert "$89.99" in result["response"]
        assert result["tools_used"] == ["catalog_search"]

    @patch("src.agents.product_agent.catalog_search_tool")
    @patch("src.agents.product_agent.audit_logger")
    def test_search_empty_results(
        self, mock_audit: MagicMock, mock_catalog: MagicMock,
    ) -> None:
        mock_catalog.search.return_value = []
        agent = ProductAgent()
        result = agent.handle("Find antigravity boots")

        assert "couldn't find" in result["response"].lower()
        assert result["tools_used"] == ["catalog_search"]

    @patch("src.agents.product_agent.catalog_search_tool")
    @patch("src.agents.product_agent.audit_logger")
    def test_search_passes_category_filter(
        self, mock_audit: MagicMock, mock_catalog: MagicMock,
    ) -> None:
        mock_catalog.search.return_value = []
        agent = ProductAgent()
        agent.handle("Find shoes", context={"category": "Footwear"})

        mock_catalog.search.assert_called_once_with(
            query="Find shoes",
            category="Footwear",
            min_price=None,
            max_price=None,
        )

    @patch("src.agents.product_agent.catalog_search_tool")
    @patch("src.agents.product_agent.audit_logger")
    def test_search_passes_price_filters(
        self, mock_audit: MagicMock, mock_catalog: MagicMock,
    ) -> None:
        mock_catalog.search.return_value = []
        agent = ProductAgent()
        agent.handle(
            "Find jackets",
            context={"min_price": 50.0, "max_price": 200.0},
        )
        mock_catalog.search.assert_called_once_with(
            query="Find jackets",
            category=None,
            min_price=50.0,
            max_price=200.0,
        )

    @patch("src.agents.product_agent.catalog_search_tool")
    @patch("src.agents.product_agent.audit_logger")
    def test_search_single_result(
        self, mock_audit: MagicMock, mock_catalog: MagicMock,
    ) -> None:
        mock_catalog.search.return_value = [
            {"id": "P1", "name": "USB Hub", "price": 24.99, "description": "4-port USB hub"},
        ]
        agent = ProductAgent()
        result = agent.handle("usb hub")

        assert "1 products" in result["response"] or "1 product" in result["response"]
        assert "USB Hub" in result["response"]


# ---------------------------------------------------------------------------
# Details handler (_handle_details)
# ---------------------------------------------------------------------------

class TestProductDetails:
    """Test _handle_details() for single product lookup."""

    @patch("src.agents.product_agent.catalog_search_tool")
    @patch("src.agents.product_agent.audit_logger")
    def test_details_with_valid_product_id(
        self, mock_audit: MagicMock, mock_catalog: MagicMock,
    ) -> None:
        mock_catalog.get_product_details.return_value = {
            "id": "PROD-5", "name": "Smart Watch", "price": 299.99,
            "category": "Electronics", "description": "Feature-rich smartwatch",
            "tags": ["watch", "smart", "wearable"],
        }
        agent = ProductAgent()
        result = agent.handle(
            "Tell me about this product",
            context={"product_id": "PROD-5"},
        )

        assert "Smart Watch" in result["response"]
        assert "$299.99" in result["response"]
        assert "wearable" in result["response"]
        assert result["products"] == [mock_catalog.get_product_details.return_value]

    @patch("src.agents.product_agent.catalog_search_tool")
    @patch("src.agents.product_agent.audit_logger")
    def test_details_falls_back_to_search_when_no_id(
        self, mock_audit: MagicMock, mock_catalog: MagicMock,
    ) -> None:
        mock_catalog.search.return_value = []
        agent = ProductAgent()
        result = agent.handle("Tell me about cool gadgets")

        mock_catalog.search.assert_called_once()
        assert "couldn't find" in result["response"].lower()

    @patch("src.agents.product_agent.catalog_search_tool")
    @patch("src.agents.product_agent.audit_logger")
    def test_details_falls_back_when_product_not_found(
        self, mock_audit: MagicMock, mock_catalog: MagicMock,
    ) -> None:
        mock_catalog.get_product_details.return_value = None
        mock_catalog.search.return_value = []
        agent = ProductAgent()
        agent.handle(
            "Tell me about this item",
            context={"product_id": "PROD-NONEXIST"},
        )
        mock_catalog.search.assert_called_once()


# ---------------------------------------------------------------------------
# Comparison handler (_handle_comparison)
# ---------------------------------------------------------------------------

class TestProductComparison:
    """Test _handle_comparison() with two products."""

    @patch("src.agents.product_agent.catalog_search_tool")
    @patch("src.agents.product_agent.audit_logger")
    def test_comparison_with_two_products(
        self, mock_audit: MagicMock, mock_catalog: MagicMock,
    ) -> None:
        mock_catalog.compare_products.return_value = [
            {"id": "P1", "name": "Phone A", "price": 799.0, "category": "Phones", "tags": ["5G"]},
            {"id": "P2", "name": "Phone B", "price": 899.0, "category": "Phones", "tags": ["5G"]},
        ]
        agent = ProductAgent()
        result = agent.handle(
            "Compare these phones",
            context={"product_ids": ["P1", "P2"]},
        )

        assert "Phone A" in result["response"]
        assert "Phone B" in result["response"]
        assert len(result["products"]) == 2

    @patch("src.agents.product_agent.catalog_search_tool")
    @patch("src.agents.product_agent.audit_logger")
    def test_comparison_falls_back_with_fewer_than_two_ids(
        self, mock_audit: MagicMock, mock_catalog: MagicMock,
    ) -> None:
        mock_catalog.search.return_value = []
        agent = ProductAgent()
        result = agent.handle(
            "Compare products",
            context={"product_ids": ["P1"]},
        )
        mock_catalog.search.assert_called_once()

    @patch("src.agents.product_agent.catalog_search_tool")
    @patch("src.agents.product_agent.audit_logger")
    def test_comparison_falls_back_with_empty_ids(
        self, mock_audit: MagicMock, mock_catalog: MagicMock,
    ) -> None:
        mock_catalog.search.return_value = []
        agent = ProductAgent()
        result = agent.handle("Compare products", context={})
        mock_catalog.search.assert_called_once()

    @patch("src.agents.product_agent.catalog_search_tool")
    @patch("src.agents.product_agent.audit_logger")
    def test_comparison_calls_compare_products(
        self, mock_audit: MagicMock, mock_catalog: MagicMock,
    ) -> None:
        mock_catalog.compare_products.return_value = [
            {"id": "A", "name": "A", "price": 10.0, "category": "X", "tags": []},
            {"id": "B", "name": "B", "price": 20.0, "category": "X", "tags": []},
        ]
        agent = ProductAgent()
        agent.handle(
            "Compare A vs B",
            context={"product_ids": ["A", "B"]},
        )
        mock_catalog.compare_products.assert_called_once_with(["A", "B"])

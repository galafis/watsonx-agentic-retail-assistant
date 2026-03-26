"""In-memory product catalog with sample data and search capabilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class ProductCatalog:
    """In-memory product catalog backed by a JSON file.

    Provides keyword search, filtering by category/price, and product retrieval.
    In production, this would be backed by a database or search index.
    """

    def __init__(self, data_path: str | None = None) -> None:
        self._data_path = Path(
            data_path
            or Path(__file__).parent.parent.parent / "data" / "sample_catalog" / "products.json"
        )
        self._products: list[dict[str, Any]] = []
        self._load_products()

    def _load_products(self) -> None:
        """Load products from JSON file."""
        if self._data_path.exists():
            with open(self._data_path) as f:
                self._products = json.load(f)
            logger.info("products_loaded", count=len(self._products))
        else:
            logger.warning("product_catalog_not_found", path=str(self._data_path))

    def get_product(self, product_id: str) -> dict[str, Any] | None:
        """Retrieve a single product by ID."""
        for product in self._products:
            if product["id"] == product_id:
                return product
        return None

    def search(
        self,
        query: str,
        category: str | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Search products by keyword with optional filters.

        Args:
            query: Search query string matched against name, description, and tags.
            category: Optional category filter.
            min_price: Optional minimum price filter.
            max_price: Optional maximum price filter.
            limit: Maximum number of results to return.

        Returns:
            List of matching products sorted by relevance score.
        """
        query_lower = query.lower()
        query_terms = query_lower.split()
        results: list[tuple[float, dict[str, Any]]] = []

        for product in self._products:
            # Apply category filter
            if category and product.get("category", "").lower() != category.lower():
                continue

            # Apply price filters
            price = product.get("price", 0)
            if min_price is not None and price < min_price:
                continue
            if max_price is not None and price > max_price:
                continue

            # Compute keyword relevance score
            score = self._compute_relevance(query_terms, product)
            if score > 0:
                results.append((score, product))

        # Sort by relevance descending
        results.sort(key=lambda x: x[0], reverse=True)
        return [product for _, product in results[:limit]]

    def get_categories(self) -> list[str]:
        """Return all unique product categories."""
        return sorted({p.get("category", "unknown") for p in self._products})

    def get_all_products(self) -> list[dict[str, Any]]:
        """Return all products in the catalog."""
        return self._products.copy()

    @staticmethod
    def _compute_relevance(query_terms: list[str], product: dict[str, Any]) -> float:
        """Compute a simple keyword relevance score for a product."""
        score = 0.0
        name = product.get("name", "").lower()
        description = product.get("description", "").lower()
        tags = [t.lower() for t in product.get("tags", [])]
        category = product.get("category", "").lower()

        for term in query_terms:
            if term in name:
                score += 3.0
            if term in description:
                score += 1.0
            if any(term in tag for tag in tags):
                score += 2.0
            if term in category:
                score += 1.5

        return score


# Module-level singleton
catalog = ProductCatalog()

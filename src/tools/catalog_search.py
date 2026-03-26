"""Product catalog search tool with hybrid semantic + keyword retrieval."""

from __future__ import annotations

from typing import Any

import structlog

from src.config import settings
from src.data.product_catalog import catalog

logger = structlog.get_logger(__name__)


class CatalogSearchTool:
    """Hybrid catalog search combining semantic similarity and keyword matching.

    Uses keyword matching on the in-memory catalog. In production, integrates
    with watsonx.ai embeddings for semantic search over a vector store.
    """

    def __init__(self) -> None:
        tools_cfg = settings.tools_config.get("catalog_search", {})
        self._hybrid_alpha: float = tools_cfg.get("hybrid_alpha", 0.7)
        self._vector_top_k: int = tools_cfg.get("vector_top_k", 10)
        self._keyword_top_k: int = tools_cfg.get("keyword_top_k", 10)

    def search(
        self,
        query: str,
        category: str | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Search the product catalog with hybrid retrieval.

        Args:
            query: Natural language search query.
            category: Optional category filter.
            min_price: Optional minimum price filter.
            max_price: Optional maximum price filter.
            limit: Maximum results to return.

        Returns:
            List of matching products with relevance metadata.
        """
        logger.info(
            "catalog_search",
            query=query,
            category=category,
            min_price=min_price,
            max_price=max_price,
        )

        # Keyword search on in-memory catalog
        results = catalog.search(
            query=query,
            category=category,
            min_price=min_price,
            max_price=max_price,
            limit=limit,
        )

        logger.info("catalog_search_results", count=len(results), query=query)
        return results

    def get_product_details(self, product_id: str) -> dict[str, Any] | None:
        """Get detailed information about a specific product.

        Args:
            product_id: The product identifier.

        Returns:
            Product details dict or None if not found.
        """
        product = catalog.get_product(product_id)
        if product:
            logger.info("product_details_retrieved", product_id=product_id)
        else:
            logger.warning("product_not_found", product_id=product_id)
        return product

    def compare_products(self, product_ids: list[str]) -> list[dict[str, Any]]:
        """Compare multiple products side by side.

        Args:
            product_ids: List of product IDs to compare.

        Returns:
            List of product details for comparison.
        """
        products = []
        for pid in product_ids:
            product = catalog.get_product(pid)
            if product:
                products.append(product)

        logger.info(
            "products_compared",
            requested=len(product_ids),
            found=len(products),
        )
        return products

    def get_categories(self) -> list[str]:
        """Return all available product categories."""
        return catalog.get_categories()


# Module-level singleton
catalog_search_tool = CatalogSearchTool()

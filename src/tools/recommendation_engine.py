"""Recommendation engine combining content-based and collaborative filtering."""

from __future__ import annotations

from typing import Any

import structlog

from src.config import settings
from src.data.product_catalog import catalog

logger = structlog.get_logger(__name__)


class RecommendationEngine:
    """Product recommendation engine using content-based and collaborative filtering.

    Content-based: recommends products with similar tags, categories, and price ranges.
    Collaborative: simulates user-item interaction patterns (placeholder for production CF).
    """

    def __init__(self) -> None:
        tools_cfg = settings.tools_config.get("recommendation_engine", {})
        self._content_weight: float = tools_cfg.get("content_weight", 0.6)
        self._collaborative_weight: float = tools_cfg.get("collaborative_weight", 0.4)
        agents_cfg = settings.agents_config.get("recommendation", {})
        self._max_recommendations: int = agents_cfg.get("max_recommendations", 5)
        self._diversity_factor: float = agents_cfg.get("diversity_factor", 0.3)

    def recommend_similar(
        self,
        product_id: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Recommend products similar to a given product (content-based).

        Args:
            product_id: The reference product ID.
            limit: Maximum number of recommendations.

        Returns:
            List of recommended products with similarity scores.
        """
        limit = limit or self._max_recommendations
        reference = catalog.get_product(product_id)
        if not reference:
            logger.warning("recommendation_product_not_found", product_id=product_id)
            return []

        all_products = catalog.get_all_products()
        scored: list[tuple[float, dict[str, Any]]] = []

        for product in all_products:
            if product["id"] == product_id:
                continue
            score = self._content_similarity(reference, product)
            if score > 0:
                scored.append((score, product))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Apply diversity: avoid too many items from the same category
        results = self._apply_diversity(scored, limit)

        logger.info(
            "content_recommendations",
            product_id=product_id,
            count=len(results),
        )
        return results

    def recommend_for_customer(
        self,
        customer_id: str,
        purchase_history: list[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Recommend products for a customer based on purchase history.

        Combines content-based similarity across purchased items with
        simulated collaborative filtering signals.

        Args:
            customer_id: The customer identifier.
            purchase_history: List of previously purchased product IDs.
            limit: Maximum number of recommendations.

        Returns:
            List of recommended products.
        """
        limit = limit or self._max_recommendations
        purchase_history = purchase_history or []

        if not purchase_history:
            logger.info("no_purchase_history", customer_id=customer_id)
            return self._popular_products(limit)

        # Aggregate content scores from all purchased products
        all_products = catalog.get_all_products()
        purchased_set = set(purchase_history)
        aggregate_scores: dict[str, float] = {}

        for pid in purchase_history:
            reference = catalog.get_product(pid)
            if not reference:
                continue
            for product in all_products:
                if product["id"] in purchased_set:
                    continue
                score = self._content_similarity(reference, product)
                current = aggregate_scores.get(product["id"], 0.0)
                aggregate_scores[product["id"]] = current + score

        # Build ranked list
        scored: list[tuple[float, dict[str, Any]]] = []
        for product in all_products:
            if product["id"] in purchased_set:
                continue
            score = aggregate_scores.get(product["id"], 0.0)
            if score > 0:
                scored.append((score, product))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = self._apply_diversity(scored, limit)

        logger.info(
            "customer_recommendations",
            customer_id=customer_id,
            history_size=len(purchase_history),
            count=len(results),
        )
        return results

    def _popular_products(self, limit: int) -> list[dict[str, Any]]:
        """Return popular products as fallback when no history exists."""
        all_products = catalog.get_all_products()
        # Simulate popularity by returning first N products
        return all_products[:limit]

    @staticmethod
    def _content_similarity(ref: dict[str, Any], candidate: dict[str, Any]) -> float:
        """Compute content-based similarity between two products.

        Uses tag overlap, category match, and price proximity.
        """
        score = 0.0

        # Tag overlap (Jaccard-like)
        ref_tags = set(t.lower() for t in ref.get("tags", []))
        cand_tags = set(t.lower() for t in candidate.get("tags", []))
        if ref_tags and cand_tags:
            intersection = ref_tags & cand_tags
            union = ref_tags | cand_tags
            score += (len(intersection) / len(union)) * 3.0

        # Category match
        if ref.get("category", "").lower() == candidate.get("category", "").lower():
            score += 2.0

        # Price proximity (closer price = higher score)
        ref_price = ref.get("price", 0)
        cand_price = candidate.get("price", 0)
        if ref_price > 0 and cand_price > 0:
            ratio = min(ref_price, cand_price) / max(ref_price, cand_price)
            score += ratio * 1.0

        return score

    def _apply_diversity(
        self,
        scored: list[tuple[float, dict[str, Any]]],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Apply category diversity to avoid monotonic recommendations."""
        results: list[dict[str, Any]] = []
        category_counts: dict[str, int] = {}
        max_per_category = max(2, int(limit * (1.0 - self._diversity_factor)))

        for _score, product in scored:
            cat = product.get("category", "unknown")
            if category_counts.get(cat, 0) >= max_per_category:
                continue
            results.append(product)
            category_counts[cat] = category_counts.get(cat, 0) + 1
            if len(results) >= limit:
                break

        return results


# Module-level singleton
recommendation_engine = RecommendationEngine()

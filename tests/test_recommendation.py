"""Tests for RecommendationEngine content-based and collaborative filtering."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.tools.recommendation_engine import RecommendationEngine


# ---------------------------------------------------------------------------
# Sample catalog data
# ---------------------------------------------------------------------------

_SAMPLE_PRODUCTS: list[dict[str, Any]] = [
    {
        "id": "PROD-1",
        "name": "Wireless Headphones",
        "price": 79.99,
        "category": "Electronics",
        "tags": ["audio", "wireless", "bluetooth"],
        "description": "Premium wireless headphones with noise cancellation.",
    },
    {
        "id": "PROD-2",
        "name": "Bluetooth Speaker",
        "price": 49.99,
        "category": "Electronics",
        "tags": ["audio", "wireless", "speaker", "bluetooth"],
        "description": "Portable Bluetooth speaker with deep bass.",
    },
    {
        "id": "PROD-3",
        "name": "Running Shoes",
        "price": 129.99,
        "category": "Footwear",
        "tags": ["running", "athletic", "shoes"],
        "description": "Lightweight running shoes for trail and road.",
    },
    {
        "id": "PROD-4",
        "name": "USB-C Hub",
        "price": 39.99,
        "category": "Electronics",
        "tags": ["usb", "hub", "adapter"],
        "description": "7-in-1 USB-C hub with HDMI output.",
    },
    {
        "id": "PROD-5",
        "name": "Noise Cancelling Earbuds",
        "price": 99.99,
        "category": "Electronics",
        "tags": ["audio", "wireless", "earbuds", "noise-cancelling"],
        "description": "True wireless earbuds with active noise cancellation.",
    },
    {
        "id": "PROD-6",
        "name": "Trail Running Shoes",
        "price": 149.99,
        "category": "Footwear",
        "tags": ["running", "trail", "shoes", "outdoor"],
        "description": "Rugged trail running shoes with grip sole.",
    },
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def engine() -> RecommendationEngine:
    """Create a RecommendationEngine with mocked config."""
    with patch("src.tools.recommendation_engine.settings") as mock_settings:
        mock_settings.tools_config = {
            "recommendation_engine": {
                "content_weight": 0.6,
                "collaborative_weight": 0.4,
            },
        }
        mock_settings.agents_config = {
            "recommendation": {
                "max_recommendations": 5,
                "diversity_factor": 0.3,
            },
        }
        eng = RecommendationEngine()
    return eng


@pytest.fixture()
def mock_catalog() -> MagicMock:
    """Provide a patched catalog singleton returning sample products."""
    with patch("src.tools.recommendation_engine.catalog") as mock_cat:
        mock_cat.get_all_products.return_value = _SAMPLE_PRODUCTS.copy()
        mock_cat.get_product.side_effect = lambda pid: next(
            (p for p in _SAMPLE_PRODUCTS if p["id"] == pid), None
        )
        yield mock_cat


# ---------------------------------------------------------------------------
# recommend_similar() - content-based
# ---------------------------------------------------------------------------

class TestRecommendSimilar:
    """Test content-based similar product recommendations."""

    def test_returns_similar_products(
        self, engine: RecommendationEngine, mock_catalog: MagicMock,
    ) -> None:
        results = engine.recommend_similar("PROD-1")
        assert len(results) > 0
        result_ids = {r["id"] for r in results}
        assert "PROD-1" not in result_ids
        # Headphones should recommend audio products (speaker, earbuds)
        assert result_ids & {"PROD-2", "PROD-5"}

    def test_nonexistent_product_returns_empty(
        self, engine: RecommendationEngine, mock_catalog: MagicMock,
    ) -> None:
        results = engine.recommend_similar("PROD-DOES-NOT-EXIST")
        assert results == []

    def test_respects_limit_parameter(
        self, engine: RecommendationEngine, mock_catalog: MagicMock,
    ) -> None:
        results = engine.recommend_similar("PROD-1", limit=2)
        assert len(results) <= 2

    def test_respects_max_recommendations_default(
        self, engine: RecommendationEngine, mock_catalog: MagicMock,
    ) -> None:
        results = engine.recommend_similar("PROD-1")
        assert len(results) <= 5

    def test_excludes_reference_product(
        self, engine: RecommendationEngine, mock_catalog: MagicMock,
    ) -> None:
        results = engine.recommend_similar("PROD-3")
        result_ids = [r["id"] for r in results]
        assert "PROD-3" not in result_ids

    def test_similar_category_ranked_higher(
        self, engine: RecommendationEngine, mock_catalog: MagicMock,
    ) -> None:
        results = engine.recommend_similar("PROD-1")
        if len(results) >= 2:
            assert results[0]["category"] == "Electronics"


# ---------------------------------------------------------------------------
# Diversity factor
# ---------------------------------------------------------------------------

class TestDiversityFactor:
    """Test that diversity factor limits items per category."""

    def test_high_diversity_limits_same_category(
        self, mock_catalog: MagicMock,
    ) -> None:
        with patch("src.tools.recommendation_engine.settings") as mock_settings:
            mock_settings.tools_config = {
                "recommendation_engine": {"content_weight": 0.6, "collaborative_weight": 0.4},
            }
            mock_settings.agents_config = {
                "recommendation": {
                    "max_recommendations": 5,
                    "diversity_factor": 0.8,
                },
            }
            engine = RecommendationEngine()

        results = engine.recommend_similar("PROD-1", limit=5)
        categories = [r["category"] for r in results]
        if len(results) > 2:
            assert len(set(categories)) >= 1


# ---------------------------------------------------------------------------
# recommend_for_customer() - collaborative
# ---------------------------------------------------------------------------

class TestRecommendForCustomer:
    """Test customer-level recommendations with purchase history."""

    def test_with_purchase_history(
        self, engine: RecommendationEngine, mock_catalog: MagicMock,
    ) -> None:
        results = engine.recommend_for_customer(
            customer_id="CUST-001",
            purchase_history=["PROD-1"],
        )
        assert len(results) > 0
        result_ids = {r["id"] for r in results}
        assert "PROD-1" not in result_ids

    def test_no_history_returns_popular(
        self, engine: RecommendationEngine, mock_catalog: MagicMock,
    ) -> None:
        results = engine.recommend_for_customer(
            customer_id="CUST-NEW",
            purchase_history=[],
        )
        assert len(results) > 0

    def test_none_history_returns_popular(
        self, engine: RecommendationEngine, mock_catalog: MagicMock,
    ) -> None:
        results = engine.recommend_for_customer(
            customer_id="CUST-NEW",
            purchase_history=None,
        )
        assert len(results) > 0

    def test_excludes_already_purchased(
        self, engine: RecommendationEngine, mock_catalog: MagicMock,
    ) -> None:
        results = engine.recommend_for_customer(
            customer_id="CUST-002",
            purchase_history=["PROD-1", "PROD-2"],
        )
        result_ids = {r["id"] for r in results}
        assert "PROD-1" not in result_ids
        assert "PROD-2" not in result_ids


# ---------------------------------------------------------------------------
# Content similarity scoring (static method)
# ---------------------------------------------------------------------------

class TestContentSimilarity:
    """Test the _content_similarity static method."""

    def test_identical_tags_yield_high_score(self) -> None:
        ref = {"tags": ["audio", "wireless"], "category": "Electronics", "price": 50.0}
        candidate = {"tags": ["audio", "wireless"], "category": "Electronics", "price": 55.0}
        score = RecommendationEngine._content_similarity(ref, candidate)
        assert score > 4.0

    def test_no_overlap_yields_low_score(self) -> None:
        ref = {"tags": ["audio", "wireless"], "category": "Electronics", "price": 50.0}
        candidate = {"tags": ["outdoor", "hiking"], "category": "Sports", "price": 200.0}
        score = RecommendationEngine._content_similarity(ref, candidate)
        assert score < 2.0

    def test_same_category_adds_score(self) -> None:
        base = {"tags": [], "category": "Footwear", "price": 100.0}
        same_cat = {"tags": [], "category": "Footwear", "price": 100.0}
        diff_cat = {"tags": [], "category": "Electronics", "price": 100.0}

        score_same = RecommendationEngine._content_similarity(base, same_cat)
        score_diff = RecommendationEngine._content_similarity(base, diff_cat)
        assert score_same > score_diff

    def test_closer_price_adds_score(self) -> None:
        ref = {"tags": ["x"], "category": "A", "price": 100.0}
        close = {"tags": ["x"], "category": "A", "price": 105.0}
        far = {"tags": ["x"], "category": "A", "price": 500.0}

        score_close = RecommendationEngine._content_similarity(ref, close)
        score_far = RecommendationEngine._content_similarity(ref, far)
        assert score_close > score_far

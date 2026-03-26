"""FAQ document RAG retriever for customer support queries."""

from __future__ import annotations

from typing import Any

import structlog

from src.config import settings

logger = structlog.get_logger(__name__)

# Built-in FAQ knowledge base for demonstration
_FAQ_ENTRIES: list[dict[str, Any]] = [
    {
        "id": "faq-001",
        "question": "What is your return policy?",
        "answer": "We offer a 30-day return policy for all items in their original condition. "
        "Items must be unworn, unwashed, and with tags attached. "
        "Refunds are processed within 5-7 business days after we receive the return.",
        "category": "returns",
        "tags": ["return", "refund", "policy", "exchange"],
    },
    {
        "id": "faq-002",
        "question": "How long does shipping take?",
        "answer": "Standard shipping takes 5-7 business days. Express shipping takes 2-3 business days. "
        "Same-day delivery is available in select metro areas for orders placed before 12 PM.",
        "category": "shipping",
        "tags": ["shipping", "delivery", "time", "express"],
    },
    {
        "id": "faq-003",
        "question": "Do you offer international shipping?",
        "answer": "Yes, we ship to over 50 countries worldwide. International orders typically "
        "take 10-15 business days. Customs duties and taxes are the responsibility of the buyer.",
        "category": "shipping",
        "tags": ["international", "shipping", "worldwide", "customs"],
    },
    {
        "id": "faq-004",
        "question": "How can I track my order?",
        "answer": "You can track your order using the tracking number provided in your shipping "
        "confirmation email. Visit our order tracking page or use the chat to check your order status.",
        "category": "orders",
        "tags": ["track", "order", "status", "tracking"],
    },
    {
        "id": "faq-005",
        "question": "What payment methods do you accept?",
        "answer": "We accept Visa, MasterCard, American Express, PayPal, Apple Pay, and Google Pay. "
        "We also offer installment payments through Affirm for orders over $50.",
        "category": "payment",
        "tags": ["payment", "credit card", "paypal", "pay"],
    },
    {
        "id": "faq-006",
        "question": "How do I change or cancel my order?",
        "answer": "Orders can be modified or cancelled within 1 hour of placement. After that, "
        "the order enters processing and cannot be changed. Contact our support team immediately "
        "if you need to make changes.",
        "category": "orders",
        "tags": ["cancel", "change", "modify", "order"],
    },
    {
        "id": "faq-007",
        "question": "Do you have a loyalty program?",
        "answer": "Yes! Our Rewards Plus program earns you 1 point per dollar spent. "
        "Accumulate 100 points to get $10 off your next purchase. Gold members (500+ points/year) "
        "get free express shipping and early access to sales.",
        "category": "loyalty",
        "tags": ["loyalty", "rewards", "points", "program", "membership"],
    },
    {
        "id": "faq-008",
        "question": "What is your price match guarantee?",
        "answer": "We offer a 14-day price match guarantee. If you find the same item at a lower "
        "price from an authorized retailer, we will match that price. Excludes clearance, "
        "marketplace sellers, and flash sales.",
        "category": "pricing",
        "tags": ["price", "match", "guarantee", "discount"],
    },
]


class FAQRetriever:
    """FAQ retrieval tool using keyword matching and simple scoring.

    For production, this would use watsonx.ai embeddings with ChromaDB
    for semantic retrieval over a larger FAQ corpus.
    """

    def __init__(self) -> None:
        agents_cfg = settings.agents_config.get("support", {})
        self._confidence_threshold: float = agents_cfg.get("faq_confidence_threshold", 0.6)
        self._faq_entries = _FAQ_ENTRIES

    def retrieve(
        self,
        query: str,
        category: str | None = None,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant FAQ entries for a customer query.

        Args:
            query: Customer's question.
            category: Optional FAQ category filter.
            limit: Maximum entries to return.

        Returns:
            List of FAQ entries with relevance scores.
        """
        query_lower = query.lower()
        query_terms = query_lower.split()
        scored: list[tuple[float, dict[str, Any]]] = []

        for faq in self._faq_entries:
            if category and faq.get("category") != category:
                continue

            score = self._compute_score(query_terms, faq)
            if score > 0:
                scored.append((score, {**faq, "relevance_score": round(score, 3)}))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [entry for _, entry in scored[:limit]]

        logger.info("faq_retrieval", query=query, results_count=len(results))
        return results

    def get_faq_by_id(self, faq_id: str) -> dict[str, Any] | None:
        """Retrieve a specific FAQ entry by ID."""
        for faq in self._faq_entries:
            if faq["id"] == faq_id:
                return faq
        return None

    @staticmethod
    def _compute_score(query_terms: list[str], faq: dict[str, Any]) -> float:
        """Compute relevance score for an FAQ entry against query terms."""
        score = 0.0
        question = faq.get("question", "").lower()
        answer = faq.get("answer", "").lower()
        tags = [t.lower() for t in faq.get("tags", [])]

        for term in query_terms:
            if term in question:
                score += 3.0
            if term in answer:
                score += 1.0
            if term in tags:
                score += 2.5

        # Normalize by number of query terms
        if query_terms:
            score = score / len(query_terms)

        return score


# Module-level singleton
faq_retriever = FAQRetriever()

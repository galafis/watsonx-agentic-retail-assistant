"""Response quality scorer based on relevance, completeness, and helpfulness."""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class QualityScorer:
    """Score response quality based on multiple dimensions.

    Evaluates responses on relevance, completeness, specificity,
    and actionability to provide an overall quality metric.
    """

    def score(
        self,
        query: str,
        response: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Score the quality of an agent response.

        Args:
            query: Original user query.
            response: Agent response text.
            context: Optional context (e.g., tools used, sources found).

        Returns:
            Quality assessment with dimension scores and overall score.
        """
        context = context or {}

        relevance = self._score_relevance(query, response)
        completeness = self._score_completeness(response, context)
        specificity = self._score_specificity(response)
        actionability = self._score_actionability(response)

        # Weighted average
        overall = relevance * 0.35 + completeness * 0.25 + specificity * 0.20 + actionability * 0.20

        result = {
            "overall_score": round(overall, 3),
            "dimensions": {
                "relevance": round(relevance, 3),
                "completeness": round(completeness, 3),
                "specificity": round(specificity, 3),
                "actionability": round(actionability, 3),
            },
            "quality_label": self._label_quality(overall),
        }

        logger.info(
            "quality_scored",
            overall=result["overall_score"],
            label=result["quality_label"],
        )
        return result

    @staticmethod
    def _score_relevance(query: str, response: str) -> float:
        """Score how relevant the response is to the query.

        Uses term overlap between query and response as a proxy.
        """
        query_terms = set(query.lower().split())
        response_lower = response.lower()

        if not query_terms:
            return 0.5

        matches = sum(1 for term in query_terms if term in response_lower)
        return min(1.0, matches / max(len(query_terms), 1))

    @staticmethod
    def _score_completeness(response: str, context: dict[str, Any]) -> float:
        """Score how complete the response is.

        Considers response length, whether tools were used, and data presence.
        """
        score = 0.0

        # Response length factor
        word_count = len(response.split())
        if word_count >= 20:
            score += 0.4
        elif word_count >= 10:
            score += 0.2

        # Tools used indicates grounding
        tools_called = context.get("tools_called", [])
        if tools_called:
            score += 0.3

        # Data presence (numbers, specifics)
        if any(char.isdigit() for char in response):
            score += 0.15

        # Has structured information
        if "$" in response or "%" in response:
            score += 0.15

        return min(1.0, score)

    @staticmethod
    def _score_specificity(response: str) -> float:
        """Score how specific vs. generic the response is."""
        # Check for specific indicators
        specificity_markers = ["$", "#", "order", "product", "item", "delivery", "return"]
        lower = response.lower()
        matches = sum(1 for marker in specificity_markers if marker in lower)

        score = min(1.0, matches * 0.2)

        # Penalize very short responses
        if len(response.split()) < 5:
            score *= 0.5

        return score

    @staticmethod
    def _score_actionability(response: str) -> float:
        """Score whether the response provides actionable information."""
        action_phrases = [
            "you can",
            "please",
            "try",
            "visit",
            "click",
            "contact",
            "here's how",
            "to do this",
            "steps",
            "follow",
        ]
        lower = response.lower()
        matches = sum(1 for phrase in action_phrases if phrase in lower)
        return min(1.0, matches * 0.25)

    @staticmethod
    def _label_quality(score: float) -> str:
        """Convert a numeric quality score to a label."""
        if score >= 0.8:
            return "excellent"
        elif score >= 0.6:
            return "good"
        elif score >= 0.4:
            return "adequate"
        elif score >= 0.2:
            return "poor"
        else:
            return "very_poor"


# Module-level singleton
quality_scorer = QualityScorer()

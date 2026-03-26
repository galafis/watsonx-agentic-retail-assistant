"""Customer sentiment analysis for escalation detection."""

from __future__ import annotations

from typing import Any

import structlog

from src.config import settings

logger = structlog.get_logger(__name__)

# Simple keyword-based sentiment lexicon
_POSITIVE_WORDS = {
    "great",
    "excellent",
    "amazing",
    "love",
    "wonderful",
    "perfect",
    "fantastic",
    "awesome",
    "happy",
    "pleased",
    "satisfied",
    "thank",
    "thanks",
    "good",
    "best",
    "helpful",
    "appreciate",
    "recommend",
    "outstanding",
    "superb",
}

_NEGATIVE_WORDS = {
    "terrible",
    "awful",
    "horrible",
    "hate",
    "worst",
    "angry",
    "furious",
    "disappointed",
    "frustrated",
    "annoyed",
    "broken",
    "defective",
    "scam",
    "unacceptable",
    "ridiculous",
    "outrageous",
    "disgusting",
    "pathetic",
    "incompetent",
    "useless",
    "garbage",
    "trash",
    "never",
    "complaint",
}


class SentimentAnalyzer:
    """Analyze customer message sentiment for escalation detection.

    Uses a simple lexicon-based approach. In production, this would use
    watsonx.ai NLP models for more accurate sentiment classification.
    """

    def __init__(self) -> None:
        tools_cfg = settings.tools_config.get("sentiment", {})
        self._negative_threshold: float = tools_cfg.get("negative_threshold", -0.3)
        self._escalation_keywords: list[str] = tools_cfg.get(
            "escalation_keywords",
            [
                "manager",
                "supervisor",
                "complaint",
                "lawyer",
                "sue",
                "unacceptable",
            ],
        )

    def analyze(self, text: str) -> dict[str, Any]:
        """Analyze sentiment of a customer message.

        Args:
            text: Customer message text.

        Returns:
            Sentiment analysis results including score, label, and escalation flag.
        """
        words = text.lower().split()
        positive_count = sum(1 for w in words if w.strip(".,!?;:") in _POSITIVE_WORDS)
        negative_count = sum(1 for w in words if w.strip(".,!?;:") in _NEGATIVE_WORDS)
        total = positive_count + negative_count

        if total == 0:
            score = 0.0
            label = "neutral"
        else:
            score = (positive_count - negative_count) / total
            if score > 0.2:
                label = "positive"
            elif score < -0.2:
                label = "negative"
            else:
                label = "neutral"

        # Check for escalation triggers
        needs_escalation = self._check_escalation(text, score)

        result = {
            "score": round(score, 3),
            "label": label,
            "positive_signals": positive_count,
            "negative_signals": negative_count,
            "needs_escalation": needs_escalation,
        }

        logger.info(
            "sentiment_analyzed",
            label=label,
            score=score,
            needs_escalation=needs_escalation,
        )
        return result

    def _check_escalation(self, text: str, sentiment_score: float) -> bool:
        """Determine if the message should trigger human escalation.

        Escalation is triggered when:
        - Sentiment score is below the negative threshold, OR
        - Message contains explicit escalation keywords.
        """
        text_lower = text.lower()

        # Check escalation keywords
        for keyword in self._escalation_keywords:
            if keyword.lower() in text_lower:
                return True

        # Check sentiment threshold
        return sentiment_score < self._negative_threshold


# Module-level singleton
sentiment_analyzer = SentimentAnalyzer()

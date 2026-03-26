"""LangGraph-based orchestrator that routes queries to specialized agents via intent classification."""

from __future__ import annotations

import re
from typing import Any, TypedDict

import structlog

from src.agents.order_agent import OrderAgent
from src.agents.product_agent import ProductAgent
from src.agents.recommendation_agent import RecommendationAgent
from src.agents.support_agent import SupportAgent
from src.config import settings
from src.governance.audit_logger import audit_logger
from src.governance.guardrails import guardrails
from src.governance.quality_scorer import quality_scorer

logger = structlog.get_logger(__name__)


class AgentState(TypedDict):
    """State managed by the LangGraph orchestrator."""

    user_message: str
    intent: str
    confidence: float
    context: dict[str, Any]
    response: str
    tools_used: list[str]
    agent_used: str
    quality_score: float
    warnings: list[str]
    needs_escalation: bool
    history: list[dict[str, str]]


# Intent classification keywords
_INTENT_KEYWORDS: dict[str, list[str]] = {
    "product": [
        "product",
        "search",
        "find",
        "looking for",
        "show me",
        "browse",
        "category",
        "catalog",
        "item",
        "compare",
        "price",
        "buy",
        "purchase",
    ],
    "order": [
        "order",
        "status",
        "tracking",
        "track",
        "delivery",
        "shipped",
        "return",
        "refund",
        "exchange",
        "cancel",
        "where is my",
    ],
    "recommendation": [
        "recommend",
        "suggest",
        "similar",
        "like this",
        "what should",
        "best",
        "popular",
        "trending",
        "you think",
        "alternatives",
    ],
    "support": [
        "help",
        "support",
        "question",
        "how do",
        "how can",
        "policy",
        "faq",
        "issue",
        "problem",
        "complaint",
        "contact",
        "payment",
        "shipping",
        "warranty",
        "loyalty",
        "account",
    ],
}

# Order ID pattern for entity extraction
_ORDER_ID_PATTERN = re.compile(r"ORD-\d+", re.IGNORECASE)
_PRODUCT_ID_PATTERN = re.compile(r"PROD-\d+", re.IGNORECASE)


class AgentOrchestrator:
    """LangGraph-based state machine that routes queries to specialized agents.

    The orchestrator performs:
    1. Input validation via guardrails
    2. Intent classification using keyword matching (Granite LLM in production)
    3. Entity extraction (order IDs, product IDs)
    4. Routing to the appropriate specialized agent
    5. Output validation and quality scoring
    6. Audit logging of the complete interaction
    """

    def __init__(self) -> None:
        self._product_agent = ProductAgent()
        self._order_agent = OrderAgent()
        self._recommendation_agent = RecommendationAgent()
        self._support_agent = SupportAgent()
        self._max_iterations: int = settings.agents_config.get("orchestrator", {}).get(
            "max_iterations", 10
        )

    def run(self, user_message: str, history: list[dict[str, str]] | None = None) -> AgentState:
        """Execute the agentic workflow for a user message.

        This implements a LangGraph-style state machine with the following nodes:
        - validate_input: Check message safety
        - classify_intent: Determine which agent to route to
        - extract_entities: Pull out order IDs, product IDs, etc.
        - route_to_agent: Execute the appropriate specialized agent
        - validate_output: Check response safety and quality
        - log_interaction: Record to audit trail

        Args:
            user_message: The customer's message.
            history: Optional conversation history.

        Returns:
            Final agent state with response and metadata.
        """
        state: AgentState = {
            "user_message": user_message,
            "intent": "",
            "confidence": 0.0,
            "context": {},
            "response": "",
            "tools_used": [],
            "agent_used": "",
            "quality_score": 0.0,
            "warnings": [],
            "needs_escalation": False,
            "history": history or [],
        }

        # Node 1: Validate input
        state = self._validate_input(state)
        if state["warnings"] and not state["response"] and any(
            "blocked" in w.lower() for w in state["warnings"]
        ):
            state["response"] = (
                "I'm sorry, but I can't process that request. "
                "Please rephrase your question and I'll be happy to help."
            )
            return state

        # Node 2: Classify intent
        state = self._classify_intent(state)

        # Node 3: Extract entities
        state = self._extract_entities(state)

        # Node 4: Route to specialized agent (conditional edge)
        state = self._route_to_agent(state)

        # Node 5: Validate output
        state = self._validate_output(state)

        # Node 6: Score quality
        state = self._score_quality(state)

        # Node 7: Log interaction
        self._log_interaction(state)

        return state

    def _validate_input(self, state: AgentState) -> AgentState:
        """Validate user input through guardrails."""
        validation = guardrails.validate_input(state["user_message"])
        if not validation["safe"]:
            state["warnings"] = validation["warnings"]
            logger.warning("input_validation_failed", warnings=validation["warnings"])
        return state

    def _classify_intent(self, state: AgentState) -> AgentState:
        """Classify the user's intent using keyword matching.

        In production, this would use IBM Granite via watsonx.ai for
        more accurate intent classification with confidence scores.
        """
        message_lower = state["user_message"].lower()
        scores: dict[str, float] = {}

        for intent, keywords in _INTENT_KEYWORDS.items():
            score = sum(1.0 for kw in keywords if kw in message_lower)
            # Normalize by number of keywords for the intent
            scores[intent] = score / len(keywords) if keywords else 0.0

        if scores:
            best_intent = max(scores, key=scores.get)  # type: ignore[arg-type]
            best_score = scores[best_intent]
        else:
            best_intent = "support"
            best_score = 0.0

        # Default to support if confidence is too low
        threshold = settings.agents_config.get("orchestrator", {}).get(
            "intent_confidence_threshold", 0.1
        )
        if best_score < threshold:
            best_intent = "support"

        state["intent"] = best_intent
        state["confidence"] = round(best_score, 3)

        audit_logger.log_agent_action(
            agent_name="orchestrator",
            action_type="intent_classification",
            input_data={"message": state["user_message"]},
            output_data={"intent": best_intent, "confidence": best_score, "all_scores": scores},
        )

        logger.info("intent_classified", intent=best_intent, confidence=best_score)
        return state

    def _extract_entities(self, state: AgentState) -> AgentState:
        """Extract structured entities from the user message."""
        message = state["user_message"]
        context = state["context"].copy()

        # Extract order IDs
        order_match = _ORDER_ID_PATTERN.search(message)
        if order_match:
            context["order_id"] = order_match.group(0).upper()

        # Extract product IDs
        product_match = _PRODUCT_ID_PATTERN.search(message)
        if product_match:
            context["product_id"] = product_match.group(0).upper()

        # Extract multiple product IDs for comparison
        all_product_ids = _PRODUCT_ID_PATTERN.findall(message)
        if len(all_product_ids) >= 2:
            context["product_ids"] = [pid.upper() for pid in all_product_ids]

        state["context"] = context
        return state

    def _route_to_agent(self, state: AgentState) -> AgentState:
        """Route the query to the appropriate specialized agent based on intent."""
        intent = state["intent"]
        query = state["user_message"]
        context = state["context"]

        agent_map = {
            "product": self._product_agent,
            "order": self._order_agent,
            "recommendation": self._recommendation_agent,
            "support": self._support_agent,
        }

        agent = agent_map.get(intent, self._support_agent)

        try:
            result = agent.handle(query, context)
            state["response"] = result.get("response", "")
            state["tools_used"] = result.get("tools_used", [])
            state["agent_used"] = result.get("agent", intent)
            state["needs_escalation"] = result.get("needs_escalation", False)

            logger.info(
                "agent_routed",
                intent=intent,
                agent=state["agent_used"],
                tools=state["tools_used"],
            )
        except Exception as e:
            logger.error("agent_execution_failed", intent=intent, error=str(e))
            state["response"] = (
                "I apologize, but I encountered an issue processing your request. "
                "Please try again or contact our support team."
            )
            state["agent_used"] = "error_fallback"
            state["warnings"].append(f"Agent error: {e!s}")

        return state

    def _validate_output(self, state: AgentState) -> AgentState:
        """Validate the agent's response through guardrails."""
        if not state["response"]:
            return state

        validation = guardrails.validate_output(state["response"])
        state["response"] = validation["response"]
        if validation["warnings"]:
            state["warnings"].extend(validation["warnings"])

        return state

    def _score_quality(self, state: AgentState) -> AgentState:
        """Score the quality of the response."""
        result = quality_scorer.score(
            query=state["user_message"],
            response=state["response"],
            context={
                "tools_called": state["tools_used"],
                "agent_used": state["agent_used"],
            },
        )
        state["quality_score"] = result["overall_score"]
        return state

    def _log_interaction(self, state: AgentState) -> None:
        """Log the complete interaction to the audit trail."""
        audit_logger.log_conversation_turn(
            user_message=state["user_message"],
            assistant_response=state["response"],
            agent_used=state["agent_used"],
            tools_called=state["tools_used"],
            quality_score=state["quality_score"],
        )


def create_orchestrator() -> AgentOrchestrator:
    """Factory function to create a configured orchestrator instance.

    In production, this would initialize the LangGraph StateGraph with
    proper conditional edges and watsonx.ai model bindings:

        from langgraph.graph import StateGraph, END

        graph = StateGraph(AgentState)
        graph.add_node("validate_input", validate_input_node)
        graph.add_node("classify_intent", classify_intent_node)
        graph.add_node("extract_entities", extract_entities_node)
        graph.add_node("product_agent", product_agent_node)
        graph.add_node("order_agent", order_agent_node)
        graph.add_node("recommendation_agent", recommendation_agent_node)
        graph.add_node("support_agent", support_agent_node)
        graph.add_node("validate_output", validate_output_node)
        graph.add_node("score_quality", score_quality_node)

        graph.add_edge("validate_input", "classify_intent")
        graph.add_edge("classify_intent", "extract_entities")
        graph.add_conditional_edges(
            "extract_entities",
            route_by_intent,
            {
                "product": "product_agent",
                "order": "order_agent",
                "recommendation": "recommendation_agent",
                "support": "support_agent",
            },
        )
        for agent_node in ["product_agent", "order_agent", ...]:
            graph.add_edge(agent_node, "validate_output")
        graph.add_edge("validate_output", "score_quality")
        graph.add_edge("score_quality", END)

        graph.set_entry_point("validate_input")
        return graph.compile()
    """
    return AgentOrchestrator()


# Module-level singleton
orchestrator = create_orchestrator()

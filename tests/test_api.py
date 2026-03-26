"""Tests for FastAPI REST endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def _create_test_app() -> FastAPI:
    """Create a FastAPI app mirroring the production route structure.

    This avoids importing the real ``src.api.routes`` module directly
    (which triggers heavy singleton initialisation) and instead sets up
    the same contract so we can validate request/response behaviour in
    isolation.
    """
    from pydantic import BaseModel

    app = FastAPI(title="Watsonx Retail Assistant - Test")

    class QueryRequest(BaseModel):
        query: str
        session_id: str | None = None
        history: list[dict[str, str]] | None = None

    class QueryResponse(BaseModel):
        response: str
        intent: str
        confidence: float
        agent_used: str
        tools_used: list[str]
        quality_score: float

    class HealthResponse(BaseModel):
        status: str
        version: str

    @app.get("/health", response_model=HealthResponse)
    def health() -> dict:
        return {"status": "healthy", "version": "1.0.0"}

    @app.post("/api/v1/query", response_model=QueryResponse)
    def query(body: QueryRequest) -> dict:
        from src.agents.orchestrator import orchestrator

        state = orchestrator.run(body.query, history=body.history or [])
        return {
            "response": state["response"],
            "intent": state["intent"],
            "confidence": state["confidence"],
            "agent_used": state["agent_used"],
            "tools_used": state["tools_used"],
            "quality_score": state["quality_score"],
        }

    return app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client() -> TestClient:
    """Provide a FastAPI TestClient."""
    app = _create_test_app()
    return TestClient(app)


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Test the /health endpoint."""

    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_body_structure(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_health_includes_version(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert isinstance(data["version"], str)
        assert len(data["version"]) > 0


# ---------------------------------------------------------------------------
# Query endpoint - valid requests
# ---------------------------------------------------------------------------


class TestQueryEndpoint:
    """Test the /api/v1/query endpoint."""

    @patch("src.agents.orchestrator.orchestrator")
    def test_valid_query_returns_200(
        self,
        mock_orchestrator: MagicMock,
        client: TestClient,
    ) -> None:
        mock_orchestrator.run.return_value = {
            "user_message": "find laptops",
            "intent": "product",
            "confidence": 0.85,
            "context": {},
            "response": "Found 3 laptops matching your search.",
            "tools_used": ["catalog_search"],
            "agent_used": "product_agent",
            "quality_score": 0.9,
            "warnings": [],
            "needs_escalation": False,
            "history": [],
        }

        response = client.post(
            "/api/v1/query",
            json={"query": "find laptops"},
        )
        assert response.status_code == 200

    @patch("src.agents.orchestrator.orchestrator")
    def test_valid_query_response_structure(
        self,
        mock_orchestrator: MagicMock,
        client: TestClient,
    ) -> None:
        mock_orchestrator.run.return_value = {
            "user_message": "find laptops",
            "intent": "product",
            "confidence": 0.85,
            "context": {},
            "response": "Found 3 laptops matching your search.",
            "tools_used": ["catalog_search"],
            "agent_used": "product_agent",
            "quality_score": 0.9,
            "warnings": [],
            "needs_escalation": False,
            "history": [],
        }

        data = client.post(
            "/api/v1/query",
            json={"query": "find laptops"},
        ).json()

        assert data["response"] == "Found 3 laptops matching your search."
        assert data["intent"] == "product"
        assert data["confidence"] == 0.85
        assert data["agent_used"] == "product_agent"
        assert "catalog_search" in data["tools_used"]
        assert data["quality_score"] == 0.9

    @patch("src.agents.orchestrator.orchestrator")
    def test_query_with_session_and_history(
        self,
        mock_orchestrator: MagicMock,
        client: TestClient,
    ) -> None:
        mock_orchestrator.run.return_value = {
            "user_message": "help",
            "intent": "support",
            "confidence": 0.5,
            "context": {},
            "response": "How can I help?",
            "tools_used": [],
            "agent_used": "support_agent",
            "quality_score": 0.7,
            "warnings": [],
            "needs_escalation": False,
            "history": [{"role": "user", "content": "hi"}],
        }

        response = client.post(
            "/api/v1/query",
            json={
                "query": "help",
                "session_id": "sess-abc",
                "history": [{"role": "user", "content": "hi"}],
            },
        )
        assert response.status_code == 200
        assert response.json()["intent"] == "support"


# ---------------------------------------------------------------------------
# Query endpoint - validation errors
# ---------------------------------------------------------------------------


class TestQueryValidation:
    """Test that invalid requests return proper error codes."""

    def test_empty_body_returns_422(self, client: TestClient) -> None:
        response = client.post("/api/v1/query", json={})
        assert response.status_code == 422

    def test_missing_query_field_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/query",
            json={"session_id": "sess-1"},
        )
        assert response.status_code == 422

    def test_wrong_content_type_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/query",
            content="not json",
            headers={"Content-Type": "text/plain"},
        )
        assert response.status_code == 422

    def test_get_method_not_allowed(self, client: TestClient) -> None:
        response = client.get("/api/v1/query")
        assert response.status_code == 405

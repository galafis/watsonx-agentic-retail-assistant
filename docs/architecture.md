# Architecture

## Overview

The Watsonx Agentic Retail Assistant is a multi-agent system built on IBM watsonx.ai
and LangGraph. It processes natural-language customer queries through a governed
pipeline of intent classification, specialised agent execution, and quality-scored
responses.

The system enforces safety guardrails on every input and output, logs all
interactions for audit compliance, and scores response quality in real time.

---

## System Flowchart

```mermaid
flowchart LR
    A[User Query] --> B[Input Guardrails]
    B -->|safe| C[Intent Classifier]
    B -->|blocked| X[Rejection Response]
    C --> D[Entity Extractor]
    D --> E{LangGraph Router}
    E -->|product| F[Product Agent]
    E -->|order| G[Order Agent]
    E -->|recommendation| H[Recommendation Agent]
    E -->|support| I[Support Agent]
    F --> J[Output Guardrails]
    G --> J
    H --> J
    I --> J
    J --> K[Quality Scorer]
    K --> L[Audit Logger]
    L --> M[Response]
```

---

## Component Details

| Component | Module | Responsibility |
|---|---|---|
| Orchestrator | `src.agents.orchestrator` | LangGraph state machine managing the full query lifecycle |
| Product Agent | `src.agents.product_agent` | Product search, comparison, and detail retrieval |
| Order Agent | `src.agents.order_agent` | Order status, tracking, returns, and cancellations |
| Recommendation Agent | `src.agents.recommendation_agent` | Content-based and collaborative product recommendations |
| Support Agent | `src.agents.support_agent` | FAQ matching, sentiment analysis, and escalation |
| Catalog Search | `src.tools.catalog_search` | Hybrid vector + keyword product search |
| Recommendation Engine | `src.tools.recommendation_engine` | Similarity scoring with tag overlap, category match, and price proximity |
| Guardrails | `src.governance.guardrails` | PII detection, prompt injection blocking, price validation |
| Quality Scorer | `src.governance.quality_scorer` | Response quality scoring across multiple dimensions |
| Audit Logger | `src.governance.audit_logger` | Structured logging of all agent actions and tool calls |
| Product Catalog | `src.data.product_catalog` | In-memory product store backed by JSON |
| Order Store | `src.data.order_store` | In-memory order data store |
| API Routes | `src.api.routes` | FastAPI REST endpoints |
| UI | `src.ui.app` | Streamlit interactive front-end |
| Config | `src.config` | Centralised settings via pydantic-settings and YAML |

---

## Agent Descriptions

### Product Agent
Handles product search, detail lookup, and side-by-side comparison.
Dispatches to sub-handlers based on keywords (`compare`, `detail`, `search`)
and delegates catalog queries to the `CatalogSearchTool`.

### Order Agent
Manages the full order lifecycle: status checks, shipment tracking,
return/exchange initiation, and cancellation requests. Extracts order IDs
from user messages via regex pattern matching.

### Recommendation Agent
Generates personalised product suggestions using a hybrid strategy:
- **Content-based filtering**: tag overlap (Jaccard similarity), category
  matching, and price proximity scoring.
- **Collaborative filtering**: aggregated scores across a customer purchase
  history (production: matrix factorisation via watsonx.ai).
- **Diversity controls**: configurable `diversity_factor` prevents
  monotonic same-category recommendations.

### Support Agent
First-line customer support combining FAQ retrieval with sentiment
analysis. Negative sentiment below a configurable threshold triggers
automatic escalation to a human agent.

---

## Data Flow Sequence

```mermaid
sequenceDiagram
    participant U as User
    participant API as FastAPI
    participant O as Orchestrator
    participant G as Guardrails
    participant IC as Intent Classifier
    participant A as Specialized Agent
    participant T as Tool
    participant QS as Quality Scorer
    participant AL as Audit Logger

    U->>API: POST /api/v1/query
    API->>O: run(user_message)
    O->>G: validate_input(message)
    G-->>O: {safe, warnings}
    O->>IC: classify_intent(state)
    IC-->>O: {intent, confidence}
    O->>O: extract_entities(state)
    O->>A: handle(query, context)
    A->>T: search / lookup / recommend
    T-->>A: results
    A-->>O: {response, tools_used}
    O->>G: validate_output(response)
    G-->>O: {sanitised_response}
    O->>QS: score(query, response)
    QS-->>O: {overall_score}
    O->>AL: log_conversation_turn(...)
    O-->>API: AgentState
    API-->>U: QueryResponse
```

---

## Deployment

The application is containerised with Docker and orchestrated via
Docker Compose. The stack includes:

| Service | Image | Port |
|---|---|---|
| API Server | `watsonx-retail-api` | 8080 |
| Streamlit UI | `watsonx-retail-ui` | 8501 |
| Redis (cache) | `redis:7-alpine` | 6379 |
| PostgreSQL | `postgres:16-alpine` | 5432 |

### Quick Start

```bash
# Build and run
docker compose up --build

# API health check
curl http://localhost:8080/health

# Run tests
make test
```

### Environment Variables

| Variable | Description | Default |
|---|---|---|
| `WATSONX_API_KEY` | IBM watsonx.ai API key | *(required in production)* |
| `WATSONX_PROJECT_ID` | watsonx.ai project ID | *(required in production)* |
| `WATSONX_URL` | watsonx.ai endpoint | `https://us-south.ml.cloud.ibm.com` |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379/0` |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://retail:retail_pass@localhost:5432/retail_assistant` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |
| `ENVIRONMENT` | Runtime environment | `development` |

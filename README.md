# Watsonx Agentic Retail Assistant

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![IBM Watsonx](https://img.shields.io/badge/IBM-Watsonx.ai-052FAD.svg)](https://www.ibm.com/watsonx)
[![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-00A67E.svg)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/galafis/watsonx-agentic-retail-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/galafis/watsonx-agentic-retail-assistant/actions)

[Portugues](#portugues) | [English](#english)

---

## Portugues

### Descricao

Assistente de varejo multi-agente inteligente construido com **IBM Watsonx Granite** e orquestracao **LangGraph**. O sistema utiliza uma arquitetura agentica com quatro agentes especializados que colaboram para atender consultas de clientes sobre produtos, pedidos, recomendacoes e suporte geral.

O orquestrador classifica a intencao do usuario, extrai entidades relevantes (IDs de pedidos, IDs de produtos) e roteia para o agente especializado apropriado. Todas as interacoes passam por guardrails de seguranca (deteccao de PII, bloqueio de prompt injection, validacao de precos) e sao registradas em um audit trail completo com scoring de qualidade.

### Caracteristicas Principais

| Componente | Descricao |
|---|---|
| **Orquestrador LangGraph** | Maquina de estados que roteia consultas para 4 agentes especializados com max 10 iteracoes |
| **Agente de Produtos** | Busca hibrida (vetor + keyword), comparacao e detalhes de produtos (top-5, similaridade 0.4) |
| **Agente de Pedidos** | Status, rastreamento de entrega e devolucoes (politica de 30 dias) |
| **Agente de Recomendacoes** | Filtragem baseada em conteudo + colaborativa com diversidade (max 5, diversidade 0.3) |
| **Agente de Suporte** | FAQ retrieval com analise de sentimento e escalonamento humano (threshold -0.5) |
| **Guardrails** | Deteccao de PII (SSN, cartao, email), bloqueio de prompt injection, validacao de precos |
| **Governanca** | Audit logger completo, quality scorer multidimensional, rastreamento de ferramentas |
| **API REST** | FastAPI com endpoints para chat, historico e metricas de governanca |
| **Interface** | Streamlit com chat interativo e painel de metricas |

### Arquitetura

```mermaid
flowchart TB
    User["Usuario"] -->|Consulta| GI["Guardrails\nValidacao de Entrada"]
    GI --> IC["Classificador de Intencao\n(Keywords / Granite LLM)"]
    IC --> Router["LangGraph Router\n(Conditional Edges)"]

    Router -->|product| PA["Agente de Produtos"]
    Router -->|order| OA["Agente de Pedidos"]
    Router -->|recommendation| RA["Agente de Recomendacoes"]
    Router -->|support| SA["Agente de Suporte"]

    PA --> CS["Catalog Search\n(Hibrido Vetor+Keyword)"]
    OA --> OM["Order Management\n(Status/Returns/Tracking)"]
    RA --> RE["Recommendation Engine\n(Content+Collaborative)"]
    SA --> FAQ["FAQ Retriever\n(RAG)"]
    SA --> SENT["Sentiment Analyzer\n(Escalonamento)"]

    CS --> GO["Guardrails\nValidacao de Saida"]
    OM --> GO
    RE --> GO
    FAQ --> GO
    SENT --> GO

    GO --> QS["Quality Scorer\n(Relevancia/Completude)"]
    QS --> AL["Audit Logger\n(Governanca)"]
    AL --> UI["Streamlit UI"]
```

### Stack Tecnologico

| Camada | Tecnologia |
|---|---|
| LLM | IBM Granite-3-8b-instruct via Watsonx.ai |
| Embeddings | IBM Slate-125m-english-rtrvr |
| Orquestracao | LangGraph (maquina de estados com conditional edges) |
| API | FastAPI + Uvicorn |
| Interface | Streamlit |
| Cache | Redis 7 |
| Banco de Dados | PostgreSQL 16 |
| Vector Store | ChromaDB |
| Containerizacao | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Qualidade | Ruff, MyPy, pytest |

### Inicio Rapido

#### Com Docker Compose

```bash
# Clonar o repositorio
git clone https://github.com/galafis/watsonx-agentic-retail-assistant.git
cd watsonx-agentic-retail-assistant

# Configurar variaveis de ambiente
cp .env.example .env
# Editar .env com suas credenciais Watsonx

# Iniciar todos os servicos
docker compose up -d

# API disponivel em http://localhost:8080
# UI disponivel em http://localhost:8501
```

### Desenvolvimento Local

```bash
# Criar ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Iniciar servicos auxiliares
docker compose up -d postgres redis

# Executar API
uvicorn src.api.routes:app --host 0.0.0.0 --port 8080 --reload

# Executar UI (em outro terminal)
streamlit run src/ui/app.py

# Executar testes
pytest tests/ -v --cov=src
```

### Estrutura do Projeto

```
watsonx-agentic-retail-assistant/
├── src/
│   ├── agents/
│   │   ├── orchestrator.py        # LangGraph state machine + intent routing
│   │   ├── product_agent.py       # Busca, comparacao e detalhes de produtos
│   │   ├── order_agent.py         # Status, devolucoes e rastreamento
│   │   ├── recommendation_agent.py # Recomendacoes personalizadas
│   │   └── support_agent.py       # FAQ + sentimento + escalonamento
│   ├── tools/
│   │   ├── catalog_search.py      # Busca hibrida no catalogo
│   │   ├── recommendation_engine.py # Motor de recomendacao
│   │   ├── order_management.py    # Gestao de pedidos
│   │   ├── faq_retriever.py       # RAG para FAQ
│   │   └── sentiment.py           # Analise de sentimento
│   ├── data/
│   │   ├── product_catalog.py     # Catalogo in-memory
│   │   └── order_store.py         # Store de pedidos
│   ├── governance/
│   │   ├── guardrails.py          # PII, prompt injection, precos
│   │   ├── audit_logger.py        # Trail de auditoria
│   │   └── quality_scorer.py      # Scoring multidimensional
│   ├── api/
│   │   ├── routes.py              # Endpoints FastAPI
│   │   └── schemas.py             # Pydantic schemas
│   ├── ui/
│   │   └── app.py                 # Interface Streamlit
│   └── config.py                  # Configuracao centralizada
├── tests/
│   ├── test_orchestrator.py       # Testes do orquestrador
│   ├── test_product_agent.py      # Testes do agente de produtos
│   ├── test_guardrails.py         # Testes dos guardrails
│   ├── test_recommendation.py     # Testes de recomendacao
│   └── test_api.py                # Testes da API
├── config/
│   └── settings.yaml              # Configuracao de agentes e ferramentas
├── notebooks/
│   └── 01_retail_assistant_demo.ipynb
├── docs/
│   └── architecture.md            # Documentacao de arquitetura
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

### API Endpoints

| Metodo | Endpoint | Descricao |
|---|---|---|
| `POST` | `/api/v1/chat` | Envia mensagem ao assistente |
| `GET` | `/api/v1/chat/history` | Historico de conversas |
| `GET` | `/api/v1/products/search` | Busca de produtos |
| `GET` | `/api/v1/products/{id}` | Detalhes do produto |
| `GET` | `/api/v1/orders/{id}/status` | Status do pedido |
| `POST` | `/api/v1/orders/{id}/return` | Iniciar devolucao |
| `GET` | `/api/v1/recommendations` | Recomendacoes personalizadas |
| `GET` | `/api/v1/governance/metrics` | Metricas de governanca |
| `GET` | `/api/v1/governance/audit` | Log de auditoria |
| `GET` | `/health` | Health check |

### Agentes

| Agente | Intencao | Ferramentas | Parametros |
|---|---|---|---|
| **Product Agent** | product, search, find, browse, compare, price | `catalog_search` | top_k=5, similarity=0.4 |
| **Order Agent** | order, status, track, return, refund, delivery | `order_management` | return_days=30 |
| **Recommendation Agent** | recommend, suggest, similar, best, popular | `recommendation_engine` | max=5, diversity=0.3 |
| **Support Agent** | help, support, faq, policy, complaint, payment | `faq_retriever`, `sentiment_analyzer` | threshold=-0.5 |

---

## English

### Description

Intelligent multi-agent retail assistant built with **IBM Watsonx Granite** and **LangGraph** orchestration. The system employs an agentic architecture with four specialized agents that collaborate to handle customer queries about products, orders, recommendations, and general support.

The orchestrator classifies user intent, extracts relevant entities (order IDs, product IDs), and routes to the appropriate specialized agent. All interactions pass through security guardrails (PII detection, prompt injection blocking, price validation) and are recorded in a complete audit trail with quality scoring.

### Key Features

| Component | Description |
|---|---|
| **LangGraph Orchestrator** | State machine routing queries to 4 specialized agents with max 10 iterations |
| **Product Agent** | Hybrid search (vector + keyword), comparison, and product details (top-5, similarity 0.4) |
| **Order Agent** | Status, delivery tracking, and returns (30-day policy) |
| **Recommendation Agent** | Content-based + collaborative filtering with diversity (max 5, diversity 0.3) |
| **Support Agent** | FAQ retrieval with sentiment analysis and human escalation (threshold -0.5) |
| **Guardrails** | PII detection (SSN, credit card, email), prompt injection blocking, price validation |
| **Governance** | Complete audit logger, multidimensional quality scorer, tool tracking |
| **REST API** | FastAPI with endpoints for chat, history, and governance metrics |
| **Interface** | Streamlit with interactive chat and metrics dashboard |

### Architecture

```mermaid
flowchart TB
    User["User"] -->|Query| GI["Guardrails\nInput Validation"]
    GI --> IC["Intent Classifier\n(Keywords / Granite LLM)"]
    IC --> Router["LangGraph Router\n(Conditional Edges)"]

    Router -->|product| PA["Product Agent"]
    Router -->|order| OA["Order Agent"]
    Router -->|recommendation| RA["Recommendation Agent"]
    Router -->|support| SA["Support Agent"]

    PA --> CS["Catalog Search\n(Hybrid Vector+Keyword)"]
    OA --> OM["Order Management\n(Status/Returns/Tracking)"]
    RA --> RE["Recommendation Engine\n(Content+Collaborative)"]
    SA --> FAQ["FAQ Retriever\n(RAG)"]
    SA --> SENT["Sentiment Analyzer\n(Escalation)"]

    CS --> GO["Guardrails\nOutput Validation"]
    OM --> GO
    RE --> GO
    FAQ --> GO
    SENT --> GO

    GO --> QS["Quality Scorer\n(Relevance/Completeness)"]
    QS --> AL["Audit Logger\n(Governance)"]
    AL --> UI["Streamlit UI"]
```

### Tech Stack

| Layer | Technology |
|---|---|
| LLM | IBM Granite-3-8b-instruct via Watsonx.ai |
| Embeddings | IBM Slate-125m-english-rtrvr |
| Orchestration | LangGraph (state machine with conditional edges) |
| API | FastAPI + Uvicorn |
| Interface | Streamlit |
| Cache | Redis 7 |
| Database | PostgreSQL 16 |
| Vector Store | ChromaDB |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Quality | Ruff, MyPy, pytest |

### Quick Start

#### With Docker Compose

```bash
# Clone the repository
git clone https://github.com/galafis/watsonx-agentic-retail-assistant.git
cd watsonx-agentic-retail-assistant

# Configure environment variables
cp .env.example .env
# Edit .env with your Watsonx credentials

# Start all services
docker compose up -d

# API available at http://localhost:8080
# UI available at http://localhost:8501
```

### Local Development

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Start auxiliary services
docker compose up -d postgres redis

# Run API
uvicorn src.api.routes:app --host 0.0.0.0 --port 8080 --reload

# Run UI (in another terminal)
streamlit run src/ui/app.py

# Run tests
pytest tests/ -v --cov=src
```

### Project Structure

```
watsonx-agentic-retail-assistant/
├── src/
│   ├── agents/
│   │   ├── orchestrator.py        # LangGraph state machine + intent routing
│   │   ├── product_agent.py       # Product search, comparison, details
│   │   ├── order_agent.py         # Status, returns, tracking
│   │   ├── recommendation_agent.py # Personalized recommendations
│   │   └── support_agent.py       # FAQ + sentiment + escalation
│   ├── tools/
│   │   ├── catalog_search.py      # Hybrid catalog search
│   │   ├── recommendation_engine.py # Recommendation engine
│   │   ├── order_management.py    # Order lifecycle management
│   │   ├── faq_retriever.py       # FAQ RAG retrieval
│   │   └── sentiment.py           # Sentiment analysis
│   ├── data/
│   │   ├── product_catalog.py     # In-memory catalog
│   │   └── order_store.py         # Order store
│   ├── governance/
│   │   ├── guardrails.py          # PII, prompt injection, pricing
│   │   ├── audit_logger.py        # Audit trail
│   │   └── quality_scorer.py      # Multidimensional scoring
│   ├── api/
│   │   ├── routes.py              # FastAPI endpoints
│   │   └── schemas.py             # Pydantic schemas
│   ├── ui/
│   │   └── app.py                 # Streamlit interface
│   └── config.py                  # Centralized configuration
├── tests/
│   ├── test_orchestrator.py       # Orchestrator tests
│   ├── test_product_agent.py      # Product agent tests
│   ├── test_guardrails.py         # Guardrails tests
│   ├── test_recommendation.py     # Recommendation tests
│   └── test_api.py                # API tests
├── config/
│   └── settings.yaml              # Agent and tool configuration
├── notebooks/
│   └── 01_retail_assistant_demo.ipynb
├── docs/
│   └── architecture.md            # Architecture documentation
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/chat` | Send message to assistant |
| `GET` | `/api/v1/chat/history` | Conversation history |
| `GET` | `/api/v1/products/search` | Product search |
| `GET` | `/api/v1/products/{id}` | Product details |
| `GET` | `/api/v1/orders/{id}/status` | Order status |
| `POST` | `/api/v1/orders/{id}/return` | Initiate return |
| `GET` | `/api/v1/recommendations` | Personalized recommendations |
| `GET` | `/api/v1/governance/metrics` | Governance metrics |
| `GET` | `/api/v1/governance/audit` | Audit log |
| `GET` | `/health` | Health check |

### Agents

| Agent | Intent | Tools | Parameters |
|---|---|---|---|
| **Product Agent** | product, search, find, browse, compare, price | `catalog_search` | top_k=5, similarity=0.4 |
| **Order Agent** | order, status, track, return, refund, delivery | `order_management` | return_days=30 |
| **Recommendation Agent** | recommend, suggest, similar, best, popular | `recommendation_engine` | max=5, diversity=0.3 |
| **Support Agent** | help, support, faq, policy, complaint, payment | `faq_retriever`, `sentiment_analyzer` | threshold=-0.5 |

---

### Environment Variables

| Variable | Description | Default |
|---|---|---|
| `WATSONX_API_KEY` | IBM Watsonx API key | - |
| `WATSONX_PROJECT_ID` | Watsonx project ID | - |
| `WATSONX_URL` | Watsonx service URL | `https://us-south.ml.cloud.ibm.com` |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://retail:retail_pass@localhost:5432/retail_assistant` |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379/0` |
| `APP_HOST` | API host | `0.0.0.0` |
| `APP_PORT` | API port | `8080` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `ENVIRONMENT` | Environment name | `development` |

---

### Autor / Author

**Gabriel Demetrios Lafis** - [GitHub](https://github.com/galafis) | [LinkedIn](https://www.linkedin.com/in/gabriel-demetrios-lafis/)

# Restaurant Booking AI Agent

Agentic AI that helps users check the restaurant menu and reserve tables.

## Tools & tech stack

- **Streamlit** – frontend
- **LangGraph** – orchestration of AI agents
- **MCP** – all tools centralized; **streamable-http** transport for security
- **Qdrant** – vector database (menu RAG)
- **Postgres** – reservations database
- **Docker** – infrastructure
- **Ollama** – local model for chat and embeddings
- **RAG** – restaurant menu stored and queried via embeddings

## Ollama models

- `smallthinker` – chat
- `nomic-embed-text` – embeddings

## Workflow

- User can ask about **menu**, **table availability**, or **reserve/cancel** a table.
- **Menu**: agent uses RAG (Qdrant) via MCP tools.
- **Reservation**: agent uses Postgres via MCP tools.

### Agent split (security)

Agents are split so each can only call a subset of tools:

| Agent           | Access   | MCP tools                                      |
|----------------|----------|-------------------------------------------------|
| Menu           | RAG only | `query_menu`, `menu_count`                      |
| Booking-read   | Read DB  | `booking_check_availability`, `booking_list`   |
| Booking-write  | Write DB | `booking_create`, `booking_cancel`              |

- All tools live in **MCP**; MCP runs as **streamable-http**.
- Only the appropriate agent can call read vs write booking tools; input is **validated inside MCP**.

## Run with Docker

```bash
docker compose up --build
```

- Streamlit: http://localhost:8501  
- Pull Ollama models (first time):  
  `docker compose exec ollama ollama pull smallthinker`  
  `docker compose exec ollama ollama pull nomic-embed-text`

### Ingest menu (RAG)

1. Prefer `./sample_menu/menu_items.json` (structured items + cuisine tags for accurate filters).  
   Or put a PDF at `./sample_menu/sample_menu.pdf` as a fallback.
2. Run ingest (profile `ingest`):  
   `docker compose --profile ingest run ingest`

## Project layout

- `streamlit/` – Streamlit UI
- `agents/` – LangGraph agents (FastAPI), MCP client with scoped tools
- `mcp_server/` – MCP server (streamable-http), menu RAG + booking tools, **input validation**
- `ingest/` – one-off job to ingest menu (JSON preferred, PDF fallback) into Qdrant
- `db/` – Postgres init (tables created by MCP server)

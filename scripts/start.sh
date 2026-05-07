#!/usr/bin/env bash
set -euo pipefail

# Start the app (excluding ingest). Ingestion is in ./ingest.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "Starting core services (ollama + postgres + mcp_server + agent + streamlit)..."
docker compose up -d --build

echo "Ensuring Ollama models are present (may take a while on first run)..."
docker compose exec -T ollama ollama pull nomic-embed-text
docker compose exec -T ollama ollama pull smallthinker

echo "Done. Streamlit should be available at http://localhost:8501"
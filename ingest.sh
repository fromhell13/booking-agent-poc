#!/usr/bin/env bash
set -euo pipefail

# One-off menu ingestion (RAG).
# Important: Qdrant "local" storage cannot be opened concurrently.
# This script runs ingest while the main stack is down to avoid lock errors.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PDF_PATH="$ROOT_DIR/sample_menu/sample_menu.pdf"
if [[ ! -f "$PDF_PATH" ]]; then
  echo "Error: menu PDF not found at: $PDF_PATH" >&2
  echo "Place your menu PDF at: sample_menu/sample_menu.pdf and try again." >&2
  exit 1
fi

echo "Stopping stack to avoid Qdrant local lock..."
docker compose down

echo "Starting Ollama (required for embeddings)..."
docker compose up -d ollama

echo "Ensuring embedding model is present..."
docker compose exec -T ollama ollama pull nomic-embed-text

echo "Running ingest job..."
docker compose --profile ingest run --rm ingest

echo "Ingest complete."

echo "Starting full stack..."
docker compose up -d --build
docker compose exec -T ollama ollama pull smallthinker

echo "Done. Streamlit should be available at http://localhost:8501"

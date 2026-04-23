import os
import sys
from pathlib import Path

import redis

from mcp_server.tools.menu.menu_rag import MenuRAG

PDF_PATH = Path("/app/sample_menu/sample_menu.pdf")
QDRANT_DIR = os.getenv("QDRANT_PATH", "/data/qdrant_local")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")

if not PDF_PATH.is_file():
    print("Error: Menu PDF not found.", file=sys.stderr)
    print(f"  Expected: {PDF_PATH}", file=sys.stderr)
    print("  Place your menu PDF at: sample_menu/sample_menu.pdf (on the host), then run ingest again.", file=sys.stderr)
    sys.exit(1)

rag = MenuRAG(collection_name="menu", qdrant_path=QDRANT_DIR, embedding_model="nomic-embed-text", ollama_base_url=OLLAMA_BASE_URL)

rag.ingest_pdf(str(PDF_PATH))
try:
    redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    # Bump cache generation so old query cache is ignored immediately after re-ingest.
    redis_client.incr("menu_cache_version")
    print("Menu cache invalidated")
except Exception as exc:
    print(f"Warning: unable to invalidate menu cache ({exc})", file=sys.stderr)

print("Ingest done")

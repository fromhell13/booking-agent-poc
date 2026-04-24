import os
import sys
from pathlib import Path

import redis

from mcp_server.tools.menu.menu_rag import MenuRAG

JSON_PATH = Path("/app/sample_menu/menu_items.json")
PDF_PATH = Path("/app/sample_menu/sample_menu.pdf")
QDRANT_DIR = os.getenv("QDRANT_PATH", "/data/qdrant_local")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")

rag = MenuRAG(collection_name="menu", qdrant_path=QDRANT_DIR, embedding_model="nomic-embed-text", ollama_base_url=OLLAMA_BASE_URL)

if JSON_PATH.is_file():
    print(f"Ingesting structured menu from {JSON_PATH}")
    rag.ingest_json(str(JSON_PATH))
elif PDF_PATH.is_file():
    print(f"Ingesting PDF menu from {PDF_PATH} (add menu_items.json for cuisine-accurate filters)")
    rag.ingest_pdf(str(PDF_PATH))
else:
    print("Error: No menu source found.", file=sys.stderr)
    print(f"  Expected either: {JSON_PATH}", file=sys.stderr)
    print(f"            or:     {PDF_PATH}", file=sys.stderr)
    sys.exit(1)

try:
    redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.incr("menu_cache_version")
    print("Menu cache invalidated")
except Exception as exc:
    print(f"Warning: unable to invalidate menu cache ({exc})", file=sys.stderr)

print("Ingest done")

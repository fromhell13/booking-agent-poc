import os
import sys
from pathlib import Path

from mcp_server.tools.menu.menu_rag import MenuRAG

PDF_PATH = Path("/app/sample_menu/sample_menu.pdf")
QDRANT_DIR = os.getenv("QDRANT_PATH", "/data/qdrant_local")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

if not PDF_PATH.is_file():
    print("Error: Menu PDF not found.", file=sys.stderr)
    print(f"  Expected: {PDF_PATH}", file=sys.stderr)
    print("  Place your menu PDF at: sample_menu/sample_menu.pdf (on the host), then run ingest again.", file=sys.stderr)
    sys.exit(1)

rag = MenuRAG(collection_name="menu", qdrant_path=QDRANT_DIR, embedding_model="nomic-embed-text", ollama_base_url=OLLAMA_BASE_URL)

rag.ingest_pdf(str(PDF_PATH))
print("Ingest done")

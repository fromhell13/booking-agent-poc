from mcp_server.tools.menu_rag import MenuRAG
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # booking-agent-poc
file_path = PROJECT_ROOT / "mcp_server" / "menu" / "sample_menu.pdf"
QDRANT_DIR = str(PROJECT_ROOT / "mcp_server" / "qdrant_local")
print(f"QDRANT_DIR={QDRANT_DIR}")

#rag = MenuRAG(collection_name="menu", qdrant_path=":memory:")
rag = MenuRAG(collection_name="menu", qdrant_path=QDRANT_DIR)
rag.ingest_pdf(file_path)
print("done")
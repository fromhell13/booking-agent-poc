from mcp_server.tools.menu_rag import MenuRAG
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QDRANT_DIR = str(PROJECT_ROOT / "qdrant_local")
BASE_DIR = Path(__file__).resolve().parent.parent
file_path = BASE_DIR / "menu" / "sample_menu.pdf"
#rag = MenuRAG(collection_name="menu", qdrant_path=":memory:")
rag = MenuRAG(collection_name="menu", qdrant_path=QDRANT_DIR)
rag.ingest_pdf(file_path)
print("done")
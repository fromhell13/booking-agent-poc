# mcp_server/server.py
from __future__ import annotations

from pathlib import Path

from fastmcp import FastMCP
from fastmcp.utilities.logging import get_logger


from mcp_server.tools.menu_rag import MenuRAG

logger = get_logger(__name__)
mcp = FastMCP("booking-agent-tools")

MCP_ROOT = Path(__file__).resolve().parent
QDRANT_DIR = str(MCP_ROOT / "qdrant_local")

menu_rag = MenuRAG(collection_name="menu", qdrant_path=QDRANT_DIR, embedding_model="nomic-embed-text")
logger.info(f"MenuRAG initialized. qdrant_path={QDRANT_DIR}")

@mcp.tool()
def menu_count() -> dict:
    return {
        "collection": menu_rag.collection_name,
        "count": menu_rag.client.count(menu_rag.collection_name, exact=True).count
    }

@mcp.tool()
def query_menu(query: str) -> dict:
    logger.info(f"Querying menu with: {query}")
    hits = menu_rag.query(query=query, top_k=3)
    # Optional: trim text to keep responses small for clients
    for h in hits:
        h["text"] = (h.get("text") or "")[:800]
    return {"query": query, "top_k": 3, "hits": hits}

if __name__ == "__main__":
    
    mcp.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=8000,
        path="/mcp",
    )

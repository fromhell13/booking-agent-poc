# mcp_server/server.py
from __future__ import annotations

import sys
import logging
from pathlib import Path

from fastmcp import FastMCP
from .tools.menu_rag import MenuRAG


logging.basicConfig(stream=sys.stderr, level=logging.INFO, force=True)
logger = logging.getLogger("booking-agent-tools")

mcp = FastMCP("booking-agent-tools")

MCP_ROOT = Path(__file__).resolve().parent 
QDRANT_DIR = str(MCP_ROOT / "qdrant_local")

menu_rag = MenuRAG(collection_name="menu", qdrant_path=QDRANT_DIR, embedding_model="nomic-embed-text")
logger.info(f"MenuRAG initialized. qdrant_path={QDRANT_DIR}")

@mcp.tool("collections_count")
def collections_count() -> dict:
    """How many collections exist in this Qdrant DB."""
    cols = menu_rag.client.get_collections().collections
    return {"collections_total": len(cols), "collections": [c.name for c in cols]}

@mcp.tool("menu_count")
def menu_count() -> dict:
    """How many items/points are in the 'menu' collection."""
    try:
        # sanity check: does collection exist?
        cols = [c.name for c in menu_rag.client.get_collections().collections]
        if menu_rag.collection_name not in cols:
            return {
                "collection": menu_rag.collection_name,
                "count": 0,
                "error": f"Collection not found in {QDRANT_DIR}. Found: {cols}"
            }

        cnt = menu_rag.client.count(menu_rag.collection_name, exact=True).count
        return {"collection": menu_rag.collection_name, "count": cnt}

    except Exception as e:
        logger.exception("menu_count failed")
        return {"collection": menu_rag.collection_name, "error": str(e), "qdrant_path": QDRANT_DIR}

@mcp.tool("query_menu")
def query_menu(query: str) -> dict:
    """Query menu items matching the query."""
    try:
        hits = menu_rag.query(query=query, top_k=3)
        return {"query": query, "top_k": 3, "hits": hits}
    except Exception as e:
        logger.exception("query_menu failed")
        return {"query": query, "error": str(e), "qdrant_path": QDRANT_DIR}

if __name__ == "__main__":
    mcp.run(transport="stdio")

# mcp_server/server.py
from __future__ import annotations

from fastmcp import FastMCP
from fastmcp.utilities.logging import get_logger
from tools.menu_rag import MenuRAG

logger = get_logger(__name__)
mcp = FastMCP("booking-agent-tools")

from pathlib import Path
from tools.menu_rag import MenuRAG
PROJECT_ROOT = Path(__file__).resolve().parents[1] 
QDRANT_DIR = str(PROJECT_ROOT / "qdrant_local")

menu_rag = MenuRAG(collection_name="menu", qdrant_path=QDRANT_DIR, embedding_model="nomic-embed-text")
#menu_rag = MenuRAG(collection_name="menu", qdrant_path=":memory:", embedding_model="nomic-embed-text")

@mcp.tool()
def menu_count() -> dict:
    return {
        "collection": menu_rag.collection_name,
        "count": menu_rag.client.count(menu_rag.collection_name, exact=True).count
    }

@mcp.tool()
async def say_hello(name:str) -> str:
    logger.info(f"logger.info: say_hello(): name={name}")
    logger.warning(f"logger.warning: say_hello(): name={name}")
    logger.debug(f"logger.debug: say_hello(): name={name}")
    logger.error(f"logger.error: say_hello(): name={name}")
    return f"hello {name}"

@mcp.tool()
def query_menu(query: str, top_k: int = 3) -> dict:
    """
    Query the menu knowledge base.
    """
    
    hits = menu_rag.query(query=query, top_k=top_k)
    logger.info(f"logger.info: {hits}")
    return {"query": query, "top_k": top_k, "hits": hits}



if __name__ == "__main__":
    mcp.run(transport='stdio')

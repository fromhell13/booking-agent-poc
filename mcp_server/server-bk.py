# mcp_server/server.py
from __future__ import annotations

from fastmcp import FastMCP
from fastmcp.utilities.logging import get_logger
from tools.menu_rag import MenuRAG

logger = get_logger(__name__)
mcp = FastMCP("booking-agent-tools")

from pathlib import Path
from tools.menu_rag import MenuRAG

'''
PROJECT_ROOT = Path(__file__).resolve().parents[1] 
QDRANT_DIR = str(PROJECT_ROOT / "qdrant_local")
'''
MCP_ROOT = Path(__file__).resolve().parent
QDRANT_DIR = str(MCP_ROOT / "qdrant_local")

menu_rag = MenuRAG(collection_name="menu", qdrant_path=QDRANT_DIR, embedding_model="nomic-embed-text")
#menu_rag = MenuRAG(collection_name="menu", qdrant_path=":memory:", embedding_model="nomic-embed-text")

logger.info("MenuRAG initialized.")

@mcp.tool()
def menu_count() -> dict:
    """
    Get mwnu collections count.
   
    Returns:
        dict: A dictionary containing the collection name and the count of items in the collection.
    """
    return {
        "collection": menu_rag.collection_name,
        "count": menu_rag.client.count(menu_rag.collection_name, exact=True).count
    }

@mcp.tool()
def query_menu(query: str) -> dict:
    """
    Get menu items matching the query.
    Args:
        query (str): The query string to search for in the menu.
        top_k (int, optional): The number of top matching items to return. Defaults to 3.
    """
    logger.info(f"Querying menu with: {query}")
    try:
        hits = menu_rag.query(query=query, top_k=3)
        logger.info(f"logger.info: {hits}")
        return {"query": query, "top_k": 3, "hits": hits}
    except Exception as e:
        logger.error(f"Error querying menu: {e}")
        raise e
    



if __name__ == "__main__":
    mcp.run(transport='stdio')

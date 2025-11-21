# mcp_server/server.py

from mcp.server.fastmcp import FastMCP
from tools.rag_tools import QdrantRAGTool

# Create FastMCP instance
mcp = FastMCP("support-agent")

#initialize tools
menu_tools = QdrantRAGTool()

@mcp.tool('search menu')
async def get_menu(query: str) -> str:
    """
    Query the restaurant menu vector database.

    Args:
        query: The search query about menu items
    """
    return menu_tools.query_menu(query)

@mcp.tool('reservation')
async def reservation()

if __name__ == "__main__":
    mcp.run(transport='stdio')

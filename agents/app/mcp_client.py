import os
from langchain_mcp_adapters.client import MultiServerMCPClient

MCP_URL = os.getenv("MCP_URL", "http://mcp_server:8000/mcp")

async def get_mcp_tools():
    client = MultiServerMCPClient(
        {
            "booking-agent-tools": {
                "transport": "streamable_http",
                "url": MCP_URL,
            }
        }
    )
    tools = await client.get_tools()
    return {t.name: t for t in tools}

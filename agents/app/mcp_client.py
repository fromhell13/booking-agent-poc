"""
MCP client with scoped tool access. Each agent only receives tools it is allowed to use.
- menu: RAG tools only (query_menu, menu_count)
- booking_read: read-only DB (booking_check_availability, booking_list)
- booking_write: insert/update/critical (booking_create, booking_cancel)
"""
import asyncio
import os
from typing import Any, Literal

from langchain_mcp_adapters.client import MultiServerMCPClient

MCP_URL = os.getenv("MCP_URL", "http://mcp_server:8000/mcp")

Scope = Literal["menu", "booking_read", "booking_write"]

TOOLS_BY_SCOPE: dict[Scope, set[str]] = {
    "menu": {"query_menu", "menu_count"},
    "booking_read": {"booking_check_availability", "booking_list"},
    "booking_write": {"booking_create", "booking_cancel"},
}

_tools_cache: dict[str, dict[str, Any]] = {}
_tools_lock = asyncio.Lock()


async def get_mcp_tools(scope: Scope | None = None):
    """
    Get MCP tools. If scope is set, only tools allowed for that agent are returned.
    Tool lists are cached per scope so each chat turn does not re-handshake with MCP.
    """
    cache_key = scope if scope is not None else "__all__"
    async with _tools_lock:
        if cache_key in _tools_cache:
            return _tools_cache[cache_key]

        client = MultiServerMCPClient(
            {
                "booking-agent-tools": {
                    "transport": "streamable_http",
                    "url": MCP_URL,
                }
            }
        )
        tools = await client.get_tools()
        name_to_tool = {t.name: t for t in tools}
        if scope:
            allowed = TOOLS_BY_SCOPE.get(scope) or set()
            result = {k: v for k, v in name_to_tool.items() if k in allowed}
        else:
            result = name_to_tool
        _tools_cache[cache_key] = result
        return result

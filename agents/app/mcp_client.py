"""
MCP client with scoped tool access. Each agent only receives tools it is allowed to use.
- menu: RAG tools only (query_menu, menu_count)
- booking_read: read-only DB (booking_check_availability, booking_list)
- booking_write: insert/update/critical (booking_create, booking_cancel)
"""
import asyncio
import os
import json
import time
from urllib import request, parse
from typing import Any, Literal

from langchain_mcp_adapters.client import MultiServerMCPClient

MCP_URL = os.getenv("MCP_URL")
MCP_OAUTH_TOKEN_URL = os.getenv("MCP_OAUTH_TOKEN_URL")
MCP_OAUTH_CLIENT_ID = os.getenv("MCP_OAUTH_CLIENT_ID")
MCP_OAUTH_CLIENT_SECRET = os.getenv("MCP_OAUTH_CLIENT_SECRET")

Scope = Literal["menu", "booking_read", "booking_write"]

TOOLS_BY_SCOPE: dict[Scope, set[str]] = {
    "menu": {"query_menu", "menu_count"},
    "booking_read": {"booking_check_availability", "booking_list"},
    "booking_write": {"booking_create", "booking_cancel"},
}

_tools_cache: dict[str, dict[str, Any]] = {}
_tools_lock = asyncio.Lock()
_token_cache: dict[str, Any] = {"access_token": None, "expires_at": 0}


def _fetch_oauth_token_sync() -> str:
    now = int(time.time())
    token = _token_cache.get("access_token")
    expires_at = int(_token_cache.get("expires_at") or 0)
    if isinstance(token, str) and token and now < (expires_at - 30):
        return token

    form = parse.urlencode(
        {
            "grant_type": "client_credentials",
            "client_id": MCP_OAUTH_CLIENT_ID,
            "client_secret": MCP_OAUTH_CLIENT_SECRET,
        }
    ).encode("utf-8")
    req = request.Request(
        MCP_OAUTH_TOKEN_URL,
        data=form,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with request.urlopen(req, timeout=15) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    access_token = payload.get("access_token")
    expires_in = int(payload.get("expires_in", 3600))
    if not isinstance(access_token, str) or not access_token:
        raise RuntimeError("MCP OAuth token response missing access_token")
    _token_cache["access_token"] = access_token
    _token_cache["expires_at"] = int(time.time()) + max(60, expires_in)
    return access_token


async def _get_oauth_token() -> str:
    return await asyncio.to_thread(_fetch_oauth_token_sync)


async def get_mcp_tools(scope: Scope | None = None):
    """
    Get MCP tools. If scope is set, only tools allowed for that agent are returned.
    Tool lists are cached per scope so each chat turn does not re-handshake with MCP.
    """
    cache_key = scope if scope is not None else "__all__"
    async with _tools_lock:
        if cache_key in _tools_cache:
            return _tools_cache[cache_key]

        token = await _get_oauth_token()
        client = MultiServerMCPClient(
            {
                "booking-agent-tools": {
                    "transport": "streamable_http",
                    "url": MCP_URL,
                    "headers": {"Authorization": f"Bearer {token}"},
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

from __future__ import annotations

import os
from pathlib import Path

from fastmcp import FastMCP
from fastmcp.utilities.logging import get_logger

from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from tools.validation import (
    validate_date,
    validate_time,
    validate_phone,
    validate_name,
    validate_pax,
    validate_reservation_id,
    validate_query,
    validate_top_k,
    validate_max_tables,
    validate_notes,
)
from tools.menu.menu_rag import MenuRAG
from tools.booking.booking_repo import (
    init_db,
    create_reservation,
    list_reservations,
    cancel_reservation,
    check_availability,
)

logger = get_logger(__name__)
mcp = FastMCP("booking-agent-tools")

MCP_ROOT = Path(__file__).resolve().parent
QDRANT_DIR = os.getenv("QDRANT_PATH", str(MCP_ROOT / "qdrant_local"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

# init DB tables on startup
init_db()
logger.info("Booking DB initialized")

menu_rag = MenuRAG(
    collection_name="menu",
    qdrant_path=QDRANT_DIR,
    embedding_model="nomic-embed-text",
    ollama_base_url=OLLAMA_BASE_URL,
)
logger.info(f"MenuRAG initialized. qdrant_path={QDRANT_DIR} ollama={OLLAMA_BASE_URL}")

# -------------------
# Menu tools
# -------------------
@mcp.tool()
def menu_count() -> dict:
    return {
        "collection": menu_rag.collection_name,
        "count": menu_rag.client.count(menu_rag.collection_name, exact=True).count
    }

@mcp.tool()
def query_menu(query: str, top_k: int = 3) -> dict:
    query = validate_query(query)
    top_k = validate_top_k(top_k)
    hits = menu_rag.query(query=query, top_k=top_k)
    for h in hits:
        h["text"] = (h.get("text") or "")[:800]
    return {"query": query, "top_k": top_k, "hits": hits}

# -------------------
# Booking tools
# -------------------
@mcp.tool()
def booking_check_availability(date: str, time: str, max_tables: int = 10) -> dict:
    date = validate_date(date)
    time = validate_time(time)
    max_tables = validate_max_tables(max_tables)
    return check_availability(date=date, time=time, max_tables=max_tables)

@mcp.tool()
def booking_create(name: str, phone: str, date: str, time: str, pax: int, notes: str = "") -> dict:
    name = validate_name(name)
    phone = validate_phone(phone)
    date = validate_date(date)
    time = validate_time(time)
    pax = validate_pax(pax)
    notes = validate_notes(notes)
    return create_reservation(name=name, phone=phone, date=date, time=time, pax=pax, notes=notes)

@mcp.tool()
def booking_list(date: str | None = None) -> dict:
    if date is not None:
        date = validate_date(date)
    return list_reservations(date=date)

@mcp.tool()
def booking_cancel(reservation_id: int) -> dict:
    reservation_id = validate_reservation_id(reservation_id)
    return cancel_reservation(reservation_id=reservation_id)
'''
if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=8000,
        path="/mcp",
    )
'''
middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Mcp-Session-Id"]
    )
]
app = mcp.http_app(path="/mcp", middleware=middleware)

from __future__ import annotations

import os
import json
import time
import hmac
import base64
from pathlib import Path
from hashlib import sha256

import redis

from fastapi import FastAPI, HTTPException, Request
from fastmcp import FastMCP
from fastmcp.utilities.logging import get_logger

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

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
    validate_optional_cuisine,
    validate_full_menu_flag,
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
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
REDIS_URL = os.getenv("REDIS_URL")
MENU_CACHE_TTL_SECONDS = int(os.getenv("MENU_CACHE_TTL_SECONDS"))
MCP_OAUTH_CLIENT_ID = os.getenv("MCP_OAUTH_CLIENT_ID")
MCP_OAUTH_CLIENT_SECRET = os.getenv("MCP_OAUTH_CLIENT_SECRET")
MCP_OAUTH_SIGNING_KEY = os.getenv("MCP_OAUTH_SIGNING_KEY")
MCP_OAUTH_TOKEN_TTL_SECONDS = int(os.getenv("MCP_OAUTH_TOKEN_TTL_SECONDS"))

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

redis_client: redis.Redis | None = None
try:
    _candidate = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    _candidate.ping()
    redis_client = _candidate
    logger.info(f"Redis initialized. redis_url={REDIS_URL}")
except Exception as exc:
    logger.warning(f"Redis unavailable, continuing without cache. reason={exc}")


def _menu_cache_version() -> str:
    if not redis_client:
        return "0"
    version = redis_client.get("menu_cache_version")
    if not version:
        # default generation if ingest hasn't initialized it yet
        version = "1"
        redis_client.set("menu_cache_version", version)
    return version


def _menu_cache_key(query: str, top_k: int, cuisine: str | None, full_menu: bool) -> str:
    version = _menu_cache_version()
    c = cuisine or ""
    digest = sha256(f"{query}|{top_k}|{c}|{int(full_menu)}".encode("utf-8")).hexdigest()
    return f"menu_cache:v{version}:{digest}"


def _issue_access_token(client_id: str) -> str:
    exp = int(time.time()) + MCP_OAUTH_TOKEN_TTL_SECONDS
    payload_obj = {"sub": client_id, "exp": exp}
    payload_json = json.dumps(payload_obj, separators=(",", ":")).encode("utf-8")
    payload_b64 = base64.urlsafe_b64encode(payload_json).decode("utf-8").rstrip("=")
    sig = hmac.new(
        MCP_OAUTH_SIGNING_KEY.encode("utf-8"),
        payload_b64.encode("utf-8"),
        sha256,
    ).hexdigest()
    return f"{payload_b64}.{sig}"


def _verify_access_token(token: str) -> bool:
    try:
        payload_b64, sig = token.split(".", 1)
    except ValueError:
        return False
    expected_sig = hmac.new(
        MCP_OAUTH_SIGNING_KEY.encode("utf-8"),
        payload_b64.encode("utf-8"),
        sha256,
    ).hexdigest()
    if not hmac.compare_digest(sig, expected_sig):
        return False
    pad = "=" * (-len(payload_b64) % 4)
    try:
        payload_json = base64.urlsafe_b64decode(f"{payload_b64}{pad}".encode("utf-8"))
        payload = json.loads(payload_json)
    except Exception:
        return False
    exp = payload.get("exp")
    sub = payload.get("sub")
    if not isinstance(exp, int) or not isinstance(sub, str):
        return False
    return exp > int(time.time())

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
def query_menu(query: str, top_k: int = 8, cuisine: str | None = None, full_menu: bool = False) -> dict:
    query = validate_query(query)
    top_k = validate_top_k(top_k)
    cuisine = validate_optional_cuisine(cuisine)
    full_menu = validate_full_menu_flag(full_menu)
    key = _menu_cache_key(query=query, top_k=top_k, cuisine=cuisine, full_menu=full_menu)

    if redis_client:
        try:
            cached = redis_client.get(key)
            if cached:
                payload = json.loads(cached)
                payload["cache"] = "hit"
                return payload
        except Exception as exc:
            logger.warning(f"Redis read failed; falling back to vector search. reason={exc}")

    if full_menu:
        hits = menu_rag.list_menu(cuisine=cuisine, limit=64)
    else:
        hits = menu_rag.query(query=query, top_k=top_k, cuisine=cuisine)
    for h in hits:
        h["text"] = (h.get("text") or "")[:800]

    payload = {
        "query": query,
        "top_k": top_k,
        "cuisine": cuisine,
        "full_menu": full_menu,
        "hits": hits,
        "cache": "miss",
    }
    if redis_client:
        try:
            redis_client.setex(key, MENU_CACHE_TTL_SECONDS, json.dumps(payload))
        except Exception as exc:
            logger.warning(f"Redis write failed; skipping cache store. reason={exc}")
    return payload

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
class MCPAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/oauth/token":
            return await call_next(request)
        if request.url.path.startswith("/mcp"):
            auth = request.headers.get("authorization", "")
            if not auth.startswith("Bearer "):
                return JSONResponse({"detail": "Missing bearer token"}, status_code=401)
            token = auth[7:].strip()
            if not _verify_access_token(token):
                return JSONResponse({"detail": "Invalid or expired token"}, status_code=401)
        return await call_next(request)


mcp_http_app = mcp.http_app(path="/", middleware=[])

app = FastAPI(lifespan=mcp_http_app.lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Mcp-Session-Id"],
)
app.add_middleware(MCPAuthMiddleware)


@app.post("/oauth/token")
async def oauth_token(request: Request):
    ctype = (request.headers.get("content-type") or "").lower()
    grant_type = client_id = client_secret = None
    if "application/x-www-form-urlencoded" in ctype or "multipart/form-data" in ctype:
        form = await request.form()
        grant_type = form.get("grant_type")
        client_id = form.get("client_id")
        client_secret = form.get("client_secret")
    else:
        body = await request.json()
        if isinstance(body, dict):
            grant_type = body.get("grant_type")
            client_id = body.get("client_id")
            client_secret = body.get("client_secret")

    if grant_type != "client_credentials":
        raise HTTPException(status_code=400, detail="Unsupported grant_type")
    if client_id != MCP_OAUTH_CLIENT_ID or client_secret != MCP_OAUTH_CLIENT_SECRET:
        raise HTTPException(status_code=401, detail="Invalid client credentials")

    token = _issue_access_token(client_id)
    return {
        "access_token": token,
        "token_type": "Bearer",
        "expires_in": MCP_OAUTH_TOKEN_TTL_SECONDS,
    }


app.mount("/mcp", mcp_http_app)

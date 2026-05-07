from __future__ import annotations

import os
import json
import time
import hmac
import base64
import re
from pathlib import Path
from hashlib import sha256

import redis
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue, PointStruct

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
MENU_QDRANT_COLLECTION = os.getenv("MENU_QDRANT_COLLECTION", "menu").strip() or "menu"
MENU_EMBEDDING_MODEL = os.getenv("MENU_EMBEDDING_MODEL", "nomic-embed-text").strip() or "nomic-embed-text"
REDIS_URL = os.getenv("REDIS_URL")
MENU_CACHE_TTL_SECONDS = int(os.getenv("MENU_CACHE_TTL_SECONDS"))
MENU_SEMANTIC_CACHE_ENABLED = os.getenv("MENU_SEMANTIC_CACHE_ENABLED", "1").strip() not in ("0", "false", "False")
MENU_SEMANTIC_CACHE_THRESHOLD = float(os.getenv("MENU_SEMANTIC_CACHE_THRESHOLD"))
MENU_SEMANTIC_CACHE_COLLECTION = os.getenv("MENU_SEMANTIC_CACHE_COLLECTION", "menu_query_cache").strip() or "menu_query_cache"
MENU_CACHE_DEBUG = os.getenv("MENU_CACHE_DEBUG", "0").strip() in ("1", "true", "True")
MCP_OAUTH_CLIENT_ID = os.getenv("MCP_OAUTH_CLIENT_ID")
MCP_OAUTH_CLIENT_SECRET = os.getenv("MCP_OAUTH_CLIENT_SECRET")
MCP_OAUTH_SIGNING_KEY = os.getenv("MCP_OAUTH_SIGNING_KEY")
MCP_OAUTH_TOKEN_TTL_SECONDS = int(os.getenv("MCP_OAUTH_TOKEN_TTL_SECONDS"))

# init DB tables on startup
init_db()
logger.info("Booking DB initialized")

menu_rag = MenuRAG(
    collection_name=MENU_QDRANT_COLLECTION,
    qdrant_path=QDRANT_DIR,
    embedding_model=MENU_EMBEDDING_MODEL,
    ollama_base_url=OLLAMA_BASE_URL,
)
logger.info(
    "MenuRAG initialized. collection=%s qdrant_path=%s embedding_model=%s ollama=%s",
    MENU_QDRANT_COLLECTION,
    QDRANT_DIR,
    MENU_EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
)

semantic_cache_client: QdrantClient | None = None
try:
    if MENU_SEMANTIC_CACHE_ENABLED:
        _sc = QdrantClient(path=QDRANT_DIR)
        if not _sc.collection_exists(MENU_SEMANTIC_CACHE_COLLECTION):
            _sc.create_collection(
                collection_name=MENU_SEMANTIC_CACHE_COLLECTION,
                vectors_config=VectorParams(size=menu_rag.vector_size, distance=Distance.COSINE),
            )
        semantic_cache_client = _sc
        logger.info(
            "Menu semantic cache enabled. collection=%s threshold=%s",
            MENU_SEMANTIC_CACHE_COLLECTION,
            MENU_SEMANTIC_CACHE_THRESHOLD,
        )
except Exception as exc:
    semantic_cache_client = None
    logger.warning("Menu semantic cache unavailable; continuing without it. reason=%s", exc)

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
    signature = _menu_cache_signature(query=query, cuisine=cuisine, full_menu=full_menu)
    digest = sha256(f"{signature}|{top_k}|{c}|{int(full_menu)}".encode("utf-8")).hexdigest()
    return f"menu_cache:v{version}:{digest}"


_MENU_QUERY_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "at",
        "can",
        "could",
        "do",
        "give",
        "for",
        "have",
        "get",
        "hey",
        "hi",
        "i",
        "in",
        "is",
        "it",
        "kindly",
        "list",
        "me",
        "menu",
        "of",
        "on",
        "please",
        "recommend",
        "show",
        "serve",
        "tell",
        "the",
        "to",
        "us",
        "what",
        "whats",
        "with",
        "would",
        "you",
        "your",
    }
)

_MENU_QUERY_TOKEN_SYNONYMS: dict[str, str] = {
    # collapse common menu intent words into one concept token
    "cuisine": "menu",
    "cuisines": "menu",
    "dish": "menu",
    "dishes": "menu",
    "food": "menu",
    "foods": "menu",
    "item": "menu",
    "items": "menu",
}

_MENU_LIST_INTENT_TOKENS = frozenset(
    {
        "list",
        "show",
        "share",
        "serve",
        "have",
        "available",
        "menu",
    }
)

_MENU_CUISINE_TOKENS = frozenset({"western", "asian", "fusion", "beverage", "beverages"})


def _menu_query_keyword_key(query: str) -> str:
    """
    Normalize query into stable keyword buckets so similar asks share cache entries.
    Example: "show me western menu" and "what western dish do you have" -> "dish western"
    """
    tokens = re.findall(r"[a-z0-9]+", query.lower())
    normalized = [_MENU_QUERY_TOKEN_SYNONYMS.get(t, t) for t in tokens]
    keywords = [t for t in normalized if t not in _MENU_QUERY_STOPWORDS and len(t) > 1]
    if not keywords:
        # fallback keeps cache deterministic for very short/stopword-only queries
        keywords = tokens[:]
    if not keywords:
        return "empty"
    # sorted unique string helps collapse rephrased but equivalent requests
    return " ".join(sorted(set(keywords)))


def _is_cuisine_listing_query(query: str) -> bool:
    """
    Detect broad cuisine-listing asks where query wording should not affect cache:
    - "can you share menu for western cuisine?"
    - "give me western cuisine menu"
    """
    raw_tokens = re.findall(r"[a-z0-9]+", query.lower())
    if not raw_tokens:
        return False
    normalized = [_MENU_QUERY_TOKEN_SYNONYMS.get(t, t) for t in raw_tokens]

    has_list_intent = any(t in _MENU_LIST_INTENT_TOKENS for t in normalized)
    has_cuisine_word = any(t in _MENU_CUISINE_TOKENS for t in normalized)
    if not (has_list_intent and has_cuisine_word):
        return False

    # Ignore broad question words; if nothing specific remains, treat as listing.
    informative = [
        t
        for t in normalized
        if t not in _MENU_QUERY_STOPWORDS and t not in _MENU_LIST_INTENT_TOKENS and t not in _MENU_CUISINE_TOKENS
    ]
    return len(informative) == 0


def _menu_cache_signature(query: str, cuisine: str | None, full_menu: bool) -> str:
    """
    Hybrid strategy:
    - Full-menu: cache by scope only (all/cuisine), not raw wording.
    - Cuisine listing asks: cache by cuisine only.
    - Other semantic asks: cache by normalized keyword key.
    """
    c = cuisine or "all"
    if full_menu:
        return f"full_menu:{c}"
    if cuisine and _is_cuisine_listing_query(query):
        return f"cuisine_list:{c}"
    return f"search:{_menu_query_keyword_key(query)}"


def _qdrant_menu_semantic_filter(cuisine: str | None, full_menu: bool) -> Filter:
    c = cuisine or ""
    return Filter(
        must=[
            FieldCondition(key="cuisine", match=MatchValue(value=c)),
            FieldCondition(key="full_menu", match=MatchValue(value=int(full_menu))),
        ]
    )


def _semantic_cache_lookup(query: str, cuisine: str | None, full_menu: bool) -> tuple[str | None, float | None]:
    """
    Semantic cache: find a previously seen query with similar embedding and reuse its Redis payload.
    Qdrant stores vectors + metadata; Redis stores the actual response with TTL.
    """
    if not semantic_cache_client:
        return None, None
    try:
        vec = menu_rag.embeddings.embed_query(query)
        flt = _qdrant_menu_semantic_filter(cuisine=cuisine, full_menu=full_menu)
        hits = semantic_cache_client.search(
            collection_name=MENU_SEMANTIC_CACHE_COLLECTION,
            query_vector=vec,
            limit=1,
            query_filter=flt,
            with_payload=True,
            with_vectors=False,
        )
        if not hits:
            return None, None
        best = hits[0]
        score = float(best.score) if best.score is not None else None
        if score is None or score < MENU_SEMANTIC_CACHE_THRESHOLD:
            return None, score
        payload = best.payload or {}
        redis_key = payload.get("redis_key")
        if isinstance(redis_key, str) and redis_key:
            return redis_key, score
        return None, score
    except Exception as exc:
        logger.warning("Menu semantic cache lookup failed; skipping. reason=%s", exc)
        return None, None


def _semantic_cache_store(redis_key: str, query: str, cuisine: str | None, full_menu: bool) -> None:
    """Store query embedding -> redis_key mapping in Qdrant for semantic reuse."""
    if not semantic_cache_client:
        return
    try:
        vec = menu_rag.embeddings.embed_query(query)
        pid = sha256(redis_key.encode("utf-8")).hexdigest()
        semantic_cache_client.upsert(
            collection_name=MENU_SEMANTIC_CACHE_COLLECTION,
            points=[
                PointStruct(
                    id=pid,
                    vector=vec,
                    payload={
                        "redis_key": redis_key,
                        "cuisine": cuisine or "",
                        "full_menu": int(full_menu),
                        "ts": int(time.time()),
                    },
                )
            ],
        )
    except Exception as exc:
        logger.warning("Menu semantic cache store failed; skipping. reason=%s", exc)


def _semantic_cache_count(cuisine: str | None, full_menu: bool) -> int | None:
    if not semantic_cache_client:
        return None
    try:
        flt = _qdrant_menu_semantic_filter(cuisine=cuisine, full_menu=full_menu)
        res = semantic_cache_client.count(
            collection_name=MENU_SEMANTIC_CACHE_COLLECTION,
            count_filter=flt,
            exact=True,
        )
        return int(res.count)
    except Exception:
        return None


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
    debug: dict | None = None
    if MENU_CACHE_DEBUG:
        debug = {
            "redis_enabled": bool(redis_client),
            "semantic_enabled": bool(semantic_cache_client),
            "semantic_threshold": MENU_SEMANTIC_CACHE_THRESHOLD,
            "semantic_collection": MENU_SEMANTIC_CACHE_COLLECTION,
        }

    if redis_client:
        try:
            cached = redis_client.get(key)
            if cached:
                payload = json.loads(cached)
                payload["cache"] = "hit"
                if debug is not None:
                    payload["cache_debug"] = {**debug, "mode": "redis_exact"}
                return payload
        except Exception as exc:
            logger.warning(f"Redis read failed; falling back to vector search. reason={exc}")

    # Semantic cache: rephrased queries can reuse a prior Redis payload (same cuisine/full_menu).
    if redis_client and semantic_cache_client:
        sem_key, sem_score = _semantic_cache_lookup(query=query, cuisine=cuisine, full_menu=full_menu)
        if debug is not None:
            debug["semantic_best_score"] = sem_score
            debug["semantic_candidate_key_found"] = bool(sem_key)
            debug["semantic_points_in_bucket"] = _semantic_cache_count(cuisine=cuisine, full_menu=full_menu)
        if sem_key:
            try:
                cached = redis_client.get(sem_key)
                if cached:
                    payload = json.loads(cached)
                    payload["cache"] = "semantic_hit"
                    if sem_score is not None:
                        payload["cache_score"] = sem_score
                    if debug is not None:
                        payload["cache_debug"] = {**debug, "mode": "semantic"}
                    return payload
            except Exception as exc:
                logger.warning("Semantic cache Redis read failed; skipping. reason=%s", exc)

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
    if debug is not None:
        payload["cache_debug"] = {**debug, "mode": "miss"}
    if redis_client:
        try:
            redis_client.setex(key, MENU_CACHE_TTL_SECONDS, json.dumps(payload))
            # Store semantic mapping for future rephrases; safe even if Redis TTL expires later.
            _semantic_cache_store(redis_key=key, query=query, cuisine=cuisine, full_menu=full_menu)
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

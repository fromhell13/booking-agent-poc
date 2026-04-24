from __future__ import annotations

import json
import os
import logging
from pathlib import Path
from typing import Any, List, Optional

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

ALLOWED_CUISINES = frozenset({"western", "asian", "beverage", "fusion"})

_CUISINE_KEYWORD_CONFIG: dict[str, Any] | None = None


def _default_cuisine_keyword_config() -> dict[str, Any]:
    """Fallback if no JSON file exists (same shape as cuisine_keywords.json)."""
    return {
        "scan_order": ["beverage", "fusion", "asian", "western"],
        "keywords": {
            "beverage": ["tea", "coffee", "latte", "cappuccino", "juice", "drink", "beverage", "soda"],
            "fusion": ["kimchi", "quesadilla", "rendang", "burger", "fusion"],
            "asian": ["nasi", "lemak", "tom yum", "teriyaki", "laksa", "ramen", "curry", "asian"],
            "western": ["steak", "carbonara", "caesar", "western", "spaghetti", "grilled ribeye"],
        },
        "default_cuisine": "fusion",
    }


def _load_cuisine_keyword_config() -> dict[str, Any]:
    """
    Load keyword→cuisine hints for PDF ingest only.
    Override path with env MENU_CUISINE_KEYWORDS_PATH (absolute or relative to CWD).
    """
    global _CUISINE_KEYWORD_CONFIG
    if _CUISINE_KEYWORD_CONFIG is not None:
        return _CUISINE_KEYWORD_CONFIG

    path = os.getenv("MENU_CUISINE_KEYWORDS_PATH", "").strip()
    if not path:
        path = str(Path(__file__).resolve().parent / "cuisine_keywords.json")

    p = Path(path)
    if p.is_file():
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("config must be a JSON object")
            _CUISINE_KEYWORD_CONFIG = raw
            logger.info("Loaded menu cuisine keyword config from %s", p)
            return _CUISINE_KEYWORD_CONFIG
        except Exception as exc:
            logger.warning("Invalid cuisine keyword config at %s (%s); using defaults", p, exc)

    _CUISINE_KEYWORD_CONFIG = _default_cuisine_keyword_config()
    logger.info("Using built-in default cuisine keyword config")
    return _CUISINE_KEYWORD_CONFIG


def reset_cuisine_keyword_config_cache() -> None:
    """Tests / hot-reload: clear cached config."""
    global _CUISINE_KEYWORD_CONFIG
    _CUISINE_KEYWORD_CONFIG = None


def _guess_cuisine_from_text(text: str) -> str:
    """
    Best-effort cuisine tag for PDF chunks when ingesting without menu_items.json.
    Not used for JSON ingest (each row already has cuisine). Tune keywords in
    cuisine_keywords.json or MENU_CUISINE_KEYWORDS_PATH instead of editing code.
    """
    cfg = _load_cuisine_keyword_config()
    t = (text or "").lower()
    kw_map = cfg.get("keywords")
    if not isinstance(kw_map, dict):
        kw_map = {}

    order = cfg.get("scan_order")
    if not isinstance(order, list) or not order:
        order = list(kw_map.keys())

    for cuisine in order:
        if not isinstance(cuisine, str):
            continue
        c = cuisine.strip().lower()
        if c not in ALLOWED_CUISINES:
            continue
        words = kw_map.get(cuisine) or kw_map.get(c)
        if not isinstance(words, list):
            continue
        for w in words:
            if isinstance(w, str) and w.lower() in t:
                return c

    default = cfg.get("default_cuisine", "fusion")
    if isinstance(default, str) and default.lower() in ALLOWED_CUISINES:
        return default.lower()
    return "fusion"


def _menu_item_document(item: dict) -> Document:
    name = (item.get("name") or "").strip()
    price = (item.get("price") or "").strip()
    currency = (item.get("currency") or "MYR").strip()
    cuisine = (item.get("cuisine") or "fusion").strip().lower()
    category = (item.get("category") or "other").strip().lower()
    if cuisine not in ALLOWED_CUISINES:
        cuisine = "fusion"
    line = f"{name}\n{currency} {price}".strip()
    return Document(
        page_content=line,
        metadata={
            "cuisine": cuisine,
            "category": category,
            "name": name,
            "source": "menu_items.json",
        },
    )


def _qdrant_cuisine_filter(cuisine: str) -> Filter:
    return Filter(
        must=[
            FieldCondition(
                key="metadata.cuisine",
                match=MatchValue(value=cuisine),
            )
        ]
    )


def _payload_to_hit(payload: dict[str, Any]) -> dict:
    text = str(payload.get("page_content") or payload.get("_page_content") or "")
    meta = payload.get("metadata")
    if not isinstance(meta, dict):
        meta = {}
    return {"text": text, "metadata": meta}


class MenuRAG:
    def __init__(
        self,
        collection_name: str = "menu",
        qdrant_path: str = "qdrant_local",
        embedding_model: str = "nomic-embed-text",
        ollama_base_url: str = "http://127.0.0.1:11434",
        keep_alive_seconds: int = 1800,
    ):
        self.collection_name = collection_name

        self.client = QdrantClient(path=qdrant_path)

        _timeout = float(os.getenv("OLLAMA_HTTP_TIMEOUT", "600"))
        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url=ollama_base_url,
            keep_alive=keep_alive_seconds,
            client_kwargs={"timeout": _timeout},
        )

        self.vector_size = 768

        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

    def reset_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
        )
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

    def ingest_json(self, json_path: str) -> dict:
        path = Path(json_path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        items = raw.get("items") if isinstance(raw, dict) else None
        if not isinstance(items, list) or not items:
            raise ValueError("menu JSON must contain a non-empty 'items' array")

        self.reset_collection()
        docs = [_menu_item_document(x) for x in items if isinstance(x, dict)]
        ids = self.vector_store.add_documents(docs)
        logger.info(f"ingest_json inserted={len(ids)} collection={self.collection_name}")
        return {"collection": self.collection_name, "inserted": len(ids), "source": str(path)}

    def ingest_pdf(self, pdf_path: str) -> dict:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=40,
            add_start_index=True,
        )
        splits = splitter.split_documents(pages)
        for doc in splits:
            c = _guess_cuisine_from_text(doc.page_content)
            doc.metadata = {**doc.metadata, "cuisine": c, "source": "sample_menu.pdf"}

        self.reset_collection()
        ids = self.vector_store.add_documents(splits)
        logger.info(f"ingest_pdf inserted={len(ids)} collection={self.collection_name}")
        return {"collection": self.collection_name, "inserted": len(ids), "source": str(pdf_path)}

    def list_menu(self, cuisine: Optional[str] = None, limit: int = 64) -> List[dict]:
        flt: Optional[Filter] = _qdrant_cuisine_filter(cuisine) if cuisine else None
        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=flt,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        hits = []
        for r in records:
            if r.payload:
                hits.append(_payload_to_hit(r.payload))
        return hits

    def query(self, query: str, top_k: int = 8, cuisine: Optional[str] = None) -> List[dict]:
        flt: Optional[Filter] = _qdrant_cuisine_filter(cuisine) if cuisine else None
        results = self.vector_store.similarity_search(query, k=top_k, filter=flt)
        return [{"text": r.page_content, "metadata": dict(r.metadata)} for r in results]

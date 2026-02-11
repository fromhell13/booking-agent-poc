from __future__ import annotations

import os
import logging
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


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

        # keep_alive expects int in your installed version
        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url=ollama_base_url,
            keep_alive=keep_alive_seconds,
        )

        # Do not call embeddings at startup to get dimension.
        # nomic-embed-text is 768-dim.
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

    def ingest_pdf(self, pdf_path: str) -> dict:
        
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )

        splits = splitter.split_documents(pages)
        ids = self.vector_store.add_documents(splits)
        logger.info(f"ingest_pdf inserted={len(ids)} collection={self.collection_name}")
        return {"collection": self.collection_name, "inserted": len(ids)}

    def query(self, query: str, top_k: int = 3) -> List[dict]:
        results = self.vector_store.similarity_search(query, k=top_k)
        return [{"text": r.page_content, "metadata": r.metadata} for r in results]

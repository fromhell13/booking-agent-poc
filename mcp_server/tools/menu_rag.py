# tools/menu_rag.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, List

from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.http import models as qm
from langchain_text_splitters import RecursiveCharacterTextSplitter




class MenuRAG:
    def __init__(
        self,
        collection_name: str = "menu",
        qdrant_path: str = "qdrant_local",
        #qdrant_path: str = ":memory:",
        embedding_model: str = "nomic-embed-text",
    ):
        self.collection_name = collection_name
        self.client = QdrantClient(path=qdrant_path)
        #self.client = QdrantClient(qdrant_path)
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.vector_size = len(self.embeddings.embed_query("dim"))

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

    def ingest_pdf(self,pdf_path: str) -> dict:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        # Split document into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
        
        all_splits = text_splitter.split_documents(pages)
        ids = self.vector_store.add_documents(all_splits)
        results = self.vector_store.similarity_search(
            "Can you show menu for western cuisine?",
        )
        print(results[0])
        print(f"collection: {self.collection_name}, inserted: {len(ids)}")
        return {
            "collection": self.collection_name,
            "inserted": len(ids),
        }

    def query(self, query: str, top_k: int = 3) -> List[dict]:
        results = self.vector_store.similarity_search(query, k=top_k)

        return [
            {
                "text": r.page_content,
                "metadata": r.metadata,
            }
            for r in results
        ]

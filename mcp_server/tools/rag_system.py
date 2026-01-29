import os
import json
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
import glob

import redis
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import Qdrant


def tool(func):
    """MCP tool decorator"""
    func._is_mcp_tool = True
    return func


class QdrantRAGConfig:
    def __init__(self):
        self.qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", 6379))
        self.redis_db = int(os.getenv("REDIS_DB", 0))
        self.collection_name = os.getenv("COLLECTION_NAME", "restaurant_menu")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.cache_ttl = int(os.getenv("CACHE_TTL", 3600))  # 1 hour
        self.pdf_directory = os.getenv("PDF_DIRECTORY", "pdf_menu")


class QdrantRAGTool:
    def __init__(self, config: Optional[QdrantRAGConfig] = None):
        self.config = config or QdrantRAGConfig()
        self.embeddings = SentenceTransformerEmbeddings(
            model_name=self.config.embedding_model
        )
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            host=self.config.qdrant_host,
            port=self.config.qdrant_port
        )
        
        # Initialize Redis client
        self.redis_client = redis.Redis(
            host=self.config.redis_host,
            port=self.config.redis_port,
            db=self.config.redis_db,
            decode_responses=True
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        self.vector_store = None
        self._ensure_collection_exists()
        self._auto_process_pdfs()

    def _ensure_collection_exists(self):
        """Auto-create Qdrant collection if it doesn't exist"""
        try:
            collections = self.qdrant_client.get_collections()
            collection_exists = any(
                col.name == self.config.collection_name 
                for col in collections.collections
            )
            
            if not collection_exists:
                # Create collection with embedding dimension
                self.qdrant_client.create_collection(
                    collection_name=self.config.collection_name,
                    vectors_config=VectorParams(
                        size=384,  # all-MiniLM-L6-v2 dimension
                        distance=Distance.COSINE
                    )
                )
                print(f"Auto-created collection: {self.config.collection_name}")
            
            # Initialize LangChain Qdrant vector store
            self.vector_store = Qdrant(
                client=self.qdrant_client,
                collection_name=self.config.collection_name,
                embeddings=self.embeddings
            )
            
        except Exception as e:
            print(f"Error setting up collection: {e}")
            raise

    def _auto_process_pdfs(self):
        """Automatically process PDFs from pdf_menu directory"""
        pdf_dir = Path(self.config.pdf_directory)
        if not pdf_dir.exists():
            print(f"PDF directory '{pdf_dir}' does not exist. Creating it...")
            pdf_dir.mkdir(parents=True, exist_ok=True)
            return
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in '{pdf_dir}' directory")
            return
        
        # Check if collection is empty
        try:
            collection_info = self.qdrant_client.get_collection(self.config.collection_name)
            if collection_info.points_count == 0:
                print("Empty collection detected. Processing PDFs...")
                for pdf_file in pdf_files:
                    self._process_single_pdf(str(pdf_file))
        except Exception as e:
            print(f"Error checking collection: {e}")

    def _process_single_pdf(self, pdf_path: str) -> int:
        """Process a single PDF and store embeddings"""
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            texts = self.text_splitter.split_documents(documents)
            
            # Add metadata
            for i, doc in enumerate(texts):
                doc.metadata.update({
                    "source": pdf_path,
                    "chunk_id": i,
                    "document_type": "menu",
                    "filename": Path(pdf_path).name
                })
            
            # Add to vector store
            self.vector_store.add_documents(texts)
            
            print(f"Processed {len(texts)} chunks from {Path(pdf_path).name}")
            return len(texts)
            
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")
            raise

    def _get_cache_key(self, query: str, top_k: int) -> str:
        """Generate cache key for query"""
        query_hash = hashlib.md5(f"{query}_{top_k}".encode()).hexdigest()
        return f"qdrant_query:{query_hash}"

    def _clear_cache(self):
        """Clear all cached queries"""
        try:
            keys = self.redis_client.keys("qdrant_query:*")
            if keys:
                self.redis_client.delete(*keys)
                print("Cache cleared")
        except Exception as e:
            print(f"Error clearing cache: {e}")

    @tool
    def query_menu(self, query: str, top_k: int = 5, use_cache: bool = True) -> Dict[str, Any]:
        """
        Query the restaurant menu vector database.
        
        Args:
            query: The search query about menu items
            top_k: Number of similar documents to retrieve (default: 5)
            use_cache: Whether to use Redis caching (default: True)
        
        Returns:
            Dict containing relevant menu information and metadata
        """
        cache_key = self._get_cache_key(query, top_k)
        
        # Check cache first
        if use_cache:
            try:
                cached_result = self.redis_client.get(cache_key)
                if cached_result:
                    result_data = json.loads(cached_result)
                    result_data['cached'] = True
                    return result_data
            except Exception as e:
                print(f"Cache read error: {e}")
        
        try:
            # Query vector database
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=top_k
            )
            
            if not results:
                return {
                    "status": "no_results",
                    "message": "No relevant menu information found",
                    "results": [],
                    "cached": False
                }
            
            # Format results
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "score": float(score),
                    "metadata": {
                        "filename": doc.metadata.get('filename', 'unknown'),
                        "source": doc.metadata.get('source', 'unknown'),
                        "chunk_id": doc.metadata.get('chunk_id', 0)
                    }
                })
            
            result_data = {
                "status": "success",
                "query": query,
                "results": formatted_results,
                "total_results": len(formatted_results),
                "cached": False
            }
            
            # Cache result
            if use_cache:
                try:
                    self.redis_client.setex(
                        cache_key,
                        self.config.cache_ttl,
                        json.dumps(result_data)
                    )
                except Exception as e:
                    print(f"Cache write error: {e}")
            
            return result_data
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error querying database: {str(e)}",
                "results": [],
                "cached": False
            }

    @tool
    def get_menu_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the menu database.
        
        Returns:
            Dict containing database statistics
        """
        try:
            # Qdrant stats
            collection_info = self.qdrant_client.get_collection(self.config.collection_name)
            
            # Redis stats
            redis_info = self.redis_client.info()
            cache_keys = len(self.redis_client.keys("qdrant_query:*"))
            
            # PDF directory stats
            pdf_dir = Path(self.config.pdf_directory)
            pdf_count = len(list(pdf_dir.glob("*.pdf"))) if pdf_dir.exists() else 0
            
            return {
                "qdrant": {
                    "collection_name": self.config.collection_name,
                    "points_count": collection_info.points_count,
                    "vectors_count": collection_info.vectors_count,
                    "status": "connected"
                },
                "redis": {
                    "connected_clients": redis_info.get("connected_clients", 0),
                    "used_memory_human": redis_info.get("used_memory_human", "0B"),
                    "cached_queries": cache_keys,
                    "status": "connected"
                },
                "pdf_files": {
                    "directory": str(pdf_dir),
                    "count": pdf_count,
                    "processed": collection_info.points_count > 0
                }
            }
        except Exception as e:
            return {"error": str(e)}

    @tool
    def refresh_menu_data(self) -> Dict[str, Any]:
        """
        Refresh menu data by reprocessing PDFs and clearing cache.
        
        Returns:
            Dict containing refresh results
        """
        try:
            # Clear existing data
            self.qdrant_client.delete_collection(self.config.collection_name)
            self._clear_cache()
            
            # Recreate collection
            self._ensure_collection_exists()
            
            # Process all PDFs
            pdf_dir = Path(self.config.pdf_directory)
            pdf_files = list(pdf_dir.glob("*.pdf"))
            
            total_chunks = 0
            processed_files = []
            
            for pdf_file in pdf_files:
                try:
                    chunks = self._process_single_pdf(str(pdf_file))
                    total_chunks += chunks
                    processed_files.append({
                        "filename": pdf_file.name,
                        "chunks": chunks,
                        "status": "success"
                    })
                except Exception as e:
                    processed_files.append({
                        "filename": pdf_file.name,
                        "chunks": 0,
                        "status": "error",
                        "error": str(e)
                    })
            
            return {
                "status": "success",
                "total_chunks_processed": total_chunks,
                "files_processed": len(processed_files),
                "files": processed_files,
                "cache_cleared": True
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    @tool
    def search_menu_items(self, item_type: str = None, keywords: List[str] = None, top_k: int = 10) -> Dict[str, Any]:
        """
        Search for specific menu items or categories.
        
        Args:
            item_type: Type of item to search for (e.g., "appetizer", "main course", "dessert")
            keywords: List of keywords to search for
            top_k: Number of results to return (default: 10)
        
        Returns:
            Dict containing search results
        """
        # Build search query
        query_parts = []
        if item_type:
            query_parts.append(item_type)
        if keywords:
            query_parts.extend(keywords)
        
        if not query_parts:
            return {
                "status": "error",
                "message": "Please provide either item_type or keywords to search"
            }
        
        search_query = " ".join(query_parts)
        return self.query_menu(search_query, top_k=top_k, use_cache=True)

    @tool
    def clear_menu_cache(self) -> Dict[str, str]:
        """
        Clear the Redis cache for menu queries.
        
        Returns:
            Dict containing clear operation status
        """
        try:
            self._clear_cache()
            return {"status": "success", "message": "Menu cache cleared successfully"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
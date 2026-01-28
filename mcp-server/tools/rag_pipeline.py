from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain.schema import Document
from qdrant_client.http import models as qm
from pathlib import Path

# Define menu headings
HEADINGS = {"Western", "Asian", "Fusion", "Beverages"}
def split_menu_by_heading(text: str) -> list[Document]:
    docs: list[Document] = []
    current = None
    buf: list[str] = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        if line in HEADINGS:
            # flush previous section
            if current and buf:
                docs.append(
                    Document(
                        page_content="\n".join(buf),
                        metadata={"category": current}
                    )
                )
            current = line
            buf = []
            continue

        # normal line (menu item / price)
        buf.append(line)

    # last section
    if current and buf:
        docs.append(Document(page_content="\n".join(buf), metadata={"category": current}))

    return docs

# Define base directory and file path
BASE_DIR = Path(__file__).resolve().parent.parent
file_path = BASE_DIR / "menu" / "sample_menu.pdf"

# Load PDF document
loader = PyPDFLoader(file_path)
docs = loader.load()

# Combine all pages into a single text
full_text = "\n".join(d.page_content for d in docs)
section_docs = split_menu_by_heading(full_text)

# Split document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
#all_splits = section_docs

# Generate embeddings using Ollama
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_1 = embeddings.embed_query(all_splits[0].page_content)

# Initialize Qdrant client and create collection
client = QdrantClient(":memory:")

vector_size = len(embeddings.embed_query("hello world"))

if not client.collection_exists("menu"):
    client.create_collection(
        collection_name="menu",
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )
vector_store = QdrantVectorStore(
    client=client,
    collection_name="menu",
    embedding=embeddings,
)

# Index document chunks into Qdrant
ids = vector_store.add_documents(documents=all_splits)

results = vector_store.similarity_search(
    "Can you show menu for western cuisine?",
)

print(results[0])

'''

query = "Can you show menu for western cuisine?"
qdrant_filter = qm.Filter(
    must=[qm.FieldCondition(
        key="metadata.category",
        match=qm.MatchValue(value="Western")
    )]
)

results = vector_store.similarity_search(query, k=3, filter=qdrant_filter)
print(results[0].page_content)
'''
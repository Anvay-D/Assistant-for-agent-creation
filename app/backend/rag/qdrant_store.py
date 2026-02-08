"""
Ingest Qdrant OpenAPI specs into Qdrant vector DB
Embedding model: all-MiniLM-L6-v2 (384 dims)
Purpose: API-aware RAG with zero hallucination
"""

import logging
from typing import List

import httpx
import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from rag.collection_store import QDRANT_COLLECTION
# -------------------------------------------------
# Logging
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# -------------------------------------------------
# Constants
# -------------------------------------------------
OPENAPI_BASE_URL = "https://raw.githubusercontent.com/qdrant/qdrant/master/openapi"

OPENAPI_FILES = [
    "openapi-collections.ytt.yaml",
    "openapi-points.ytt.yaml",
    "openapi-main.ytt.yaml",
    "openapi-shards.ytt.yaml",
    "openapi-shard-snapshots.ytt.yaml",
    "openapi-snapshots.ytt.yaml",
    "openapi-service.ytt.yaml",
    "openapi-cluster.ytt.yaml",
]

QDRANT_URL = "http://localhost:6333"


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DIM = 384

# -------------------------------------------------
# Embeddings
# -------------------------------------------------
logger.info("🔤 Loading embedding model")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# -------------------------------------------------
# Qdrant helpers
# -------------------------------------------------
def ensure_collection_exists() -> None:
    """Create Qdrant collection if it does not exist"""
    client = QdrantClient(url=QDRANT_URL)

    if client.collection_exists(QDRANT_COLLECTION):
        logger.info(f"✅ Collection '{QDRANT_COLLECTION}' already exists")
        return

    logger.info(f"🆕 Creating collection '{QDRANT_COLLECTION}'")
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(
            size=VECTOR_DIM,
            distance=Distance.COSINE,
        ),
    )
    logger.info("✅ Collection created")


# -------------------------------------------------
# OpenAPI loaders
# -------------------------------------------------
def load_openapi_yaml(filename: str) -> dict:
    url = f"{OPENAPI_BASE_URL}/{filename}"
    logger.info(f"📥 Fetching OpenAPI spec: {filename}")
    response = httpx.get(url, timeout=30, follow_redirects=True)
    response.raise_for_status()
    return yaml.safe_load(response.text)


def openapi_to_documents(spec: dict, source_file: str) -> List[Document]:
    """Convert OpenAPI spec into LangChain Documents"""
    documents: List[Document] = []

    for path, methods in spec.get("paths", {}).items():
        for method, meta in methods.items():
            document = Document(
                page_content=f"""
ENDPOINT:
{method.upper()} {path}

OPERATION ID:
{meta.get("operationId", "")}

SUMMARY:
{meta.get("summary", "")}

DESCRIPTION:
{meta.get("description", "")}

REQUEST BODY:
{meta.get("requestBody", {})}

RESPONSES:
{meta.get("responses", {})}
""".strip(),
                metadata={
                    "source": source_file,
                    "path": path,
                    "method": method.upper(),
                    "operation_id": meta.get("operationId", ""),
                    "deprecated": meta.get("deprecated", False),
                    "type": "qdrant_api",
                },
            )
            documents.append(document)

    return documents


# -------------------------------------------------
# Ingestion pipeline
# -------------------------------------------------
def ingest_openapi_documents() -> List[Document]:
    """Load and chunk all OpenAPI specs"""
    all_docs: List[Document] = []

    for file in OPENAPI_FILES:
        spec = load_openapi_yaml(file)
        all_docs.extend(openapi_to_documents(spec, file))

    logger.info(f"📄 Collected {len(all_docs)} endpoint documents")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks = splitter.split_documents(all_docs)
    logger.info(f"✂️ Created {len(chunks)} text chunks")

    return chunks


# -------------------------------------------------
# Store in Qdrant
# -------------------------------------------------
def store_qdrant_api_docs() -> None:
    ensure_collection_exists()

    chunks = ingest_openapi_documents()

    logger.info(f"📤 Storing {len(chunks)} chunks in Qdrant")
    Qdrant.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=QDRANT_COLLECTION,
    )

    logger.info("🎉 Qdrant API ingestion completed successfully")


# -------------------------------------------------
# Entry point
# -------------------------------------------------
if __name__ == "__main__":
    store_qdrant_api_docs()
# -------------------------------------------------
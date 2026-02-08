import re
import logging
from typing import List

import requests
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from rag.collection_store import (LANGCHAIN_COLLECTION, LANGGRAPH_COLLECTION)
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
LLMS_TXT_URL = "https://docs.langchain.com/llms.txt"
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
def ensure_collection_exists(collection_name: str) -> None:
    client = QdrantClient(url=QDRANT_URL)

    if client.collection_exists(collection_name):
        logger.info(f"✅ Collection '{collection_name}' already exists")
        return

    logger.info(f"🆕 Creating collection '{collection_name}'")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=VECTOR_DIM,
            distance=Distance.COSINE,
        ),
    )
    logger.info("✅ Collection created")

# -------------------------------------------------
# Fetch helpers
# -------------------------------------------------
def fetch_text(url: str) -> str:
    logger.info(f"Fetching: {url}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.text


def extract_urls_from_llms_txt(text: str) -> List[str]:
    # Markdown-style links: [title](url)
    urls = re.findall(r"\((https?://[^)]+)\)", text)
    logger.info(f"🔗 Extracted {len(urls)} URLs from llms.txt")
    return urls

# -------------------------------------------------
# Ingestion logic
# -------------------------------------------------
def ingest_docs(filter_keyword: str) -> List[Document]:
    """
    filter_keyword:
        "langchain"  -> LangChain docs
        "langgraph"  -> LangGraph docs
    """
    llms_txt = fetch_text(LLMS_TXT_URL)
    urls = extract_urls_from_llms_txt(llms_txt)

    docs: List[Document] = []

    for i, url in enumerate(urls, start=1):
        if filter_keyword not in url.lower():
            continue

        try:
            logger.info(f"[{i}] Processing {url}")
            content = fetch_text(url)

            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": url,
                        "project": filter_keyword,
                    },
                )
            )
        except Exception as e:
            logger.warning(f"Skipping {url}: {e}")

    logger.info(f"📄 Collected {len(docs)} raw documents")
    return docs

# -------------------------------------------------
# Chunking
# -------------------------------------------------
def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks = splitter.split_documents(docs)
    chunks = [c for c in chunks if c.page_content.strip()]
    logger.info(f"✂️ Created {len(chunks)} chunks")
    return chunks

# -------------------------------------------------
# Store pipeline
# -------------------------------------------------
def store_docs(collection_name: str, chunks: List[Document]) -> None:
    ensure_collection_exists(collection_name)

    Qdrant.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=collection_name,
    )

    logger.info(f"🎉 Stored {len(chunks)} chunks in '{collection_name}'")

# -------------------------------------------------
# Entry point
# -------------------------------------------------
if __name__ == "__main__":
    logger.info("🚀 Starting LangChain + LangGraph ingestion")

    # LangChain
    langchain_docs = ingest_docs("langchain")
    langchain_chunks = chunk_documents(langchain_docs)
    store_docs(LANGCHAIN_COLLECTION, langchain_chunks)

    # LangGraph
    langgraph_docs = ingest_docs("langgraph")
    langgraph_chunks = chunk_documents(langgraph_docs)
    store_docs(LANGGRAPH_COLLECTION, langgraph_chunks)

    logger.info("✅ Ingestion completed successfully")

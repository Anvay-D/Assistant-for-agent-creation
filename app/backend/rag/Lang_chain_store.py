import re
import requests
import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# NEW (recommended)
from langchain_huggingface import HuggingFaceEmbeddings  # pip install -U langchain-huggingface sentence-transformers
from langchain_community.vectorstores import Qdrant


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

LLMS_TXT_URL = "https://docs.langchain.com/llms.txt"
COLLECTION = "langchain_docs"
QDRANT_URL = "http://localhost:6333"


def fetch_text(url: str) -> str:
    logger.info(f"Fetching URL: {url}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    logger.info(f"Fetched {len(r.text)} characters from {url}")
    return r.text


def extract_urls_from_llms_txt(text: str) -> List[str]:
    urls = re.findall(r"\((https?://[^)]+)\)", text)
    logger.info(f"Extracted {len(urls)} URLs from llms.txt")
    return urls


def html_to_text(html_or_md: str) -> str:
    return html_or_md


logger.info("Starting LangChain docs ingestion")

logger.info("Downloading llms.txt")
llms_txt = fetch_text(LLMS_TXT_URL)
urls = extract_urls_from_llms_txt(llms_txt)

docs: List[Document] = []
for i, u in enumerate(urls, start=1):
    try:
        logger.info(f"[{i}/{len(urls)}] Processing {u}")
        raw = fetch_text(u)
        docs.append(Document(page_content=html_to_text(raw), metadata={"source": u}))
    except Exception as e:
        logger.warning(f"Skipping {u} due to error: {e}")

logger.info(f"Total raw documents collected: {len(docs)}")

logger.info("Splitting documents into chunks")
splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)

chunks = splitter.split_documents(docs)
chunks = [c for c in chunks if c.page_content.strip()]
logger.info(f"Total chunks after splitting & cleanup: {len(chunks)}")

logger.info("Loading embedding model: all-MiniLM-L6-v2")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)  # init style per docs [web:29][web:30]

logger.info(f"Storing chunks in Qdrant collection '{COLLECTION}' at {QDRANT_URL}")
vs = Qdrant.from_documents(
    documents=chunks,
    embedding=embeddings,
    url=QDRANT_URL,
    collection_name=COLLECTION,
)

logger.info("Ingestion completed successfully")
logger.info(f"Ingested chunks count: {len(chunks)}")

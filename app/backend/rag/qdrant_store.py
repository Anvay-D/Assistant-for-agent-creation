import re
import requests
import logging
from typing import List
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient  # ADD THIS
from qdrant_client.models import Distance, VectorParams  # ADD THIS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant

# -------------------------------------------------
# Logging setup
# -------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# -------------------------------------------------
# Constants
# -------------------------------------------------
QDRANT_BASE_URL = "https://api.qdrant.tech"
QDRANT_SECTIONS = [
    "https://api.qdrant.tech/api-reference/collections/",
    "https://api.qdrant.tech/api-reference/points/",
    "https://api.qdrant.tech/api-reference/search/",
    "https://api.qdrant.tech/api-reference/indexes/",
    "https://api.qdrant.tech/api-reference/snapshots/",
    "https://api.qdrant.tech/api-reference/aliases/",
    "https://api.qdrant.tech/api-reference/distributed/",
    "https://api.qdrant.tech/api-reference/service/",
]

COLLECTION_QDRANT = "qdrant_api_complete"
QDRANT_URL = "http://localhost:6333"

# -------------------------------------------------
# ✅ AUTO CREATE COLLECTION IF NEEDED
# -------------------------------------------------
def ensure_collection_exists():
    """Create collection if it doesn't exist (384 dims = all-MiniLM-L6-v2)"""
    client = QdrantClient(QDRANT_URL)
    
    if client.collection_exists(COLLECTION_QDRANT):
        logger.info(f"✅ Collection '{COLLECTION_QDRANT}' already exists")
        return
    
    # Create new collection with correct dimensions
    vectors_config = VectorParams(size=384, distance=Distance.COSINE)
    
    client.create_collection(
        collection_name=COLLECTION_QDRANT,
        vectors_config=vectors_config
    )
    logger.info(f"✅ Created collection '{COLLECTION_QDRANT}' (384-dim COSINE)")

# -------------------------------------------------
# Your existing helpers (unchanged)
# -------------------------------------------------
def fetch_text(url: str) -> str:
    logger.info(f"Fetching: {url}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.text

def extract_all_subpages(section_url: str) -> List[str]:
    html = fetch_text(section_url)
    soup = BeautifulSoup(html, 'html.parser')
    
    urls = set()
    for link in soup.find_all('a', href=True):
        href = link['href']
        if (href.startswith('/api-reference/') and 
            '/api-reference/' + section_url.split('/api-reference/')[1] in href and
            not href.endswith('/')):
            full_url = urljoin(QDRANT_BASE_URL, href)
            urls.add(full_url)
    
    content_links = soup.find_all('a', href=re.compile(r'/api-reference/'))
    for link in content_links:
        href = link['href']
        if href.startswith('/'):
            full_url = urljoin(QDRANT_BASE_URL, href)
            if any(section.split('/api-reference/')[1] in full_url for section in QDRANT_SECTIONS):
                urls.add(full_url)
    
    logger.info(f"Found {len(urls)} subpages in {section_url}")
    return list(urls)

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, 'html.parser')
    for unwanted in soup(['nav', 'header', 'footer', 'aside', 'script', 'style']):
        unwanted.decompose()
    
    main_content = soup.find(['main', '.content', '[class*="content"]', 'article'])
    if main_content:
        soup = main_content
    
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = ' '.join(chunk for chunk in chunks if len(chunk) > 3)
    
    return text[:30000]

# -------------------------------------------------
# FIXED Embeddings
# -------------------------------------------------
logger.info("Loading embedding model: all-MiniLM-L6-v2")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# -------------------------------------------------
# MAIN INGESTION (unchanged)
# -------------------------------------------------
def ingest_qdrant_api_complete():
    logger.info("=== QDRANT API COMPLETE INGESTION ===")
    
    all_urls = set()
    for section in QDRANT_SECTIONS:
        logger.info(f"🔍 Scraping section: {section}")
        subpages = extract_all_subpages(section)
        all_urls.update(subpages)
    
    all_urls = list(all_urls)
    logger.info(f"📋 Total unique Qdrant API pages: {len(all_urls)}")
    
    docs: List[Document] = []
    for i, url in enumerate(all_urls, 1):
        try:
            logger.info(f"[{i}/{len(all_urls)}] Processing: {url}")
            html = fetch_text(url)
            cleaned_text = html_to_text(html)
            
            if len(cleaned_text.strip()) > 300:
                docs.append(Document(
                    page_content=cleaned_text,
                    metadata={
                        "source": url,
                        "type": "qdrant_api",
                        "section": url.split('/api-reference/')[1].split('/')[0]
                    }
                ))
        except Exception as e:
            logger.warning(f"❌ Skip {url}: {e}")
    
    logger.info(f"✅ Collected {len(docs)} valid documents")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = splitter.split_documents(docs)
    chunks = [c for c in chunks if len(c.page_content.strip()) > 50]
    
    logger.info(f"✅ Final chunks: {len(chunks)}")
    return chunks

# -------------------------------------------------
# ✅ FIXED Store function
# -------------------------------------------------
def store_qdrant_api():
    # STEP 1: Ensure collection exists
    ensure_collection_exists()
    
    # STEP 2: Ingest documents
    chunks = ingest_qdrant_api_complete()
    
    # STEP 3: Use SAFE from_documents (now collection exists)
    logger.info(f"📤 Storing {len(chunks)} chunks to '{COLLECTION_QDRANT}'")
    vs = Qdrant.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=QDRANT_URL,  # ✅ FIXED: use 'url' not 'path'
        collection_name=COLLECTION_QDRANT,
    )
    logger.info("🎉 QDRANT API INGESTION COMPLETE!")
    return vs

# -------------------------------------------------
# RUN
# -------------------------------------------------
if __name__ == "__main__":
    store_qdrant_api()

# app/rag/qdrant_client.py
from qdrant_client import QdrantClient
from config import QDRANT_URL

qdrant_client = QdrantClient(url=QDRANT_URL)

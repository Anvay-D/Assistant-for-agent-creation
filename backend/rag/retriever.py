from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter


client = QdrantClient("http://localhost:6333")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

client.recreate_collection(
    "Langchain_db",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

loader = WebBaseLoader("https://docs.langchain.com/oss/python/integrations")
docs = loader.load()

print("Raw docs:", len(docs))

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(docs)
docs = [d for d in docs if d.page_content.strip()]

print("Final chunks:", len(docs))

store = Qdrant(
    client=client,
    collection_name="Langchain_db",
    embeddings=embeddings,
)

ids = store.add_documents(docs)
print("Inserted:", len(ids))

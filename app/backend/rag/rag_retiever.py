# retriever_qdrant.py
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

COLLECTION = "langchain_docs"

client = QdrantClient("http://localhost:6333")
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve_context(query: str, top_k: int = 5) -> str:
    # 1. Ensure the query isn't empty before encoding
    if not query.strip():
        return "No query provided."

    query_vector = embedder.encode(query).tolist()

    response = client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    # 2. Access .points explicitly for clarity
    hits = response.points if hasattr(response, 'points') else response

    contexts = []
    for point in hits:
        payload = point.payload or {}
        # 3. Standard Qdrant attribute is .score
        score = point.score if hasattr(point, 'score') else 0.0
        
        # 4. Use .get() with a fallback to avoid NoneType errors
        text = payload.get("page_content") or payload.get("text") or ""
        metadata = payload.get("metadata") or {}
        source = metadata.get("source", "unknown")
        
        if not text.strip():
            continue
            
        contexts.append(
            f"[Source: {source} | Score: {score:.3f}]\n{text[:800]}..."
        )

    return "\n\n---\n\n".join(contexts) if contexts else "No relevant context found."
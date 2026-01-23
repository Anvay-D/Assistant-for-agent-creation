from qdrant_client import QdrantClient

client = QdrantClient("http://localhost:6333")

collections = client.get_collections()
print(collections)


info = client.get_collection("Langchain_db")
print("Points count:", info.points_count)
print("Vectors:", info.config.params.vectors)

points, _ = client.scroll(
    collection_name="Langchain_db",
    limit=5,
    with_payload=True,
    with_vectors=False,
)

for p in points:
    print("ID:", p.id)
    print("Payload keys:", p.payload.keys())
    print("Text preview:", p.payload.get("page_content", "")[:300])
    print("-" * 50)

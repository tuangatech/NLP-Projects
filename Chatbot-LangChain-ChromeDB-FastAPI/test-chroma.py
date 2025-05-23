import chromadb
from chromadb.config import Settings
import json

# Connect to ChromaDB (running locally or in Docker)
client = chromadb.HttpClient(
    host="localhost",  # or IP if remote
    port=8001,
    settings=Settings(anonymized_telemetry=False)
)

# Check connection
client.heartbeat()
print("Connected to ChromaDB")

collections = client.list_collections()
print("Available collections:", collections)

collection_name = "rag_documents"
collection = client.get_collection(collection_name)
print(f"Number of docs in `{collection.name}`: {collection.count()}")  # Should be > 0

# Peek at first 5 records
results = collection.peek(3)

# Print results
print("Peeked records:")

for i in range(len(results['ids'])):
    record = {
        "id": results["ids"][i],
        "metadata": results["metadatas"][i] if results["metadatas"] else None,
        "document": results["documents"][i] if results["documents"] else None
    }
    print(json.dumps(record, indent=2))
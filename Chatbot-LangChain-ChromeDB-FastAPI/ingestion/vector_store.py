# vector_store.py
import chromadb
from chromadb.utils import embedding_functions
import hashlib
from datetime import datetime
from typing import List, Dict, Any
import logging
from config import CHROMA_HOST, CHROMA_PORT, COLLECTION_NAME, OPENAI_API_KEY, EMBEDDING_MODEL, CHROMA_BATCH_SIZE

logger = logging.getLogger(__name__)

def connect_to_chromadb():
    try:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        client.heartbeat()
        logger.info(f"Connected to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}")
    except Exception as e:
        logging.error(f"Failed to connect to ChromaDB: {e}")
        return
    return client

def get_or_create_collection(client, force_reingest=False):
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBEDDING_MODEL
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"}
    )
    if force_reingest:
        # Get all IDs and delete them. This is safer than deleting the collection if it has specific metadata.
        count = collection.count()
        if count > 0:
            logging.info(f"Found {count} existing items in collection. Deleting...")
            ids = collection.get(include=[])['ids']
            collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} items from '{COLLECTION_NAME}'.")
    return collection


def generate_chunk_id(chunk: Dict[str, Any]) -> str:
    """
    Generate a deterministic, short ID for a chunk using SHA-256 hash of its content and metadata.
    Ensures uniqueness and stability across runs.
    """
    combined = (
        chunk["metadata"]["file_name"] +
        chunk["metadata"].get("section_title", "Unknown Section") +
        chunk["text"]
    )
    content_hash = hashlib.sha256(combined.encode()).hexdigest()[:12]
    return f"{content_hash}"

def add_chunks_to_collection(collection, chunks):
    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [generate_chunk_id(c) for c in chunks]

    embeddings = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBEDDING_MODEL
    )(documents)

    for i in range(0, len(documents), CHROMA_BATCH_SIZE):
        batch = slice(i, i + CHROMA_BATCH_SIZE)
        collection.add(
            documents=documents[batch],
            embeddings=embeddings[batch],
            metadatas=metadatas[batch],
            ids=ids[batch]
        )
        logger.info(f"Added batch {i // CHROMA_BATCH_SIZE + 1}")

    try:
        collection.modify(metadata={
            "last_ingested_at": datetime.now().isoformat(),
            "total_chunks": str(len(chunks)),
            "source_files": ", ".join(set(m["file_name"] for m in metadatas))
        })
    except Exception as e:
        logging.warning(f"Failed to update collection metadata: {e}")
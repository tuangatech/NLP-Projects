# config.py
import os
from pathlib import Path
from dotenv import load_dotenv
import logging

load_dotenv()
# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# --- Directories ---
PDF_DIR = Path(os.getenv("PDF_DIR", "/app/pdfs"))

# --- ChromaDB Config ---
CHROMA_HOST = os.getenv("CHROMA_HOST", "chromadb")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_documents")

# --- OpenAI Config ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")

# --- Text Splitter Config ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
CHROMA_BATCH_SIZE = int(os.getenv("CHROMA_BATCH_SIZE", 100))


if not PDF_DIR.exists() or not any(PDF_DIR.iterdir()):
    logging.warning(f"PDF directory {PDF_DIR} is empty or does not exist. Skipping ingestion.")

if not OPENAI_API_KEY:
    logging.error("OpenAI API Key is not set. Cannot proceed with embedding.")
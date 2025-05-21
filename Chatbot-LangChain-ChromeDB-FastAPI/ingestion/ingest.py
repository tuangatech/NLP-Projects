# ingestion/ingest.py
# Handles the ingestion pipeline: loading PDFs, splitting text, creating embeddings, and storing in ChromaDB.

import os
import re
import json
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import fitz  # PyMuPDF

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
# from langchain_openai import OpenAIEmbeddings # Alternative way to use OpenAI embeddings

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv("/app/.env")

# --- Configuration Constants ---
PDF_DIR = Path(os.getenv("PDF_DIR", "/app/pdfs"))  # Directory where PDFs are stored (mounted volume in Docker)

# ChromaDB Configuration
CHROMA_HOST = os.getenv("CHROMA_HOST", "chromadb") # Service name from docker-compose
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000")) # Port for ChromaDB service
COLLECTION_NAME = os.getenv("COLLECTION_NAME")    # "rag_documents"

print(f"COLLECTION_NAME: {COLLECTION_NAME}")

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logging.error("OPENAI_API_KEY not found in environment variables.")
    raise ValueError("OPENAI_API_KEY not found.")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")

# LangChain Text Splitter Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))

# Batch Size for ChromaDB writes
BATCH_SIZE = int(os.getenv("CHROMA_BATCH_SIZE", 100))

used_toc_titles = set()  # Set to track used section titles, skip them after first use

# --- Helper Functions ---

def get_section_title_heuristic(line_text: str) -> Optional[str]:
    """
    A simple heuristic to detect section titles based on line properties.
    """
    line_content = line_text.strip()

    # Rule 1: Line is empty or too short/long
    if not line_content or len(line_content) < 3 or len(line_content) > 80:
        return None

    # Rule 2: Starts with common pattern (e.g., "1.", "1.1", "Chapter 2")
    section_pattern = re.compile(r"^(?:[IVXLCDM]+\.|\d+(?:\.\d+)*\s+|[A-Z][a-z]+\s+\d+|Part\s+\d+|Section\s+\d+)", re.IGNORECASE)
    if section_pattern.match(line_content) and len(line_content.split()) <= 8:
        return line_content

    # Rule 3: Title Case with few words (e.g., "Introduction", "Related Work")
    if line_content.istitle() and len(line_content.split()) <= 6:
        return line_content

    # Rule 4: Ends with colon (often indicates headings in some formats)
    # if line_content.endswith(":") and len(line_content.split()) <= 5:
    #     return line_content

    # Rule 5: All caps and short (common in some document styles)
    if line_content.isupper() and 3 < len(line_content) < 50 and len(line_content.split()) < 8:
        return line_content

    return None

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

def create_toc_dict(toc: List[Tuple[int, str, int]]) -> dict:
    toc_dict = {}
    for item in toc:
        level, title, page = item
        # Normalize title by removing extra whitespace and making case-insensitive
        normalized_title = ' '.join(title.strip().split()).lower()
        if normalized_title not in toc_dict:
            toc_dict[normalized_title] = []
        toc_dict[normalized_title].append((page, level))
    return toc_dict

def is_section_title(text: str, toc_dict: dict) -> tuple[bool, Optional[str], Optional[int]]:
    """Check if text matches any TOC entry title (regardless of page)."""
    normalized_text = ' '.join(text.strip().split()).lower()
    if normalized_text in toc_dict:
        for page, level in toc_dict[normalized_text]:
            key = (normalized_text, page)       # to skip duplicate titles in TOC
            if key not in used_toc_titles:
                used_toc_titles.add(key)
                return True, text.strip(), level
    return False, None, None

def fix_broken_text(text):
    """
    Apply heuristics to fix common broken text issues from PDF extraction.
    
    Args:
        text (str): The raw text extracted from PDF
        
    Returns:
        str: Cleaned text with fewer breaks and better formatting
    """
    # 1. Fix line breaks in the middle of sentences
    text = re.sub(r'(?<!\n)\n(?!\n|\.\s|$|\d)', ' ', text)  # Replace single newlines with space
    
    # 2. Fix hyphenated words broken across lines
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)  # Join hyphenated words
    
    # 3. Fix numbered lists broken across lines
    text = re.sub(r'(\d+)\s*\n\s*([A-Z])', r'\1 \2', text)  # Fix "3\nThe" -> "3 The"
    
    # 4. Fix spaces before punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    
    # 5. Fix multiple spaces
    text = re.sub(r' +', ' ', text)
    
    # 6. Fix orphaned single letters at line starts (common in PDFs)
    text = re.sub(r'\n(\w)\s+', r' \1 ', text)
    
    # 7. Fix common PDF artifacts (page numbers, headers/footers)
    text = re.sub(r'\n\d+\s*\n', '\n', text)  # Remove standalone page numbers
    
    # 8. Fix capitalization after line breaks
    def cap_correct(match):
        return match.group(1).lower()
    text = re.sub(r'\.\s+([A-Z])', lambda m: f'. {m.group(1).lower()}', text)
    
    # 9. Remove excessive newlines (keep max 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 10. Fix space after opening parentheses/before closing
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)
    
    return text.strip()

def process_pdf(file_path: Path) -> List[Dict[str, Any]]:
    """
    Loads a single PDF, extracts text and metadata, attempts section-aware chunking,
    and splits into chunks.
    """
    logging.info(f"Processing PDF: {file_path.name}")
    
    # Use PyMuPDF (fitz) directly for more control over text extraction and layout analysis
    doc = fitz.open(file_path)
    toc = doc.get_toc()  # TODO: Use this to create a more accurate section title list
    toc_dict = create_toc_dict(toc)  # Create a dictionary for TOC entries
    
    docs_for_splitting: List[Document] = []
    # current_section_title = "Introduction" # Default section title    

    current_section_title: Optional[str] = None
    current_section_content: str = ""
    current_section_pages: List[int] = []

    for page_num_fitz in range(1, 5):  # Process only first 5 pages for testing  range(5):  # 5 = len(doc)
        page = doc.load_page(page_num_fitz)
        page_number = page_num_fitz + 1        
        # Get text blocks with bounding box information
        blocks = page.get_text("blocks")  # list of (x0, y0, x1, y1, "lines in block", block_no, block_type)
        logging.info(f"> Processing page {page_number}, found {len(blocks)} blocks.")  
        for block in blocks:
            block_text = block[4].strip()  
            logging.info(f"> Block text: {block_text[:70]}...")  # Log first 50 chars of block text
        
        page_content = ""
        
        for block in blocks:
            block_text = block[4].strip() # The text content of the block
            if not block_text:  # Skip empty blocks
                continue
            lines = block_text.split('\n')
            first_line = lines[0] if lines else ""

            # Check if first line matches a TOC entry
            is_title, title_text, _ = is_section_title(first_line, toc_dict)

            if is_title:
                # If we already have a section being built, finalize it before starting new one
                if current_section_title:
                    docs_for_splitting.append(Document(
                        page_content=current_section_content.strip(),
                        metadata={
                            "file_name": file_path.name,
                            "section_title": current_section_title,
                            "start_page": current_section_pages[0],
                            "end_page": current_section_pages[-1],
                        }
                    ))

                # Start new section
                current_section_title = title_text
                current_section_content = ""  # Reset content, skipping the title itself
                current_section_pages = [page_number]
                # Append remaining lines after the title
                if len(lines) > 1:
                    current_section_content += "\n".join(lines[1:]) + "\n"
                continue

            # Accumulate content into current section or as part of page content
            if current_section_title:
                current_section_content += block_text + "\n"
                current_section_pages.append(page_number)
            else:
                page_content += block_text + "\n"

        # If no section detected on this page, but some content exists without a section
        if not current_section_title and page_content.strip():
            # Fallback to adding as unclassified page-level content
            docs_for_splitting.append(Document(
                page_content=current_section_content.strip(),
                metadata={
                    "file_name": file_path.name,
                    "section_title": "Unclassified",
                    "start_page": 0,
                    "end_page": 0,
                }
            ))

    # Finalize any remaining section at end of document
    if current_section_title and current_section_content.strip():
        docs_for_splitting.append(Document(
            page_content=current_section_content.strip(),
            metadata={
                "file_name": file_path.name,
                "section_title": current_section_title,
                "start_page": current_section_pages[0],
                "end_page": current_section_pages[-1],
            }
        ))

    doc.close()

    if not docs_for_splitting:
        logging.warning(f"No text content extracted from {file_path.name}")
        return []

    # Fix broken text in all documents before splitting
    fixed_documents = []
    
    for doc in docs_for_splitting:
        # Apply text fixes to the content
        fixed_content = fix_broken_text(doc.page_content)
        
        # Create a new document with fixed content and original metadata
        fixed_doc = Document(
            page_content=fixed_content,
            metadata=doc.metadata.copy()  # Preserve all original metadata
        )
        
        fixed_documents.append(fixed_doc)

    # Now split all collected documents (which are now more section-aware)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,          # Maximum size of each chunk
        chunk_overlap=CHUNK_OVERLAP,    # How much overlap between chunks
        length_function=len,            # How to measure chunk size (standard len() for characters)
        is_separator_regex=False,       # Use simple string separators, not regex patterns
    )
    
    split_chunks_docs = text_splitter.split_documents(fixed_documents)  # docs_for_splitting
    
    # Prepare chunks with refined metadata for ChromaDB
    final_chunks_for_db = []
    for i, chunk_doc in enumerate(split_chunks_docs):
        chunk_metadata = {
            "file_name": chunk_doc.metadata.get("file_name", file_path.name),
            "start_page": chunk_doc.metadata.get("start_page", 0),  # Should be set from above
            "end_page": chunk_doc.metadata.get("end_page", 0),      # Should be set from above
            "section_title": chunk_doc.metadata.get("section_title", "Unknown Section"), # Should be set
            "chunk_index_in_doc": i # Add a chunk index for potential unique ID generation
        }
        final_chunks_for_db.append({
            "text": chunk_doc.page_content,
            "metadata": chunk_metadata
        })
        
    logging.info(f"Finished processing {file_path.name}, created {len(final_chunks_for_db)} chunks.")
    return final_chunks_for_db


# --- Main Ingestion Logic ---

def run_ingestion(force_reingest: bool = False):
    """
    Main function to run the ingestion pipeline.
    """
    logging.info("Starting ingestion process...")

    if not PDF_DIR.exists() or not any(PDF_DIR.iterdir()):
        logging.warning(f"PDF directory {PDF_DIR} is empty or does not exist. Skipping ingestion.")
        return
    
    if not OPENAI_API_KEY:
        logging.error("OpenAI API Key is not set. Cannot proceed with embedding.")
        return

    # Initialize ChromaDB client (connecting to the service)
    try:
        # Ensure ChromaDB is ready (Docker Compose depends_on with healthcheck helps)
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT, settings=chromadb.Settings(anonymized_telemetry=False))
        client.heartbeat() # Check connection
        logging.info(f"Successfully connected to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}")
    except Exception as e:
        logging.error(f"Failed to connect to ChromaDB: {e}")
        return

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBEDDING_MODEL
    )

    try:
        logging.info(f"Getting or creating collection: {COLLECTION_NAME}")
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=openai_ef, 
            metadata={"hnsw:space": "cosine"} 
        )
        logging.info(f"Collection '{COLLECTION_NAME}' ready.")
    except Exception as e:
        logging.error(f"Failed to get or create ChromaDB collection: {e}")
        return

    if force_reingest:
        logging.info(f"Force re-ingest: Clearing existing documents from collection '{COLLECTION_NAME}'.")
        try:
            # Get all IDs and delete them. This is safer than deleting the collection if it has specific metadata.
            count = collection.count()
            if count > 0:
                logging.info(f"Found {count} existing items in collection. Deleting...")
                # Fetching all IDs from the collection
                existing_ids = collection.get(include=[])['ids'] # Only get IDs
                if existing_ids:
                    collection.delete(ids=existing_ids)
                    logging.info(f"Deleted {len(existing_ids)} items from '{COLLECTION_NAME}'.")
                else:
                    logging.info("No items to delete, or get() returned no IDs.")
            else:
                 logging.info(f"Collection '{COLLECTION_NAME}' is already empty.")
        except Exception as e:
            logging.error(f"Error during force re-ingest clearing of collection items: {e}")


    all_pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not all_pdf_files:
        logging.warning(f"No PDF files found in {PDF_DIR}. Nothing to ingest.")
        return
        
    all_chunks_for_db: List[Dict[str, Any]] = []
    processed_file_count = 0

    for pdf_file in all_pdf_files:
        try:
            chunks = process_pdf(pdf_file)
            if chunks:
                all_chunks_for_db.extend(chunks)
            processed_file_count += 1
        except Exception as e:
            logging.error(f"Failed to process {pdf_file.name}: {e}", exc_info=True)

    if not all_chunks_for_db:
        logging.warning("No chunks were generated from any PDF files.")
        return

    logging.info(f"Total chunks to embed and add: {len(all_chunks_for_db)}")

    documents_to_add = [chunk['text'] for chunk in all_chunks_for_db]
    # documents_to_add = []
    # for chunk in all_chunks_for_db:
    #     text = chunk.get('text', None)
    #     if isinstance(text, str) and len(text.strip()) > 0:
    #         documents_to_add.append(text)
    #     else:
    #         logging.warning(f"Skipping invalid or empty document: {chunk}")

    # if not documents_to_add:
    #     logging.error("No valid documents to embed after filtering.")
    #     return
    
    metadatas_to_add = [chunk['metadata'] for chunk in all_chunks_for_db]
    ids_to_add = [generate_chunk_id(chunk) for chunk in all_chunks_for_db]
    
    logging.info("Generating embeddings for all chunks...")
    try:
        # temp_docs = ["Hello tax world" for chunk in all_chunks_for_db]
        embeddings_list = openai_ef(documents_to_add)  # documents_to_add   # $$ ************************************** $$$$
        logging.info(f"Successfully generated {len(embeddings_list)} embeddings.")
    except Exception as e:
        logging.error(f"Error generating embeddings with OpenAI: {e}", exc_info=True)
        return

    batch_size = 50 
    for i in range(0, len(documents_to_add), batch_size):
        batch_documents = documents_to_add[i:i+batch_size]
        batch_metadatas = metadatas_to_add[i:i+batch_size]
        batch_ids = ids_to_add[i:i+batch_size]
        batch_embeddings = embeddings_list[i:i+batch_size]
        
        try:
            logging.info(f"Adding batch of {len(batch_documents)} chunks to ChromaDB (Batch {i//batch_size + 1})...")
            collection.add(
                documents=batch_documents,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            logging.info(f"Successfully added batch {i//batch_size + 1} to ChromaDB.")
        except Exception as e:
            logging.error(f"Failed to add batch to ChromaDB: {e}", exc_info=True)

    # Update collection metadata with ingestion timestamp and source files to indicate the ingestion is complete
    try:
        from datetime import datetime
        collection.modify(metadata={
            "last_ingested_at": datetime.now().isoformat(),
            "source_files": ", ".join([f.name for f in all_pdf_files]),
            "total_chunks": str(len(all_chunks_for_db)),
            "chunk_size": str(CHUNK_SIZE),
            "chunk_overlap": str(CHUNK_OVERLAP),
        })
        logging.info("Updated collection metadata with ingestion info.")
    except Exception as e:
        logging.warning(f"Failed to update collection metadata: {e}")

    # Peek at first 6 records
    results = collection.peek(16)
    # Print results
    logging.info("Peeked records:")
    for i in range(len(results['ids'])):
        record = {
            "id": results["ids"][i],
            "metadata": results["metadatas"][i] if results["metadatas"] else None,
            "document": results["documents"][i] if results["documents"] else None
        }
        # logging.info(json.dumps(record, indent=2))
        logging.info(f"\nRecord {i+1}: ID: {record['id']}, Metadata: {record['metadata']},\n Document: {record['document'][:200]}...")

    logging.info(f"Successfully processed {processed_file_count} PDF files and attempted to add {len(all_chunks_for_db)} chunks to ChromaDB.")
    logging.info("Ingestion process completed.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingestion pipeline for RAG chatbot.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-ingestion even if data exists."
    )
    args = parser.parse_args()

    run_ingestion(force_reingest=args.force)
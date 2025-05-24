# Handles the ingestion pipeline: loading PDFs, splitting text, creating embeddings, and storing in ChromaDB.
import logging
from config import PDF_DIR
from pdf_processor import process_pdf
from vector_store import connect_to_chromadb, create_collection, add_chunks_to_collection

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_ingestion(force_reingest: bool = False):
    logging.info("Starting ingestion process...")
    if not PDF_DIR.exists() or not any(PDF_DIR.iterdir()):
        logging.warning("No PDF files found. Exiting.")
        return

    all_chunks = []
    all_pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not all_pdf_files:
        logging.warning(f"No PDF files found in {PDF_DIR}. Nothing to ingest.")
        return

    for pdf_file in all_pdf_files:
        try:
            logging.info(f"Processing {pdf_file.name}")
            chunks = process_pdf(pdf_file)
            if chunks:
                all_chunks.extend(chunks)
        except Exception as e:
            logging.error(f"Failed to process {pdf_file.name}: {e}", exc_info=True)

    if not all_chunks:
        logging.warning("No chunks were generated from any PDF files.")
        return

    logging.info(f"Total chunks to embed and add: {len(all_chunks)}")
    
    client = connect_to_chromadb()
    collection = create_collection(client, force_reingest=force_reingest)
    add_chunks_to_collection(collection, all_chunks)

    logging.info("Ingestion completed successfully.")

    # Peek at first records
    results = collection.peek(16)
    logging.info("Peeked records:")
    for i in range(len(results['ids'])):
        record = {
            "id": results["ids"][i],
            "metadata": results["metadatas"][i] if results["metadatas"] else None,
            "document": results["documents"][i] if results["documents"] else None
        }
        logging.info(f"\nRecord {i+1}: ID: {record['id']}, Metadata: {record['metadata']},\n Document: {record['document'][:200]}...")


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
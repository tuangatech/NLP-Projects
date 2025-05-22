from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Tuple, Optional, Any
import fitz
import logging
import re
from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

def create_toc_dict(toc: List[Tuple[int, str, int]]) -> dict:
    toc_dict = {}
    for item in toc:
        level, title, page = item
        normalized_title = ' '.join(title.strip().split()).lower()
        if normalized_title not in toc_dict:
            toc_dict[normalized_title] = []
        toc_dict[normalized_title].append((page, level))
    return toc_dict

def is_section_title(text: str, toc_dict: dict) -> tuple[bool, Optional[str], Optional[int]]:
    normalized_text = ' '.join(text.strip().split()).lower()
    if normalized_text in toc_dict:
        for page, level in toc_dict[normalized_text]:
            key = (normalized_text, page)
            if key not in used_toc_titles:
                used_toc_titles.add(key)
                return True, text.strip(), level
    return False, None, None

# Apply heuristics to fix common broken text issues from PDF extraction.
def fix_broken_text(text):
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

used_toc_titles = set()

def process_pdf(file_path: Path) -> List[Dict[str, Any]]:
    logger.info(f"Processing PDF: {file_path.name}")
    doc = fitz.open(file_path)
    toc = doc.get_toc()
    toc_dict = create_toc_dict(toc)

    docs_for_splitting = []
    current_section_title = None
    current_section_content = ""
    current_section_pages = []

    for page_num_fitz in range(1, 5):  # Process only first 5 pages for testing 5 = len(doc)
        page = doc.load_page(page_num_fitz)
        page_number = page_num_fitz + 1
        blocks = page.get_text("blocks")
        for block in blocks:
            block_text = block[4].strip()
            if not block_text:
                continue
            lines = block_text.split('\n')
            first_line = lines[0]
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
                current_section_content = "\n".join(lines[1:]) + "\n" # Reset content, skipping the title lines[0]
                current_section_pages = [page_number]
            else:
                # Accumulate content into current section as part of page content
                if current_section_title:
                    current_section_content += block_text + "\n"
                    current_section_pages.append(page_number)
        # If no section detected on this page, but some content exists without a section
        if not current_section_title and block_text.strip():
            docs_for_splitting.append(Document(
                page_content=block_text.strip(),
                metadata={"file_name": file_path.name, "section_title": "Unclassified", "start_page": 0, "end_page": 0}
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

    fixed_docs = [
        Document(page_content=fix_broken_text(d.page_content), metadata=d.metadata.copy())
        for d in docs_for_splitting
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,          # Maximum size of each chunk
        chunk_overlap=CHUNK_OVERLAP,    # How much overlap between chunks
        length_function=len
    )
    split_chunks = splitter.split_documents(fixed_docs)

    final_chunks = []
    for i, chunk in enumerate(split_chunks):
        final_chunks.append({
            "text": chunk.page_content,
            "metadata": {
                "file_name": chunk.metadata["file_name"],
                "start_page": chunk.metadata["start_page"],
                "end_page": chunk.metadata["end_page"],
                "section_title": chunk.metadata["section_title"],
                "chunk_index_in_doc": i
            }
        })

    logger.info(f"Created {len(final_chunks)} chunks from {file_path.name}")
    return final_chunks
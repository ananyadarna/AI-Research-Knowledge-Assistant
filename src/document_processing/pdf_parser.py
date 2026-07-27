import fitz  # PyMuPDF
import re

def extract_pdf_pages(file_path: str) -> list[dict]:
    """
    Extracts text page-by-page from a PDF file.
    Returns a list of dicts, each containing:
      - page_number: int (1-indexed)
      - text: str (cleaned page text)
    """
    doc = fitz.open(file_path)
    pages = []
    
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        text = page.get_text("text")
        cleaned_text = clean_text(text)
        
        pages.append({
            "page_number": page_idx + 1,
            "text": cleaned_text
        })
        
    return pages

def clean_text(text: str) -> str:
    """
    Performs basic text cleaning:
      - Replaces consecutive whitespaces/newlines with single spaces
      - Removes non-printable control characters
    """
    if not text:
        return ""
    # Replace multiple whitespaces/newlines with a single space
    text = re.sub(r'\s+', ' ', text)
    # Strip non-printable or null bytes
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
    return text.strip()

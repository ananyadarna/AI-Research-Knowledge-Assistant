from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_pages(pages: list[dict], chunk_size: int = 800, chunk_overlap: int = 100) -> list[dict]:
    """
    Splits page-by-page text into chunks.
    Splitting page-by-page guarantees that each chunk is mapped to exactly one page,
    making page citation in the RAG pipeline highly accurate.
    
    Returns a list of dicts, each containing:
      - chunk_id: str
      - page_number: int (source page number)
      - text: str (chunk text)
    """
    # Recursive splitter for fine-grained sub-splitting within page limits
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = []
    
    for page in pages:
        page_num = page["page_number"]
        page_text = page["text"]
        
        if not page_text.strip():
            continue
            
        page_chunks = splitter.split_text(page_text)
        
        for idx, chunk_text in enumerate(page_chunks):
            chunks.append({
                "chunk_id": f"p{page_num}_c{idx}",
                "page_number": page_num,
                "text": chunk_text
            })
            
    return chunks

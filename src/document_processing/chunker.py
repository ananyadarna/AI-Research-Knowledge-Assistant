class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100, length_function=len, separators=None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def split_text(self, text: str) -> list[str]:
        if not text:
            return []
            
        if self.length_function(text) <= self.chunk_size:
            return [text]

        separator = self.separators[-1]
        for s in self.separators:
            if s == "" or s in text:
                separator = s
                break

        splits = text.split(separator) if separator != "" else list(text)
        
        final_chunks = []
        current_chunk = []
        current_length = 0

        for s in splits:
            s_len = self.length_function(s)
            sep_len = self.length_function(separator) if current_chunk else 0

            if current_length + s_len + sep_len > self.chunk_size and current_chunk:
                joined = separator.join(current_chunk)
                final_chunks.append(joined)

                overlap_size = 0
                overlap_chunk = []
                for item in reversed(current_chunk):
                    item_len = self.length_function(item)
                    if overlap_size + item_len <= self.chunk_overlap:
                        overlap_chunk.insert(0, item)
                        overlap_size += item_len
                    else:
                        break
                current_chunk = overlap_chunk
                current_length = sum(self.length_function(x) for x in current_chunk)
            
            current_chunk.append(s)
            current_length += s_len

        if current_chunk:
            final_chunks.append(separator.join(current_chunk))

        return final_chunks

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

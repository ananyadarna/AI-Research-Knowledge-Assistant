from src.document_processing.pdf_parser import clean_text
from src.document_processing.chunker import chunk_pages

def test_clean_text():
    """
    Verifies that duplicate whitespaces and special non-printable control
    characters are removed from the extracted text string.
    """
    raw_input = "Hello \n\n  world!\x00 This is   a line\x08 of text."
    expected_output = "Hello world! This is a line of text."
    assert clean_text(raw_input) == expected_output

def test_chunk_pages():
    """
    Validates that the page splitter divides text into reasonable overlapping blocks
    while accurately preserving 1-indexed page number mappings for citations.
    """
    pages_data = [
        {"page_number": 1, "text": "Artificial Intelligence is accelerating. " * 40},  # ~1600 characters
        {"page_number": 2, "text": "Robotics dynamics involve physical actuators. " * 10}  # ~450 characters
    ]
    
    # Run chunker with custom limits
    chunks = chunk_pages(pages_data, chunk_size=500, chunk_overlap=50)
    
    assert len(chunks) > 0
    for chunk in chunks:
        assert "chunk_id" in chunk
        assert "page_number" in chunk
        assert "text" in chunk
        
        # Verify page mapping remains accurate
        if "actuators" in chunk["text"]:
            assert chunk["page_number"] == 2
        else:
            assert chunk["page_number"] == 1
            
        # Verify chunk size constraint
        assert len(chunk["text"]) <= 550

import os
import shutil
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.database.base import Base
from src.database.models import DocumentMetadata, ChatSession, ChatMessage
from src.vector_store.manager import VectorSearchManager
from src.rag.qa_chain import RAGQAEngine
from src.rag.summarizer import DocumentSummarizer
from src.rag.comparator import DocumentComparator
from config.settings import settings

# Temporary directory locations for test isolation
TEST_DB_PATH = "./data/test_assistant.db"
TEST_VECTOR_DIR = "./data/test_vector_db"

@pytest.fixture(scope="module", autouse=True)
def override_settings():
    # Save original environment paths
    orig_db = settings.database_url
    orig_vector = settings.vector_db_dir
    
    settings.database_url = f"sqlite:///{TEST_DB_PATH}"
    settings.vector_db_dir = TEST_VECTOR_DIR
    
    # Clear any previous test residues
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    if os.path.exists(TEST_VECTOR_DIR):
        shutil.rmtree(TEST_VECTOR_DIR)
        
    yield
    
    # Restore original configurations and wipe test outputs
    settings.database_url = orig_db
    settings.vector_db_dir = orig_vector
    
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except Exception:
            pass
    if os.path.exists(TEST_VECTOR_DIR):
        try:
            shutil.rmtree(TEST_VECTOR_DIR)
        except Exception:
            pass

@pytest.fixture(scope="function")
def db_session():
    # Initialize SQLite database file for testing
    engine = create_engine(settings.database_url)
    Base.metadata.create_all(bind=engine)
    SessionClass = sessionmaker(bind=engine)
    session = SessionClass()
    
    yield session
    
    session.close()
    Base.metadata.drop_all(bind=engine)

def test_vector_indexing_and_rag_qa_pipeline(db_session):
    """
    End-to-end integration test verifying chunk indexing, semantic queries,
    BM25 text searches, hybrid rank fusion, QA answer citations, and summarization.
    """
    # 1. Initialize vector store manager
    manager = VectorSearchManager()
    
    doc_id = "test-doc-1"
    file_name = "artificial_intelligence_overview.pdf"
    chunks = [
        {"chunk_id": "p1_c0", "page_number": 1, "text": "Deep neural networks learn hierarchal representations from raw input datasets."},
        {"chunk_id": "p2_c0", "page_number": 2, "text": "Intrusion detection systems prevent zero day exploits and cybersecurity attacks."}
    ]
    
    # Index document chunks into Chroma DB
    manager.add_document_chunks(doc_id, file_name, chunks)

    # 2. Semantic Search validation
    semantic_results = manager.search(query="neural network", search_mode="semantic", top_k=1)
    assert len(semantic_results) > 0
    assert "hierarchal" in semantic_results[0]["text"]
    assert semantic_results[0]["metadata"]["page_number"] == 1

    # 3. Keyword Search validation (BM25)
    keyword_results = manager.search(query="cybersecurity zero day", search_mode="keyword", top_k=1)
    assert len(keyword_results) > 0
    assert "zero day" in keyword_results[0]["text"]
    assert keyword_results[0]["metadata"]["page_number"] == 2

    # 4. Hybrid Search validation (RRF Score Fusion)
    hybrid_results = manager.search(query="neural network cybersecurity", search_mode="hybrid", top_k=2)
    assert len(hybrid_results) == 2

    # 5. Insert Document metadata to match Chroma content in testing ORM
    doc_meta = DocumentMetadata(
        doc_id=doc_id,
        file_name=file_name,
        total_pages=2,
        total_chunks=2,
        category="Computer Vision",
        processing_status="PROCESSED",
        file_path="mock/test_doc.pdf"
    )
    db_session.add(doc_meta)
    db_session.commit()

    # 6. Contextual QA & Citations validation (using Mock Mode fallback)
    qa_engine = RAGQAEngine(db_session)
    session_id = "test-session-123"
    
    qa_response = qa_engine.answer_question(
        session_id=session_id,
        query="Tell me about neural networks",
        doc_ids=[doc_id],
        search_mode="semantic"
    )

    assert "final_answer" in qa_response
    assert len(qa_response["sources"]) > 0
    assert qa_response["sources"][0]["document_name"] == file_name
    assert qa_response["sources"][0]["page_number"] == 1

    # 7. Document Summarization validation
    summarizer = DocumentSummarizer(db_session)
    summary = summarizer.generate_summary(doc_id)
    assert "executive_summary" in summary
    assert "technical_summary" in summary
    assert len(summary["bullet_points"]) > 0
    assert len(summary["key_takeaways"]) > 0

    # 8. Document Comparison validation
    # Create second document
    doc_id_2 = "test-doc-2"
    file_name_2 = "cyber_security_handbook.pdf"
    doc_meta_2 = DocumentMetadata(
        doc_id=doc_id_2,
        file_name=file_name_2,
        total_pages=1,
        total_chunks=1,
        category="Cyber Security",
        processing_status="PROCESSED",
        file_path="mock/test_doc_2.pdf"
    )
    db_session.add(doc_meta_2)
    db_session.commit()
    
    manager.add_document_chunks(doc_id_2, file_name_2, [
        {"chunk_id": "p1_c0", "page_number": 1, "text": "Securing cloud database instances via network security firewalls."}
    ])

    comparator = DocumentComparator(db_session)
    comparison = comparator.compare_documents([doc_id, doc_id_2])
    assert "methodologies" in comparison
    assert "pros_cons" in comparison
    assert len(comparison["similarities"]) > 0
    assert len(comparison["differences"]) > 0

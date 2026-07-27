import os
import shutil
import pytest
from fastapi.testclient import TestClient
from main import app
from src.database.base import Base
from sqlalchemy import create_engine
from config.settings import settings

TEST_DB_PATH = "./data/test_api_assistant.db"
TEST_VECTOR_DIR = "./data/test_api_vector_db"
TEST_UPLOAD_DIR = "./data/test_api_uploads"

@pytest.fixture(scope="module", autouse=True)
def override_settings():
    # Save original configurations
    orig_db = settings.database_url
    orig_vector = settings.vector_db_dir
    orig_upload = settings.upload_dir
    
    settings.database_url = f"sqlite:///{TEST_DB_PATH}"
    settings.vector_db_dir = TEST_VECTOR_DIR
    settings.upload_dir = TEST_UPLOAD_DIR
    
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.vector_db_dir, exist_ok=True)
    
    # Initialize DB tables
    engine = create_engine(settings.database_url)
    Base.metadata.create_all(bind=engine)
    
    yield
    
    # Restore original configurations and tear down test outputs
    settings.database_url = orig_db
    settings.vector_db_dir = orig_vector
    settings.upload_dir = orig_upload
    
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
    if os.path.exists(TEST_UPLOAD_DIR):
        try:
            shutil.rmtree(TEST_UPLOAD_DIR)
        except Exception:
            pass

client = TestClient(app)

def test_root_and_analytics_stats():
    # Test Root Status Endpoint
    response = client.get("/")
    assert response.status_code == 200
    assert "version" in response.json()

    # Test Analytics Stats Dashboard Endpoint
    response = client.get("/analytics/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_documents" in data
    assert "total_chunks" in data
    assert "most_queried_documents" in data


def test_document_upload_lifecycle():
    # Create a dummy mock PDF file content
    dummy_pdf_content = b"%PDF-1.4 ... mock contents of research paper ..."
    
    # 1. Upload Document via Endpoint
    files = {"file": ("api_test_doc.pdf", dummy_pdf_content, "application/pdf")}
    response = client.post("/documents/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "doc_id" in data
    assert data["processing_status"] == "PENDING"
    
    doc_id = data["doc_id"]
    
    # 2. Get List of Documents
    response = client.get("/documents")
    assert response.status_code == 200
    docs = response.json()
    assert len(docs) > 0
    assert any(d["doc_id"] == doc_id for d in docs)
    
    # 3. Get Specific Document Details
    response = client.get(f"/documents/{doc_id}")
    assert response.status_code == 200
    details = response.json()
    assert details["doc_id"] == doc_id
    
    # 4. Trigger Reprocess endpoint
    response = client.post(f"/documents/{doc_id}/reprocess")
    assert response.status_code == 200
    reprocess_data = response.json()
    assert reprocess_data["doc_id"] == doc_id
    assert reprocess_data["processing_status"] == "PENDING"

    # 5. Delete Document
    response = client.delete(f"/documents/{doc_id}")
    assert response.status_code == 200
    assert "message" in response.json()

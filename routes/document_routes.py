import os
import uuid
import logging
import shutil
from datetime import datetime
from fastapi import APIRouter, Depends, UploadFile, File, BackgroundTasks, HTTPException
from sqlalchemy.orm import Session
from config.settings import settings
from src.database.base import get_db
from src.database.models import DocumentMetadata
from src.document_processing.pdf_parser import extract_pdf_pages
from src.document_processing.chunker import chunk_pages
from src.ml.predictor import DocumentClassifier
from src.vector_store.manager import VectorSearchManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

# Instantiate singletons
classifier = DocumentClassifier()
vector_manager = VectorSearchManager()

def process_document_background(doc_id: str, file_path: str, file_name: str, db: Session):
    """
    Runs the document ingestion pipeline asynchronously:
    1. Parse PDF text page-by-page.
    2. Run TensorFlow classifier to auto-categorize the text.
    3. Segment the pages into chunk sequences.
    4. Write embeddings & metadata into the Chroma vector DB.
    5. Update document status in SQL database to PROCESSED/FAILED.
    """
    db_record = db.query(DocumentMetadata).filter(DocumentMetadata.doc_id == doc_id).first()
    if not db_record:
        logger.error(f"Async ingestion failed: Document ID {doc_id} not found in DB.")
        return

    try:
        # Step 1: Parse PDF
        pages = extract_pdf_pages(file_path)
        total_pages = len(pages)
        
        # Step 2: Auto-classify using TensorFlow model
        category = "Other/General"
        try:
            category = classifier.classify_document(pages)
        except Exception as ml_err:
            logger.warning(f"TF Classification failed for {file_name}: {ml_err}. Falling back to default.")

        # Step 3: Chunking
        chunks = chunk_pages(pages)
        total_chunks = len(chunks)

        # Step 4: Index into Chroma DB
        vector_manager.add_document_chunks(
            doc_id=doc_id,
            file_name=file_name,
            chunks=chunks
        )

        # Step 5: Complete transaction
        db_record.total_pages = total_pages
        db_record.total_chunks = total_chunks
        db_record.category = category
        db_record.processing_status = "PROCESSED"
        db.commit()
        logger.info(f"Ingestion successful for document {file_name} (ID: {doc_id})")

    except Exception as e:
        logger.exception(f"Pipeline error processing document {file_name}: {e}")
        db_record.processing_status = "FAILED"
        db.commit()

@router.post("/upload", response_model=dict)
def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Uploads a PDF document and runs the ingestion pipeline in the background.
    Returns the document ID and details immediately with a status of 'PENDING'.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    doc_id = str(uuid.uuid4())
    file_name = file.filename
    file_path = os.path.join(settings.upload_dir, f"{doc_id}_{file_name}")

    # Write file to disk
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Create pending record in database
    db_record = DocumentMetadata(
        doc_id=doc_id,
        file_name=file_name,
        file_path=file_path,
        upload_timestamp=datetime.utcnow(),
        processing_status="PENDING"
    )
    db.add(db_record)
    db.commit()

    # Enqueue background pipeline task
    background_tasks.add_task(
        process_document_background,
        doc_id=doc_id,
        file_path=file_path,
        file_name=file_name,
        db=db
    )

    return {
        "doc_id": doc_id,
        "file_name": file_name,
        "processing_status": "PENDING",
        "message": "Upload complete. Processing started in the background."
    }

@router.get("", response_model=list[dict])
def list_documents(db: Session = Depends(get_db)):
    """
    Lists all documents and their processing status.
    """
    documents = db.query(DocumentMetadata).all()
    return [
        {
            "doc_id": doc.doc_id,
            "file_name": doc.file_name,
            "upload_timestamp": doc.upload_timestamp.isoformat(),
            "total_pages": doc.total_pages,
            "total_chunks": doc.total_chunks,
            "category": doc.category,
            "processing_status": doc.processing_status
        }
        for doc in documents
    ]

@router.get("/{doc_id}", response_model=dict)
def get_document_details(doc_id: str, db: Session = Depends(get_db)):
    """
    Fetches the details and processing status of a single document.
    """
    doc = db.query(DocumentMetadata).filter(DocumentMetadata.doc_id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    
    return {
        "doc_id": doc.doc_id,
        "file_name": doc.file_name,
        "upload_timestamp": doc.upload_timestamp.isoformat(),
        "total_pages": doc.total_pages,
        "total_chunks": doc.total_chunks,
        "category": doc.category,
        "processing_status": doc.processing_status
    }

@router.delete("/{doc_id}", response_model=dict)
def delete_document(doc_id: str, db: Session = Depends(get_db)):
    """
    Deletes the document's metadata from the SQL DB, deletes its chunks from Chroma DB,
    and removes the uploaded file from the disk.
    """
    doc = db.query(DocumentMetadata).filter(DocumentMetadata.doc_id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    # Remove vector chunks from Chroma DB
    try:
        vector_manager.delete_document_chunks(doc_id)
    except Exception as e:
        logger.error(f"Error removing vector chunks for {doc_id}: {e}")

    # Remove raw file from disk
    if os.path.exists(doc.file_path):
        try:
            os.remove(doc.file_path)
        except Exception as e:
            logger.error(f"Error removing file {doc.file_path}: {e}")

    # Remove SQL database record
    db.delete(doc)
    db.commit()

    return {
        "doc_id": doc_id,
        "message": f"Document {doc.file_name} successfully deleted."
    }

@router.post("/{doc_id}/reprocess", response_model=dict)
def reprocess_document(
    doc_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Cleans up old vector store chunks and re-triggers the document ingestion pipeline.
    """
    doc = db.query(DocumentMetadata).filter(DocumentMetadata.doc_id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    if not os.path.exists(doc.file_path):
        raise HTTPException(status_code=400, detail="Source PDF file not found on disk. Cannot reprocess.")

    # 1. Delete old chunks from Chroma DB
    try:
        vector_manager.delete_document_chunks(doc_id)
    except Exception as e:
        logger.error(f"Error removing old vector chunks for reprocessing {doc_id}: {e}")

    # 2. Reset database state
    doc.processing_status = "PENDING"
    doc.upload_timestamp = datetime.utcnow()
    db.commit()

    # 3. Re-enqueue background pipeline task
    background_tasks.add_task(
        process_document_background,
        doc_id=doc_id,
        file_path=doc.file_path,
        file_name=doc.file_name,
        db=db
    )

    return {
        "doc_id": doc_id,
        "processing_status": "PENDING",
        "message": f"Reprocessing for document {doc.file_name} started in the background."
    }


import json
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from src.database.base import get_db
from src.database.models import QueryLog, ChatSession
from src.vector_store.manager import VectorSearchManager
from src.rag.qa_chain import RAGQAEngine

router = APIRouter(prefix="/search", tags=["search"])
vector_manager = VectorSearchManager()

class QueryRequest(BaseModel):
    query: str = Field(..., description="The query string to search for")
    doc_ids: list[str] = Field(None, description="Optional scope filter to search only specific document IDs")
    search_mode: str = Field("semantic", description="Search strategy to use: 'semantic', 'keyword', or 'hybrid'")
    top_k: int = Field(5, description="Number of results to return")

class QARequest(BaseModel):
    session_id: str = Field(..., description="Chat session ID for conversational memory persistence")
    query: str = Field(..., description="The question to ask the AI assistant")
    doc_ids: list[str] = Field(None, description="Optional scope filter to constrain RAG context to specific document IDs")
    search_mode: str = Field("semantic", description="Search strategy to use: 'semantic', 'keyword', or 'hybrid'")

@router.post("/query", response_model=list[dict])
def query_documents(request: QueryRequest, db: Session = Depends(get_db)):
    """
    Performs retrieval based on Search Mode (semantic, keyword, or hybrid RRF).
    Logs the query and matched document IDs into the QueryLog for analytics telemetry.
    """
    if request.search_mode.lower() not in ["semantic", "keyword", "hybrid"]:
        raise HTTPException(status_code=400, detail="Search mode must be 'semantic', 'keyword', or 'hybrid'")

    results = vector_manager.search(
        query=request.query,
        search_mode=request.search_mode,
        doc_ids=request.doc_ids,
        top_k=request.top_k
    )

    # Log query and capture referenced document IDs
    matched_doc_ids = list(set(res["metadata"]["doc_id"] for res in results if "metadata" in res))
    
    query_log = QueryLog(
        query_text=request.query,
        timestamp=datetime.utcnow(),
        doc_ids_referenced=json.dumps(matched_doc_ids)
    )
    db.add(query_log)
    db.commit()

    return results

@router.post("/qa", response_model=dict)
def rag_qa(request: QARequest, db: Session = Depends(get_db)):
    """
    Runs the contextual RAG QA loop. Answers user queries grounded on document chunks,
    maintains session memory, logs query telemetry, and returns structured page-number citations.
    """
    # Verify session exists or create it
    session = db.query(ChatSession).filter(ChatSession.session_id == request.session_id).first()
    if not session:
        session = ChatSession(session_id=request.session_id)
        db.add(session)
        db.commit()

    # Invoke RAG engine
    rag_engine = RAGQAEngine(db)
    response = rag_engine.answer_question(
        session_id=request.session_id,
        query=request.query,
        doc_ids=request.doc_ids,
        search_mode=request.search_mode
    )

    # Log query and capture referenced document IDs
    matched_doc_ids = list(set(ctx["document_name"] for ctx in response["retrieved_context"]))
    
    query_log = QueryLog(
        query_text=request.query,
        timestamp=datetime.utcnow(),
        doc_ids_referenced=json.dumps(matched_doc_ids)
    )
    db.add(query_log)
    db.commit()

    return response

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from src.database.base import get_db
from src.rag.summarizer import DocumentSummarizer
from src.rag.comparator import DocumentComparator

router = APIRouter(prefix="/analysis", tags=["analysis"])

class CompareRequest(BaseModel):
    doc_ids: list[str] = Field(..., min_items=2, description="List of document IDs to compare (minimum of 2)")

@router.post("/summarize/{doc_id}", response_model=dict)
def summarize_document(doc_id: str, db: Session = Depends(get_db)):
    """
    Generates a structured multi-part summary of the document (Executive Summary,
    Technical Summary, Bullet Points, Key Takeaways).
    """
    try:
        summarizer = DocumentSummarizer(db)
        summary = summarizer.generate_summary(doc_id)
        return summary
    except FileNotFoundError as fnf_err:
        raise HTTPException(status_code=404, detail=str(fnf_err))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")

@router.post("/compare", response_model=dict)
def compare_documents(request: CompareRequest, db: Session = Depends(get_db)):
    """
    Compares two or more documents on methodologies, pros/cons, similarities,
    differences, and conclusions.
    """
    try:
        comparator = DocumentComparator(db)
        comparison = comparator.compare_documents(request.doc_ids)
        return comparison
    except FileNotFoundError as fnf_err:
        raise HTTPException(status_code=404, detail=str(fnf_err))
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate comparison: {str(e)}")

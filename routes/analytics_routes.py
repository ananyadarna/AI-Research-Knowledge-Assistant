from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from src.database.base import get_db
from src.analytics.metrics import get_system_metrics

router = APIRouter(prefix="/analytics", tags=["analytics"])

@router.get("/stats", response_model=dict)
def get_analytics(db: Session = Depends(get_db)):
    """
    Returns high-level system analytics including total documents, total chunks,
    total embeddings, category distribution, and top-queried documents.
    """
    return get_system_metrics(db)

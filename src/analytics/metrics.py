import json
from collections import Counter
from sqlalchemy.orm import Session
from src.database.models import DocumentMetadata, ChatMessage, QueryLog

def get_system_metrics(db: Session) -> dict:
    """
    Computes system-wide statistics for the AI Research Assistant dashboard,
    including document counts, processing chunks, categories, and query telemetry.
    """
    # 1. Total documents in the database
    docs = db.query(DocumentMetadata).all()
    total_docs = len(docs)

    # 2. Total chunks across all processed documents
    total_chunks = sum(doc.total_chunks or 0 for doc in docs)

    # 3. Total embeddings generated
    # In this RAG architecture, each text chunk corresponds to exactly one vector embedding.
    total_embeddings = total_chunks

    # 4. Total questions answered
    # Calculated as the count of assistant replies saved in the ChatMessage history.
    total_questions = db.query(ChatMessage).filter(ChatMessage.role == "assistant").count()

    # 5. Distribution of documents by TensorFlow categories
    categories = [doc.category for doc in docs if doc.category]
    category_distribution = dict(Counter(categories))

    # 6. Most queried documents
    # Parse the QueryLog records. doc_ids_referenced stores doc IDs as a JSON list.
    query_logs = db.query(QueryLog).all()
    queried_doc_ids = []
    
    for log in query_logs:
        if log.doc_ids_referenced:
            try:
                doc_list = json.loads(log.doc_ids_referenced)
                if isinstance(doc_list, list):
                    queried_doc_ids.extend(doc_list)
            except Exception:
                pass  # Skip corrupted log formats

    # Map database IDs to display filenames and aggregate counts
    doc_id_counts = Counter(queried_doc_ids)
    doc_name_map = {doc.doc_id: doc.file_name for doc in docs}

    most_queried = []
    for doc_id, count in doc_id_counts.most_common(5):
        if doc_id in doc_name_map:
            most_queried.append({
                "document_name": doc_name_map[doc_id],
                "queries_count": count
            })

    return {
        "total_documents": total_docs,
        "total_chunks": total_chunks,
        "total_embeddings_generated": total_embeddings,
        "total_questions_answered": total_questions,
        "category_distribution": category_distribution,
        "most_queried_documents": most_queried
    }

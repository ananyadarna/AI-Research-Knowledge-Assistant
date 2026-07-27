from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from src.database.base import Base

class DocumentMetadata(Base):
    __tablename__ = "document_metadata"

    doc_id = Column(String, primary_key=True, index=True)
    file_name = Column(String, nullable=False)
    upload_timestamp = Column(DateTime, default=datetime.utcnow)
    total_pages = Column(Integer, nullable=True)
    total_chunks = Column(Integer, default=0)
    processing_status = Column(String, default="PENDING")  # PENDING, PROCESSED, FAILED
    category = Column(String, nullable=True)  # Populated by TensorFlow classifier
    file_path = Column(String, nullable=False)

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    session_id = Column(String, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    message_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(String, ForeignKey("chat_sessions.session_id"), nullable=False)
    role = Column(String, nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    citations = Column(Text, nullable=True)  # JSON string representing source citations

    session = relationship("ChatSession", back_populates="messages")

class QueryLog(Base):
    __tablename__ = "query_logs"

    log_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    query_text = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    doc_ids_referenced = Column(Text, nullable=True)  # JSON list of referenced doc_ids

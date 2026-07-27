import json
import logging
from sqlalchemy.orm import Session
from src.database.models import DocumentMetadata
from src.vector_store.manager import VectorSearchManager
from src.rag.qa_chain import get_llm

logger = logging.getLogger(__name__)

class DocumentSummarizer:
    def __init__(self, db_session: Session):
        self.db = db_session
        self.search_manager = VectorSearchManager()
        self.llm = get_llm()

    def generate_summary(self, doc_id: str) -> dict:
        """
        Retrieves text from the document and generates a structured summary:
        Executive Summary, Technical Summary, Bullet Points, and Key Takeaways.
        """
        # Verify document exists in database
        doc = self.db.query(DocumentMetadata).filter(DocumentMetadata.doc_id == doc_id).first()
        if not doc:
            raise FileNotFoundError(f"Document with ID {doc_id} not found in database.")

        # Get document chunks from Chroma DB
        chunks = self.search_manager.collection.get(
            where={"doc_id": doc_id},
            include=["documents"]
        )
        
        if not chunks or not chunks["documents"]:
            return {
                "executive_summary": "No text content found for this document.",
                "technical_summary": "No text content found for this document.",
                "bullet_points": [],
                "key_takeaways": []
            }

        # Concatenate text up to a reasonable token limit (~6000-8000 characters)
        text_corpus = "\n".join(chunks["documents"])
        text_limit = 8000
        if len(text_corpus) > text_limit:
            text_corpus = text_corpus[:text_limit] + "\n...[Content truncated for length]..."

        system_prompt = (
            "You are an expert academic and technical writer.\n"
            "Analyze the provided document text and generate a structured summary in JSON format.\n"
            "The JSON object must contain exactly these four keys:\n"
            "1. 'executive_summary': (string) A concise, non-technical overview of the document's purpose and findings.\n"
            "2. 'technical_summary': (string) A deep technical explanation of the methodologies, algorithms, or systems described.\n"
            "3. 'bullet_points': (list of strings) A set of bullet points summarizing key details.\n"
            "4. 'key_takeaways': (list of strings) The major conclusions and actionable insights.\n\n"
            "Format the response strictly as valid JSON, with no extra markdown formatting."
        )

        user_content = f"Document Text:\n{text_corpus}"

        if self.llm:
            try:
                response = self.llm.invoke([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ])
                content = response.content
                if isinstance(content, list):
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict) and "text" in block:
                            text_parts.append(block["text"])
                        elif isinstance(block, str):
                            text_parts.append(block)
                    content = "".join(text_parts)
                content = content.strip()

                if content.startswith("```json"):
                    content = content.split("```json")[1].split("```")[0].strip()
                elif content.startswith("```"):
                    content = content.split("```")[1].split("```")[0].strip()

                return json.loads(content)
            except Exception as e:
                logger.error(f"Error invoking LLM for summarization: {e}")
                return {
                    "executive_summary": "Error generating summary using LLM.",
                    "technical_summary": "Error generating summary using LLM.",
                    "bullet_points": [str(e)],
                    "key_takeaways": []
                }
        else:
            # Mock mode
            logger.warning("No LLM API keys configured. Running in Mock Mode.")
            return {
                "executive_summary": f"[Mock Summary] Executive overview of the document '{doc.file_name}'. This document covers research and concepts relating to {doc.category or 'technical fields'}.",
                "technical_summary": f"[Mock Summary] Deep dive technical summary of methodology and architectural layouts of '{doc.file_name}'.",
                "bullet_points": [
                    f"First key point extracted from the text corpus of {doc.file_name}.",
                    f"Second main highlight focusing on category {doc.category}."
                ],
                "key_takeaways": [
                    "Synthetically generated takeaway A.",
                    "Synthetically generated takeaway B."
                ]
            }

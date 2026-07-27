import json
import logging
from sqlalchemy.orm import Session
from src.database.models import DocumentMetadata
from src.vector_store.manager import VectorSearchManager
from src.rag.qa_chain import get_llm

logger = logging.getLogger(__name__)

class DocumentComparator:
    def __init__(self, db_session: Session):
        self.db = db_session
        self.search_manager = VectorSearchManager()
        self.llm = get_llm()

    def compare_documents(self, doc_ids: list[str]) -> dict:
        """
        Gathers key text chunks from the selected documents, passes them to the LLM,
        and generates a detailed comparative breakdown (methodology, pros/cons, similarities, differences).
        """
        if not doc_ids or len(doc_ids) < 2:
            raise ValueError("At least two document IDs must be provided for comparison.")

        # Resolve document metadata
        docs = self.db.query(DocumentMetadata).filter(DocumentMetadata.doc_id.in_(doc_ids)).all()
        if len(docs) != len(doc_ids):
            found_ids = [d.doc_id for d in docs]
            missing_ids = list(set(doc_ids) - set(found_ids))
            raise FileNotFoundError(f"Some documents were not found: {missing_ids}")

        # Retrieve a subset of chunks (e.g. the first 5 chunks of each document to stay within context limits)
        document_inputs = []
        for doc in docs:
            chunks = self.search_manager.collection.get(
                where={"doc_id": doc.doc_id},
                include=["documents"]
            )
            text_corpus = ""
            if chunks and chunks["documents"]:
                text_corpus = "\n".join(chunks["documents"][:5])
                
            document_inputs.append(
                f"Document Name: {doc.file_name}\n"
                f"Category: {doc.category or 'Unknown'}\n"
                f"Content Snippet:\n{text_corpus}\n"
                f"====================================\n"
            )

        combined_corpus = "\n".join(document_inputs)

        system_prompt = (
            "You are a critical research reviewer and technical analyst.\n"
            "Compare the provided documents and output a structured comparative study in JSON format.\n"
            "The JSON object must contain exactly these five keys:\n"
            "1. 'methodologies': (list of objects) each with keys 'document_name' and 'methodology' explaining how the research was conducted.\n"
            "2. 'pros_cons': (list of objects) each with keys 'document_name', 'advantages' (list of strings), and 'disadvantages' (list of strings).\n"
            "3. 'similarities': (list of strings) list of common methods, goals, or conclusions between the documents.\n"
            "4. 'differences': (list of strings) key points of divergence in methodologies, technologies, or outcomes.\n"
            "5. 'conclusions': (list of objects) each with keys 'document_name' and 'conclusion' summarizing each document's core findings.\n\n"
            "Format the response strictly as valid JSON, with no extra markdown formatting."
        )

        if self.llm:
            try:
                response = self.llm.invoke([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": combined_corpus}
                ])
                content = response.content.strip()

                if content.startswith("```json"):
                    content = content.split("```json")[1].split("```")[0].strip()
                elif content.startswith("```"):
                    content = content.split("```")[1].split("```")[0].strip()

                return json.loads(content)
            except Exception as e:
                logger.error(f"Error invoking LLM for comparison: {e}")
                return {
                    "methodologies": [],
                    "pros_cons": [],
                    "similarities": ["Error generating comparison via LLM."],
                    "differences": [str(e)],
                    "conclusions": []
                }
        else:
            # Mock mode
            logger.warning("No LLM API keys configured. Running in Mock Mode.")
            
            methodologies = []
            pros_cons = []
            conclusions = []
            for doc in docs:
                methodologies.append({
                    "document_name": doc.file_name,
                    "methodology": f"Research methodology described in {doc.file_name} focusing on {doc.category}."
                })
                pros_cons.append({
                    "document_name": doc.file_name,
                    "advantages": ["Automated feature scaling", "Domain-specific parsing"],
                    "disadvantages": ["High computational complexity", "Sensitive to noise"]
                })
                conclusions.append({
                    "document_name": doc.file_name,
                    "conclusion": f"Core conclusion reached by the authors of {doc.file_name}."
                })
                
            return {
                "methodologies": methodologies,
                "pros_cons": pros_cons,
                "similarities": [
                    "Both documents focus on advanced technical systems.",
                    "Both studies utilize automated preprocessing pipelines."
                ],
                "differences": [
                    f"The first paper emphasizes {docs[0].category} while the second covers {docs[1].category}."
                ],
                "conclusions": conclusions
            }

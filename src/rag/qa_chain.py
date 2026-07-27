import json
import logging
from datetime import datetime
from sqlalchemy.orm import Session
from config.settings import settings
from src.database.models import ChatMessage
from src.vector_store.manager import VectorSearchManager

logger = logging.getLogger(__name__)

def get_llm():
    """
    Returns the configured LLM based on environment settings.
    If no keys are configured, returns None to trigger a mock responder for test environments.
    """
    if settings.llm_provider.lower() == "openai" and settings.openai_api_key:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            openai_api_key=settings.openai_api_key,
            model_name=settings.openai_model_name,
            temperature=0.0
        )
    elif settings.llm_provider.lower() == "gemini" and settings.google_api_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            google_api_key=settings.google_api_key,
            model=settings.gemini_model_name,
            temperature=0.0
        )
    return None

class RAGQAEngine:
    def __init__(self, db_session: Session):
        self.db = db_session
        self.search_manager = VectorSearchManager()
        self.llm = get_llm()

    def _get_chat_history(self, session_id: str, limit: int = 5) -> str:
        """
        Retrieves the last `limit` messages in the session to construct short-term conversational memory.
        """
        messages = (
            self.db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.timestamp.asc())
            .all()
        )
        # Take last N messages
        recent_messages = messages[-limit:]
        history_str = ""
        for msg in recent_messages:
            role_label = "User" if msg.role == "user" else "Assistant"
            history_str += f"{role_label}: {msg.content}\n"
        return history_str

    def answer_question(self, session_id: str, query: str, doc_ids: list[str] = None, search_mode: str = "semantic") -> dict:
        """
        Retrieves relevant context, formats conversational history, invokes the LLM,
        and saves both the user query and generated response to the database.
        """
        # 1. Retrieve relevant chunks from vector store
        retrieved_chunks = self.search_manager.search(
            query=query,
            search_mode=search_mode,
            doc_ids=doc_ids,
            top_k=4
        )
        
        # 2. Extract context string & source metadata
        context_blocks = []
        retrieved_context_data = []
        possible_sources = []
        
        for idx, chunk in enumerate(retrieved_chunks):
            doc_name = chunk["metadata"]["file_name"]
            page_num = chunk["metadata"]["page_number"]
            context_blocks.append(f"Source: {doc_name} (Page {page_num})\nContent: {chunk['text']}\n---")
            
            retrieved_context_data.append({
                "text": chunk["text"],
                "document_name": doc_name,
                "page_number": page_num
            })
            
            possible_sources.append({
                "document_name": doc_name,
                "page_number": page_num
            })
            
        context_str = "\n".join(context_blocks)
        
        # 3. Retrieve conversational memory
        history_str = self._get_chat_history(session_id)
        
        # 4. Formulate the prompt
        system_prompt = (
            "You are a helpful, professional research assistant.\n"
            "Answer the user's question using ONLY the provided context blocks. Do not make up facts.\n"
            "If the answer cannot be determined from the retrieved context, you MUST respond EXACTLY with:\n"
            "\"I cannot determine the answer from the provided documents.\"\n\n"
            "You must respond with a JSON object containing two keys:\n"
            "1. 'final_answer': (string) your comprehensive answer to the user query.\n"
            "2. 'citations': (list of objects) each containing 'document_name' and 'page_number' matching the sources you used.\n\n"
            "Example format:\n"
            "{\n"
            "  \"final_answer\": \"The company achieved a 12% revenue growth.\",\n"
            "  \"citations\": [\n"
            "     {\"document_name\": \"financials.pdf\", \"page_number\": 4}\n"
            "  ]\n"
            "}\n\n"
            f"Context:\n{context_str}\n\n"
            f"Chat History:\n{history_str}\n"
        )
        
        # 5. Get Answer (handle LLM vs. Mock fallback)
        final_answer = ""
        citations = []
        
        if self.llm:
            try:
                # Call LLM
                response = self.llm.invoke([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
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
                
                # Attempt to parse JSON response
                try:
                    # Clean markdown wrappers if LLM returned them
                    if content.startswith("```json"):
                        content = content.split("```json")[1].split("```")[0].strip()
                    elif content.startswith("```"):
                        content = content.split("```")[1].split("```")[0].strip()
                        
                    parsed = json.loads(content)
                    final_answer = parsed.get("final_answer", "")
                    citations = parsed.get("citations", [])
                except Exception:
                    # Fallback if parsing fails: treat whole content as answer and use retrieved sources
                    final_answer = content
                    citations = possible_sources[:1] if "I cannot determine" not in content else []
            except Exception as e:
                logger.error(f"Error calling LLM: {e}")
                final_answer = "Error generating response from LLM provider."
                citations = []
        else:
            # Mock mode for testing without active API keys
            logger.warning("No LLM API keys configured. Running in Mock Mode.")
            if not retrieved_chunks:
                final_answer = "I cannot determine the answer from the provided documents."
                citations = []
            else:
                first_chunk = retrieved_chunks[0]
                final_answer = f"[Mock Mode] Based on {first_chunk['metadata']['file_name']} (Page {first_chunk['metadata']['page_number']}), the text mentions: '{first_chunk['text'][:100]}...'"
                citations = [{
                    "document_name": first_chunk["metadata"]["file_name"],
                    "page_number": first_chunk["metadata"]["page_number"]
                }]

        # 6. Save message exchange to ChatMessage database
        user_msg = ChatMessage(
            session_id=session_id,
            role="user",
            content=query
        )
        assistant_msg = ChatMessage(
            session_id=session_id,
            role="assistant",
            content=final_answer,
            citations=json.dumps(citations)
        )
        self.db.add(user_msg)
        self.db.add(assistant_msg)
        self.db.commit()
        
        return {
            "final_answer": final_answer,
            "sources": citations,
            "retrieved_context": retrieved_context_data
        }

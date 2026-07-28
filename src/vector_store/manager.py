import os
import sqlite3
import logging
from rank_bm25 import BM25Okapi
from config.settings import settings

logger = logging.getLogger(__name__)

# Attempt to load Chroma DB, fall back to SQLite Vector Store if C-DLLs are blocked by OS policies
try:
    import chromadb
    from chromadb.utils import embedding_functions
    HAS_CHROMADB = True
except Exception as e:
    logger.warning(f"ChromaDB import failed ({e}). Falling back to SQLite persistent vector store.")
    HAS_CHROMADB = False

class FallbackCollection:
    def __init__(self, db_path: str):
        self.db_path = os.path.join(db_path, "vector_store.db")
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    doc_id TEXT,
                    file_name TEXT,
                    page_number INTEGER,
                    text TEXT
                )
            """)
            conn.commit()

    def add(self, ids: list[str], documents: list[str], metadatas: list[dict]):
        with sqlite3.connect(self.db_path) as conn:
            for chunk_id, doc_text, meta in zip(ids, documents, metadatas):
                conn.execute(
                    "INSERT OR REPLACE INTO chunks (id, doc_id, file_name, page_number, text) VALUES (?, ?, ?, ?, ?)",
                    (chunk_id, meta.get("doc_id"), meta.get("file_name"), int(meta.get("page_number", 1)), doc_text)
                )
            conn.commit()

    def delete(self, where: dict):
        if not where:
            return
        doc_id = where.get("doc_id")
        with sqlite3.connect(self.db_path) as conn:
            if isinstance(doc_id, dict) and "$in" in doc_id:
                in_ids = doc_id["$in"]
                placeholders = ",".join("?" for _ in in_ids)
                conn.execute(f"DELETE FROM chunks WHERE doc_id IN ({placeholders})", in_ids)
            elif doc_id:
                conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            conn.commit()

    def get(self, where: dict = None, include: list[str] = None):
        query = "SELECT id, doc_id, file_name, page_number, text FROM chunks"
        params = []
        if where and "doc_id" in where:
            doc_id = where["doc_id"]
            if isinstance(doc_id, dict) and "$in" in doc_id:
                in_ids = doc_id["$in"]
                placeholders = ",".join("?" for _ in in_ids)
                query += f" WHERE doc_id IN ({placeholders})"
                params = list(in_ids)
            else:
                query += " WHERE doc_id = ?"
                params = [doc_id]

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            rows = cursor.execute(query, params).fetchall()

        ids = [r[0] for r in rows]
        docs = [r[4] for r in rows]
        metas = [{"doc_id": r[1], "file_name": r[2], "page_number": r[3]} for r in rows]

        return {
            "ids": ids,
            "documents": docs,
            "metadatas": metas
        }

    def query(self, query_texts: list[str], n_results: int = 5, where: dict = None):
        query_str = query_texts[0] if query_texts else ""
        data = self.get(where=where)
        if not data["documents"]:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        docs = data["documents"]
        ids = data["ids"]
        metas = data["metadatas"]

        query_words = set(query_str.lower().split())
        scored = []
        for idx, doc_text in enumerate(docs):
            doc_words = doc_text.lower().split()
            overlap = sum(1 for w in doc_words if w in query_words)
            score = overlap / (len(doc_words) + 1.0)
            dist = 1.0 - score
            scored.append((dist, ids[idx], docs[idx], metas[idx]))

        scored.sort(key=lambda x: x[0])
        top = scored[:n_results]

        return {
            "ids": [[x[1] for x in top]],
            "documents": [[x[2] for x in top]],
            "metadatas": [[x[3] for x in top]],
            "distances": [[x[0] for x in top]]
        }


class VectorSearchManager:
    def __init__(self):
        if HAS_CHROMADB:
            try:
                self.client = chromadb.PersistentClient(path=settings.vector_db_dir)
                self.embedding_fn = self._get_embedding_function()
                self.collection = self.client.get_or_create_collection(
                    name="document_chunks",
                    embedding_function=self.embedding_fn
                )
            except Exception as e:
                logger.warning(f"ChromaDB initialization failed: {e}. Using SQLite fallback.")
                self.collection = FallbackCollection(settings.vector_db_dir)
        else:
            self.collection = FallbackCollection(settings.vector_db_dir)

    def _get_embedding_function(self):
        provider = settings.embedding_provider.lower()
        if provider == "openai" and settings.openai_api_key:
            return embedding_functions.OpenAIEmbeddingFunction(
                api_key=settings.openai_api_key,
                model_name="text-embedding-3-small"
            )
        try:
            return embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        except Exception:
            return None

    def add_document_chunks(self, doc_id: str, file_name: str, chunks: list[dict]):
        if not chunks:
            return

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            ids.append(f"{doc_id}_{chunk['chunk_id']}")
            documents.append(chunk["text"])
            metadatas.append({
                "doc_id": doc_id,
                "file_name": file_name,
                "page_number": int(chunk["page_number"])
            })

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

    def delete_document_chunks(self, doc_id: str):
        self.collection.delete(where={"doc_id": doc_id})

    def _semantic_query(self, query: str, doc_ids: list[str] = None, top_k: int = 5) -> list[dict]:
        filter_dict = {}
        if doc_ids:
            if len(doc_ids) == 1:
                filter_dict = {"doc_id": doc_ids[0]}
            else:
                filter_dict = {"doc_id": {"$in": doc_ids}}

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=filter_dict if doc_ids else None
        )

        parsed = []
        if results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            ids = results["ids"][0]
            distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)

            for idx, text in enumerate(docs):
                parsed.append({
                    "id": ids[idx],
                    "text": text,
                    "metadata": metas[idx],
                    "score": float(distances[idx])
                })
        return parsed

    def _keyword_query(self, query: str, doc_ids: list[str] = None, top_k: int = 5) -> list[dict]:
        filter_dict = {}
        if doc_ids:
            if len(doc_ids) == 1:
                filter_dict = {"doc_id": doc_ids[0]}
            else:
                filter_dict = {"doc_id": {"$in": doc_ids}}

        all_chunks = self.collection.get(
            where=filter_dict if doc_ids else None,
            include=["documents", "metadatas"]
        )

        if not all_chunks or not all_chunks["documents"]:
            return []

        docs = all_chunks["documents"]
        metas = all_chunks["metadatas"]
        ids = all_chunks["ids"]

        corpus_tokenized = [doc.lower().split(" ") for doc in docs]
        query_tokenized = query.lower().split(" ")

        bm25 = BM25Okapi(corpus_tokenized)
        scores = bm25.get_scores(query_tokenized)

        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        for idx in sorted_indices:
            doc_words = set(corpus_tokenized[idx])
            has_overlap = any(qw in doc_words for qw in query_tokenized)

            if scores[idx] <= 0 and not has_overlap:
                continue

            score = scores[idx]
            if score <= 0 and has_overlap:
                overlap_count = sum(1 for qw in query_tokenized if qw in doc_words)
                score = 1e-5 * overlap_count

            results.append({
                "id": ids[idx],
                "text": docs[idx],
                "metadata": metas[idx],
                "score": float(score)
            })

        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def _hybrid_query(self, query: str, doc_ids: list[str] = None, top_k: int = 5) -> list[dict]:
        candidate_k = max(20, top_k * 2)

        semantic_results = self._semantic_query(query, doc_ids, top_k=candidate_k)
        keyword_results = self._keyword_query(query, doc_ids, top_k=candidate_k)

        rrf_scores = {}
        chunk_map = {}

        for rank, res in enumerate(semantic_results):
            chunk_id = res["id"]
            chunk_map[chunk_id] = res
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (60.0 + rank + 1))

        for rank, res in enumerate(keyword_results):
            chunk_id = res["id"]
            chunk_map[chunk_id] = res
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (60.0 + rank + 1))

        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        results = []
        for cid in sorted_ids[:top_k]:
            res = chunk_map[cid]
            res["score"] = rrf_scores[cid]
            results.append(res)

        return results

    def search(self, query: str, search_mode: str = "semantic", doc_ids: list[str] = None, top_k: int = 5) -> list[dict]:
        mode = search_mode.lower()
        if mode == "keyword":
            return self._keyword_query(query, doc_ids, top_k)
        elif mode == "hybrid":
            return self._hybrid_query(query, doc_ids, top_k)
        else:
            return self._semantic_query(query, doc_ids, top_k)

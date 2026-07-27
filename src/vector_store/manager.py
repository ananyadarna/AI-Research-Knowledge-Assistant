import os
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from config.settings import settings

class VectorSearchManager:
    def __init__(self):
        # 1. Initialize persistent Chroma client
        self.client = chromadb.PersistentClient(path=settings.vector_db_dir)
        
        # 2. Select embedding function based on settings
        self.embedding_fn = self._get_embedding_function()
        
        # 3. Get or create the main collection
        self.collection = self.client.get_or_create_collection(
            name="document_chunks",
            embedding_function=self.embedding_fn
        )

    def _get_embedding_function(self):
        """
        Loads the configured embedding function. Falls back to a local
        sentence-transformers model if API keys or local model aren't fully configured.
        """
        provider = settings.embedding_provider.lower()
        
        if provider == "openai" and settings.openai_api_key:
            return embedding_functions.OpenAIEmbeddingFunction(
                api_key=settings.openai_api_key,
                model_name="text-embedding-3-small"
            )
        elif provider == "gemini" and settings.google_api_key:
            return embedding_functions.GoogleGenerativeAiEmbeddingFunction(
                api_key=settings.google_api_key
            )
        
        # Local fallback (sentence-transformers/all-MiniLM-L6-v2)
        # Chroma will download this model automatically if not cached
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

    def add_document_chunks(self, doc_id: str, file_name: str, chunks: list[dict]):
        """
        Adds text chunks to the Chroma DB collection with proper metadata.
        """
        if not chunks:
            return
            
        ids = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            # Format a unique ID per chunk
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
        """
        Removes all chunks associated with a specific document ID.
        """
        self.collection.delete(where={"doc_id": doc_id})

    def _semantic_query(self, query: str, doc_ids: list[str] = None, top_k: int = 5) -> list[dict]:
        """
        Retrieves top_k chunks using cosine similarity from Chroma DB.
        """
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
            # Chroma returns distances; smaller distance means closer similarity
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
        """
        Retrieves top_k chunks using BM25 token-matching algorithm in memory.
        """
        filter_dict = {}
        if doc_ids:
            if len(doc_ids) == 1:
                filter_dict = {"doc_id": doc_ids[0]}
            else:
                filter_dict = {"doc_id": {"$in": doc_ids}}
                
        # Fetch all matching documents in the scope
        all_chunks = self.collection.get(
            where=filter_dict if doc_ids else None,
            include=["documents", "metadatas"]
        )
        print("DEBUG_KEYWORD_GET:", all_chunks)
        
        if not all_chunks or not all_chunks["documents"]:
            return []
            
        docs = all_chunks["documents"]
        metas = all_chunks["metadatas"]
        ids = all_chunks["ids"]
        
        # Tokenize corpus and query
        corpus_tokenized = [doc.lower().split(" ") for doc in docs]
        query_tokenized = query.lower().split(" ")
        
        bm25 = BM25Okapi(corpus_tokenized)
        scores = bm25.get_scores(query_tokenized)
        
        # Sort indices by score descending
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        results = []
        for idx in sorted_indices:
            # Check for actual token intersection to handle zero IDF on tiny corpora (N=2)
            doc_words = set(corpus_tokenized[idx])
            has_overlap = any(qw in doc_words for qw in query_tokenized)
            
            if scores[idx] <= 0 and not has_overlap:
                continue  # Ignore if it has no match and non-positive score
                
            score = scores[idx]
            if score <= 0 and has_overlap:
                # Assign a tiny positive score based on match count so RRF ranking can rank it
                overlap_count = sum(1 for qw in query_tokenized if qw in doc_words)
                score = 1e-5 * overlap_count

            results.append({
                "id": ids[idx],
                "text": docs[idx],
                "metadata": metas[idx],
                "score": float(score)
            })
            
        # Re-sort results to ensure overlap-scored items rank above zero-scored items
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return results[:top_k]



    def _hybrid_query(self, query: str, doc_ids: list[str] = None, top_k: int = 5) -> list[dict]:
        """
        Combines Semantic & BM25 results using Reciprocal Rank Fusion (RRF).
        RRF formula: score = sum( 1 / (60 + rank) )
        """
        # Fetch larger pool of candidates from both search modes to get good overlap
        candidate_k = max(20, top_k * 2)
        
        semantic_results = self._semantic_query(query, doc_ids, top_k=candidate_k)
        keyword_results = self._keyword_query(query, doc_ids, top_k=candidate_k)
        
        rrf_scores = {}
        chunk_map = {}
        
        # Calculate scores from semantic rankings
        for rank, res in enumerate(semantic_results):
            chunk_id = res["id"]
            chunk_map[chunk_id] = res
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (60.0 + rank + 1))
            
        # Add scores from keyword rankings
        for rank, res in enumerate(keyword_results):
            chunk_id = res["id"]
            chunk_map[chunk_id] = res
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (60.0 + rank + 1))
            
        # Sort by RRF score descending
        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        
        results = []
        for cid in sorted_ids[:top_k]:
            res = chunk_map[cid]
            res["score"] = rrf_scores[cid]  # Override with RRF score
            results.append(res)
            
        return results

    def search(self, query: str, search_mode: str = "semantic", doc_ids: list[str] = None, top_k: int = 5) -> list[dict]:
        """
        Dispatches search to specified mode: semantic, keyword, or hybrid.
        """
        mode = search_mode.lower()
        if mode == "keyword":
            return self._keyword_query(query, doc_ids, top_k)
        elif mode == "hybrid":
            return self._hybrid_query(query, doc_ids, top_k)
        else:
            return self._semantic_query(query, doc_ids, top_k)

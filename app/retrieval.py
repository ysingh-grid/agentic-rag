import asyncio
import json
import logging
import os

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app.config import settings

logger = logging.getLogger(__name__)

class RetrieverSingleton:
    """Manages models and indexes so they aren't loaded per request."""
    def __init__(self):
        self.embedding_model = None
        self.cross_encoder = None
        self.chroma_client = None
        self.collection = None
        self.bm25 = None
        self.doc_chunks: list[dict] = []  # parallel list for BM25 index

    def load_models(self):
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        self.embedding_model = HuggingFaceEmbedding(model_name=settings.EMBEDDING_MODEL)
        self.cross_encoder = CrossEncoder(settings.RERANKER_MODEL, max_length=512)

        os.makedirs(settings.CHROMA_DB_DIR, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DB_DIR)
        self.collection = self.chroma_client.get_or_create_collection(name="rag_documents")

        self._load_bm25()

    def _load_bm25(self):
        if os.path.exists(settings.BM25_INDEX_PATH):
            try:
                with open(settings.BM25_INDEX_PATH, 'r') as f:
                    data = json.load(f)
                self.doc_chunks = data.get("docs", [])
                corpus = [doc.get("content", "").split() for doc in self.doc_chunks]
                if corpus:
                    self.bm25 = BM25Okapi(corpus)
                    logger.info("BM25 index loaded from disk (%d docs).", len(self.doc_chunks))
            except Exception as e:
                logger.error("Failed to load BM25 index: %s", e)
                self.bm25 = None
        else:
            logger.warning("No BM25 index found on disk — serving vector-only (degraded) until first ingestion.")
            self.bm25 = None

retriever = RetrieverSingleton()

def reciprocal_rank_fusion(
    vector_results: list[tuple[dict, float]],
    bm25_results: list[tuple[dict, float]],
    k: int = settings.RRF_K,
) -> list[dict]:
    """
    Combines two ranked lists via Reciprocal Rank Fusion.
    score(d) = Σ 1 / (k + rank_i(d))     k=60 is the empirically validated constant.
    Uses rank position only — raw BM25 / cosine scores are discarded (scale-invariant).
    """
    rrf_scores: dict[str, dict] = {}

    for rank, (doc, _score) in enumerate(vector_results, start=1):
        doc_id = doc.get("id") if isinstance(doc, dict) else str(id(doc))
        entry = rrf_scores.setdefault(doc_id, {"doc": doc, "score": 0.0})
        entry["score"] += 1.0 / (k + rank)

    for rank, (doc, _score) in enumerate(bm25_results, start=1):
        doc_id = doc.get("id") if isinstance(doc, dict) else str(id(doc))
        entry = rrf_scores.setdefault(doc_id, {"doc": doc, "score": 0.0})
        entry["score"] += 1.0 / (k + rank)

    sorted_docs = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in sorted_docs][:settings.RETRIEVAL_TOP_K]

async def _vector_search(query: str, query_embedding: list[float], top_k: int) -> list[tuple[dict, float]]:
    """Query ChromaDB. Guards against requesting more results than the collection contains."""
    try:
        # ChromaDB raises if n_results > number of items in the collection
        count = await asyncio.to_thread(lambda: retriever.collection.count())
        if count == 0:
            return []
        safe_k = min(top_k, count)

        results = await asyncio.to_thread(
            retriever.collection.query,
            query_embeddings=[query_embedding],
            n_results=safe_k,
        )
    except Exception as e:
        logger.error("ChromaDB query failed: %s", e)
        return []

    docs = []
    if results and results.get('documents') and results['documents'][0]:
        for idx, doc_text in enumerate(results['documents'][0]):
            metadata = (results['metadatas'][0][idx] if results.get('metadatas') else {})
            score = (results['distances'][0][idx] if results.get('distances') else 0.0)
            docs.append(({"content": doc_text, "metadata": metadata, "id": results['ids'][0][idx]}, score))
    return docs

async def _bm25_search(query: str, top_k: int):
    if not retriever.bm25 or not retriever.doc_chunks:
        return []
        
    tokenized_query = query.split()
    scores = retriever.bm25.get_scores(tokenized_query)
    
    # Get top_k indices
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    
    docs = []
    for idx in top_indices:
        if scores[idx] > 0:
            docs.append((retriever.doc_chunks[idx], scores[idx]))
    return docs

async def hybrid_retrieve(query: str, query_embedding: list[float], top_k: int = settings.RETRIEVAL_TOP_K):
    # Run searches in parallel
    vector_task = asyncio.create_task(_vector_search(query, query_embedding, top_k))
    bm25_task = asyncio.create_task(_bm25_search(query, top_k))
    
    vector_results, bm25_results = await asyncio.gather(vector_task, bm25_task)
    
    # RRF
    fused_docs = reciprocal_rank_fusion(vector_results, bm25_results)
    return fused_docs

async def rerank_candidates(query: str, candidates: list[dict], top_n: int = settings.RERANK_TOP_N):
    if not candidates:
        return []

    if not retriever.cross_encoder:
        return candidates[:top_n]

    cross_input = [[query, doc.get("content", "")] for doc in candidates]

    # Batched inference
    scores = await asyncio.to_thread(retriever.cross_encoder.predict, cross_input)

    # Assign scores and sort
    scored_candidates = []
    for doc, score in zip(candidates, scores):
        doc["rerank_score"] = float(score)
        scored_candidates.append(doc)

    scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored_candidates[:top_n]


async def session_vector_search(
    query_embedding: list[float],
    session_docs: list[dict],
    top_k: int,
) -> list[dict]:
    """In-memory cosine similarity search over per-session document chunks.

    session_docs entries must have an "embedding" key (list[float]).
    Returns up to top_k docs sorted by descending cosine similarity.
    This never touches ChromaDB — results are ephemeral to the session.
    """
    if not session_docs:
        return []

    import numpy as np

    q = np.array(query_embedding, dtype=np.float32)
    q_norm = q / (np.linalg.norm(q) + 1e-8)

    doc_embs = np.array(
        [d["embedding"] for d in session_docs], dtype=np.float32
    )
    norms = np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-8
    doc_norm = doc_embs / norms

    scores = doc_norm @ q_norm  # shape (N,)
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        {**{k: v for k, v in session_docs[i].items() if k != "embedding"},
         "session_score": float(scores[i])}
        for i in top_indices
        if scores[i] > 0.0
    ]

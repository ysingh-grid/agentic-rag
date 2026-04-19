import asyncio
import hashlib
import json
import logging
import os
import time

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from llama_index.core.node_parser import SentenceSplitter

from app.auth import get_admin_key
from app.config import settings
from app.retrieval import retriever

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Ingestion"])

def _make_doc_id(filename: str) -> str:
    """Deterministic doc_id from filename. Stable across restarts."""
    return hashlib.sha256(filename.encode("utf-8")).hexdigest()[:16]

async def process_document_bg(text: str, source_name: str, doc_id: str):
    """Background task: chunk, embed, index into ChromaDB + BM25."""
    try:
        parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        chunks = parser.split_text(text)

        timestamp = time.time()
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        metadatas = [
            {"source": source_name, "chunk_index": i, "doc_id": doc_id, "ingested_at": timestamp}
            for i in range(len(chunks))
        ]

        def chroma_upsert():
            # 1. Delete all existing chunks for this doc_id to prevent duplicates on re-ingestion
            existing = retriever.collection.get(where={"doc_id": doc_id})
            if existing and existing.get("ids"):
                retriever.collection.delete(ids=existing["ids"])
                logger.info("Deleted %d existing chunks for doc_id=%s", len(existing["ids"]), doc_id)

            # 2. Generate embeddings and insert fresh chunks
            embeddings = [
                retriever.embedding_model.get_text_embedding(c) for c in chunks
            ]
            retriever.collection.add(
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )

        await asyncio.to_thread(chroma_upsert)

        # 3. Remove old BM25 chunks for this doc and append new ones
        retriever.doc_chunks = [
            c for c in retriever.doc_chunks
            if c.get("metadata", {}).get("doc_id") != doc_id
        ]
        for i, chunk_text in enumerate(chunks):
            retriever.doc_chunks.append({
                "id": ids[i],
                "content": chunk_text,
                "metadata": metadatas[i],
            })

        # 4. Rebuild BM25 index in a thread (CPU-bound)
        def rebuild_bm25():
            from rank_bm25 import BM25Okapi
            corpus = [doc["content"].split() for doc in retriever.doc_chunks]
            retriever.bm25 = BM25Okapi(corpus)

            # Persist to disk so restarts don't need full rebuild
            os.makedirs(os.path.dirname(settings.BM25_INDEX_PATH), exist_ok=True)
            with open(settings.BM25_INDEX_PATH, "w") as f:
                json.dump({"docs": retriever.doc_chunks}, f)

        await asyncio.to_thread(rebuild_bm25)

        logger.info("Document '%s' ingested: %d chunks.", source_name, len(chunks))

        # 5. Invalidate retrieval cache for any queries that may have hit stale results.
        #    We clear ALL retrieval cache keys (broad invalidation) since doc update
        #    can affect any query. TTL will clean stragglers within 1h anyway.
        try:
            from app.cache import cache_layer
            if cache_layer.enabled:
                async for key in cache_layer.redis.scan_iter("retrieval_cache:*", count=100):
                    await cache_layer.redis.delete(key)
                logger.info("Retrieval cache invalidated after ingestion of %s.", source_name)
        except Exception as e:
            logger.warning("Cache invalidation failed: %s", e)

    except Exception as e:
        logger.error("Failed to ingest document '%s': %s", source_name, e)


@router.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    admin_user: str = Depends(get_admin_key),
):
    """Admin-only endpoint for document ingestion."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    try:
        content_bytes = await file.read()
        content = content_bytes.decode("utf-8")
    except Exception:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded text")

    if not content.strip():
        raise HTTPException(status_code=400, detail="File appears to be empty")

    # Deterministic doc_id — same filename always maps to same ID
    doc_id = _make_doc_id(file.filename)

    # Non-blocking: ingestion runs in the event loop background
    asyncio.create_task(process_document_bg(content, file.filename, doc_id))

    return {
        "message": f"Document '{file.filename}' queued for ingestion.",
        "doc_id": doc_id,
    }

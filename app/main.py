import sys
import asyncio
import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, Field

import app.session as session_module
from app.config import settings
from app.health import router as health_router, set_redis_client as health_set_redis
from app.ingestion import router as ingestion_router
from app.auth import get_api_key
from app.session import get_session, clear_session, RedisSessionStore
from app.cache import cache_layer
from app.retrieval import retriever, hybrid_retrieve, rerank_candidates
from app.context import assemble_prompt
from app.agent import stream_llm_response
from app.observability import tracer

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI):
    _redis_client = None
    # Try connecting to Redis
    try:
        from redis.asyncio import Redis
        _redis_client = Redis.from_url(settings.REDIS_URL, decode_responses=False)
        await _redis_client.ping()

        # Update the module-level session_store so all get_session() calls use Redis
        session_module.session_store = RedisSessionStore(_redis_client)

        # Wire Redis into health-check and cache
        health_set_redis(_redis_client)
        cache_layer.setup(_redis_client)
        print("Redis connected successfully.")
    except Exception as e:
        print(f"Failed to connect to Redis. Running in-memory (degraded). Error: {e}", file=sys.stderr)

    retriever.load_models()
    print("Models and indexes loaded.")

    yield

    # Graceful shutdown: drain active background tasks then close Redis
    pending = [t for t in asyncio.all_tasks() if not t.done()]
    if pending:
        try:
            await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=30)
        except asyncio.TimeoutError:
            pass

    if _redis_client is not None:
        await _redis_client.aclose()


app = FastAPI(
    title=settings.APP_NAME,
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None
)

# Exception handlers
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# Routers
app.include_router(health_router)
app.include_router(ingestion_router)

# Request Models
class ChatRequest(BaseModel):
    query: str = Field(..., max_length=settings.MAX_QUERY_LENGTH)

@app.post("/chat")
@limiter.limit("10/minute")
async def chat_endpoint(
    request: Request,
    response: Response,
    payload: ChatRequest,
    user: str = Depends(get_api_key)
):
    session = await get_session(request, response)

    # Observability main trace
    trace = tracer.trace(session_id=session.session_id, name="rag-query", query=payload.query)

    try:
        # 1. Response Cache Check — key uses history BEFORE this turn is appended
        cached_response = await cache_layer.get_response(
            session.session_id, payload.query, session.chat_history
        )
        if cached_response:
            session.chat_history.append({"role": "user", "content": payload.query})
            session.chat_history.append({"role": "assistant", "content": cached_response})
            await session_module.session_store.set(session.session_id, session)

            async def _cached_gen():
                yield cached_response

            return StreamingResponse(_cached_gen(), media_type="text/event-stream")

        # 2. Embedding Cache Check & Generation
        query_emb = await cache_layer.get_embedding(payload.query)
        if not query_emb:
            query_emb = await asyncio.to_thread(
                retriever.embedding_model.get_text_embedding, payload.query
            )
            await cache_layer.set_embedding(payload.query, query_emb)

        # 3. Retrieval Cache Check & Hybrid Retrieval
        retrieved_docs = await cache_layer.get_retrieval(
            payload.query, settings.RETRIEVAL_TOP_K, settings.RRF_K
        )
        if not retrieved_docs:
            retrieved_docs = await hybrid_retrieve(payload.query, query_emb, settings.RETRIEVAL_TOP_K)
            if retrieved_docs:
                await cache_layer.set_retrieval(
                    payload.query, settings.RETRIEVAL_TOP_K, settings.RRF_K, retrieved_docs
                )

        # 4. Cross-Encoder Reranking
        reranked_docs = await rerank_candidates(payload.query, retrieved_docs, settings.RERANK_TOP_N)

        # 5. Assemble prompt (passes trimmed history into the prompt string)
        messages, trimmed_history = assemble_prompt(payload.query, session, reranked_docs)

        # Snapshot history BEFORE the new turn for the response-cache key
        history_snapshot = list(session.chat_history)

        # 6. Stream and persist
        async def tracking_generator():
            full_response = ""
            async for token in stream_llm_response(messages):
                yield token
                full_response += token

            # Only persist a clean (non-error) response
            if full_response.strip() and "LLM_TIMEOUT" not in full_response:
                session.chat_history.append({"role": "user", "content": payload.query})
                session.chat_history.append({"role": "assistant", "content": full_response})
                # Enforce sliding window on stored history too
                if len(session.chat_history) > 10:  # 5 turns × 2 messages
                    session.chat_history = session.chat_history[-10:]
                await session_module.session_store.set(session.session_id, session)
                # Key uses pre-turn snapshot so get_response() can reconstruct the correct key
                await cache_layer.set_response(
                    session.session_id, payload.query, history_snapshot, full_response
                )

        return StreamingResponse(tracking_generator(), media_type="text/event-stream")

    finally:
        trace.end()

@app.post("/clear-session")
async def clear_session_endpoint(
    request: Request,
    user: str = Depends(get_api_key)
):
    cleared = await clear_session(request)
    # Response-cache keys are tied to the session history hash;
    # clearing history means all old keys will never match again — no explicit scan needed.
    if cleared:
        return {"status": "success", "message": "Session history cleared"}
    raise HTTPException(status_code=404, detail="No active session found")

# Mount static frontend LAST so it never shadows API routes.
# A catch-all static mount registered before routers blocks /chat, /health etc.
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/ui", StaticFiles(directory=frontend_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

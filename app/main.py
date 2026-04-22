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
from app.retrieval import retriever, hybrid_retrieve, rerank_candidates, session_vector_search, reciprocal_rank_fusion
from app.context import assemble_prompt
from app.agent import stream_llm_response
from app.observability import tracer

from app.multi_query import generate_queries  # 🔥 NEW

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI):
    _redis_client = None
    try:
        from redis.asyncio import Redis
        _redis_client = Redis.from_url(settings.REDIS_URL, decode_responses=False)
        await _redis_client.ping()

        session_module.session_store = RedisSessionStore(_redis_client)
        health_set_redis(_redis_client)
        cache_layer.setup(_redis_client)

        print("Redis connected successfully.")
    except Exception as e:
        print(f"Redis unavailable. Using in-memory. Error: {e}", file=sys.stderr)

    retriever.load_models()
    print("Models loaded.")

    yield

    if _redis_client:
        try:
            await _redis_client.aclose()
        except Exception:
            pass


app = FastAPI(
    title=settings.APP_NAME,
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

app.include_router(health_router)
app.include_router(ingestion_router)


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
    trace = tracer.trace(session_id=session.session_id, name="rag-query", query=payload.query)

    try:
        # 1. Response Cache
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

        # 2. MULTI-QUERY 🔥
        queries = generate_queries(payload.query, num_queries=3)

        all_results = []

        for q in queries:
            # try retrieval cache per query
            cached = await cache_layer.get_retrieval(
                q, settings.RETRIEVAL_TOP_K, settings.RRF_K
            )

            if cached:
                results = cached
            else:
                q_emb = await asyncio.to_thread(
                    retriever.embedding_model.get_text_embedding, q
                )

                if session.session_docs:
                    results = await session_vector_search(
                        q_emb, session.session_docs, settings.RETRIEVAL_TOP_K
                    )
                else:
                    results = await hybrid_retrieve(q, q_emb, settings.RETRIEVAL_TOP_K)

                if results:
                    await cache_layer.set_retrieval(
                        q, settings.RETRIEVAL_TOP_K, settings.RRF_K, results
                    )

            if results:
                all_results.extend(results)

        # 3. DEDUP MULTI-QUERY RESULTS 🔥
        seen_ids = set()
        retrieved_docs = []
        for doc in all_results:
            doc_id = doc.get("id") if isinstance(doc, dict) else str(id(doc))
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                retrieved_docs.append(doc)

        # 4. RERANK
        reranked_docs = await rerank_candidates(
            payload.query, retrieved_docs, settings.RERANK_TOP_N
        )

        # 5. PROMPT
        messages, trimmed_history = assemble_prompt(payload.query, session, reranked_docs)
        history_snapshot = list(session.chat_history)

        # 6. STREAM RESPONSE
        async def tracking_generator():
            full_response = ""

            async for token in stream_llm_response(messages):
                yield token
                full_response += token

            if full_response.strip() and "LLM_TIMEOUT" not in full_response:
                session.chat_history.append({"role": "user", "content": payload.query})
                session.chat_history.append({"role": "assistant", "content": full_response})

                if len(session.chat_history) > 10:
                    session.chat_history = session.chat_history[-10:]

                await session_module.session_store.set(session.session_id, session)

                await cache_layer.set_response(
                    session.session_id,
                    payload.query,
                    history_snapshot,
                    full_response
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

    if cleared:
        return {"status": "success", "message": "Session history cleared"}

    raise HTTPException(status_code=404, detail="No active session found")


# Static frontend
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
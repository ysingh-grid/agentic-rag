import httpx
from fastapi import APIRouter
from pydantic import BaseModel

from app.config import settings

router = APIRouter(tags=["Health"])

# Injected at startup by main.py once Redis is confirmed live
_redis_client = None

def set_redis_client(client) -> None:
    global _redis_client
    _redis_client = client

class HealthResponse(BaseModel):
    status: str
    components: dict[str, str]

@router.get("/health", response_model=HealthResponse)
async def health_check():
    components: dict[str, str] = {"api": "ok"}

    # --- Redis ---
    if _redis_client is not None:
        try:
            await _redis_client.ping()
            components["redis"] = "ok"
        except Exception as e:
            components["redis"] = f"error: {e}"
    else:
        components["redis"] = "degraded (in-memory fallback)"

    # --- LLM (LM Studio) ---
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{settings.LLM_BASE_URL}/models")
            if resp.status_code == 200:
                models = resp.json().get("data", [])
                if models:
                    components["llm"] = f"ok ({len(models)} model(s) loaded)"
                else:
                    components["llm"] = "no_models_loaded"
            else:
                components["llm"] = f"error_status_{resp.status_code}"
    except Exception as e:
        components["llm"] = f"error: {e}"

    # --- ChromaDB + BM25 ---
    try:
        from app.retrieval import retriever
        if retriever.collection is not None:
            count = retriever.collection.count()
            components["chromadb"] = f"ok ({count} chunks indexed)"
        else:
            components["chromadb"] = "not initialised"

        if retriever.bm25 is not None:
            components["bm25"] = f"ok ({len(retriever.doc_chunks)} docs)"
        else:
            components["bm25"] = "degraded (vector-only until first ingestion)"
    except Exception as e:
        components["chromadb"] = f"error: {e}"

    overall_status = "ok" if all(
        v.startswith("ok") or v.startswith("degraded") for v in components.values()
    ) else "degraded"

    return HealthResponse(status=overall_status, components=components)

import asyncio
import json
import logging
from typing import AsyncGenerator

from fastapi import HTTPException
import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# Concurrency Gate: limits simultaneous LLM calls to prevent VRAM exhaustion
llm_semaphore = asyncio.Semaphore(settings.LLM_CONCURRENCY)

_LLM_CALL_TIMEOUT = 45.0
_LLM_RETRY_BACKOFF = 5.0
_SLOW_WARNING_AFTER = 20.0  # seconds before "Still generating..." token

async def _call_lm_studio(messages: list[dict], attempt: int = 1) -> AsyncGenerator[str, None]:
    """Single streaming call to LM Studio. Raises on non-200 or timeout.
    
    Note: reasoning models typically emit thinking tokens
    via delta.reasoning_content before the final answer in delta.content.
    max_tokens must be large enough to cover BOTH phases.
    """
    payload = {
        "model": settings.LLM_MODEL_NAME,
        "messages": messages,
        "stream": True,
        "temperature": 0.3,
        # 8192 gives reasoning models enough room to think AND answer.
        # Reasoning models will easily exhaust 1024 tokens just on their <think> block.
        "max_tokens": 8192,
    }
    async with httpx.AsyncClient(timeout=_LLM_CALL_TIMEOUT) as client:
        async with client.stream(
            "POST",
            f"{settings.LLM_BASE_URL}/chat/completions",
            json=payload,
        ) as response:
            if response.status_code != 200:
                raise RuntimeError(f"LM Studio returned HTTP {response.status_code}")

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_chunk = line[6:]
                if data_chunk == "[DONE]":
                    return
                try:
                    data_js = json.loads(data_chunk)
                    delta = data_js["choices"][0]["delta"]
                    # Prefer visible answer content; fall back to reasoning_content
                    # so the user sees *something* while the model thinks.
                    token = delta.get("content") or delta.get("reasoning_content") or ""
                    if token:
                        yield token
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

async def stream_llm_response(messages: list[dict]) -> AsyncGenerator[str, None]:
    """Streams response from LM Studio with:
    - Concurrency gate (Semaphore + queue timeout)
    - Retry once on timeout with 5-second backoff
    - "Still generating..." nudge after 20 s of silence
    - Semaphore always released (try/finally)
    """
    # --- Concurrency gate ---
    try:
        await asyncio.wait_for(llm_semaphore.acquire(), timeout=settings.QUEUE_TIMEOUT)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=503,
            detail="System at capacity. Please retry in 30 seconds.",
            headers={"Retry-After": str(settings.QUEUE_TIMEOUT)},
        )

    try:
        for attempt in (1, 2):
            try:
                first_token = True
                last_token_time = asyncio.get_event_loop().time()

                async for token in _call_lm_studio(messages, attempt=attempt):
                    now = asyncio.get_event_loop().time()
                    # Inject slow-generation nudge exactly once after 20 s of silence
                    if first_token and (now - last_token_time) >= _SLOW_WARNING_AFTER:
                        yield "\u23f3 Still generating..."
                    first_token = False
                    last_token_time = now
                    yield token

                return  # clean exit after successful stream

            except (httpx.ReadTimeout, httpx.TimeoutException) as exc:
                if attempt == 1:
                    logger.warning("LLM timeout on attempt 1 — retrying after %ss", _LLM_RETRY_BACKOFF)
                    await asyncio.sleep(_LLM_RETRY_BACKOFF)
                    continue
                # Second timeout: return structured error token
                logger.error("LLM timeout on attempt 2 — giving up.")
                yield json.dumps({"error": "LLM_TIMEOUT", "message": "The model is taking too long. Please try again."})
                return

            except RuntimeError as exc:
                logger.error("LLM call failed: %s", exc)
                yield json.dumps({"error": "LLM_ERROR", "message": str(exc)})
                return
    finally:
        # Semaphore is ALWAYS released even if the generator is abandoned mid-stream
        llm_semaphore.release()

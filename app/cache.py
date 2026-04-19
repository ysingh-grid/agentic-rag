import hashlib
import json
import logging
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)

class CacheLayer:
    """Three-namespace Redis cache layer (embedding, retrieval, response)."""

    def __init__(self):
        self.redis = None
        self.enabled = False

    def setup(self, redis_client) -> None:
        self.redis = redis_client
        self.enabled = True

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # A. Embedding cache (global, no TTL, LRU eviction via Redis config)
    # ------------------------------------------------------------------

    async def get_embedding(self, text: str) -> Optional[list[float]]:
        if not self.enabled:
            return None
        try:
            raw = await self.redis.get(f"embedding_cache:{self._hash(text)}")
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:
            logger.warning("embedding cache read error: %s", exc)
            return None

    async def set_embedding(self, text: str, embedding: list[float]) -> None:
        if not self.enabled:
            return
        try:
            await self.redis.set(
                f"embedding_cache:{self._hash(text)}",
                json.dumps(embedding),
            )
        except Exception as exc:
            logger.warning("embedding cache write error: %s", exc)

    # ------------------------------------------------------------------
    # B. Retrieval cache (global, 1-hour TTL)
    # ------------------------------------------------------------------

    async def get_retrieval(self, query: str, top_k: int, rrf_k: int) -> Optional[list[dict]]:
        if not self.enabled:
            return None
        try:
            key_str = f"{query}_{top_k}_{rrf_k}"
            raw = await self.redis.get(f"retrieval_cache:{self._hash(key_str)}")
            if raw is None:
                return None
            # raw is bytes when decode_responses=False
            return json.loads(raw if isinstance(raw, str) else raw.decode("utf-8"))
        except Exception as exc:
            logger.warning("retrieval cache read error: %s", exc)
            return None

    async def set_retrieval(self, query: str, top_k: int, rrf_k: int, docs: list[dict]) -> None:
        if not self.enabled:
            return
        try:
            key_str = f"{query}_{top_k}_{rrf_k}"
            await self.redis.setex(
                f"retrieval_cache:{self._hash(key_str)}",
                3600,
                json.dumps(docs),
            )
        except Exception as exc:
            logger.warning("retrieval cache write error: %s", exc)

    # ------------------------------------------------------------------
    # C. Response cache (session-scoped, 30-min TTL)
    # Key format: response_cache:{session_id}:{content_hash}
    # This prefix structure allows targeted SCAN on /clear-session.
    # ------------------------------------------------------------------

    def _history_hash(self, chat_history: list[dict]) -> str:
        recent = chat_history[-3:] if len(chat_history) > 3 else chat_history
        return self._hash(json.dumps(recent, sort_keys=True))

    def _response_key(self, session_id: str, query: str, chat_history: list[dict]) -> str:
        content_hash = self._hash(f"{query}_{self._history_hash(chat_history)}")
        return f"response_cache:{session_id}:{content_hash}"

    async def get_response(
        self, session_id: str, query: str, chat_history: list[dict]
    ) -> Optional[str]:
        if not self.enabled:
            return None
        try:
            raw = await self.redis.get(self._response_key(session_id, query, chat_history))
            if raw is None:
                return None
            return raw.decode("utf-8") if isinstance(raw, bytes) else raw
        except Exception as exc:
            logger.warning("response cache read error: %s", exc)
            return None

    async def set_response(
        self, session_id: str, query: str, chat_history: list[dict], response: str
    ) -> None:
        if not self.enabled:
            return
        try:
            await self.redis.setex(
                self._response_key(session_id, query, chat_history),
                settings.SESSION_TTL,
                response,
            )
        except Exception as exc:
            logger.warning("response cache write error: %s", exc)

    async def clear_session_responses(self, session_id: str) -> int:
        """Delete all response_cache entries for a session using SCAN.
        Because keys are prefixed with `response_cache:{session_id}:`,
        we can match them without scanning the entire keyspace.
        """
        if not self.enabled:
            return 0
        deleted = 0
        try:
            pattern = f"response_cache:{session_id}:*"
            async for key in self.redis.scan_iter(pattern, count=100):
                await self.redis.delete(key)
                deleted += 1
        except Exception as exc:
            logger.warning("response cache clear error for session %s: %s", session_id, exc)
        return deleted

cache_layer = CacheLayer()

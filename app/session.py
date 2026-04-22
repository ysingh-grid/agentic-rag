import json
import secrets
import time
from typing import Optional

from fastapi import Request, Response
from pydantic import BaseModel

from app.config import settings


class SessionData(BaseModel):
    session_id: str
    chat_history: list[dict] = []
    metadata: dict = {}
    last_activity: float
    created_at: float
    # Per-session document chunks. Each entry: {id, content, embedding, metadata}.
    # Populated by /chat/upload — scoped to this session only, never written to ChromaDB.
    session_docs: list[dict] = []


class SessionStore:
    async def get(self, session_id: str) -> Optional[SessionData]:
        raise NotImplementedError

    async def set(self, session_id: str, data: SessionData) -> None:
        raise NotImplementedError

    async def delete(self, session_id: str) -> None:
        raise NotImplementedError


class InMemorySessionStore(SessionStore):
    def __init__(self):
        self._sessions: dict[str, SessionData] = {}

    async def get(self, session_id: str) -> Optional[SessionData]:
        data = self._sessions.get(session_id)
        if data:
            if time.time() - data.last_activity > settings.SESSION_TTL:
                await self.delete(session_id)
                return None
        return data

    async def set(self, session_id: str, data: SessionData) -> None:
        data.last_activity = time.time()
        self._sessions[session_id] = data

    async def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


class RedisSessionStore(SessionStore):
    def __init__(self, redis_client):
        self.redis = redis_client

    async def get(self, session_id: str) -> Optional[SessionData]:
        raw = await self.redis.get(f"session:{session_id}")
        if raw is None:
            return None
            
        data = SessionData.model_validate_json(raw)
        
        # Reset TTL
        await self.redis.expire(f"session:{session_id}", settings.SESSION_TTL)
        return data

    async def set(self, session_id: str, data: SessionData) -> None:
        data.last_activity = time.time()
        # Serialize to JSON (never pickle)
        await self.redis.setex(
            f"session:{session_id}", 
            settings.SESSION_TTL, 
            data.model_dump_json()
        )

    async def delete(self, session_id: str) -> None:
        await self.redis.delete(f"session:{session_id}")


# Instance will be injected or defined depending on startup
# For now, fallback to in-memory, to be patched during app startup if redis is available.
session_store: SessionStore = InMemorySessionStore()

def generate_session_id() -> str:
    # Use secure token, never uuid4
    return secrets.token_urlsafe(32)

async def get_session(request: Request, response: Response) -> SessionData:
    session_id = request.cookies.get("session_id")

    if session_id:
        session_data = await session_store.get(session_id)
        if session_data:
            # Valid existing session — do NOT re-set the cookie (no unnecessary Set-Cookie overhead)
            return session_data

    # No cookie or session expired — create a new one
    session_id = generate_session_id()
    current_time = time.time()

    session_data = SessionData(
        session_id=session_id,
        last_activity=current_time,
        created_at=current_time,
    )

    await session_store.set(session_id, session_data)

    # secure=True requires HTTPS; disable for localhost development
    is_secure = not settings.DEBUG

    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        secure=is_secure,
        samesite="strict",
        max_age=settings.SESSION_TTL,
        path="/",
    )

    return session_data

async def clear_session(request: Request) -> bool:
    session_id = request.cookies.get("session_id")
    if session_id:
        session_data = await session_store.get(session_id)
        if session_data:
            session_data.chat_history = []
            session_data.last_activity = time.time()
            await session_store.set(session_id, session_data)
            return True
    return False

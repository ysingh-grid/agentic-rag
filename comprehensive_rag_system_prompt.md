# Production-Grade Agentic RAG System — Comprehensive Design Prompt

> This prompt incorporates the original system specification alongside 23 identified architectural gaps, each patched with industry-standard practices. All patched sections are marked with ⚠️ PATCHED.

---

## 🎯 Objective

Design and implement a production-quality, network-exposed agentic RAG-based system where multiple users access the application over a network, and each user gets an isolated session with independent memory, while maintaining efficiency on a local LLM.

---

## ⚙️ 1. Backend

- Use Python with FastAPI
- Fully async architecture — no blocking calls
  - ⚠️ PATCHED: All synchronous LlamaIndex operations (BM25 retrieval, index queries, document ingestion) MUST be wrapped in `asyncio.to_thread()` to avoid blocking the event loop. Verify the LM Studio client uses `acreate` (async). Use `httpx.AsyncClient` for direct HTTP calls to LM Studio.
- Expose the following endpoints:
  - `POST /chat`
  - `POST /clear-session`
  - `GET /health`
  - `POST /ingest` ⚠️ PATCHED: Admin-only endpoint for document ingestion. Must use separate auth scope from user-facing endpoints.
- ⚠️ PATCHED — Graceful shutdown: Use FastAPI's `lifespan` async context manager. On startup: load models, warm up Redis and vector store connections, rebuild BM25 index if not found on disk. On shutdown: stop accepting new requests, drain active requests within a 30-second window, then close Redis client cleanly.

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await warm_up_connections()
    yield
    # Shutdown
    await asyncio.wait_for(asyncio.gather(*active_tasks), timeout=30)
    await redis_client.aclose()
```

---

## 🔐 2. Authentication & Security

⚠️ PATCHED — This section was entirely absent from the original design. Authentication is mandatory for any network-exposed application.

### A. Authentication Middleware

- Implement JWT-based authentication with short-lived access tokens (15-minute expiry) and refresh tokens.
- For simpler closed-network deployments, use static API key validation via `Authorization: Bearer <key>` header.
- The auth middleware MUST run before session handling. Unauthenticated requests must be rejected with `401` before any session logic executes.

### B. Session ID Security

- ⚠️ PATCHED: Use `secrets.token_urlsafe(32)` for session ID generation. Do NOT use `uuid4` — it is not designed as a security token.
- ⚠️ PATCHED: Optionally HMAC-sign the session ID cookie value with a server-side secret so the backend can detect tampering before hitting Redis.
- ⚠️ PATCHED — Session fixation prevention: Always regenerate and rotate the session ID upon any privilege change or initial session establishment. Immediately invalidate the old ID in the session store.

### C. Cookie Security

Set all of the following attributes on session cookies:

```
Set-Cookie: session_id=<id>; HttpOnly; Secure; SameSite=Strict; Path=/; Max-Age=1800
```

- `HttpOnly` — prevents JavaScript access
- `Secure` — HTTPS only
- `SameSite=Strict` — ⚠️ PATCHED: prevents CSRF. The original spec omitted this, leaving a direct cross-site request forgery vector.
- `Max-Age=1800` — 30-minute TTL aligned with session TTL

### D. Input Validation

⚠️ PATCHED — The original spec listed "input validation" as a requirement without specifying it.

- Query length hard limit: 2000 characters. Reject with `400 Bad Request` beyond this.
- Use Pydantic models for all request bodies. FastAPI auto-rejects malformed requests.
- Prompt injection defense: Wrap all user-provided input in explicit delimiters inside the system prompt:
  ```
  <user_query>{sanitized_query}</user_query>
  ```
  Instruct the model in the system prompt to treat content inside these tags as user input only, never as instructions.
- ⚠️ PATCHED — Rate limiting: Use `slowapi` with a Redis backend. Enforce per-IP limits: 10 requests/minute, 100 requests/hour. Return `429 Too Many Requests` with a `Retry-After` header.

### E. CORS Configuration

⚠️ PATCHED — Entirely absent from the original design. Without this, browsers block cross-origin requests with cookies.

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Exact origins — never "*" with credentials
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)
```

---

## 🤖 3. LLM Setup

- Use a local LLM via LM Studio (OpenAI-compatible API)
- Assume limited throughput: 2–5 concurrent requests maximum
- Design for: latency management, request queuing, graceful degradation under load

### Concurrency Gate

⚠️ PATCHED — The original design specified a semaphore but omitted timeout handling, leaving the queue open-ended under load.

```python
semaphore = asyncio.Semaphore(3)
QUEUE_TIMEOUT = 30  # seconds

async def call_llm_with_backpressure(prompt: str):
    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=QUEUE_TIMEOUT)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=503,
            detail="System at capacity. Please retry in 30 seconds.",
            headers={"Retry-After": "30"}
        )
    try:
        return await llm_call(prompt)
    finally:
        semaphore.release()
```

- Use `asyncio.Queue(maxsize=10)` to buffer requests with per-item timeouts if a larger queue is needed.
- Return `503 Service Unavailable` with `Retry-After` header when capacity is exceeded — never a silent hang.
- The frontend must handle `503` with a visible user-facing message, not a generic error state.

### Streaming

⚠️ PATCHED — The original design returned responses synchronously. A 7B–14B local model takes 10–30 seconds per response. Synchronous responses cause 30-second blank loading states that degrade UX severely.

- Enable `stream=True` on all LM Studio calls.
- Return `StreamingResponse` with `media_type="text/event-stream"` (Server-Sent Events).
- The frontend consumes the stream using `EventSource` or `fetch` with `ReadableStream`, rendering tokens as they arrive.
- This does not reduce total generation time but reduces perceived latency from 30 seconds to ~2 seconds (time-to-first-token).

---

## 🧠 4. RAG + Agent

### Embedding Model

⚠️ PATCHED — The original design assumed LlamaIndex would have an embedding model available without specifying one. LM Studio does NOT expose an embeddings endpoint by default.

- Run a dedicated local embedding model, completely decoupled from the LM Studio LLM.
- Recommended options:
  - `BAAI/bge-large-en-v1.5` — best quality for retrieval tasks
  - `BAAI/bge-small-en-v1.5` — faster, lower memory, slightly lower quality
  - `nomic-embed-text` via Ollama — if Ollama is already in the stack
- Configure LlamaIndex's `ServiceContext` with `HuggingFaceEmbeddings` as the embedding model.

### Vector Store

⚠️ PATCHED — The original design did not specify a vector store backend, which is a fundamental architectural decision.

- Use **ChromaDB** for local deployments: persistent, embedded (no external service), supports metadata filtering.
- Avoid FAISS for multi-user scenarios — it lacks built-in filtering and requires full index reload on updates.
- Alternative: **Qdrant** (Docker-based, better filtering, production-ready API).
- ChromaDB persists to disk automatically and survives application restarts.

### Document Ingestion Pipeline

⚠️ PATCHED — The original design jumped to retrieval without defining how documents enter the system.

- Expose `POST /ingest` (admin-only, separate auth scope).
- Chunking strategy: recursive character text splitter, `chunk_size=512`, `chunk_overlap=50`. The overlap prevents context loss at chunk boundaries.
- Store the following metadata on every chunk: `source`, `chunk_index`, `doc_id`, `ingested_at`.
- Update strategy: documents are identified by `doc_id`. On update, delete all chunks matching that `doc_id` from ChromaDB, then re-ingest. Trigger BM25 index rebuild and invalidate all retrieval cache keys associated with the updated document.
- Ingestion must run as a background task (`asyncio.create_task`) and must not block the query-handling path.

### Agent Factory

⚠️ PATCHED — The original design correctly said "reconstruct agent per request" but did not address the overhead of reconstructing tool objects on every request.

- Cache static tool objects (retriever tool, utility tool) as module-level singletons. These don't change between requests.
- Only the memory and chat history portions are re-instantiated per request, initialized from the session store.
- Use a factory pattern:

```python
# Module-level singletons (initialized once at startup)
_retriever_tool = build_retriever_tool()
_utility_tool = build_utility_tool()

def build_agent(session_state: SessionData) -> ReActAgent:
    memory = ChatMemoryBuffer.from_dict(session_state.memory)
    return ReActAgent.from_tools(
        tools=[_retriever_tool, _utility_tool],
        llm=llm,
        memory=memory,
        verbose=False,
    )
```

### Memory Management

⚠️ PATCHED — The original design said "memory per session" without addressing context window overflow. A 7B–14B local model has a 4K–8K token context window. After 10–15 turns, chat history alone can saturate the context window before retrieved documents are appended.

- Implement a **sliding window**: keep last N=5 turns of chat history. Discard older turns.
- Alternative: **summarization memory** (`SummaryMemory` in LlamaIndex) — periodically compress older turns into a summary block. Adds one LLM call per summarization cycle.
- Enforce a hard token budget on every request:

```
system_prompt + chat_history + retrieved_context + query ≤ context_limit × 0.85
```

If budget is exceeded, trim the oldest turns first until within budget. Log a warning when trimming occurs.

---

## 🔥 5. Advanced Retrieval (MANDATORY)

### A. Hybrid Retrieval

Combine vector search (embeddings) and keyword-based search (BM25) into a unified retrieval step.

**Fusion strategy — use Reciprocal Rank Fusion (RRF):**

⚠️ PATCHED — The original design listed RRF as an option but did not specify it as required. Raw score fusion (the naive alternative) is broken by default because BM25 scores (unbounded positive floats, typically 0–25+) and cosine similarity scores (bounded `[-1, 1]`) are on incompatible scales. Weighted fusion on raw scores causes BM25 to dominate unconditionally.

RRF formula:
```
RRF_score(d) = Σ  1 / (k + rank_i(d))     where k = 60
```

- Each retriever produces a ranked list of documents.
- Every document receives a rank position from each retriever.
- Final scores are harmonic sums of rank positions — not raw similarity scores.
- Documents appearing in both retrieval lists receive a significant score boost.
- Documents appearing in only one list still receive a fair score.
- `k=60` is the empirically validated constant from the original RRF paper.
- No normalization step is required — RRF is scale-invariant by design.

If weighted score fusion is required instead: apply min-max normalization **within each retriever's result set** before weighting. Never normalize globally across documents not in the result set.

**BM25 index persistence:**

⚠️ PATCHED — BM25 is in-memory by default. Application restarts wipe the index entirely.

- On startup, check for a serialized BM25 index on disk (JSON format).
- If absent, build from corpus and serialize before accepting traffic.
- For large corpora where rebuild takes > 10 seconds, serve vector-only retrieval as a degraded fallback until BM25 is ready. Log a warning to observability.
- On document updates, invalidate and schedule a background rebuild.

### B. Reranking

Implement reranking after hybrid retrieval to improve context quality passed to the LLM.

⚠️ PATCHED — The original design said "implement reranking" without specifying the model or inference approach. Without this, the reranker either doesn't exist or is implemented as a slow LLM call that saturates the concurrency gate.

**Recommended:** `BAAI/bge-reranker-base` (HuggingFace cross-encoder)

- Batch all candidates in a single inference call: `model.predict([(query, doc) for doc in candidates])`
- ~100ms for 15 candidates on CPU — acceptable latency for local deployment
- Significantly better relevance ranking than BM25 or cosine similarity alone

**Alternatives:**
- `LLMRerank` from LlamaIndex: uses the existing LLM to score relevance. Slower, consumes a semaphore slot, but requires no additional model.
- Cohere Rerank API: offloads compute but adds network dependency and external cost.

**Retrieval sizing:**
- Initial retrieval size: `top_k = 15` (from hybrid retriever)
- Post-rerank output: `top_n = 4` (passed to context builder)
- Quality gain from reranking plateaus beyond `top_n = 5`

**Why reranking improves RAG:** Embedding similarity and BM25 both capture topical relevance but not query-specific precision. A cross-encoder reads the query and each candidate together — it can distinguish "contains the keywords" from "actually answers the question." This reduces noise in the context window, which directly improves answer faithfulness.

### C. Retrieval Pipeline

```
Query
  → Embedding cache check (Redis)
  → [Cache miss] Embed query (bge-large-en-v1.5)
  → Store embedding in cache
  → Retrieval cache check (Redis, key: SHA256(query + top_k + weights))
  → [Cache miss] Parallel: Vector search (ChromaDB) + BM25 keyword search
  → RRF fusion (k=60) → top_k=15 candidates
  → Store in retrieval cache (TTL=1hr)
  → Cross-encoder reranker (bge-reranker-base, batched)
  → top_n=4 final chunks
  → Context builder (token budget check)
  → LLM (LM Studio, stream=True)
  → SSE stream to frontend
```

Data at each stage:
- **Query → Embedding cache:** raw query string
- **Embedding cache → Vector search:** 1536-dim float vector (or model-specific dimension)
- **Vector search + BM25 → RRF:** two ranked lists of `(doc_chunk, score)` tuples
- **RRF → Reranker:** list of `(query, doc_chunk)` pairs, top_k=15
- **Reranker → Context builder:** list of `(doc_chunk, relevance_score)` pairs, top_n=4
- **Context builder → LLM:** formatted string: system prompt + chat history + retrieved chunks + current query, verified within token budget

### D. Retrieval Configuration

Allow runtime tuning of the following parameters. Store defaults in a config file:

```python
RETRIEVAL_TOP_K: int = 15          # Initial retrieval size (pre-rerank)
RERANK_TOP_N: int = 4              # Post-rerank context size
VECTOR_WEIGHT: float = 0.6         # Used only if switching from RRF to weighted fusion
KEYWORD_WEIGHT: float = 0.4        # Used only if switching from RRF to weighted fusion
RRF_K: int = 60                    # RRF constant
EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
RERANKER_MODEL: str = "BAAI/bge-reranker-base"
```

---

## 🧩 6. Session Management (COOKIE-BASED, MANDATORY)

### A. Cookie Handling

- Use HTTP-only cookies (see Section 2C for full cookie spec including `SameSite=Strict`)
- Automatically generate `session_id` using `secrets.token_urlsafe(32)` if cookie is absent
- Cookie must persist across requests with TTL aligned to session TTL (30 minutes)
- Do NOT accept `session_id` from the request body — cookie only

### B. Session Isolation

Each session must maintain:
- Independent chat history (no cross-user access)
- Independent reconstructed memory (LlamaIndex `ChatMemoryBuffer`)
- No shared mutable state between sessions

### C. Session Store

**In-memory (development):**
```python
sessions: Dict[str, SessionData] = {}
```

**Redis (production):**

⚠️ PATCHED — Serialize session data as JSON only. Never use `pickle` — pickle deserialization is a remote code execution vector if Redis is compromised or data is tampered with.

```python
# Store
await redis.setex(f"session:{session_id}", TTL, session_data.model_dump_json())
# Retrieve
raw = await redis.get(f"session:{session_id}")
session_data = SessionData.model_validate_json(raw) if raw else None
```

Define a typed `SessionData` Pydantic model. Only JSON-serializable types go into session state (strings, lists of dicts, numbers). No Python objects.

```python
class SessionData(BaseModel):
    session_id: str
    chat_history: list[dict]       # [{"role": "user"|"assistant", "content": "..."}]
    metadata: dict                 # arbitrary session-level metadata
    last_activity: float           # Unix timestamp
    created_at: float              # Unix timestamp
```

### D. Session Lifecycle

- TTL: 30 minutes of inactivity
- On every request: reset the TTL (`EXPIRE session:{id} 1800`)
- Automatic cleanup: Redis handles TTL expiry natively. No manual cleanup loop required.
- Memory management: on session expiry, session data is removed from Redis. No other cleanup needed. The agent is reconstructed per-request, so no live agent state persists.

---

## 💾 7. Caching (MANDATORY, MULTI-LAYER)

Three isolated cache namespaces in Redis:

### A. Embedding Cache (Global, Persistent)

```
embedding_cache:{SHA256(text)}  →  serialized float vector
```

- Shared across all users
- No TTL (embeddings don't change unless the embedding model changes)
- Eviction policy: LRU (`maxmemory-policy allkeys-lru` in Redis config)
- Invalidate on embedding model upgrade

### B. Retrieval Cache (Global, TTL-based)

```
retrieval_cache:{SHA256(query + top_k + rrf_k)}  →  list of doc chunk IDs + content
```

- Shared across all users — retrieval results are corpus facts, not personalized
- TTL: 1 hour
- Invalidate on document updates: use Redis `SCAN` with prefix pattern to delete all affected keys
- Why global: two users asking the same question should hit the same retrieved chunks. Scoping this to sessions wastes cache space and misses deduplication.

### C. Response Cache (Session-Scoped, Short TTL)

⚠️ PATCHED — The original design keyed the response cache on `session_id + query` only. This is broken for conversational sessions: the same query asked at different points in the conversation has a different correct answer as chat history changes.

```
response_cache:{SHA256(session_id + query + hash(last_3_turns))}  →  response string
```

- Session-scoped (never shared across users)
- TTL: 30 minutes (aligned with session TTL)
- Use sparingly: most conversational queries will not hit this cache due to evolving history.
- Invalidate on `/clear-session`.

### D. Cache Invalidation Triggers

| Event | Invalidates |
|---|---|
| Document update (re-ingestion) | All `retrieval_cache:*` keys for affected queries; all `embedding_cache:*` for updated doc chunks |
| Session reset (`/clear-session`) | `response_cache:{session_id}*` |
| Embedding model upgrade | All `embedding_cache:*` keys |
| TTL expiry | Handled automatically by Redis |

### E. Redis Configuration

```
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1          # RDB snapshot for durability
appendonly yes      # AOF for crash recovery (optional, adds write overhead)
```

Use Redis connection pooling. Do not open a new connection per request.

---

## ⚡ 8. Concurrency & Load Handling

See Section 3 (LLM Setup) for the semaphore implementation with timeout.

Key principles:
- `asyncio.Semaphore(3)` limits concurrent LLM calls to 3 (tune based on GPU VRAM)
- 30-second queue timeout before returning `503`
- `Retry-After: 30` header on all `503` responses
- Non-LLM operations (cache reads, retrieval, reranking) do not consume a semaphore slot
- All blocking I/O (BM25, disk, synchronous LlamaIndex calls) runs in `asyncio.to_thread()`

---

## 🧪 9. Evaluation

⚠️ PATCHED — The original design ran evaluation against the production LLM instance, competing for the same semaphore slots and skewing latency measurements.

Use RAGAS or LlamaIndex evals in a dedicated offline process:

- Evaluation must NOT run in the production API process
- Use a separate lighter LLM (GPT-3.5 API or equivalent cloud call) for RAGAS scoring — this avoids self-evaluation bias and decouples evaluation from local LLM load
- Fixed query set: 50 queries with ground-truth context and answers

**Evaluation methodology:**
1. Prepare `eval_dataset.json` with `(query, ground_truth_context, ground_truth_answer)` tuples
2. Run offline: `python scripts/evaluate.py --mode [vector_only | hybrid | hybrid_reranked]`
3. Compare three configurations:
   - Vector-only retrieval, no reranking
   - Hybrid retrieval (RRF), no reranking
   - Hybrid retrieval (RRF) + cross-encoder reranking
4. Store timestamped results JSON for trend analysis

**Metrics:**
- Context precision (retrieved chunks relevant to query)
- Context recall (relevant chunks not missed)
- Answer faithfulness (answer grounded in retrieved context — no hallucination)
- Answer relevance (answer addresses the query)

---

## 📊 10. Observability

Integrate Langfuse (self-hosted recommended for local deployments to avoid sending traces to external services).

Track on every request:
- Full request trace with `trace_id` tied to `session_id`
- Step-level spans:
  - `embedding_lookup` — cache hit/miss, latency
  - `hybrid_retrieval` — vector latency, BM25 latency, candidate count
  - `reranking` — input size, output size, latency
  - `context_build` — token count, truncation occurred (true/false)
  - `llm_call` — prompt tokens, completion tokens, latency, stream=true
- Failures and retry counts per step
- Session ID and user ID (if auth implemented) as trace metadata

Expose a `/metrics` endpoint for Prometheus scraping (use `prometheus-fastapi-instrumentator`).

---

## 🔒 11. Security Summary

| Control | Implementation |
|---|---|
| Authentication | JWT middleware, before session handling |
| Rate limiting | `slowapi` + Redis, 10 req/min per IP |
| Input validation | Pydantic + 2000 char limit + prompt delimiters |
| Session ID generation | `secrets.token_urlsafe(32)` |
| Cookie security | `HttpOnly; Secure; SameSite=Strict` |
| Session fixation | Rotate session ID on auth events |
| Redis serialization | JSON only, never pickle |
| CORS | Explicit origins, `allow_credentials=True`, no wildcard |

---

## ❗ 12. Failure Handling (MANDATORY)

### Empty Retrieval

If the retrieval pipeline returns zero chunks:
- Do NOT silently pass an empty context to the LLM
- Log a warning to observability
- Pass a fallback instruction to the LLM: `"No relevant documents were found. Answer based on general knowledge and clearly state you are not drawing from the knowledge base."`
- Prepend the response with a visible disclaimer to the user

### LLM Timeout

- Set an explicit timeout on the LM Studio call (e.g., 45 seconds)
- On timeout: retry once with a backoff of 5 seconds
- On second failure: return a structured error response — `{"error": "LLM_TIMEOUT", "message": "The model is taking too long. Please try again."}`
- Do NOT store a partial or failed response in session memory

### Tool Failure (Agent)

- If a tool call fails, the ReAct agent should catch the exception and continue reasoning with the information it has
- Log the failure to observability with tool name and error type
- The agent's final response should acknowledge it could not access a specific tool if relevant to the answer

### BM25 Index Not Ready

- Serve vector-only retrieval as degraded fallback
- Log a warning on every request served in degraded mode
- Do NOT raise an error to the user — degraded retrieval is better than no retrieval

### Redis Unavailable

- Fall back to in-memory session store for the duration of the outage
- Log a critical error to observability
- Rate limiting and response caching silently degrade (acceptable) — do not block requests

---

## 🌐 13. Frontend Requirements

- Lightweight single-page interface (plain HTML + JS or minimal React)
- Must rely on cookies automatically — no manual session ID management in client code
- Must NOT include session ID in request bodies

### Required behaviors:
- On page load: send a request to `/health`. Display a "system offline" message if health check fails.
- Submit queries via `fetch` with `credentials: "include"` to send cookies automatically.
- Consume SSE stream: render tokens to the chat window as they arrive. Show a blinking cursor during generation.
- Show a loading indicator immediately on submit, before the first token arrives (covers retrieval + rerank latency).
- Handle `503` with a user-visible message: "System is at capacity. Retrying in 30 seconds..." with automatic retry.
- Handle `401` by redirecting to login or prompting for API key.
- "Clear chat" button: calls `POST /clear-session`. On success, clears the message list in the UI. The session cookie persists — only memory is cleared.
- Display timestamps on messages. Indicate when the system is using degraded retrieval (if the API returns a degraded flag in the response).

---

## 🏗️ 14. System Architecture & Data Flow

```
Browser
  │  HTTPS + SameSite=Strict cookie
  ▼
[Auth middleware]  →  401 if unauthenticated
  │
[Rate limiter]  →  429 if exceeded
  │
[Input validator]  →  400 if invalid
  │
[Session manager]
  │  Extract session_id from cookie
  │  Fetch SessionData from Redis (JSON)
  │  Rotate session_id if needed
  ▼
[Concurrency gate]  →  503 if queue timeout
  │  asyncio.Semaphore(3) with 30s timeout
  ▼
[Agent factory]
  │  Static tools loaded from module-level singletons
  │  ChatMemoryBuffer initialized from session history
  │  Context budget calculated: system + history + future_context ≤ 85% ctx_limit
  ▼
[Retrieval pipeline]
  │  1. Check response cache → return immediately if hit
  │  2. Check embedding cache → embed query if miss
  │  3. Check retrieval cache → run hybrid retrieval if miss
  │     a. Vector search (ChromaDB, bge-large-en-v1.5)
  │     b. BM25 keyword search (disk-persisted index)
  │     c. RRF fusion (k=60) → top_k=15
  │     d. Store in retrieval cache (TTL=1hr)
  │  4. Cross-encoder reranker (bge-reranker-base, batched) → top_n=4
  │  5. Context builder: assemble prompt within token budget
  ▼
[LM Studio — local LLM]
  │  stream=True, called via asyncio.to_thread if SDK is sync
  │  Token-by-token SSE stream to frontend
  ▼
[Post-response]
  │  Append turn to session history (enforce sliding window N=5)
  │  Store updated session in Redis (JSON, TTL reset)
  │  Store in response cache (session-scoped, history-hashed key)
  │  Emit trace to Langfuse
  ▼
Browser renders tokens as SSE stream arrives
```

---

## 🌐 15. End-to-End User Flow

### A. First-Time User Flow

1. User opens the application in a browser.
2. Browser sends `GET /` — no session cookie present.
3. Auth middleware checks for JWT/API key. If absent, redirect to login.
4. After auth, the session manager checks for a session cookie. None is found.
5. Backend generates a new `session_id = secrets.token_urlsafe(32)`.
6. An empty `SessionData` record is created and stored in Redis with TTL=1800s.
7. Response includes `Set-Cookie: session_id=<id>; HttpOnly; Secure; SameSite=Strict; Max-Age=1800`.
8. Browser stores the cookie. All subsequent requests from this browser automatically include it.
9. The user is now uniquely identifiable for the duration of the session without any client-side session management code.

### B. Returning User Flow

1. User returns to the application (within 30-minute TTL window).
2. Browser automatically sends the session cookie on every request.
3. Session manager extracts `session_id` from cookie.
4. Fetches `SessionData` from Redis: `GET session:{session_id}`.
5. If found: deserializes JSON → `SessionData`. Chat history and memory are restored.
6. If not found (TTL expired): treat as a new user (see Section E below).
7. TTL is reset on every successful request: `EXPIRE session:{id} 1800`.

Session continuity: chat history is stored as a list of `{role, content}` dicts in Redis. The `ChatMemoryBuffer` is reconstructed from this list on every request. The user's conversation context is preserved as long as the session is active.

### C. Query Processing Flow

1. User types a query and submits.
2. Frontend sends `POST /chat` with `credentials: "include"`. Request body: `{"query": "user's message"}`. Session cookie is sent automatically.
3. Auth middleware validates JWT/API key.
4. Rate limiter checks request count for this IP.
5. Input validator: checks query length ≤ 2000 chars, validates Pydantic schema.
6. Session manager: extracts `session_id`, fetches `SessionData` from Redis.
7. Concurrency gate: acquires semaphore slot (or returns `503` after 30s timeout).
8. Agent factory: builds `ReActAgent` with static tools and memory initialized from session history.
9. Retrieval pipeline executes:
   - Check response cache → if hit, skip to step 14.
   - Check embedding cache → embed query if miss, store embedding.
   - Check retrieval cache → if miss, run vector + BM25 search in parallel, apply RRF fusion, store result.
   - Run cross-encoder reranker on top_k=15 → select top_n=4 chunks.
   - Context builder assembles final prompt within token budget.
10. LM Studio generates response with `stream=True`.
11. Tokens are streamed to the frontend via SSE as they are generated.
12. After full response: append `(user_query, assistant_response)` turn to `chat_history`. Enforce sliding window (N=5 turns).
13. Updated `SessionData` is written back to Redis (JSON, TTL reset).
14. Response is stored in response cache (key includes session + query + history hash).
15. Trace is emitted to Langfuse.
16. Frontend renders the complete message.

### D. Caching Interaction Flow

**Response cache hit:**
- After step 6 (session loaded), check response cache.
- Key: `SHA256(session_id + query + hash(last_3_turns))`
- If hit: skip retrieval, reranking, context building, and LLM call entirely. Return cached response immediately.
- Session history is still updated (the turn is appended) to maintain conversation continuity.

**Retrieval cache hit:**
- During step 9, after embedding is ready, check retrieval cache.
- If hit: skip vector search, BM25 search, and RRF fusion. Proceed directly to reranking.
- Note: reranking still runs on cached retrieval results because it is query-dependent and fast.

**Embedding cache hit:**
- During step 9, before any retrieval, check embedding cache.
- If hit: skip embedding model inference. Use cached vector directly.

**No cache hit:**
- Full pipeline executes: embed → retrieve (vector + BM25) → RRF → rerank → context build → LLM.

### E. Session Expiry Flow

1. User is inactive for 30 minutes. Redis TTL expires and deletes `session:{session_id}`.
2. User submits a new query. Browser sends the old session cookie.
3. Session manager calls `GET session:{old_id}` from Redis. Returns null.
4. Treat as new user: generate new `session_id`, create empty `SessionData`.
5. Set new session cookie in response.
6. The query is processed with no chat history — the agent has no memory of prior conversation.

User experience impact: the user does not see an error. The application responds normally. However, the conversational context is lost. The response may feel disconnected from a prior conversation the user remembers. If the application intends to support long-lived sessions, consider increasing TTL or implementing persistent session storage with explicit user accounts.

### F. Error / Failure Flow

**LLM timeout:**
- User submitted a query. The LM Studio call times out after 45 seconds.
- Retry once after 5 seconds.
- If second attempt also times out: return `{"error": "LLM_TIMEOUT", "message": "The model is taking too long. Please try again."}`.
- Frontend shows: "Response timed out. Please try again." with a retry button.
- The failed turn is NOT appended to session history.

**Empty retrieval:**
- Retrieval returns zero chunks. LLM responds with a general knowledge answer.
- Frontend receives the response with a header or flag indicating degraded mode.
- UI shows a subtle indicator: "Answer based on general knowledge — no documents found."

**System at capacity (503):**
- Concurrency gate timeout exceeded.
- Response: `503 Service Unavailable` with `Retry-After: 30`.
- Frontend shows: "System is at capacity. Retrying in 30 seconds..." and automatically retries after the specified delay.

**Rate limit exceeded (429):**
- `slowapi` returns `429 Too Many Requests`.
- Frontend shows: "Too many requests. Please wait a moment."

**Invalid input (400):**
- Query exceeds 2000 characters or fails Pydantic validation.
- Frontend shows: "Your message is too long. Please shorten it." (if length) or a generic validation error.

### G. Session Reset Flow

1. User clicks "Clear chat" in the frontend.
2. Frontend sends `POST /clear-session` with session cookie.
3. Backend: fetches session from Redis. Clears `chat_history` list. Resets `last_activity`. Writes updated session back to Redis.
4. Invalidates all `response_cache:*` keys for this session in Redis.
5. Returns `200 OK`.
6. Frontend clears the message list. The input field is ready for a new conversation.
7. The session cookie is NOT changed — the session persists, only the memory is cleared.

### H. UX Considerations

**Response latency expectations:**
- Time to first token: 2–5 seconds (retrieval + rerank + LLM prefill)
- Time to complete response: 10–30 seconds for a typical answer (dependent on response length and model speed)
- With SSE streaming, users see tokens arriving within 2–5 seconds, which is acceptable. Without streaming, they wait 10–30 seconds for a blank screen, which is not.

**Loading indicators:**
- Show a spinner immediately on query submit (covers retrieval + rerank latency before first token).
- Replace spinner with a blinking cursor when the first token arrives.
- Remove cursor and finalize message layout when the stream closes.

**Slow LLM handling:**
- Do not show "this is taking longer than expected" messages for the first 15 seconds — this is normal for a 7B–14B local model.
- After 20 seconds without a token, show: "Still generating..." to reassure the user.
- After 45 seconds, trigger the timeout flow.

**Conversational continuity:**
- Chat history is restored from Redis on every request. The user can close and reopen the browser within the 30-minute TTL window and resume the conversation without any action on their part.
- After session expiry, the conversation history is gone. Consider notifying the user: "Your previous session has expired. Starting a new conversation."

---

## 🏗️ 16. Performance Constraints & Optimization

Assume: local LLM (7B–14B), 2–5 concurrent users, limited GPU VRAM.

**Irreducible bottleneck:** The local LLM. Even with Semaphore(3), if each request takes 20 seconds at 7B, the P95 latency for the 3rd concurrent user is 60 seconds. Streaming (Section 3) is the single highest-impact optimization — it does not reduce total generation time but transforms UX from "30-second blank wait" to "tokens appearing in 2 seconds."

**Optimization priority order:**
1. SSE streaming — highest UX impact, zero cost
2. Retrieval and embedding caching — eliminates redundant model inference for repeated queries
3. Cross-encoder reranker batching — single forward pass for all candidates vs. N sequential calls
4. BM25 disk persistence — eliminates cold-start rebuild time
5. Module-level tool singletons — reduces per-request agent construction overhead
6. Sliding window memory — prevents context overflow, keeps prompts short

**Known bottlenecks:**
- First request after BM25 rebuild: higher latency
- First request for a new query (no cache): full retrieval + rerank pipeline (~500ms–1s before LLM)
- Summarization memory cycle: adds one LLM call when triggered (use sliding window unless quality degradation is observed)

---

## 📦 17. Output Format for Implementation

Deliver the implementation in the following structure:

```
/app
  main.py              # FastAPI app, lifespan, middleware registration
  config.py            # All tunable parameters (top_k, top_n, TTL, model names)
  auth.py              # JWT / API key middleware
  session.py           # SessionData model, Redis store, cookie handling
  agent.py             # AgentFactory, static tool singletons
  retrieval.py         # Hybrid retriever, RRF fusion, reranker
  context.py           # Context builder with token budget enforcement
  cache.py             # Three-namespace Redis cache layer
  ingestion.py         # Document chunking, ingestion pipeline, /ingest endpoint
  health.py            # Deep health check endpoint
  observability.py     # Langfuse trace integration
/scripts
  evaluate.py          # Offline RAGAS evaluation
/frontend
  index.html           # SSE-consuming frontend
```

Implementation priorities in order:
1. Reliability — session isolation, error handling, graceful degradation
2. Correctness — RRF fusion (not concatenation), JSON-only Redis, history-hashed cache keys
3. Performance — streaming, caching, async wrappers
4. Observability — trace every step before going to production
5. Evaluation — run offline evals after the system is stable

Build this as a production system, not a demo. Every design decision above has a specific reason. Do not substitute simpler alternatives without understanding the tradeoff being accepted.

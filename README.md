# Agentic Scalable RAG System

A production-grade, network-exposed Agentic RAG system built with FastAPI, Redis, ChromaDB, and local LLMs (via LM Studio).

## Prerequisites

1.  **Python Packages:** You've successfully synced using `uv sync`.
2.  **LM Studio:**    - Set CORS domains explicitly in LM Studio to allow `http://localhost:3000` 
    - Go to LM Studio, load your model (`llama-3.2-3b-instruct`).
    - Start the server (default runs on `http://localhost:1234`)./v1` as the base URL).
3.  **Docker:** Required for running the Redis cache layer.

## How to Run

### Step 1: Start Redis
The system uses Redis for 3-tier caching (embeddings, retrieval, responses) and session management. Start the pre-configured Redis container:
```bash
docker-compose up -d
```

### Step 2: Start the Backend server
Run the FastAPI app using Uvicorn. The dependencies are downloaded, so you can start it right away:
```bash
# Activate your uv environment first if not already done:
source .venv/bin/activate

# Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
*Note: The first time you run this, it will download the BGE embedding and reranking models from HuggingFace, which may take a minute.*

### Step 3: Access the UI
Open your browser and navigate to the frontend:
[http://localhost:8000/ui](http://localhost:8000/ui)

## How to Use the System

### 1. Ingesting Documents
To make the RAG system useful, you need to feed it data. You can do this via the `/ingest` API endpoint using `curl` (or Postman).

Open a new terminal window and run:
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Authorization: Bearer dev_api_key_123" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/document.txt"
```
*(Replace `/path/to/your/document.txt` with an actual text file on your machine).*

The system will chunk the text, compute embeddings, store them in ChromaDB, rebuild the BM25 index, and invalidate the retrieval cache in the background.

### 2. Chatting
Go to the UI at `http://localhost:8000/ui` and start asking questions about the document you just ingested. The system will use hybrid search (BM25 + ChromaDB + Reciprocal Rank Fusion) and cross-encoder reranking to fetch the most relevant context and stream the answer token-by-token from LM Studio.

## Architecture Highlights
- **Rate Limiting:** Managed by `slowapi` to prevent abuse.
- **Session Auth:** Uses HTTP-Only/Secure cookies + Redis TTL.
- **Caching:** Heavy computations (embeddings, RRF) are cached in Redis.
- **Observability:** Prepared for Langfuse tracing inside `rag-query` spans.
- **Hybrid Search:** Combines keyword (BM25) and semantic (ChromaDB) with an explicit HuggingFace reranker.

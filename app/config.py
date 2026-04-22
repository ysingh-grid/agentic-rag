import os
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "Agentic Scalable RAG"
    DEBUG: bool = True
    
    # Retrieval configuration
    RETRIEVAL_TOP_K: int = 15          # Initial retrieval size (pre-rerank)
    RERANK_TOP_N: int = 4              # Post-rerank context size
    RRF_K: int = 60                    # RRF constant
    
    # Models
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    
    # LLM
    LLM_BASE_URL: str = "http://localhost:1234/v1"
    LLM_MODEL_NAME: str = "llama-3.2-3b-instruct"
    
    # Concurrency & load handling
    LLM_CONCURRENCY: int = 3
    QUEUE_TIMEOUT: int = 30
    
    # Data Stores
    REDIS_URL: str = "redis://localhost:6379/0"
    CHROMA_DB_DIR: str = "./db/chroma"
    BM25_INDEX_PATH: str = "./db/bm25_index.json"
    
    # Session Management
    SESSION_TTL: int = 1800  # 30 minutes
    
    # Validation
    MAX_QUERY_LENGTH: int = 2000
    
    # Auth
    AUTH_MODE: Literal["apikey", "jwt", "none"] = "apikey"
    API_KEY: str = "dev_api_key_123"
    JWT_SECRET: str = "your-jwt-secret-here"
    COOKIE_SECRET: str = "your-cookie-secret-here"
    
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore"
    )

settings = Settings()

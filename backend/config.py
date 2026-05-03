from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Local embedding model — no API key required
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Cross-encoder for Stage-2 re-ranking (fastembed ONNX model name)
    RERANKER_MODEL: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # ChromaDB persistent storage
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "rag_documents"
    KNOWLEDGE_COLLECTION_NAME: str = "nexus_knowledge"

    # Semantic chunking parameters
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    # Hybrid retrieval config
    TOP_K_RETRIEVAL: int = 40   # Stage 1: hybrid (vector+BM25) candidate pool
    TOP_K_RERANK: int = 5       # Stage 2: cross-encoder top-K sent to LLM
    BM25_WEIGHT: float = 0.5    # RRF weight for BM25 leg (vector gets 1-weight)

    # Knowledge store — recalled chunks from past interactions
    TOP_K_KNOWLEDGE: int = 5    # Past-interaction chunks merged into candidate pool

    # LLM (OpenRouter — OpenAI-compatible, routes to Nemotron)
    NVIDIA_API_KEY: str = ""
    NVIDIA_BASE_URL: str = "https://openrouter.ai/api/v1"
    LLM_MODEL: str = "nvidia/nemotron-3-super-120b-a12b:free"
    LLM_MAX_TOKENS: int = 4096

    # Upload limits
    UPLOAD_DIR: str = "./uploads"
    MAX_FILE_SIZE_MB: int = 50

    class Config:
        env_file = ".env"


settings = Settings()

# Ensure required directories exist on import
Path(settings.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

"""
FastAPI entry point.

Endpoints
---------
GET  /health   — liveness check + collection stats
GET  /documents — list ingested files
POST /upload   — ingest a PDF (multipart/form-data)
POST /query    — two-stage RAG query, SSE streaming response
"""

import json
import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import settings
from ingestion import DocumentIngestionPipeline
from llm import LLMGenerator
from reranker import CrossEncoderReranker
from retrieval import HybridRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-20s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global model singletons — initialised in lifespan, not at import time
# ---------------------------------------------------------------------------

_pipeline: Optional[DocumentIngestionPipeline] = None
_retriever: Optional[HybridRetriever] = None
_reranker: Optional[CrossEncoderReranker] = None
_generator: Optional[LLMGenerator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models once at startup; release on shutdown."""
    global _pipeline, _retriever, _reranker, _generator

    logger.info("=== Loading ingestion pipeline ===")
    _pipeline = DocumentIngestionPipeline()

    logger.info("=== Loading hybrid retriever (vector + BM25) ===")
    _retriever = HybridRetriever()

    logger.info("=== Loading cross-encoder reranker ===")
    try:
        _reranker = CrossEncoderReranker()
    except Exception as exc:
        logger.warning(
            "Cross-encoder failed to load (%s). "
            "Stage-2 re-ranking disabled — will use top-%d from vector search.",
            exc,
            settings.TOP_K_RERANK,
        )
        _reranker = None

    logger.info("=== Loading LLM generator ===")
    _generator = LLMGenerator()

    logger.info("=== Startup complete ===")
    yield

    logger.info("Shutting down")


app = FastAPI(
    title="RAG System API",
    description="Two-stage retrieval (bi-encoder → cross-encoder) + Nemotron generation",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------

def get_pipeline() -> DocumentIngestionPipeline:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Ingestion pipeline not ready")
    return _pipeline

def get_retriever() -> HybridRetriever:
    if _retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not ready")
    return _retriever

def get_generator() -> LLMGenerator:
    if _generator is None:
        raise HTTPException(status_code=503, detail="LLM generator not ready")
    return _generator


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    top_k_retrieval: int = settings.TOP_K_RETRIEVAL
    top_k_rerank: int = settings.TOP_K_RERANK


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    pipeline = get_pipeline()
    stats = pipeline.get_stats()
    return {
        "status": "healthy",
        "reranker_available": _reranker is not None,
        **stats,
    }


@app.get("/documents")
async def list_documents():
    pipeline = get_pipeline()
    files = sorted(Path(settings.UPLOAD_DIR).glob("*.pdf"))
    return {
        "documents": [f.name for f in files],
        "count": len(files),
        **pipeline.get_stats(),
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Receive a PDF and run the full ingestion pipeline:
    Load → Chunk → Embed (all-MiniLM-L6-v2 via ONNX) → Store (ChromaDB)
    """
    pipeline = get_pipeline()

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    dest = Path(settings.UPLOAD_DIR) / file.filename

    try:
        content = await file.read()
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds {settings.MAX_FILE_SIZE_MB} MB limit",
            )

        dest.write_bytes(content)
        logger.info("Saved upload: %s (%d bytes)", file.filename, len(content))

        result = pipeline.ingest(str(dest))

        # Invalidate the BM25 index so the new chunks are included on next query
        retriever = get_retriever()
        retriever.invalidate_bm25()

        return {"message": f"Ingested '{file.filename}' successfully", **result}

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Ingestion failed for %s", file.filename)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/query")
async def query(request: QueryRequest):
    """
    Two-stage RAG query with SSE streaming.

    Stage 1  Vector search   → top-{top_k_retrieval} candidates
    Stage 2  Cross-encoder   → top-{top_k_rerank} refined chunks (skipped if reranker unavailable)
    Stage 3  LLM generation  → grounded answer streamed via SSE
    """
    retriever = get_retriever()
    generator = get_generator()

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Stage 1 — dense retrieval
    candidates = retriever.retrieve(request.query, top_k=request.top_k_retrieval)
    if not candidates:
        raise HTTPException(
            status_code=404,
            detail="No documents indexed. Upload PDFs first.",
        )

    # Stage 2 — cross-encoder re-ranking (graceful fallback to top-K from Stage 1)
    if _reranker is not None:
        top_chunks = _reranker.rerank(
            request.query, candidates, top_k=request.top_k_rerank
        )
    else:
        top_chunks = candidates[: request.top_k_rerank]
        for c in top_chunks:
            c.setdefault("rerank_score", c.get("similarity_score", 0.0))

    sources = [
        {
            "text": c["text"][:350] + ("…" if len(c["text"]) > 350 else ""),
            "source": c["metadata"].get("source", "unknown"),
            "page": c["metadata"].get("page", 0) + 1,
            "similarity_score": round(c.get("similarity_score", 0.0), 4),
            "bm25_score": round(c.get("bm25_score", 0.0), 4),
            "rrf_score": round(c.get("rrf_score", 0.0), 6),
            "rerank_score": round(c.get("rerank_score", 0.0), 4),
        }
        for c in top_chunks
    ]

    async def event_stream():
        yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"

        async for token in generator.stream(request.query, top_chunks):
            yield f"data: {json.dumps({'type': 'token', 'data': token})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

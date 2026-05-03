"""
FastAPI entry point.

Endpoints
---------
GET  /health   — liveness check + collection stats
GET  /documents — list ingested files
POST /upload   — ingest a PDF (multipart/form-data)
POST /query    — two-stage RAG query, SSE streaming response
"""

import asyncio
import json
import logging
import os
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
from knowledge import KnowledgeStore
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
_knowledge: Optional[KnowledgeStore] = None
_ingest_lock = asyncio.Lock()  # prevents concurrent OCR from corrupting Tesseract state


def _wipe_all() -> None:
    """Delete all indexed documents, knowledge entries, and uploaded files."""
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    client = chromadb.PersistentClient(
        path=settings.CHROMA_PERSIST_DIR,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    for name in [settings.CHROMA_COLLECTION_NAME, settings.KNOWLEDGE_COLLECTION_NAME]:
        try:
            client.delete_collection(name)
            logger.info("Wiped collection: %s", name)
        except Exception:
            pass

    upload_dir = Path(settings.UPLOAD_DIR)
    for f in upload_dir.glob("*.pdf"):
        f.unlink()
    logger.info("Cleared uploads directory")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models once at startup; wipe state for a fresh session."""
    global _pipeline, _retriever, _reranker, _generator, _knowledge

    logger.info("=== Wiping previous session data ===")
    _wipe_all()

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

    logger.info("=== Loading adaptive knowledge store ===")
    _knowledge = KnowledgeStore()

    logger.info("=== Startup complete ===")
    yield

    logger.info("Shutting down")


app = FastAPI(
    title="RAG System API",
    description="Two-stage retrieval (bi-encoder → cross-encoder) + Nemotron generation",
    version="1.0.0",
    lifespan=lifespan,
)

_cors_origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
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

SUMMARY_MAX_CHUNKS_PER_PART = 30  # ~4K tokens of context per part — keeps LLM fast


class QueryRequest(BaseModel):
    query: str
    top_k_retrieval: int = settings.TOP_K_RETRIEVAL
    top_k_rerank: int = settings.TOP_K_RERANK


class SummarizeRequest(BaseModel):
    sources: list[str]  # filenames to summarise; empty list = all documents
    part: int = 1       # which part to generate (1-indexed)


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
    result = pipeline.collection.get(include=["metadatas"])

    source_chunks: dict[str, int] = {}
    for meta in result["metadatas"]:
        src = meta.get("source", "unknown")
        source_chunks[src] = source_chunks.get(src, 0) + 1

    doc_details = [
        {"name": name, "chunks": chunks}
        for name, chunks in sorted(source_chunks.items())
    ]
    return {
        "documents": list(source_chunks.keys()),
        "document_details": doc_details,
        "count": len(doc_details),
        "total_chunks": pipeline.collection.count(),
        "collection_name": settings.CHROMA_COLLECTION_NAME,
    }


@app.post("/reset")
async def reset_all():
    """Wipe all indexed documents, knowledge store, and uploads — fresh start."""
    pipeline = get_pipeline()
    retriever = get_retriever()

    # Clear main collection
    all_ids = pipeline.collection.get()["ids"]
    if all_ids:
        pipeline.collection.delete(ids=all_ids)

    # Clear knowledge store
    if _knowledge is not None:
        all_k = _knowledge.collection.get()["ids"]
        if all_k:
            _knowledge.collection.delete(ids=all_k)

    # Clear uploads
    for f in Path(settings.UPLOAD_DIR).glob("*.pdf"):
        f.unlink()

    retriever.invalidate_bm25()
    logger.info("Reset: all data wiped")
    return {"message": "Reset complete"}


@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Remove a PDF and all its indexed chunks from the system."""
    pipeline = get_pipeline()
    retriever = get_retriever()

    existing = pipeline.collection.get(where={"source": {"$eq": filename}})
    if not existing["ids"]:
        raise HTTPException(status_code=404, detail=f"'{filename}' not found in index")

    chunks_removed = len(existing["ids"])
    pipeline.collection.delete(ids=existing["ids"])

    if _knowledge is not None:
        _knowledge.delete_source(filename)

    file_path = Path(settings.UPLOAD_DIR) / filename
    if file_path.exists():
        file_path.unlink()

    retriever.invalidate_bm25()

    logger.info("Deleted '%s' — %d chunks removed", filename, chunks_removed)
    return {"message": f"Deleted '{filename}'", "chunks_removed": chunks_removed}


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

        async with _ingest_lock:
            result = await asyncio.to_thread(pipeline.ingest, str(dest))

        # Invalidate the BM25 index so the new chunks are included on next query
        retriever = get_retriever()
        retriever.invalidate_bm25()

        return {"message": f"Ingested '{file.filename}' successfully", **result}

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Ingestion failed for %s", file.filename)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/summarize")
async def summarize_documents(request: SummarizeRequest):
    """
    Fetch ALL chunks for the requested source(s) ordered by page and stream
    a plain-language summary.  If the document is large, splits into parts
    and tells the client how many parts exist so it can request the next one.
    """
    pipeline = get_pipeline()
    generator = get_generator()

    # Resolve which sources to summarise
    if not request.sources:
        all_meta = await asyncio.to_thread(
            pipeline.collection.get, include=["metadatas"]
        )
        sources = sorted({m["source"] for m in all_meta["metadatas"]})
    else:
        sources = request.sources

    if not sources:
        raise HTTPException(status_code=404, detail="No documents uploaded yet.")

    # Fetch all chunks for each source — run in thread to avoid blocking event loop
    def _fetch_all_chunks() -> list[dict]:
        chunks: list[dict] = []
        for src in sources:
            result = pipeline.collection.get(
                where={"source": {"$eq": src}},
                include=["documents", "metadatas"],
            )
            for doc, meta in zip(result["documents"], result["metadatas"]):
                chunks.append({"text": doc, "metadata": meta})
        return chunks

    all_chunks = await asyncio.to_thread(_fetch_all_chunks)

    if not all_chunks:
        raise HTTPException(status_code=404, detail="No indexed chunks found for the selected documents.")

    # Sort by (source, page) so the LLM reads in reading order
    all_chunks.sort(key=lambda c: (c["metadata"].get("source", ""), c["metadata"].get("page", 0)))

    total_chunks = len(all_chunks)
    total_parts = max(1, (total_chunks + SUMMARY_MAX_CHUNKS_PER_PART - 1) // SUMMARY_MAX_CHUNKS_PER_PART)
    part = max(1, min(request.part, total_parts))

    start = (part - 1) * SUMMARY_MAX_CHUNKS_PER_PART
    end = start + SUMMARY_MAX_CHUNKS_PER_PART
    chunks_for_part = all_chunks[start:end]

    source_label = (
        sources[0] if len(sources) == 1
        else ", ".join(sources[:2]) + (f" +{len(sources)-2} more" if len(sources) > 2 else "")
    )

    async def event_stream():
        yield f"data: {json.dumps({'type': 'meta', 'data': {'part': part, 'total_parts': total_parts, 'total_chunks': total_chunks, 'sources': sources, 'source_label': source_label}})}\n\n"

        try:
            async for token in generator.stream_summary(chunks_for_part, part, total_parts, source_label):
                yield f"data: {json.dumps({'type': 'token', 'data': token})}\n\n"
        except Exception as exc:
            logger.exception("Summary LLM error for '%s' part %d", source_label, part)
            yield f"data: {json.dumps({'type': 'error', 'data': str(exc)})}\n\n"
            return

        yield f"data: {json.dumps({'type': 'done', 'data': {'total_parts': total_parts}})}\n\n"

    logger.info(
        "Summarize '%s' — %d total chunks, part %d/%d (%d chunks)",
        source_label, total_chunks, part, total_parts, len(chunks_for_part),
    )
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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

    # Stage 1 — hybrid retrieval
    candidates = retriever.retrieve(request.query, top_k=request.top_k_retrieval)
    if not candidates:
        raise HTTPException(
            status_code=404,
            detail="No documents indexed. Upload PDFs first.",
        )

    # Stage 1b — inject recalled chunks from past interactions (self-improving)
    if _knowledge is not None:
        recalled = _knowledge.recall(request.query)
        if recalled:
            seen = {
                f"{c['metadata']['source']}::{c['text'][:80]}"
                for c in candidates
            }
            for rc in recalled:
                key = f"{rc['metadata']['source']}::{rc['text'][:80]}"
                if key not in seen:
                    candidates.append(rc)
                    seen.add(key)

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

    # Collect reference-list chunks for every source file in the top results
    source_files = {c["metadata"]["source"] for c in top_chunks}
    ref_chunks: list[dict] = []
    for src_file in source_files:
        ref_chunks.extend(retriever.get_reference_chunks(src_file))

    async def event_stream():
        yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"

        async for token in generator.stream(request.query, top_chunks, ref_chunks):
            yield f"data: {json.dumps({'type': 'token', 'data': token})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

        # Store this interaction for future retrieval improvement (background)
        if _knowledge is not None:
            asyncio.create_task(
                asyncio.to_thread(_knowledge.store, request.query, top_chunks)
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

# Nexus — Technical Intelligence Engine

A production-grade Retrieval-Augmented Generation (RAG) system built for deep technical document analysis. Nexus combines hybrid search, cross-encoder re-ranking, and a structured LLM extraction engine to transform research papers and technical documents into machine-readable intelligence — complete tables, taxonomies, and resolved reference lists.

---

## Architecture

```
PDF Upload
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Ingestion Pipeline (backend/ingestion.py)           │
│                                                     │
│  PyMuPDF direct extraction                         │
│      │ garble rate > 15%?                          │
│      ▼                                             │
│  Tesseract OCR @ 300 DPI  ──→  keep best result   │
│      │ both > 50% garbled?                         │
│      ▼                                             │
│  Skip page                                         │
│                                                     │
│  RecursiveCharacterTextSplitter (512 / 64 overlap) │
│  fastembed all-MiniLM-L6-v2 (ONNX, no torch)      │
│  ChromaDB upsert (cosine HNSW)                     │
│  Reference-list chunks tagged  is_references=1     │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Stage 1 — Hybrid Retrieval (backend/retrieval.py)  │
│                                                     │
│  Dense vector search  ─┐                           │
│  BM25Okapi keyword     ─┴─→  RRF fusion (k=60)    │
│                                                     │
│  Returns top-25 candidates                         │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Stage 2 — Cross-Encoder Re-ranking (reranker.py)  │
│                                                     │
│  Xenova/ms-marco-MiniLM-L-6-v2 (ONNX via          │
│  fastembed)  →  top-3 chunks                       │
└─────────────────────────────────────────────────────┘
    │        │
    │        └─ Reference chunks fetched for each source
    ▼
┌─────────────────────────────────────────────────────┐
│  Stage 3 — LLM Generation (backend/llm.py)         │
│                                                     │
│  NVIDIA Nemotron 3 Super 120B via OpenRouter        │
│  System: Technical Intelligence Engine persona     │
│  Context: top-3 chunks + reference list block      │
│  Output: structured Markdown with tables,          │
│  tiered lists, resolved inline citations           │
└─────────────────────────────────────────────────────┘
    │
    ▼
Server-Sent Events → Next.js 15 → React streaming UI
```

---

## Features

- **Hybrid Search** — BM25 keyword matching + dense vector search fused via Reciprocal Rank Fusion. Catches both semantic similarity and exact term matches.
- **Smart OCR fallback** — Per-page garble detection selects best extraction (direct vs Tesseract OCR) automatically. Skips pages only when both methods fail.
- **Cross-encoder re-ranking** — Two-stage retrieval: 25 candidates → cross-encoder → top 3 sent to LLM.
- **Reference resolution** — Paper inline citations like `[1]`, `[4]` are automatically resolved to full author/title from the paper's reference section.
- **Structured extraction** — LLM forced to render all taxonomies, classification schemes, and comparisons as Markdown tables and numbered lists — never prose.
- **SSE streaming** — Tokens stream from FastAPI → Next.js Route Handler → React in real time.
- **No PyTorch** — Entire ML stack runs on ONNX Runtime via fastembed, sharing the same runtime that ChromaDB already pulls in.

---

## Stack

| Layer | Technology |
|-------|-----------|
| LLM | NVIDIA Nemotron 3 Super 120B via OpenRouter |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (fastembed ONNX) |
| Re-ranker | `Xenova/ms-marco-MiniLM-L-6-v2` (fastembed ONNX) |
| Vector DB | ChromaDB (persistent, cosine HNSW) |
| Keyword search | BM25Okapi (rank-bm25) |
| PDF extraction | PyMuPDF + Tesseract OCR fallback |
| Backend | FastAPI + uvicorn (async SSE) |
| Frontend | Next.js 15 (App Router, TypeScript) |
| Styling | Tailwind CSS |

---

## Project Structure

```
rag-system/
├── backend/
│   ├── main.py          # FastAPI app, routes, lifespan model loading
│   ├── ingestion.py     # PDF load → OCR → chunk → embed → store
│   ├── retrieval.py     # Hybrid BM25 + vector search with RRF
│   ├── reranker.py      # Cross-encoder Stage-2 re-ranking
│   ├── llm.py           # LLM client, system prompt, SSE generation
│   ├── config.py        # Pydantic settings (reads .env)
│   ├── requirements.txt
│   └── .env             # API keys (gitignored)
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── api/[...slug]/route.ts  # Streaming proxy to backend
│   │   │   ├── page.tsx
│   │   │   └── layout.tsx
│   │   ├── components/
│   │   │   ├── ChatInterface.tsx       # SSE consumer, message renderer
│   │   │   ├── SourceCitation.tsx      # Score bars, retrieval badges
│   │   │   └── UploadZone.tsx          # Drag-and-drop PDF upload
│   │   └── lib/api.ts                  # Typed fetch wrappers
│   └── package.json
├── start_backend.sh     # Auto-creates venv, installs deps, starts uvicorn
└── streamlit_app.py     # Alternative Streamlit UI
```

---

## Setup

### Prerequisites

- Python 3.12 (not 3.13+, ML packages require ≤3.12)
- Node.js 18+
- Tesseract OCR: `brew install tesseract`
- An OpenRouter API key — get one at [openrouter.ai/keys](https://openrouter.ai/keys)

### Backend

```bash
cd backend

# Create venv with Python 3.12
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and set: NVIDIA_API_KEY=sk-or-v1-...

# Start server
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Or use the helper script from the repo root:

```bash
./start_backend.sh
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

---

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness + collection stats |
| `GET` | `/documents` | List ingested files |
| `POST` | `/upload` | Ingest a PDF (`multipart/form-data`, field `file`) |
| `POST` | `/query` | RAG query — returns SSE stream |

### Query SSE Protocol

```
POST /query
Content-Type: application/json

{ "query": "What classification scheme is used?" }
```

Response is a stream of `text/event-stream` events:

```
data: {"type": "sources", "data": [...]}   ← citation data, first event

data: {"type": "token",   "data": "###"}   ← LLM tokens, many events

data: {"type": "done"}                      ← stream end
```

Each source object:

```json
{
  "text": "...",
  "source": "paper.pdf",
  "page": 3,
  "similarity_score": 0.412,
  "bm25_score": 7.21,
  "rrf_score": 0.0318,
  "rerank_score": 2.14
}
```

---

## Configuration

All settings are in `backend/config.py` and can be overridden via environment variables or `backend/.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `NVIDIA_API_KEY` | _(required)_ | OpenRouter API key |
| `NVIDIA_BASE_URL` | `https://openrouter.ai/api/v1` | LLM endpoint |
| `LLM_MODEL` | `nvidia/nemotron-3-super-120b-a12b:free` | Model ID |
| `LLM_MAX_TOKENS` | `2048` | Max response tokens |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `RERANKER_MODEL` | `Xenova/ms-marco-MiniLM-L-6-v2` | Cross-encoder model |
| `TOP_K_RETRIEVAL` | `25` | Hybrid search candidate pool |
| `TOP_K_RERANK` | `3` | Chunks forwarded to LLM |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |

---

## LLM Output Format

Every response follows this structure:

```markdown
### Summary
Direct answer with [Source N] citations.

### Detailed Findings
Tables, numbered lists, and definitions extracted from sources.
All taxonomies and classification schemes rendered as structured data.

### Key Technical Terms
Glossary with exact definitions from the document.

### Reference Resolution
[1] Author(s), *Title*, Venue/Year.
[4] Author(s), *Title*, Venue/Year.

### Source Coverage
Which chunks contributed to which part of the answer.
```

---

## How Reference Resolution Works

During ingestion, chunks are tagged `is_references=1` when they contain 3+ lines matching the `[N]` citation pattern. At query time, reference chunks for each source file are fetched separately and passed to the LLM as a `<reference_list>` block alongside the retrieved context. The LLM is instructed to resolve every inline citation found in the evidence against this list.

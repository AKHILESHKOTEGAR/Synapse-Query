"""
Streamlit fallback UI for the RAG system.

Demonstrates the full deep-learning pipeline (ingestion, two-stage
retrieval, streaming generation) without requiring the Next.js frontend.

Run from the project root:
    cd rag-system
    streamlit run streamlit_app.py
"""

import sys
import tempfile
from pathlib import Path

# Make backend modules importable when run from project root
sys.path.insert(0, str(Path(__file__).parent / "backend"))

import streamlit as st

st.set_page_config(
    page_title="RAG System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Cached singletons — models load once per session
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading embedding model…")
def get_pipeline():
    from ingestion import DocumentIngestionPipeline
    return DocumentIngestionPipeline()


@st.cache_resource(show_spinner="Loading retriever…")
def get_retriever():
    from retrieval import VectorRetriever
    return VectorRetriever()


@st.cache_resource(show_spinner="Loading cross-encoder…")
def get_reranker():
    from reranker import CrossEncoderReranker
    return CrossEncoderReranker()


@st.cache_resource(show_spinner=False)
def get_generator():
    from llm import LLMGenerator
    return LLMGenerator()


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.title("🧠 RAG System — Deep Learning Demo")
st.caption(
    "Stage 1: Dense vector search (all-MiniLM-L6-v2)  →  "
    "Stage 2: Cross-encoder re-rank (ms-marco-MiniLM-L-6-v2)  →  "
    "Stage 3: Grounded Claude generation"
)
st.divider()

sidebar, main_col = st.columns([1, 2], gap="large")

# ---------------------------------------------------------------------------
# Sidebar — ingestion + stats + architecture notes
# ---------------------------------------------------------------------------

with sidebar:
    st.subheader("📄 Document Ingestion")

    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded:
        if st.button("Ingest Document", type="primary", use_container_width=True):
            pipeline = get_pipeline()
            with st.spinner(f"Running ingestion pipeline for {uploaded.name}…"):
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = tmp.name
                result = pipeline.ingest(tmp_path)
            st.success(
                f"✓ {result['pages_processed']} pages → "
                f"{result['chunks_stored']} chunks stored"
            )
            with st.expander("Ingestion details"):
                st.json(result)

    st.divider()

    pipeline = get_pipeline()
    stats = pipeline.get_stats()
    st.metric("Chunks Indexed", stats["total_chunks"])

    st.divider()
    st.subheader("⚙️ Architecture")
    st.markdown(
        """
**Stage 1 — Bi-encoder (fast)**
- `all-MiniLM-L6-v2` — 384-dim sentence embeddings
- ANN search via ChromaDB HNSW index
- Retrieves top-10 candidates in O(log N)

**Stage 2 — Cross-encoder (accurate)**
- `ms-marco-MiniLM-L-6-v2` — joint (query, doc) encoding
- Full attention across both inputs → better relevance
- Scores top-10 → selects top-3 for LLM

**Stage 3 — Grounded generation**
- Only 3 chunks sent to Claude
- Reduces hallucination; forces citation
- Response streamed token-by-token
        """
    )

# ---------------------------------------------------------------------------
# Main column — chat interface
# ---------------------------------------------------------------------------

with main_col:
    st.subheader("💬 Query Interface")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📚 Sources ({len(msg['sources'])} re-ranked chunks)"):
                    for i, src in enumerate(msg["sources"], 1):
                        c1, c2 = st.columns(2)
                        c1.markdown(f"**[Source {i}]** `{src['source']}` — p.{src['page']}")
                        c2.markdown(
                            f"Rerank `{src['rerank_score']:.4f}` · "
                            f"Vector `{src['similarity_score']:.4f}`"
                        )
                        st.caption(src["text"])
                        if i < len(msg["sources"]):
                            st.divider()

    # New query
    if query := st.chat_input("Ask anything about your documents…"):
        pipeline = get_pipeline()

        if pipeline.get_stats()["total_chunks"] == 0:
            st.error("No documents indexed yet — upload a PDF on the left first.")
            st.stop()

        # Append user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            retriever = get_retriever()
            reranker = get_reranker()
            generator = get_generator()

            # Stage 1
            with st.status("Stage 1 — vector retrieval…") as s1:
                candidates = retriever.retrieve(query, top_k=10)
                s1.update(
                    label=f"Stage 1 complete — {len(candidates)} candidates",
                    state="complete",
                )

            # Stage 2
            with st.status("Stage 2 — cross-encoder re-ranking…") as s2:
                top_chunks = reranker.rerank(query, candidates, top_k=3)
                s2.update(
                    label=f"Stage 2 complete — top-{len(top_chunks)} refined",
                    state="complete",
                )

            # Stage 3 — streaming
            response = st.write_stream(
                generator.stream_sync(query, top_chunks)
            )

            # Citations
            sources = [
                {
                    "text": c["text"][:400],
                    "source": c["metadata"].get("source", "unknown"),
                    "page": c["metadata"].get("page", 0) + 1,
                    "similarity_score": round(c.get("similarity_score", 0.0), 4),
                    "rerank_score": round(c.get("rerank_score", 0.0), 4),
                }
                for c in top_chunks
            ]
            with st.expander(f"📚 Sources ({len(sources)} re-ranked chunks)"):
                for i, src in enumerate(sources, 1):
                    c1, c2 = st.columns(2)
                    c1.markdown(f"**[Source {i}]** `{src['source']}` — p.{src['page']}")
                    c2.markdown(
                        f"Rerank `{src['rerank_score']:.4f}` · "
                        f"Vector `{src['similarity_score']:.4f}`"
                    )
                    st.caption(src["text"])
                    if i < len(sources):
                        st.divider()

        st.session_state.messages.append(
            {"role": "assistant", "content": response, "sources": sources}
        )

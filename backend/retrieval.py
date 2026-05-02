"""
Stage 1 — Hybrid retrieval: Dense Vector Search ⊕ BM25 → RRF Fusion.

Why hybrid?
-----------
  Bi-encoder (vector)  great at semantic similarity / paraphrases.
                        Fails on rare proper nouns, model numbers, exact terms
                        whose vectors land in unexpected neighbourhoods.

  BM25 (keyword)       exact token overlap — catches any term verbatim.
                        Blind to meaning; "car" won't surface "automobile".

  Reciprocal Rank Fusion combines both ranked lists without needing
  calibrated scores from either.  Standard constant k=60 (Cormack 2009).
  A chunk earns RRF credit from each list it appears in, so chunks that
  rank moderately well in *both* beat chunks dominant in only one.

Result: top-20 diverse candidates forwarded to Stage-2 cross-encoder.
"""

import logging
from typing import Any

from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.config import Settings as ChromaSettings

from config import settings

logger = logging.getLogger(__name__)

# Standard RRF constant — lower values amplify rank-1 bonus
_RRF_K = 60


class HybridRetriever:
    def __init__(self) -> None:
        self.embedding_model = TextEmbedding(
            model_name=settings.EMBEDDING_MODEL,
            cache_dir="./.model_cache",
        )

        client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        # BM25 index is built lazily and invalidated after each ingestion
        self._bm25: BM25Okapi | None = None
        self._bm25_corpus: list[tuple[str, dict]] = []   # (text, metadata)

    # ------------------------------------------------------------------
    # BM25 index management
    # ------------------------------------------------------------------

    def invalidate_bm25(self) -> None:
        """Drop the in-memory BM25 index so it rebuilds on the next query."""
        self._bm25 = None
        self._bm25_corpus = []
        logger.info("BM25 index invalidated — will rebuild on next query")

    def _build_bm25_index(self) -> None:
        """
        Load every chunk from ChromaDB and build a BM25Okapi index.
        Called lazily on first query and after each document ingestion.
        """
        total = self.collection.count()
        if total == 0:
            logger.warning("Collection is empty — BM25 index skipped")
            return

        result = self.collection.get(include=["documents", "metadatas"])
        texts: list[str] = result["documents"]
        metas: list[dict] = result["metadatas"]

        self._bm25_corpus = list(zip(texts, metas))
        tokenized = [text.lower().split() for text in texts]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built — %d documents", len(texts))

    # ------------------------------------------------------------------
    # Individual search legs
    # ------------------------------------------------------------------

    def _vector_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        total = self.collection.count()
        if total == 0:
            return []

        query_vec = list(self.embedding_model.embed([query]))[0].tolist()

        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=min(top_k, total),
            include=["documents", "metadatas", "distances"],
        )

        return [
            {
                "text": doc,
                "metadata": meta,
                "similarity_score": round(1.0 - dist, 6),
                "bm25_score": 0.0,
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def _bm25_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if self._bm25 is None:
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        # Sort by score descending, keep top_k with positive score
        ranked_indices = sorted(
            (i for i in range(len(scores)) if scores[i] > 0),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        return [
            {
                "text": self._bm25_corpus[i][0],
                "metadata": self._bm25_corpus[i][1],
                "similarity_score": 0.0,
                "bm25_score": float(scores[i]),
            }
            for i in ranked_indices
        ]

    # ------------------------------------------------------------------
    # RRF fusion
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_key(metadata: dict) -> str:
        """Stable unique key for deduplication across result lists."""
        return f"{metadata.get('source', '')}::chunk_{metadata.get('chunk_index', '')}"

    def _rrf_fusion(
        self,
        vector_results: list[dict[str, Any]],
        bm25_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Reciprocal Rank Fusion.

        score(d) = Σ  1 / (k + rank_in_list_i)

        Chunks appearing in both lists accumulate credit from each,
        naturally surfacing results that are both semantically relevant
        and keyword-matched.
        """
        rrf_scores: dict[str, float] = {}
        merged: dict[str, dict] = {}

        for rank, chunk in enumerate(vector_results):
            key = self._chunk_key(chunk["metadata"])
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
            merged[key] = chunk

        for rank, chunk in enumerate(bm25_results):
            key = self._chunk_key(chunk["metadata"])
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
            if key not in merged:
                merged[key] = chunk
            else:
                # Carry the BM25 score into the entry that already has similarity_score
                merged[key] = {**merged[key], "bm25_score": chunk["bm25_score"]}

        sorted_keys = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
        return [
            {**merged[key], "rrf_score": round(rrf_scores[key], 6)}
            for key in sorted_keys
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """
        Hybrid retrieve: vector ∪ BM25 fused via RRF → top_k candidates.

        BM25 index is rebuilt lazily; call `invalidate_bm25()` after ingestion
        to ensure new documents appear in keyword results.
        """
        if top_k is None:
            top_k = settings.TOP_K_RETRIEVAL  # default 20

        if self.collection.count() == 0:
            logger.warning("Collection empty — no documents ingested yet")
            return []

        # Lazy BM25 index build
        if self._bm25 is None:
            self._build_bm25_index()

        vector_results = self._vector_search(query, top_k)
        bm25_results = self._bm25_search(query, top_k)

        fused = self._rrf_fusion(vector_results, bm25_results)
        top = fused[:top_k]

        logger.info(
            "Hybrid retrieval — vector: %d  BM25: %d  fused: %d  returning: %d",
            len(vector_results), len(bm25_results), len(fused), len(top),
        )
        return top

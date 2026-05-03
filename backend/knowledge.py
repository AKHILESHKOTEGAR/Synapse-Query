"""
Adaptive Knowledge Store — makes Nexus self-improving over time.

Every time a user asks a question, the query and its top retrieved chunks
are stored here.  On future queries, semantically similar past interactions
are recalled and injected back into the candidate pool before re-ranking.

This means:
- Chunks that proved relevant for a topic surface more easily for related
  future questions, even if raw vector/BM25 scores rank them lower.
- The system gets more useful the more it is queried — without any model
  fine-tuning or external service.

It is NOT a chat history or a general-purpose AI memory.  Every piece of
recalled context is still grounded in the uploaded documents.
"""

import hashlib
import logging
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from fastembed import TextEmbedding

from config import settings

logger = logging.getLogger(__name__)


class KnowledgeStore:
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
            name=settings.KNOWLEDGE_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Knowledge store ready — %d interactions stored",
            self.collection.count(),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> list[float]:
        return list(self.embedding_model.embed([text]))[0].tolist()

    @staticmethod
    def _entry_id(query: str, chunk_text: str) -> str:
        return hashlib.md5(f"{query}\x00{chunk_text[:200]}".encode()).hexdigest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(self, query: str, top_chunks: list[dict[str, Any]]) -> None:
        """
        Persist a successful retrieval interaction.

        Each top chunk is stored embedded under the query vector so that
        future semantically similar queries can recall it.
        """
        if not top_chunks:
            return

        ids, documents, embeddings, metadatas = [], [], [], []

        # Embed the query once
        query_vec = self._embed(query)

        for chunk in top_chunks:
            entry_id = self._entry_id(query, chunk["text"])
            meta = chunk.get("metadata", {})

            ids.append(entry_id)
            documents.append(chunk["text"])
            embeddings.append(query_vec)
            metadatas.append({
                "source": meta.get("source", "unknown"),
                "page": meta.get("page", 0),
                "extraction": meta.get("extraction", "direct"),
                "origin_query": query[:300],
            })

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info(
            "Knowledge store: saved %d chunks for query '%s...' (total: %d)",
            len(ids), query[:60], self.collection.count(),
        )

    def delete_source(self, source: str) -> None:
        """Remove all knowledge entries that originated from a given source file."""
        if self.collection.count() == 0:
            return
        result = self.collection.get(where={"source": {"$eq": source}})
        if result["ids"]:
            self.collection.delete(ids=result["ids"])
            logger.info(
                "Knowledge store: removed %d entries for '%s'",
                len(result["ids"]), source,
            )

    def recall(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """
        Return chunks from past interactions that are relevant to this query.

        Results are formatted identically to retrieval.py output so they can
        be merged directly into the candidate pool.
        """
        if self.collection.count() == 0:
            return []

        if top_k is None:
            top_k = settings.TOP_K_KNOWLEDGE

        query_vec = self._embed(query)
        n = min(top_k, self.collection.count())

        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        recalled = [
            {
                "text": doc,
                "metadata": meta,
                "similarity_score": round(1.0 - dist, 6),
                "bm25_score": 0.0,
                "rrf_score": 0.0,
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

        if recalled:
            logger.info(
                "Knowledge store: recalled %d chunks for query '%s...'",
                len(recalled), query[:60],
            )

        return recalled

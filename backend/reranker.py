"""
Stage 2 — Cross-encoder re-ranking via fastembed TextCrossEncoder.

Uses Xenova/ms-marco-MiniLM-L-6-v2 (ONNX export of the original HuggingFace
cross-encoder/ms-marco-MiniLM-L-6-v2).  No torch required — runs on the
same ONNX Runtime that chromadb already installs.

A cross-encoder jointly encodes (query, document) pairs through a single
transformer, letting every attention head attend across both inputs.  This
is significantly more accurate than bi-encoder scoring, at the cost of
O(K) forward passes — feasible because we only score the top-10 candidates
from Stage 1.
"""

import logging
from typing import Any

from fastembed.rerank.cross_encoder import TextCrossEncoder

from config import settings

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self) -> None:
        logger.info("Loading cross-encoder: %s", settings.RERANKER_MODEL)
        self.model = TextCrossEncoder(
            model_name=settings.RERANKER_MODEL,
            cache_dir="./.model_cache",
        )
        logger.info("Cross-encoder ready")

    def rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Score (query, chunk) pairs and return top-K by rerank score.

        Parameters
        ----------
        query   : user query string
        chunks  : Stage-1 candidates [{text, metadata, similarity_score}, ...]
        top_k   : how many to return (default: settings.TOP_K_RERANK)

        Returns
        -------
        Top-K chunks sorted by cross-encoder score descending,
        each augmented with a `rerank_score` field.
        """
        if top_k is None:
            top_k = settings.TOP_K_RERANK

        if not chunks:
            return []

        documents = [c["text"] for c in chunks]
        # TextCrossEncoder.rerank returns an Iterable[float]
        scores = list(self.model.rerank(query=query, documents=documents))

        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        reranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
        top = reranked[:top_k]

        logger.info(
            "Stage-2 re-ranked %d → top-%d  |  best=%.4f  worst=%.4f",
            len(chunks), len(top),
            top[0]["rerank_score"],
            top[-1]["rerank_score"],
        )
        return top

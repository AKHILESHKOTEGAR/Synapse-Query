"""
Stage 3 — Grounded answer generation via Nemotron Super on OpenRouter.

OpenRouter exposes an OpenAI-compatible API at https://openrouter.ai/api/v1.
We pass HTTP-Referer and X-Title headers as required by OpenRouter's terms.

Only the top-3 re-ranked chunks are forwarded to the LLM.  The system
prompt positions the model as a Technical Auditor — forcing structured
Markdown output, multi-tier logic extraction, and mandatory citation.
"""

import logging
from collections.abc import AsyncGenerator, Generator
from typing import Any

from openai import AsyncOpenAI, OpenAI

from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Technical Auditor system prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are **Nexus — Technical Document Auditor**, an enterprise-grade intelligence \
system built for precision extraction from research papers, technical \
specifications, and regulatory documents.

## Absolute Rules

1. **GROUNDED ONLY** — Every factual claim must end with [Source N].
   If a claim cannot be cited from the provided context, do not make it.
   Do not blend general knowledge with source content.

2. **CITATION DENSITY** — Aim for a [Source N] tag on every sentence that \
asserts a fact, value, classification, or process step.

3. **PRESERVE EXACT TERMINOLOGY** — Never paraphrase acronyms, model names, \
class labels, identifiers, or threshold values.
   Example: "HFW", "CW", "Exact Match", "0.87 F1-score" must appear verbatim.

4. **INSUFFICIENT CONTEXT PROTOCOL** — If the context lacks the answer, output:
   > ⚠️ Insufficient context: the sources do not contain information about \
[specific missing element]. Consider uploading additional documents.

---

## Structured Extraction Mandate

Detect and render the following patterns from the source text:

| Pattern | Required Output Format |
|---------|----------------------|
| Classification scheme / taxonomy (N tiers) | Markdown table with all tiers, criteria, and examples |
| Sequential process / pipeline | Numbered list with sub-steps |
| Comparative metrics (models, methods, scores) | Markdown table |
| Definition of a technical term | Bold term + blockquote definition + [Source N] |
| Enumerated categories (e.g., 4 match types) | Numbered list — one entry per category, all sub-criteria included |
| Threshold / cutoff values | `inline code` for the exact value |

**Never collapse a multi-tier classification into prose.**
If a document defines four pattern-matching categories, output all four — \
not a summary sentence.

---

## Response Structure

Use this template for every technical query:

### Summary
_One or two sentences — direct answer to the question with citations._

### Detailed Findings
_Structured evidence — tables, lists, definitions extracted verbatim from \
sources.  This is the bulk of the response._

### Key Technical Terms
_Glossary of domain-specific terms found in the sources with exact definitions._

### Source Coverage
_Brief note on which [Source N] chunks were used and what aspect each covers._

---

Respond in GitHub-Flavored Markdown.  Use tables wherever comparative or \
categorical data appears.  Do not add caveats beyond what is stated above.\
"""

# ---------------------------------------------------------------------------
# User message template
# ---------------------------------------------------------------------------

_USER_TEMPLATE = """\
<retrieved_context>
{context}
</retrieved_context>

<query>{query}</query>

## Extraction Instructions

Perform a **structured technical audit** of the retrieved context above.

1. Identify every classification scheme, taxonomy, or multi-tier logic \
present in the sources and render each as a Markdown table or numbered list.
2. Extract all numeric values, thresholds, scores, and identifiers verbatim — \
wrap them in `backticks`.
3. Cite every claim with [Source N].  A claim without a citation is a violation.
4. If the query asks about categories or types, list **all** of them — \
do not omit tiers.
5. If comparative data is available (e.g., multiple models or methods), \
format it as a table.

Respond using the Technical Auditor structure defined in your system prompt.\
"""

_OPENROUTER_HEADERS = {
    "HTTP-Referer": "http://localhost:3000",
    "X-Title": "Nexus RAG - Technical Auditor",
}


class LLMGenerator:
    def __init__(self) -> None:
        self._async_client = AsyncOpenAI(
            base_url=settings.NVIDIA_BASE_URL,
            api_key=settings.NVIDIA_API_KEY,
            default_headers=_OPENROUTER_HEADERS,
        )
        self._sync_client = OpenAI(
            base_url=settings.NVIDIA_BASE_URL,
            api_key=settings.NVIDIA_API_KEY,
            default_headers=_OPENROUTER_HEADERS,
        )

    # ------------------------------------------------------------------
    # Context builder — numbered, sourced, with metadata header
    # ------------------------------------------------------------------

    def _build_context_block(self, chunks: list[dict[str, Any]]) -> str:
        """
        Format top-K chunks into a numbered context block.

        Each block header includes source file, page, and retrieval scores
        so the LLM can reference them precisely.
        """
        parts = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk["metadata"]
            src = meta.get("source", "unknown")
            page = meta.get("page", 0) + 1
            extraction = meta.get("extraction", "direct")
            rerank = chunk.get("rerank_score", chunk.get("rrf_score", 0))
            header = (
                f"[Source {i}] {src} | Page {page} | "
                f"Re-rank: {rerank:.3f} | Extraction: {extraction}"
            )
            parts.append(f"{header}\n{chunk['text']}")
        return "\n\n---\n\n".join(parts)

    def _messages(self, query: str, chunks: list[dict[str, Any]]) -> list[dict]:
        context = self._build_context_block(chunks)
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _USER_TEMPLATE.format(context=context, query=query),
            },
        ]

    # ------------------------------------------------------------------
    # Async streaming — FastAPI
    # ------------------------------------------------------------------

    async def stream(
        self, query: str, chunks: list[dict[str, Any]]
    ) -> AsyncGenerator[str, None]:
        response = await self._async_client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=self._messages(query, chunks),
            max_tokens=settings.LLM_MAX_TOKENS,
            stream=True,
        )
        async for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    # ------------------------------------------------------------------
    # Sync streaming — Streamlit write_stream
    # ------------------------------------------------------------------

    def stream_sync(
        self, query: str, chunks: list[dict[str, Any]]
    ) -> Generator[str, None, None]:
        response = self._sync_client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=self._messages(query, chunks),
            max_tokens=settings.LLM_MAX_TOKENS,
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    # ------------------------------------------------------------------
    # Non-streaming — utility / testing
    # ------------------------------------------------------------------

    def generate(self, query: str, chunks: list[dict[str, Any]]) -> str:
        response = self._sync_client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=self._messages(query, chunks),
            max_tokens=settings.LLM_MAX_TOKENS,
        )
        return response.choices[0].message.content or ""

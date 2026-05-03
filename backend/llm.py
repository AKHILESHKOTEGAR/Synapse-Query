"""
Stage 3 — Grounded answer generation via Nemotron Super on OpenRouter.

OpenRouter exposes an OpenAI-compatible API at https://openrouter.ai/api/v1.
We pass HTTP-Referer and X-Title headers as required by OpenRouter's terms.

Only the top-3 re-ranked chunks are forwarded to the LLM.  The system
prompt positions the model as a Technical Auditor — forcing structured
Markdown output, multi-tier logic extraction, mandatory citation, and
inline-reference resolution against the paper's reference list.
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
You are **Nexus — Technical Document Auditor**, a professional-grade discovery \
engine built for precision extraction from research papers, technical \
specifications, and regulatory documents.  Your role is to transform \
unstructured source text into a structured, verifiable knowledge base — \
every answer fully supported by the source, with a complete paper trail of \
page references and retrieval confidence scores.

## Absolute Rules

1. **GROUNDED ONLY** — Every factual claim must end with [Source N].
   If a claim cannot be cited from the provided context, do not make it.
   Do not blend general knowledge with source content.

2. **CITATION DENSITY** — Aim for a [Source N] tag on every sentence that \
asserts a fact, value, classification, or process step.

3. **PRESERVE EXACT TERMINOLOGY** — Never paraphrase acronyms, model names, \
class labels, identifiers, or threshold values.
   Example: "HFW" (High-Frequency Words), "CW" (Content Words), \
"Exact Match", "0.87 F1-score" must appear verbatim exactly as in the source.

4. **EXPLAIN THEN STRUCTURE** — For every taxonomy, classification scheme, or \
multi-tier system found in the sources:
   a. First write a short explanatory paragraph describing what the system is \
and why it exists, with citations.
   b. Then render the full structure as a Markdown table or numbered list — \
never skip either part.
   Example: if a document defines a four-tier matching system \
(Exact Match, Sparse Match, Incomplete Match, No Match), \
explain the overall pattern-matching approach first, \
then output all four tiers as a table with criteria and conditions.

5. **REFERENCE RESOLUTION** — When the source text contains inline numeric \
citations such as [1], [4], [12], and a `<reference_list>` is provided, \
resolve each cited number to its paper title and authors. Include a \
**Referenced Papers** section at the end listing only the references \
actually mentioned in your response, in the format:
   > **[N]** Author(s) — *Title*

6. **INSUFFICIENT CONTEXT PROTOCOL** — If the context lacks the answer, output:
   > ⚠️ Insufficient context: the sources do not contain information about \
[specific missing element]. Consider uploading additional documents.

---

## Structured Extraction Mandate

Detect and render the following patterns from the source text:

| Pattern | Required Output Format |
|---------|----------------------|
| Classification scheme / taxonomy (N tiers) | Explanation paragraph + Markdown table: all tiers, criteria, examples |
| Four-tier match system (Exact / Sparse / Incomplete / No Match) | Explanation paragraph + table with all four tiers and their exact criteria |
| Sequential process / pipeline | Explanation + numbered list with sub-steps |
| Comparative metrics (models, methods, scores) | Markdown table: Model/Method, Metric, Score, Source |
| Definition of a technical term | **Bold term** + blockquote definition + [Source N] |
| Threshold / cutoff values | `inline code` for the exact value |

**Rule:** Every taxonomy or classification output must open with an explanation \
paragraph before the table or list. Never collapse multi-tier logic into a \
single prose sentence. Never skip tiers.

---

## Response Structure

Use this template for every technical query:

### Summary
_One or two sentences — direct answer to the question with citations._

### Detailed Findings
_Explanation paragraphs followed by structured evidence — tables, numbered \
lists, definitions extracted verbatim.  Every taxonomy or classification \
gets its own explanation + table block._

### Key Technical Terms
_Glossary of domain-specific terms found in the sources with exact definitions._

### Referenced Papers
_Only include if inline citations like [1], [4] appear in the source text \
and a reference list is provided. List each resolved reference once._

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
{reference_section}
<query>{query}</query>

## Extraction Instructions

Perform a **structured technical audit** of the retrieved context above.

1. Identify every classification scheme, taxonomy, or multi-tier logic present \
in the sources.  For each one: write a short explanation of what it is and why \
it matters, then render the full structure as a Markdown table — all tiers, \
criteria, special conditions.
2. If the sources define a match-type system (e.g., Exact Match, Sparse Match, \
Incomplete Match, No Match), output **all** tiers in a table — explain the \
system first, then table every tier with its exact criteria and conditions.
3. Extract all numeric values, thresholds, scores, and identifiers verbatim — \
wrap them in `backticks`.
4. Cite every claim with [Source N].  A claim without a citation is a violation.
5. If comparative data is available (e.g., multiple models or methods with \
scores), render it as a Markdown table: columns for Model/Method, Metric, \
Score, and Source.
6. Scan the source text for inline numeric citations like [1], [4], [12]. \
If a `<reference_list>` block is provided, include a **Referenced Papers** \
section listing each resolved number → full author + title. \
If no inline citations appear in the sources, omit this section entirely.

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
    # Context builders
    # ------------------------------------------------------------------

    def _build_context_block(self, chunks: list[dict[str, Any]]) -> str:
        """Format top-K chunks into a numbered context block with metadata headers."""
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

    def _build_reference_block(self, ref_chunks: list[dict[str, Any]]) -> str:
        """
        Concatenate reference-list chunks into a <reference_list> block.
        Returns empty string when no reference chunks are available.
        """
        if not ref_chunks:
            return ""
        combined = "\n\n".join(c["text"] for c in ref_chunks)
        return f"\n<reference_list>\n{combined}\n</reference_list>\n"

    def _messages(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        ref_chunks: list[dict[str, Any]] | None = None,
    ) -> list[dict]:
        context = self._build_context_block(chunks)
        reference_section = self._build_reference_block(ref_chunks or [])
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _USER_TEMPLATE.format(
                    context=context,
                    reference_section=reference_section,
                    query=query,
                ),
            },
        ]

    # ------------------------------------------------------------------
    # Async streaming — FastAPI
    # ------------------------------------------------------------------

    async def stream(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        ref_chunks: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[str, None]:
        response = await self._async_client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=self._messages(query, chunks, ref_chunks),
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
        self,
        query: str,
        chunks: list[dict[str, Any]],
        ref_chunks: list[dict[str, Any]] | None = None,
    ) -> Generator[str, None, None]:
        response = self._sync_client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=self._messages(query, chunks, ref_chunks),
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

    def generate(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        ref_chunks: list[dict[str, Any]] | None = None,
    ) -> str:
        response = self._sync_client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=self._messages(query, chunks, ref_chunks),
            max_tokens=settings.LLM_MAX_TOKENS,
        )
        return response.choices[0].message.content or ""

"""
Modular PDF ingestion pipeline.

Stages
------
1. Load    – PyMuPDF; per-page garble detection → OCR fallback via Tesseract
2. Clean   – ligature repair, control-char strip, whitespace normalisation
3. Chunk   – RecursiveCharacterTextSplitter
4. Embed   – fastembed TextEmbedding (ONNX, no torch)
5. Store   – ChromaDB

Text extraction strategy (per page)
------------------------------------
  Pass 1 — PyMuPDF direct extraction with TEXT_DEHYPHENATE | TEXT_MEDIABOX_CLIP.
            Fast, preserves layout.  Works for 99 % of born-digital PDFs.

  Pass 2 — Garble detection: if the extracted text has >15 % Unicode Private
            Use Area chars, replacement chars (�), or is near-empty, the
            page is considered unreadable and we fall through to Pass 3.

  Pass 3 — OCR via Tesseract 5.  PyMuPDF renders the page to a 300-DPI
            RGB pixmap; pytesseract runs Tesseract on the image.  Handles:
              • Scanned PDFs with no real text layer
              • PDFs whose fonts use a custom / Private-Use-Area encoding
              • PDFs with CIDFont glyphs PyMuPDF can't map to Unicode

  OCR is optional: if Tesseract is not installed, Pages that fail Pass 1
  are skipped with a warning instead of crashing.
"""

import io
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastembed import TextEmbedding
import chromadb
from chromadb.config import Settings as ChromaSettings

from config import settings

logger = logging.getLogger(__name__)

# Ligatures not caught by NFKC
_LIGATURE_MAP = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl",
    "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
}

# Try to import OCR deps — fall back gracefully if missing
try:
    import pytesseract
    from PIL import Image
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False


class DocumentIngestionPipeline:
    def __init__(self) -> None:
        logger.info("Initialising ingestion pipeline...")

        if not _OCR_AVAILABLE:
            logger.warning(
                "pytesseract / Pillow not installed — OCR fallback disabled. "
                "Scanned PDFs or custom-font PDFs may produce garbled text."
            )

        self.embedding_model = TextEmbedding(
            model_name=settings.EMBEDDING_MODEL,
            cache_dir="./.model_cache",
        )
        logger.info("Embedding model loaded: %s", settings.EMBEDDING_MODEL)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            length_function=len,
        )

        self._chroma_client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self._chroma_client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection '%s' ready (%d chunks)",
            settings.CHROMA_COLLECTION_NAME,
            self.collection.count(),
        )

    # ------------------------------------------------------------------
    # Text cleaning
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_text(text: str) -> str:
        """NFKC normalise, repair ligatures, strip control chars."""
        text = unicodedata.normalize("NFKC", text)
        for lig, rep in _LIGATURE_MAP.items():
            text = text.replace(lig, rep)
        # Strip control chars except \t and \n
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _garble_rate(text: str) -> float:
        stripped = text.strip()
        if not stripped:
            return 1.0
        bad = sum(
            1 for c in stripped
            if c == "�"
            or 0xE000 <= ord(c) <= 0xF8FF
            or 0xF0000 <= ord(c) <= 0xFFFFF
        )
        return bad / len(stripped)

    @classmethod
    def _is_garbled(cls, text: str, threshold: float = 0.15) -> bool:
        if len(text.strip()) < 20:
            return True
        return cls._garble_rate(text) > threshold

    # ------------------------------------------------------------------
    # Per-page extraction with OCR fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_direct(page) -> str:
        """Pass 1 — PyMuPDF native text extraction."""
        return page.get_text(
            "text",
            flags=(
                fitz.TEXT_PRESERVE_WHITESPACE
                | fitz.TEXT_DEHYPHENATE
                | fitz.TEXT_MEDIABOX_CLIP
            ),
        )

    @staticmethod
    def _extract_ocr(page) -> str:
        """
        Pass 3 — Tesseract OCR on a 300-DPI render of the page.

        300 DPI is the minimum recommended for reliable OCR on body text.
        PSM 3 = fully automatic page segmentation (no prior on orientation).
        """
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return pytesseract.image_to_string(img, lang="eng", config="--psm 3")

    def _extract_page(self, page, page_num: int) -> tuple[str, str]:
        """
        Return (clean_text, method_used) for a single PDF page.
        method_used is 'direct', 'ocr', or 'skipped'.

        Strategy: always prefer whichever extraction has the lower garble rate.
        OCR is only attempted when direct extraction is >15% garbled.
        A page is skipped only when both methods exceed 50% garbled.
        """
        raw = self._extract_direct(page)
        text = self._clean_text(raw)
        direct_rate = self._garble_rate(text)

        if direct_rate <= 0.15:
            return text, "direct"

        # Direct is garbled — try OCR if available
        if not _OCR_AVAILABLE:
            logger.warning(
                "Page %d: %.0f%% garbled, OCR unavailable — keeping direct",
                page_num, direct_rate * 100,
            )
            return (text, "direct") if direct_rate < 0.50 else ("", "skipped")

        logger.info("Page %d: %.0f%% garbled — trying OCR", page_num, direct_rate * 100)
        ocr_text = self._clean_text(self._extract_ocr(page))
        ocr_rate = self._garble_rate(ocr_text)

        # Use whichever extraction is cleaner
        if ocr_rate < direct_rate:
            best_text, best_rate, best_method = ocr_text, ocr_rate, "ocr"
        else:
            best_text, best_rate, best_method = text, direct_rate, "direct"

        if best_rate > 0.50:
            logger.warning("Page %d: both extractions >50%% garbled — skipping", page_num)
            return "", "skipped"

        logger.info("Page %d: using %s (%.0f%% garbled)", page_num, best_method, best_rate * 100)
        return best_text, best_method

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _load_pdf(self, file_path: str) -> list[Document]:
        doc = fitz.open(file_path)
        pages: list[Document] = []
        ocr_count = 0

        for page_num, page in enumerate(doc):
            text, method = self._extract_page(page, page_num)
            if method == "ocr":
                ocr_count += 1
            if len(text) < 20:
                continue
            pages.append(Document(
                page_content=text,
                metadata={
                    "page": page_num,
                    "source": Path(file_path).name,
                    "extraction": method,
                },
            ))

        doc.close()
        logger.info(
            "Loaded %d pages from %s  (direct=%d  ocr=%d)",
            len(pages), Path(file_path).name,
            len(pages) - ocr_count, ocr_count,
        )
        return pages

    def _chunk(self, documents: list[Document]) -> list[Document]:
        chunks = self.text_splitter.split_documents(documents)
        logger.info("Chunked %d pages → %d chunks", len(documents), len(chunks))
        return chunks

    def _embed(self, chunks: list[Document]) -> list[list[float]]:
        texts = [c.page_content for c in chunks]
        return [v.tolist() for v in self.embedding_model.embed(texts)]

    def _store(
        self,
        chunks: list[Document],
        embeddings: list[list[float]],
        source_filename: str,
    ) -> dict[str, Any]:
        ids, documents, metadatas = [], [], []

        for i, chunk in enumerate(chunks):
            ids.append(f"{source_filename}::chunk_{i}")
            documents.append(chunk.page_content)
            metadatas.append({
                "source": source_filename,
                "page": chunk.metadata.get("page", 0),
                "chunk_index": i,
                "chunk_size": len(chunk.page_content),
                "extraction": chunk.metadata.get("extraction", "direct"),
            })

        # Delete all existing chunks for this source before upserting to prevent
        # stale chunks from prior ingestions persisting if chunk count changes.
        existing = self.collection.get(where={"source": source_filename})
        if existing["ids"]:
            self.collection.delete(ids=existing["ids"])
            logger.info("Deleted %d stale chunks for '%s'", len(existing["ids"]), source_filename)

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info(
            "Stored %d chunks for '%s' (collection total: %d)",
            len(ids), source_filename, self.collection.count(),
        )
        return {
            "chunks_stored": len(ids),
            "source": source_filename,
            "collection_total": self.collection.count(),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, file_path: str) -> dict[str, Any]:
        """Run Load → Clean → Chunk → Embed → Store."""
        filename = Path(file_path).name
        logger.info("Starting ingestion: %s", filename)

        pages = self._load_pdf(file_path)
        if not pages:
            raise ValueError(
                f"No extractable text in {filename}. "
                "Check the file is not password-protected or purely image-based."
            )

        chunks = self._chunk(pages)
        embeddings = self._embed(chunks)
        result = self._store(chunks, embeddings, filename)

        return {**result, "pages_processed": len(pages), "status": "success"}

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_chunks": self.collection.count(),
            "collection_name": settings.CHROMA_COLLECTION_NAME,
        }

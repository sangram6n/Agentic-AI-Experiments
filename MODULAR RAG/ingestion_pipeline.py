# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 19:48:27 2026

@author: Sangram Mishra
"""

# day2_ingestion_pipeline.py
# DSA standards:
#   - Generators for memory efficiency (lazy evaluation)
#   - bisect for O(log n) chapter lookup instead of O(n) linear scan
#   - dict for O(1) extension routing instead of if-elif chain
#   - Sets for O(1) heading style membership checks
#   - Short-circuit evaluation to avoid full document traversal
#   - Explicit resource management (context managers)
#   - Typed signatures throughout

from __future__ import annotations

import base64
import bisect
import io
import logging
import os
from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from typing import Callable

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# BASE METADATA
# O(1) — stat() is a single syscall
# ─────────────────────────────────────────────────────────────────

def get_file_meta(filepath: str) -> dict:
    """Common metadata for all file types. Single stat() call."""
    p = Path(filepath)
    stat = p.stat()
    return {
        "source":        str(filepath),
        "filename":      p.name,
        "file_type":     p.suffix.lower(),
        "file_size_kb":  round(stat.st_size / 1024, 2),
        "modified_date": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d"),
    }

# ─────────────────────────────────────────────────────────────────
# THRESHOLD CHECK
# Short-circuit O(n) at threshold instead of full join O(n) always
# ─────────────────────────────────────────────────────────────────

def exceeds_text_threshold(docs: list[Document], threshold: int) -> bool:
    """
    Stop counting once threshold is met.
    Avoids joining all page_content strings into one giant string.
    O(n) worst case, O(1) best case.
    """
    total = 0
    for doc in docs:
        total += len(doc.page_content)
        if total >= threshold:
            return True
    return False


# ─────────────────────────────────────────────────────────────────
# PDF — NORMAL
# bisect for O(log n) chapter lookup
# Generator per page — no full doc in RAM
# ─────────────────────────────────────────────────────────────────

def _extract_heading_from_page(page) -> str:
    """
    Extract first heading-sized text span from a PDF page.
    Nested generators — exits immediately on first match.
    O(spans on page) worst case, O(1) best case.
    """
    try:
        blocks = page.get_text("dict").get("blocks", [])
        return next(
            (
                span["text"].strip()
                for block in blocks
                if block.get("type") == 0          # text block only
                for line in block.get("lines", [])
                for span in line.get("spans", [])
                if span.get("size", 0) > 12 and span.get("text", "").strip()
            ),
            ""  # default if no heading found
        )
    except Exception:
        return ""


def load_pdf(filepath: str) -> Generator[Document, None, None]:
    """
    Extract text from digital PDF page by page.
    Chapter lookup: O(log n) via bisect_right.
    Font-size heuristic for section heading: O(spans per page).
    Yields one Document per page — O(1) peak memory per page.
    """
    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz
        except ImportError:
            fitz = None

    base_meta = get_file_meta(filepath)

    if fitz is not None:
        try:
            with fitz.open(filepath) as pdf:
                total_pages = len(pdf)
                toc = pdf.get_toc()

                raw_chapter_map: dict[int, str] = {
                    page: title
                    for level, title, page in toc
                    if level == 1
                }

                chapter_pages: list[int] = sorted(raw_chapter_map)
                chapter_names: list[str] = [raw_chapter_map[p] for p in chapter_pages]

                def get_chapter(page_num: int) -> str:
                    idx = bisect.bisect_right(chapter_pages, page_num) - 1
                    return chapter_names[idx] if idx >= 0 else "Unknown"

                for page_num in range(total_pages):
                    page = pdf[page_num]
                    text = page.get_text()

                    if not text.strip():
                        continue

                    section_name = _extract_heading_from_page(page)

                    yield Document(
                        page_content=text,
                        metadata={
                            **base_meta,
                            "page":        page_num + 1,
                            "total_pages": total_pages,
                            "chapter":     get_chapter(page_num + 1),
                            "section":     section_name,
                            "word_count":  len(text.split()),
                            "method":      "pymupdf",
                        }
                    )
                return
        except Exception as e:
            logger.warning("  ⚠️ PyMuPDF extraction issue on %s: %s", Path(filepath).name, e)

    # Fallback to pypdf if PyMuPDF unavailable or encountered an error
    try:
        import pypdf
        reader = pypdf.PdfReader(filepath)
        total_pages = len(reader.pages)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                yield Document(
                    page_content=text,
                    metadata={
                        **base_meta,
                        "page":        page_num + 1,
                        "total_pages": total_pages,
                        "chapter":     "Unknown",
                        "section":     "",
                        "word_count":  len(text.split()),
                        "method":      "pypdf",
                    }
                )
    except Exception as e:
        logger.warning("  ⚠️ pypdf fallback failed on %s: %s", Path(filepath).name, e)


# ─────────────────────────────────────────────────────────────────
# PDF — SCANNED via DOCLING
# Yields per section — O(1) memory per section
# ─────────────────────────────────────────────────────────────────

def load_pdf_docling(filepath: str) -> Generator[Document, None, None]:
    """
    Convert scanned PDF using Docling (IBM).
    Yields one Document per logical section.
    pip install docling
    """
    from docling.document_converter import DocumentConverter

    base_meta = get_file_meta(filepath)
    result = DocumentConverter().convert(filepath)
    doc = result.document

    children = getattr(doc.body, "children", [])

    if not children:
        # Fallback: yield entire doc as one Document
        full_text = doc.export_to_markdown()
        if full_text.strip():
            yield Document(
                page_content=full_text,
                metadata={**base_meta, "page": 1, "method": "docling_full"}
            )
        return

    for i, item in enumerate(children):
        text = (
            item.export_to_markdown()
            if hasattr(item, "export_to_markdown")
            else str(item)
        )
        if not text.strip():
            continue

        # prov is a list; grab first entry's page safely
        # Handles both Pydantic model objects (Docling v2+) and plain dicts
        prov_list = getattr(item, "prov", [])
        if prov_list:
            entry = prov_list[0]
            if isinstance(entry, dict):
                page_no = entry.get("page", i + 1)
            else:
                page_no = getattr(entry, "page", i + 1)
        else:
            page_no = i + 1
        heading = getattr(item, "heading", "") or ""

        yield Document(
            page_content=text,
            metadata={
                **base_meta,
                "page":       page_no,
                "section":    heading,
                "word_count": len(text.split()),
                "method":     "docling",
            }
        )


# ─────────────────────────────────────────────────────────────────
# PDF — SCANNED via TESSERACT (last fallback)
# Yields per page
# ─────────────────────────────────────────────────────────────────

def load_pdf_tesseract(filepath: str) -> Generator[Document, None, None]:
    """
    Last-resort OCR using Tesseract + pdf2image.
    pip install pdf2image pytesseract
    Poppler required on PATH (Windows: https://github.com/oschwartz10612/poppler-windows)
    Processes one page at a time to avoid OOM on large scanned PDFs.
    """
    try:
        import pytesseract
        from pdf2image import convert_from_path, pdfinfo_from_path
    except ImportError:
        logger.warning("  ⚠️ pytesseract or pdf2image not installed for OCR fallback.")
        return

    base_meta = get_file_meta(filepath)

    # Get page count without loading images — avoids full-doc RAM spike
    total_pages = pdfinfo_from_path(filepath).get("Pages", 1)

    for i in range(1, total_pages + 1):
        # One page at a time — O(1) peak memory per page
        images = convert_from_path(
            filepath, dpi=300, first_page=i, last_page=i
        )
        if not images:
            continue
        page_image = images[0]
        try:
            text = pytesseract.image_to_string(page_image)
            if text.strip():
                yield Document(
                    page_content=text,
                    metadata={
                        **base_meta,
                        "page":       i,
                        "word_count": len(text.split()),
                        "method":     "tesseract",
                    }
                )
        finally:
            page_image.close()  # release PIL memory immediately


# ─────────────────────────────────────────────────────────────────
# PDF — SMART ROUTER
# Try cheapest strategy first, escalate only on failure
# Strategy cost: pymupdf < docling < tesseract < vision_llm
# ─────────────────────────────────────────────────────────────────

def _collect_with_threshold(
    gen: Generator[Document, None, None],
    threshold: int,
) -> tuple[list[Document], bool]:
    """
    Consume generator into list, returning early once threshold is met.
    Avoids joining all page_content strings into one giant string.
    """
    docs: list[Document] = []
    total_chars = 0
    for doc in gen:
        docs.append(doc)
        total_chars += len(doc.page_content)
        if total_chars >= threshold:
            docs.extend(gen)  # drain remaining
            return docs, True
    return docs, total_chars >= threshold


def load_pdf_smart(
    filepath: str,
    text_threshold: int = 100
) -> Generator[Document, None, None]:
    """
    Escalating strategy chain — cheapest first.
    Short-circuits as soon as a strategy produces enough text.

    Strategy order (cost ascending):
      1. PyMuPDF    → digital PDF, free, instant
      2. Docling    → scanned PDF, free, seconds
      3. Tesseract  → fallback OCR, free, slow
    """
    # Strategy 1: PyMuPDF
    docs, sufficient = _collect_with_threshold(load_pdf(filepath), text_threshold)
    if sufficient:
        logger.info("  ✅ Digital PDF: %s", Path(filepath).name)
        yield from docs
        return

    # Strategy 2: Docling
    logger.info("  🔁 Scanned → Docling: %s", Path(filepath).name)
    try:
        docs, sufficient = _collect_with_threshold(
            load_pdf_docling(filepath), text_threshold
        )
        if sufficient:
            yield from docs
            return
    except Exception as e:
        logger.warning("  ⚠️  Docling failed: %s", e)

    # Strategy 3: Tesseract
    logger.info("  🔁 Docling insufficient → Tesseract: %s", Path(filepath).name)
    yield from load_pdf_tesseract(filepath)


# ─────────────────────────────────────────────────────────────────
# DOCX
# Set for O(1) heading style membership check
# Section-boundary accumulator pattern — O(n) single pass
# ─────────────────────────────────────────────────────────────────

def load_docx(filepath: str) -> Generator[Document, None, None]:
    """
    Extract DOCX by section using heading styles as boundaries.
    Set for O(1) style lookup.
    Single O(n) pass over paragraphs.
    Yields one Document per section.
    pip install python-docx
    """
    import docx

    base_meta = get_file_meta(filepath)

    # Set for O(1) membership — vs list O(n)
    HEADING_STYLES: frozenset[str] = frozenset({
        "Heading 1", "Heading 2", "Heading 3",
        "Title", "Subtitle"
    })

    document = docx.Document(filepath)

    current_heading: str = "Introduction"
    current_lines: list[str] = []
    section_index: int = 1

    def flush_section() -> Document | None:
        """Flush accumulated lines as one Document."""
        nonlocal section_index
        if not current_lines:
            return None
        text = "\n".join(current_lines)
        doc = Document(
            page_content=text,
            metadata={
                **base_meta,
                "section":       current_heading,
                "section_index": section_index,
                "word_count":    len(text.split()),
                "method":        "docx",
            }
        )
        section_index += 1
        return doc

    for para in document.paragraphs:
        if para.style.name in HEADING_STYLES:       # O(1) set lookup
            doc = flush_section()
            if doc:
                yield doc
            current_heading = para.text.strip() or current_heading
            current_lines = []
        elif para.text.strip():
            current_lines.append(para.text.strip())

    # Final section
    doc = flush_section()
    if doc:
        yield doc


_OCR_ENGINE = None

def _get_ocr():
    """Singleton cached instance of RapidOCR."""
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _OCR_ENGINE = RapidOCR()
        except Exception:
            _OCR_ENGINE = False
    return _OCR_ENGINE if _OCR_ENGINE is not False else None


# ─────────────────────────────────────────────────────────────────
# PPTX
# Placeholder idx stored in a frozenset for O(1) title detection
# Single pass per slide
# ─────────────────────────────────────────────────────────────────

def load_pptx(filepath: str) -> Generator[Document, None, None]:
    """
    Extract PPTX slide-by-slide.
    Title placeholder detected via frozenset idx lookup — O(1).
    Single O(shapes) pass per slide.
    pip install python-pptx
    """
    from pptx import Presentation

    base_meta = get_file_meta(filepath)

    # Placeholder idx 0 = title, 1 = center title — frozenset for O(1)
    TITLE_IDX: frozenset[int] = frozenset({0, 1})

    prs = Presentation(filepath)
    total_slides = len(prs.slides)

    for i, slide in enumerate(prs.slides):
        slide_title: str = ""
        body_lines: list[str] = []

        for shape in slide.shapes:
            # Handle tables
            if getattr(shape, "has_table", False):
                table_lines = []
                for row in shape.table.rows:
                    row_cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                    if any(row_cells):
                        table_lines.append(" | ".join(row_cells))
                if table_lines:
                    body_lines.append("\n".join(table_lines))
                continue

            # Handle text shapes
            if not hasattr(shape, "text"):
                continue
            text = shape.text.strip()
            if not text:
                continue

            if getattr(shape, "is_placeholder", False):
                try:
                    ph = shape.placeholder_format
                    if ph and ph.idx in TITLE_IDX:     # O(1) frozenset lookup
                        slide_title = text
                    else:
                        body_lines.append(text)
                except (ValueError, AttributeError):
                    body_lines.append(text)
            else:
                body_lines.append(text)

        full_text = "\n".join(body_lines)
        content = f"{slide_title}\n{full_text}".strip()

        if not content:
            # Check for slide images (e.g. image-based presentation screenshots)
            image_texts = []
            ocr = _get_ocr()
            if ocr is not None:
                for shape in slide.shapes:
                    if hasattr(shape, "image"):
                        try:
                            res, _ = ocr(shape.image.blob)
                            if res:
                                image_texts.append("\n".join(line[1] for line in res))
                        except Exception:
                            pass
            if image_texts:
                content = "\n\n".join(image_texts).strip()

        if not content:
            continue

        layout_name = (
            slide.slide_layout.name
            if slide.slide_layout else ""
        )

        yield Document(
            page_content=content,
            metadata={
                **base_meta,
                "slide":        i + 1,
                "total_slides": total_slides,
                "slide_title":  slide_title,
                "layout":       layout_name,
                "word_count":   len(content.split()),
                "method":       "pptx",
            }
        )


# ─────────────────────────────────────────────────────────────────
# IMAGE — OCR
# ─────────────────────────────────────────────────────────────────

def load_image_ocr(filepath: str) -> Document | None:
    """
    Extract text from image using RapidOCR (or fallback to Tesseract).
    Returns None (not empty list) to allow clean short-circuit in caller.
    """
    base_meta = get_file_meta(filepath)
    text = ""
    width, height = 0, 0

    try:
        from PIL import Image
        with Image.open(filepath) as image:
            width, height = image.size
    except Exception:
        pass

    ocr = _get_ocr()
    if ocr is not None:
        try:
            res, _ = ocr(filepath)
            if res:
                text = "\n".join(line[1] for line in res)
        except Exception:
            pass

    if not text.strip():
        try:
            import pytesseract
            from PIL import Image
            with Image.open(filepath) as image:
                text = pytesseract.image_to_string(image)
        except Exception:
            pass

    if not text.strip():
        return None

    return Document(
        page_content=text,
        metadata={
            **base_meta,
            "image_width":  width,
            "image_height": height,
            "word_count":   len(text.split()),
            "method":       "ocr",
        }
    )


# ─────────────────────────────────────────────────────────────────
# IMAGE — VISION LLM
# Encode once, reuse bytes — no double read
# ─────────────────────────────────────────────────────────────────

# Lookup dict: O(1) mime type resolution — vs if-elif chain O(n)
_MIME_MAP: dict[str, str] = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".webp": "image/webp",
    ".bmp":  "image/bmp",
}

_VISION_PROMPT = (
    "Extract all text verbatim. "
    "Describe all charts, tables, and diagrams with their data and labels. "
    "Be thorough and structured."
)


def _encode_image(filepath: str) -> tuple[str, str]:
    """
    Read file once, return (base64_string, mime_type).
    Single file read — no double I/O.
    """
    ext = Path(filepath).suffix.lower()
    mime_type = _MIME_MAP.get(ext, "image/png")

    with open(filepath, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    return image_data, mime_type


def load_image_vision(
    filepath: str,
    llm_client: str = "ollama"
) -> Document:
    """
    Understand image via Vision LLM.
    Image encoded once and reused — O(1) I/O.
    llm_client: 'ollama' | 'anthropic' | 'openai'
    """
    base_meta = get_file_meta(filepath)
    image_data, mime_type = _encode_image(filepath)  # single read

    # Build message — dict lookup O(1) for client selection
    message_builders: dict[str, Callable] = {
        "ollama":    _build_ollama_vision_message,
        "anthropic": _build_anthropic_vision_message,
        "openai":    _build_openai_vision_message,
    }

    builder = message_builders.get(llm_client)
    if builder is None:
        raise ValueError(f"Unknown llm_client: {llm_client}. "
                         f"Choose from: {list(message_builders)}")

    llm, message = builder(image_data, mime_type)
    response = llm.invoke([message])

    return Document(
        page_content=response.content,
        metadata={
            **base_meta,
            "word_count": len(response.content.split()),
            "method":     f"vision_{llm_client}",
        }
    )


def _build_ollama_vision_message(image_data: str, mime_type: str):
    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage
    llm = ChatOllama(model="llava:7b")   # swap: moondream, llava-phi3
    msg = HumanMessage(content=[
        {"type": "image_url",
         "image_url": {"url": f"data:{mime_type};base64,{image_data}"}},
        {"type": "text", "text": _VISION_PROMPT},
    ])
    return llm, msg


def _build_anthropic_vision_message(image_data: str, mime_type: str):
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    msg = HumanMessage(content=[
        {"type": "image",
         "source": {"type": "base64", "media_type": mime_type, "data": image_data}},
        {"type": "text", "text": _VISION_PROMPT},
    ])
    return llm, msg


def _build_openai_vision_message(image_data: str, mime_type: str):
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    llm = ChatOpenAI(model="gpt-4o")
    msg = HumanMessage(content=[
        {"type": "image_url",
         "image_url": {"url": f"data:{mime_type};base64,{image_data}"}},
        {"type": "text", "text": _VISION_PROMPT},
    ])
    return llm, msg


# ─────────────────────────────────────────────────────────────────
# IMAGE — SMART ROUTER
# OCR first (free, fast) → Vision LLM only on failure
# ─────────────────────────────────────────────────────────────────

def load_image_smart(
    filepath: str,
    ocr_min_chars: int = 50,
    llm_client: str = "ollama"
) -> Generator[Document, None, None]:
    """
    Cost-ascending strategy:
      1. Tesseract OCR  → free, instant
      2. Vision LLM     → costs tokens, slower

    Short-circuits at Step 1 if OCR returns enough text.
    """
    doc = load_image_ocr(filepath)

    if doc and len(doc.page_content) >= ocr_min_chars:
        logger.info("  ✅ OCR (%d chars): %s", len(doc.page_content), Path(filepath).name)
        yield doc
        return

    logger.info("  🔁 OCR insufficient → Vision LLM: %s", Path(filepath).name)
    yield load_image_vision(filepath, llm_client=llm_client)


# ─────────────────────────────────────────────────────────────────
# EXTENSION ROUTER
# dict dispatch: O(1) vs if-elif chain O(n)
# ─────────────────────────────────────────────────────────────────

# Map extension → loader function
# All loaders return Generator[Document] for uniform interface
_EXTENSION_ROUTER: dict[str, Callable[[str], Generator[Document, None, None]]] = {
    ".pdf":  load_pdf_smart,
    ".docx": load_docx,
    ".pptx": load_pptx,
    ".jpg":  load_image_smart,
    ".jpeg": load_image_smart,
    ".png":  load_image_smart,
    ".webp": load_image_smart,
    ".bmp":  load_image_smart,
}

# Supported extensions as frozenset for O(1) existence check
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(_EXTENSION_ROUTER)


def load_file(filepath: str) -> Generator[Document, None, None]:
    """
    Route file to correct loader via O(1) dict lookup.
    Yields Documents lazily — caller controls memory.
    """
    ext = Path(filepath).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:         # O(1) frozenset check
        logger.warning("  ⚠️  Unsupported type [%s]: %s", ext, Path(filepath).name)
        return

    loader = _EXTENSION_ROUTER[ext]             # O(1) dict lookup
    logger.info("  📄 [%s] %s", ext, Path(filepath).name)

    try:
        yield from loader(filepath)
    except Exception as e:
        logger.error("  ❌ Failed [%s] %s: %s", ext, Path(filepath).name, e)


# ─────────────────────────────────────────────────────────────────
# DIRECTORY LOADER
# rglob generator — never loads all paths into RAM at once
# Yields Documents — O(1) peak memory regardless of folder size
# ─────────────────────────────────────────────────────────────────

def load_directory(
    directory: str,
    extensions: frozenset[str] | None = None
) -> Generator[Document, None, None]:
    """
    Walk directory recursively, yield Documents one at a time.

    Args:
        directory:  Root folder to scan.
        extensions: Optional filter, e.g. frozenset({'.pdf', '.docx'}).
                    Defaults to all supported types.

    Memory: O(1) — generator chain, no list accumulation.
    Time:   O(files × pages) — unavoidable, we must read every file.
    """
    allowed = extensions or SUPPORTED_EXTENSIONS
    root = Path(directory)

    IGNORED_DIRS: frozenset[str] = frozenset({
        "chroma_db", "chroma_db_day2", ".git", ".venv", "venv", "__pycache__", ".ipynb_checkpoints"
    })

    # rglob returns a generator — paths not loaded all at once
    file_gen = (
        p for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in allowed
        and not any(part in IGNORED_DIRS or (part.startswith(".") and part not in {".", ".."}) for part in p.parts)
    )

    # Deduplicate by resolved path (handles symlinks, relative aliases)
    seen: set[Path] = set()
    count = 0
    for filepath in file_gen:
        resolved = filepath.resolve()
        if resolved in seen:
            logger.debug("  ⏭️  Skipping duplicate: %s", filepath.name)
            continue
        seen.add(resolved)
        for doc in load_file(str(filepath)):
            count += 1
            yield doc

    logger.info("✅ Total documents yielded: %d", count)
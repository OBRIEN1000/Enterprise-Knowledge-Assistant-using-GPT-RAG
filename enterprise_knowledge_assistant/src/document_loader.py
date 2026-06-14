"""Document loading.

Extracts plain text from uploaded files (PDF, DOCX, TXT, MD) and returns a
list of ``(page_number, text)`` pairs so that citations can reference the
original location inside a document.
"""

from __future__ import annotations

import io
from typing import BinaryIO, List, Tuple

from pypdf import PdfReader
from docx import Document as DocxDocument


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

# Each "page" is a logical unit referenced in citations. For non-paginated
# formats (txt / md / docx) we fall back to a single page numbered 1.
Page = Tuple[int, str]


def _load_pdf(stream: BinaryIO) -> List[Page]:
    reader = PdfReader(stream)
    pages: List[Page] = []
    for index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((index, text))
    return pages


def _load_docx(stream: BinaryIO) -> List[Page]:
    document = DocxDocument(stream)
    paragraphs = [p.text for p in document.paragraphs if p.text and p.text.strip()]
    text = "\n".join(paragraphs).strip()
    return [(1, text)] if text else []


def _load_text(stream: BinaryIO) -> List[Page]:
    raw = stream.read()
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="ignore")
    else:
        text = str(raw)
    text = text.strip()
    return [(1, text)] if text else []


def load_document(filename: str, content: bytes) -> List[Page]:
    """Return a list of ``(page, text)`` extracted from *content*.

    The dispatch is based on the file extension. Unsupported types raise
    ``ValueError`` so the UI can show a friendly message.
    """
    lower = filename.lower()
    stream = io.BytesIO(content)

    if lower.endswith(".pdf"):
        return _load_pdf(stream)
    if lower.endswith(".docx"):
        return _load_docx(stream)
    if lower.endswith(".txt") or lower.endswith(".md"):
        return _load_text(stream)

    raise ValueError(f"Unsupported file type: {filename}")

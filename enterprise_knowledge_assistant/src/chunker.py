"""Text chunking.

Splits a document into overlapping chunks small enough to embed while
preserving enough context for retrieval. The splitter is paragraph-aware
to avoid breaking sentences whenever possible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass
class Chunk:
    document: str
    page: int
    index: int
    text: str


_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_long_paragraph(paragraph: str, max_chars: int) -> List[str]:
    sentences = _SENTENCE_SPLIT.split(paragraph)
    blocks: List[str] = []
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        if len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                blocks.append(current)
            current = sentence
    if current:
        blocks.append(current)
    return blocks


def _chunk_text(text: str, size: int, overlap: int) -> List[str]:
    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]
    units: List[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= size:
            units.append(paragraph)
        else:
            units.extend(_split_long_paragraph(paragraph, size))

    chunks: List[str] = []
    buffer = ""
    for unit in units:
        candidate = f"{buffer}\n\n{unit}".strip() if buffer else unit
        if len(candidate) <= size:
            buffer = candidate
            continue
        if buffer:
            chunks.append(buffer)
        # carry the tail of the previous chunk to keep context across boundaries
        tail = buffer[-overlap:] if overlap and buffer else ""
        buffer = f"{tail}\n{unit}".strip() if tail else unit

    if buffer:
        chunks.append(buffer)
    return chunks


def build_chunks(
    document_name: str,
    pages: Iterable[Tuple[int, str]],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Chunk]:
    output: List[Chunk] = []
    running_index = 0
    for page_number, page_text in pages:
        for piece in _chunk_text(page_text, chunk_size, chunk_overlap):
            output.append(
                Chunk(
                    document=document_name,
                    page=page_number,
                    index=running_index,
                    text=piece,
                )
            )
            running_index += 1
    return output

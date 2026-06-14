"""In-memory vector store.

A lightweight replacement for ChromaDB or FAISS that works reliably on
Streamlit Community Cloud (no native dependencies, no SQLite quirks).
The index is held in the user's session, so each visitor gets an isolated
knowledge base.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .chunker import Chunk


@dataclass
class SearchHit:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(self) -> None:
        self._chunks: List[Chunk] = []
        self._matrix: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def documents(self) -> List[str]:
        return sorted({chunk.document for chunk in self._chunks})

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) == 0:
            return
        if vectors.shape[0] != len(chunks):
            raise ValueError("Chunk / vector length mismatch")

        self._chunks.extend(chunks)
        self._matrix = vectors if self._matrix is None else np.vstack([self._matrix, vectors])

    def remove_document(self, document: str) -> None:
        if self._matrix is None:
            return
        keep = [i for i, chunk in enumerate(self._chunks) if chunk.document != document]
        self._chunks = [self._chunks[i] for i in keep]
        self._matrix = self._matrix[keep] if keep else None

    def clear(self) -> None:
        self._chunks = []
        self._matrix = None

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[SearchHit]:
        if self._matrix is None or len(self._chunks) == 0:
            return []
        scores = self._matrix @ query_vector
        order = np.argsort(-scores)[: max(1, top_k)]
        return [SearchHit(chunk=self._chunks[i], score=float(scores[i])) for i in order]

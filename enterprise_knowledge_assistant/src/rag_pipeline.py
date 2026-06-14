"""End-to-end RAG orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Sequence

from .chunker import Chunk
from .embeddings import EmbeddingsClient
from .llm_client import LLMClient
from .vector_store import SearchHit, VectorStore


SYSTEM_PROMPT = (
    "You are an enterprise knowledge assistant. Answer the user's question "
    "using only the context excerpts provided. If the context is "
    "insufficient, say so plainly. Always cite the supporting excerpts "
    "using the [n] markers that appear in the context. Keep answers "
    "concise, factual, and free of speculation."
)


@dataclass
class RagAnswer:
    sources: List[SearchHit]
    iterator: Iterator[str]


class RagPipeline:
    def __init__(
        self,
        store: VectorStore,
        embeddings: EmbeddingsClient,
        llm: LLMClient,
        top_k: int = 5,
    ) -> None:
        self._store = store
        self._embeddings = embeddings
        self._llm = llm
        self._top_k = top_k

    def index(self, chunks: Sequence[Chunk]) -> None:
        vectors = self._embeddings.embed([chunk.text for chunk in chunks])
        self._store.add(chunks, vectors)

    def answer(self, question: str) -> RagAnswer:
        if len(self._store) == 0:
            def empty() -> Iterator[str]:
                yield (
                    "No documents are indexed yet. Upload one or more files "
                    "in the sidebar to start asking questions."
                )
            return RagAnswer(sources=[], iterator=empty())

        query_vector = self._embeddings.embed_query(question)
        hits = self._store.search(query_vector, top_k=self._top_k)

        context_blocks = []
        for position, hit in enumerate(hits, start=1):
            header = f"[{position}] {hit.chunk.document} — page {hit.chunk.page}"
            context_blocks.append(f"{header}\n{hit.chunk.text}")
        context = "\n\n".join(context_blocks)

        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Context excerpts:\n{context}\n\n"
            "Write the answer below. Cite excerpts with their bracketed numbers."
        )

        return RagAnswer(
            sources=hits,
            iterator=self._llm.stream_answer(SYSTEM_PROMPT, user_prompt),
        )

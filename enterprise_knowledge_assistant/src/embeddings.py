"""Gemini embeddings.

Wraps the Google Gen AI client to produce dense vectors for a batch of
texts. We default to ``text-embedding-004`` which returns 768-dimensional
vectors and is well supported by Streamlit Community Cloud.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
from google import genai
from google.genai import types


_BATCH_SIZE = 64


class EmbeddingsClient:
    def __init__(self, api_key: str, model: str = "gemini-embedding-001") -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for embeddings.")
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def embed(self, texts: Sequence[str], task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        vectors: List[List[float]] = []
        for start in range(0, len(texts), _BATCH_SIZE):
            batch = list(texts[start : start + _BATCH_SIZE])
            response = self._client.models.embed_content(
                model=self._model,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=768,
                ),
            )
            for embedding in response.embeddings:
                vectors.append(list(embedding.values))

        matrix = np.asarray(vectors, dtype=np.float32)
        # Normalise so cosine similarity reduces to a dot product
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text], task_type="RETRIEVAL_QUERY")[0]

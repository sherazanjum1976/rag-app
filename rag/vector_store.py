"""
vector_store.py

FAISS-backed in-memory vector store. Holds chunk embeddings plus their
associated metadata (text + page numbers) so retrieved vectors can be
traced back to their source chunk.

The index is kept entirely in memory / Streamlit session state - nothing
is written to disk, which matches Streamlit Community Cloud's ephemeral
filesystem (see README "Limitations" section).
"""

from dataclasses import dataclass
from typing import List

import numpy as np

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'faiss-cpu' package is not installed. Please check requirements.txt."
    ) from exc

from rag.text_chunker import Chunk


class VectorStoreError(Exception):
    """Raised on FAISS index build/search failures."""


@dataclass
class SearchResult:
    """A single retrieved chunk with its similarity score."""
    chunk: Chunk
    score: float


class FAISSVectorStore:
    """
    Wraps a FAISS IndexFlatIP (inner product) index.

    Embeddings are expected to already be L2-normalized (see
    embeddings.EmbeddingModel), which makes inner-product search
    equivalent to cosine similarity.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: List[Chunk] = []

    def build(self, embeddings: np.ndarray, chunks: List[Chunk]) -> None:
        """
        Build the index from scratch with the given embeddings and chunks.

        Args:
            embeddings: float32 array of shape (n, dimension).
            chunks: list of Chunk objects, same length and order as embeddings.
        """
        if embeddings.shape[0] != len(chunks):
            raise VectorStoreError(
                "Number of embeddings does not match number of chunks "
                f"({embeddings.shape[0]} vs {len(chunks)})."
            )
        if embeddings.shape[0] == 0:
            raise VectorStoreError("No chunks were produced from this document.")
        if embeddings.shape[1] != self.dimension:
            raise VectorStoreError(
                f"Embedding dimension mismatch: index expects {self.dimension}, "
                f"got {embeddings.shape[1]}."
            )

        try:
            # Reset in case build() is called again (e.g. new document uploaded).
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(embeddings)
            self.chunks = list(chunks)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Failed to build FAISS index: {exc}") from exc

    @property
    def is_empty(self) -> bool:
        return self.index.ntotal == 0

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> List[SearchResult]:
        """
        Search for the top_k most similar chunks to the query embedding.

        Returns an empty list if the store has not been built yet.
        """
        if self.is_empty:
            return []

        query_vector = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
        k = min(top_k, self.index.ntotal)

        try:
            scores, indices = self.index.search(query_vector, k)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"FAISS search failed: {exc}") from exc

        results: List[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(SearchResult(chunk=self.chunks[idx], score=float(score)))
        return results

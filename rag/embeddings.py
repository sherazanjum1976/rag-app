"""
embeddings.py

Generates vector embeddings for text using an open-source Sentence
Transformers model. Model loading is cached (via Streamlit's resource
cache) so the ~90MB model is downloaded/loaded only once per session,
not on every rerun.
"""

from typing import List

import numpy as np
import streamlit as st

import config


class EmbeddingError(Exception):
    """Raised when the embedding model fails to load or encode text."""


@st.cache_resource(show_spinner="Loading embedding model (first run only)...")
def _load_model(model_name: str):
    """
    Load and cache a SentenceTransformer model.

    Cached with st.cache_resource so the model is loaded into memory only
    once per app process, not once per user interaction/rerun.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbeddingError(
            "The 'sentence-transformers' package is not installed. "
            "Please check requirements.txt."
        ) from exc

    try:
        return SentenceTransformer(model_name)
    except Exception as exc:  # noqa: BLE001
        raise EmbeddingError(
            f"Failed to load embedding model '{model_name}': {exc}"
        ) from exc


class EmbeddingModel:
    """Thin wrapper around a cached SentenceTransformer model."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self._model = _load_model(self.model_name)

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of texts (e.g. document chunks).

        Returns:
            A float32 numpy array of shape (len(texts), dimension),
            L2-normalized so that inner-product search is equivalent to
            cosine similarity.
        """
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")
        try:
            embeddings = self._model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(f"Failed to generate embeddings: {exc}") from exc

        return embeddings.astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single user query. Returns a float32 array of shape (dimension,)."""
        return self.embed_texts([query])[0]

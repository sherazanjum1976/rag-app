"""
retriever.py

Given a user question, embeds the query and retrieves the most relevant
chunks from the FAISS vector store. Covers: Query Embedding -> Similarity
Search -> Relevant Context Retrieval.
"""

from typing import List

import config
from rag.embeddings import EmbeddingModel
from rag.vector_store import FAISSVectorStore, SearchResult


class Retriever:
    """Combines an embedding model and a vector store to answer retrieval queries."""

    def __init__(self, embedding_model: EmbeddingModel, vector_store: FAISSVectorStore):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = config.TOP_K,
        min_score: float = config.MIN_SIMILARITY_SCORE,
    ) -> List[SearchResult]:
        """
        Retrieve the top_k most relevant chunks for a query, filtering out
        results below min_score (a weak similarity usually means the
        document does not actually contain relevant information).
        """
        if not query or not query.strip():
            return []

        query_embedding = self.embedding_model.embed_query(query)
        results = self.vector_store.search(query_embedding, top_k=top_k)
        return [r for r in results if r.score >= min_score]

"""
pipeline.py

Ties together every stage of the RAG workflow into a single object that
the Streamlit app can drive:

PDF Upload -> PDF Extraction -> Text Cleaning -> Chunking ->
Tokenization/Length Control -> Embeddings -> FAISS Vector Store ->
User Query -> Query Embedding -> Similarity Search -> Relevant Context
Retrieval -> Prompt Construction -> Groq LLM -> Final Answer
"""

from dataclasses import dataclass
from typing import BinaryIO, List

import config
from rag.embeddings import EmbeddingError, EmbeddingModel
from rag.llm import GroqLLM, LLMError
from rag.pdf_processor import PDFProcessingError, extract_pages
from rag.retriever import Retriever
from rag.text_chunker import Chunk, chunk_pages
from rag.vector_store import FAISSVectorStore, SearchResult, VectorStoreError


class PipelineError(Exception):
    """A user-facing error raised at any stage of the RAG pipeline."""


@dataclass
class ProcessingStats:
    """Summary statistics returned after processing a document."""
    num_pages: int
    num_chunks: int
    filename: str


@dataclass
class AnswerResult:
    """The result of answering a question: the answer text plus its sources."""
    answer: str
    sources: List[SearchResult]


class RAGPipeline:
    """
    Stateful RAG pipeline for a single document session.

    A new RAGPipeline (or a call to process_document) should be created
    each time the user uploads a new PDF; the previous FAISS index and
    chunks are discarded.
    """

    def __init__(self):
        self.embedding_model = EmbeddingModel(config.EMBEDDING_MODEL)
        self.vector_store = FAISSVectorStore(dimension=self.embedding_model.dimension)
        self.retriever = Retriever(self.embedding_model, self.vector_store)
        self.chunks: List[Chunk] = []
        self.is_ready: bool = False
        self.document_name: str | None = None

    def process_document(
        self,
        file_obj: BinaryIO,
        filename: str,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
    ) -> ProcessingStats:
        """
        Run the full ingestion pipeline on an uploaded PDF: extract, clean,
        chunk, embed, and index. Raises PipelineError on any failure with a
        user-friendly message.
        """
        try:
            pages = extract_pages(file_obj, max_size_mb=config.MAX_PDF_SIZE_MB)
        except PDFProcessingError as exc:
            raise PipelineError(str(exc)) from exc

        chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            raise PipelineError(
                "No usable text chunks could be produced from this document."
            )

        try:
            embeddings = self.embedding_model.embed_texts([c.text for c in chunks])
        except EmbeddingError as exc:
            raise PipelineError(str(exc)) from exc

        try:
            self.vector_store.build(embeddings, chunks)
        except VectorStoreError as exc:
            raise PipelineError(str(exc)) from exc

        self.chunks = chunks
        self.is_ready = True
        self.document_name = filename

        return ProcessingStats(
            num_pages=len(pages), num_chunks=len(chunks), filename=filename
        )

    def answer_question(
        self,
        question: str,
        groq_api_key: str | None = None,
        top_k: int = config.TOP_K,
    ) -> AnswerResult:
        """
        Retrieve relevant context for `question` and generate a grounded
        answer via Groq. Raises PipelineError on any failure.
        """
        if not question or not question.strip():
            raise PipelineError("Please enter a question.")

        if not self.is_ready or self.vector_store.is_empty:
            raise PipelineError(
                "Please upload and process a PDF document before asking questions."
            )

        try:
            results = self.retriever.retrieve(question, top_k=top_k)
        except (EmbeddingError, VectorStoreError) as exc:
            raise PipelineError(str(exc)) from exc

        try:
            llm = GroqLLM(api_key=groq_api_key)
            answer = llm.generate_answer(question, results)
        except LLMError as exc:
            raise PipelineError(str(exc)) from exc

        return AnswerResult(answer=answer, sources=results)

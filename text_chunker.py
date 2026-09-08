"""
text_chunker.py

Splits cleaned page text into overlapping, appropriately-sized chunks
ready for embedding. This stage covers: Chunking -> Tokenization/Length
Control in the RAG workflow.

Token length is estimated using a simple, dependency-free heuristic
(~4 characters per token for English text) rather than a model-specific
tokenizer. This avoids requiring a network call to download tokenizer
files at runtime (important for a reliable Streamlit Cloud deployment)
while still giving a reasonably accurate control over chunk size.
"""

import re
from dataclasses import dataclass
from typing import List

from rag.pdf_processor import PageContent

CHARS_PER_TOKEN_ESTIMATE = 4


@dataclass
class Chunk:
    """A chunk of text ready for embedding, with traceable page metadata."""
    chunk_id: int
    text: str
    start_page: int
    end_page: int

    @property
    def page_label(self) -> str:
        if self.start_page == self.end_page:
            return f"page {self.start_page}"
        return f"pages {self.start_page}-{self.end_page}"


def estimate_tokens(text: str) -> int:
    """Rough token-count estimate (~4 chars/token) used for length control."""
    return max(1, len(text) // CHARS_PER_TOKEN_ESTIMATE)


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def _split_into_sentences(text: str) -> List[str]:
    """Best-effort sentence splitter (regex-based, no extra dependencies)."""
    if not text:
        return []
    sentences = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_pages(
    pages: List[PageContent],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Chunk]:
    """
    Convert extracted page text into overlapping chunks.

    Args:
        pages: List of PageContent from pdf_processor.extract_pages.
        chunk_size: Target maximum chunk size in estimated tokens.
        chunk_overlap: Number of estimated tokens to carry over from the
            end of one chunk into the start of the next, for continuity.

    Returns:
        List of Chunk objects, each tagged with the page(s) it was drawn from.
    """
    if chunk_overlap >= chunk_size:
        # Guard against misconfiguration that would cause an infinite loop
        # or non-progressing overlap.
        chunk_overlap = max(0, chunk_size // 4)

    # Flatten to a list of (page_number, sentence) preserving document order.
    sentence_units: List[tuple[int, str]] = []
    for page in pages:
        if not page.text:
            continue
        for sentence in _split_into_sentences(page.text):
            sentence_units.append((page.page_number, sentence))

    chunks: List[Chunk] = []
    if not sentence_units:
        return chunks

    current_sentences: List[tuple[int, str]] = []
    current_tokens = 0
    chunk_id = 0

    def flush_chunk() -> List[tuple[int, str]]:
        """Finalize current_sentences into a Chunk; return the overlap tail."""
        nonlocal chunk_id
        if not current_sentences:
            return []

        text = " ".join(s for _, s in current_sentences)
        pages_in_chunk = [p for p, _ in current_sentences]
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=text,
                start_page=min(pages_in_chunk),
                end_page=max(pages_in_chunk),
            )
        )
        chunk_id += 1

        # Build the overlap tail: walk backwards from the end, keeping
        # sentences until we reach ~chunk_overlap tokens.
        tail: List[tuple[int, str]] = []
        tail_tokens = 0
        for item in reversed(current_sentences):
            t = estimate_tokens(item[1])
            if tail_tokens + t > chunk_overlap and tail:
                break
            tail.insert(0, item)
            tail_tokens += t
        return tail

    for page_number, sentence in sentence_units:
        sentence_tokens = estimate_tokens(sentence)

        # If a single sentence is larger than the whole chunk size, hard-split
        # it on whitespace so we never produce an unembeddable oversized chunk.
        if sentence_tokens > chunk_size:
            words = sentence.split(" ")
            piece: List[str] = []
            piece_tokens = 0
            for word in words:
                wt = estimate_tokens(word + " ")
                if piece_tokens + wt > chunk_size and piece:
                    current_sentences.append((page_number, " ".join(piece)))
                    current_tokens += piece_tokens
                    tail = flush_chunk()
                    current_sentences = tail
                    current_tokens = sum(estimate_tokens(s) for _, s in tail)
                    piece, piece_tokens = [], 0
                piece.append(word)
                piece_tokens += wt
            if piece:
                sentence = " ".join(piece)
                sentence_tokens = piece_tokens
            else:
                continue

        if current_tokens + sentence_tokens > chunk_size and current_sentences:
            tail = flush_chunk()
            current_sentences = tail
            current_tokens = sum(estimate_tokens(s) for _, s in tail)

        current_sentences.append((page_number, sentence))
        current_tokens += sentence_tokens

    # Flush whatever remains.
    if current_sentences:
        text = " ".join(s for _, s in current_sentences)
        pages_in_chunk = [p for p, _ in current_sentences]
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=text,
                start_page=min(pages_in_chunk),
                end_page=max(pages_in_chunk),
            )
        )

    return chunks

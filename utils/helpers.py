"""
helpers.py

Small, dependency-free helper functions used by the Streamlit UI layer
(app.py) for formatting and validation. Kept separate from the RAG
pipeline modules to preserve separation of concerns.
"""

from typing import List

from rag.vector_store import SearchResult


def validate_uploaded_file(uploaded_file) -> str | None:
    """
    Basic client-side validation of a Streamlit UploadedFile before it is
    handed to the pipeline.

    Returns:
        None if the file looks valid, otherwise a human-readable error message.
    """
    if uploaded_file is None:
        return "No file was uploaded."

    name = uploaded_file.name or ""
    if not name.lower().endswith(".pdf"):
        return "Please upload a file with a .pdf extension."

    if uploaded_file.size == 0:
        return "The uploaded file is empty."

    return None


def format_sources_markdown(sources: List[SearchResult]) -> str:
    """
    Format retrieved chunks as a Markdown block for display under an answer,
    e.g.:

    **Sources:**
    - Excerpt 1 (page 3) - similarity 0.82
    - Excerpt 2 (pages 4-5) - similarity 0.77
    """
    if not sources:
        return "_No supporting excerpts met the relevance threshold._"

    lines = []
    for i, result in enumerate(sources, start=1):
        lines.append(
            f"- Excerpt {i} ({result.chunk.page_label}) "
            f"- relevance {result.score:.2f}"
        )
    return "\n".join(lines)


def truncate(text: str, max_chars: int = 400) -> str:
    """Truncate text to max_chars, appending an ellipsis if shortened."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def format_file_size(size_bytes: int) -> str:
    """Format a byte count as a human-readable string (e.g. '2.3 MB')."""
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"

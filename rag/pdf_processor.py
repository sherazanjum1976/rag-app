"""
pdf_processor.py

Responsible for extracting text from an uploaded PDF file, page by page,
and cleaning/normalizing that text. This is the first stage of the RAG
pipeline: PDF Upload -> PDF Extraction -> Text Cleaning.
"""

import re
from dataclasses import dataclass
from typing import BinaryIO, List

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class PDFProcessingError(Exception):
    """Raised when a PDF cannot be read or contains no usable text."""


@dataclass
class PageContent:
    """Extracted and cleaned text for a single PDF page."""
    page_number: int  # 1-indexed, for human-friendly display/citation
    text: str


def _clean_text(raw_text: str) -> str:
    """
    Normalize extracted PDF text:
      - Collapse repeated whitespace/newlines introduced by PDF layout.
      - Strip common ligature/encoding artifacts.
      - Trim leading/trailing whitespace.
    """
    if not raw_text:
        return ""

    text = raw_text.replace("\x00", " ")
    # Join hyphenated line-breaks: "exam-\nple" -> "example"
    text = re.sub(r"-\n(?=[a-z])", "", text)
    # Collapse newlines into spaces (PDF text layout often breaks mid-sentence)
    text = re.sub(r"[ \t]*\n[ \t]*", " ", text)
    # Collapse multiple spaces/tabs into a single space
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Remove stray control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

    return text.strip()


def extract_pages(file_obj: BinaryIO, max_size_mb: int = 50) -> List[PageContent]:
    """
    Extract and clean text from every page of a PDF.

    Args:
        file_obj: A file-like object (e.g. Streamlit's UploadedFile) opened
            in binary mode, positioned at the start of the file.
        max_size_mb: Maximum allowed file size in megabytes.

    Returns:
        A list of PageContent objects, one per page (including pages with
        empty text, so page numbering stays accurate).

    Raises:
        PDFProcessingError: if the file is too large, not a valid PDF,
            encrypted without a usable password, or contains no
            extractable text at all (e.g. a fully scanned/image-only PDF).
    """
    # --- Size check -------------------------------------------------------
    file_obj.seek(0, 2)  # seek to end
    size_bytes = file_obj.tell()
    file_obj.seek(0)
    size_mb = size_bytes / (1024 * 1024)
    if size_bytes == 0:
        raise PDFProcessingError("The uploaded file is empty.")
    if size_mb > max_size_mb:
        raise PDFProcessingError(
            f"The uploaded PDF is {size_mb:.1f} MB, which exceeds the "
            f"{max_size_mb} MB limit. Please upload a smaller file."
        )

    # --- Parse PDF ----------------------------------------------------------
    try:
        reader = PdfReader(file_obj)
    except PdfReadError as exc:
        raise PDFProcessingError(
            "This file could not be read as a PDF. It may be corrupted or "
            "not a valid PDF file."
        ) from exc
    except Exception as exc:  # noqa: BLE001 - surface any parser failure clearly
        raise PDFProcessingError(f"Failed to open the PDF file: {exc}") from exc

    if reader.is_encrypted:
        # Try an empty password first (some PDFs are "encrypted" only to
        # restrict editing, not to require a password to open/read).
        try:
            reader.decrypt("")
        except Exception:
            pass
        if reader.is_encrypted:
            raise PDFProcessingError(
                "This PDF is password-protected. Please upload an "
                "unprotected PDF."
            )

    if len(reader.pages) == 0:
        raise PDFProcessingError("This PDF contains no pages.")

    pages: List[PageContent] = []
    total_chars = 0
    for i, page in enumerate(reader.pages):
        try:
            raw_text = page.extract_text() or ""
        except Exception:
            raw_text = ""
        cleaned = _clean_text(raw_text)
        total_chars += len(cleaned)
        pages.append(PageContent(page_number=i + 1, text=cleaned))

    if total_chars < 20:
        raise PDFProcessingError(
            "No readable text could be extracted from this PDF. It may be "
            "a scanned/image-only document. Please upload a PDF with "
            "selectable text, or run OCR on it first."
        )

    return pages

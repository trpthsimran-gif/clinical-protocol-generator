"""
pdf_ingest.py

Handles the "Clinical Protocol PDFs + Other Medical Documents" part of
the pipeline: extract text from an uploaded PDF so it can be chunked
and embedded into the vector database (vector_store.py).
"""

from typing import List, Dict
from pypdf import PdfReader

from vector_store import chunk_text, add_documents


def extract_text_from_pdf(file) -> str:
    """
    Extract all text from an uploaded PDF file (Streamlit's UploadedFile
    object, or any file-like object PdfReader can accept).
    """
    reader = PdfReader(file)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)
    return "\n".join(pages_text)


def ingest_pdf(file, source_name: str) -> int:
    """
    Full pipeline for one PDF: extract -> chunk -> embed -> store.
    Returns the number of chunks added to the vector database.
    """
    raw_text = extract_text_from_pdf(file)
    if not raw_text.strip():
        return 0

    chunks = chunk_text(raw_text, chunk_size=500, overlap=50)
    metadatas: List[Dict] = [
        {"source": source_name, "type": "pdf", "chunk_index": i}
        for i in range(len(chunks))
    ]
    return add_documents(chunks, metadatas)

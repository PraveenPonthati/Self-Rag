"""
PDF loading and chunking module.

Uses PyMuPDF (fitz) to extract text from PDF files, then splits the text
into overlapping chunks using LangChain's RecursiveCharacterTextSplitter.
"""

import os
from typing import List

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def extract_text_from_pdf(file_path: str) -> List[Document]:
    """
    Open a PDF with PyMuPDF and extract all text, one Document per page.

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        A list of LangChain Document objects, one per page, with metadata
        containing the source filename and page number.
    """
    documents: List[Document] = []

    # Open the PDF file using PyMuPDF
    pdf = fitz.open(file_path)
    filename = os.path.basename(file_path)

    for page_num in range(len(pdf)):
        page = pdf[page_num]
        text = page.get_text("text")  # extract plain text from the page

        # Skip blank pages
        if not text.strip():
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": filename,
                    "page": page_num + 1,  # 1-indexed for display
                },
            )
        )

    pdf.close()
    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split page-level Documents into smaller overlapping chunks.

    Smaller chunks improve retrieval precision while the overlap prevents
    important context from being cut at a boundary.

    Args:
        documents: List of Documents (usually page-level).

    Returns:
        List of smaller chunk Documents, preserving source metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Try to split on paragraph/sentence boundaries first
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    return chunks


def load_and_chunk_pdf(file_path: str) -> List[Document]:
    """
    Convenience wrapper: load a PDF and return ready-to-index chunks.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of chunk Documents ready to be added to ChromaDB.
    """
    pages = extract_text_from_pdf(file_path)
    chunks = chunk_documents(pages)
    return chunks

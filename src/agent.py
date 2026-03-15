"""
Agent runner module.

Provides a clean interface to:
  - Index PDF files into ChromaDB
  - Run the Self-RAG LangGraph pipeline against a user question
  - Return a structured result dict for the Streamlit frontend
"""

import os
from typing import Any, Dict, List

from langchain_core.documents import Document

from src.config import MAX_RETRIES
from src.graph import rag_graph
from src.graph_state import GraphState
from src.pdf_loader import load_and_chunk_pdf
from src.vector_store import add_documents, get_indexed_sources


def index_pdf(file_path: str) -> Dict[str, Any]:
    """
    Load, chunk, and index a PDF file into ChromaDB.

    Args:
        file_path: Path to the uploaded PDF file.

    Returns:
        A dict with:
          - success (bool)
          - chunks_indexed (int)
          - filename (str)
          - error (str | None)
    """
    try:
        chunks = load_and_chunk_pdf(file_path)
        if not chunks:
            return {
                "success": False,
                "chunks_indexed": 0,
                "filename": os.path.basename(file_path),
                "error": "No text could be extracted from the PDF.",
            }

        count = add_documents(chunks)
        return {
            "success": True,
            "chunks_indexed": count,
            "filename": os.path.basename(file_path),
            "error": None,
        }
    except Exception as exc:
        return {
            "success": False,
            "chunks_indexed": 0,
            "filename": os.path.basename(file_path),
            "error": str(exc),
        }


def run_rag_query(question: str) -> Dict[str, Any]:
    """
    Run the Self-RAG LangGraph pipeline for a user question.

    Args:
        question: The natural language question from the user.

    Returns:
        A dict with:
          - answer (str)              — final generated answer
          - sources (List[Document])  — relevant chunks used
          - hallucination_score (str) — 'yes' = grounded, 'no' = hallucinated
          - answer_grade (str)        — 'yes' = addresses question
          - retry_count (int)         — how many rewrites occurred
          - final_question (str)      — the (possibly rewritten) question used
          - error (str | None)        — error message if something went wrong
    """
    # Build the initial state
    initial_state: GraphState = {
        "question": question,
        "original_question": question,
        "documents": [],
        "relevant_documents": [],
        "generation": "",
        "hallucination_score": "",
        "answer_grade": "",
        "retry_count": 0,
        "error": None,
    }

    try:
        # Invoke the compiled LangGraph app
        final_state = rag_graph.invoke(initial_state)

        generation = final_state.get("generation", "")
        relevant_docs = final_state.get("relevant_documents", [])

        # If no answer was generated, compose a helpful fallback message
        if not generation:
            generation = (
                "I could not find relevant information in the uploaded documents "
                "to answer your question. Please try rephrasing or upload "
                "additional PDFs."
            )

        return {
            "answer": generation,
            "sources": relevant_docs,
            "hallucination_score": final_state.get("hallucination_score", ""),
            "answer_grade": final_state.get("answer_grade", ""),
            "retry_count": final_state.get("retry_count", 0),
            "final_question": final_state.get("question", question),
            "error": final_state.get("error"),
        }

    except Exception as exc:
        return {
            "answer": "",
            "sources": [],
            "hallucination_score": "",
            "answer_grade": "",
            "retry_count": 0,
            "final_question": question,
            "error": str(exc),
        }


def list_indexed_sources() -> List[str]:
    """Return a list of PDF filenames currently indexed in ChromaDB."""
    return get_indexed_sources()

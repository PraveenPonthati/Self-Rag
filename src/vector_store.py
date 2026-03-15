"""
Vector store module — manages ChromaDB for document storage and retrieval.

ChromaDB is used as a local, persistent vector database.
Google's embedding-001 model produces the embeddings via langchain-google-genai.
"""

from typing import List

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    GOOGLE_API_KEY,
    TOP_K,
)

# ── Embedding function (shared across all operations) ────────────────────────
_embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=GOOGLE_API_KEY,
)


def get_vector_store() -> Chroma:
    """
    Return a Chroma vector store instance backed by the local persist directory.
    Creates the directory and collection on first use.
    """
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=_embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )


def add_documents(documents: List[Document]) -> int:
    """
    Add a list of Document chunks to ChromaDB.

    Args:
        documents: Chunked Documents to index.

    Returns:
        Number of documents successfully added.
    """
    store = get_vector_store()
    store.add_documents(documents)
    return len(documents)


def retrieve_documents(query: str, k: int = TOP_K) -> List[Document]:
    """
    Perform a similarity search and return the top-k matching chunks.

    Args:
        query: Natural language question.
        k: Number of results to return.

    Returns:
        List of the most relevant Document chunks.
    """
    store = get_vector_store()
    return store.similarity_search(query, k=k)


def get_indexed_sources() -> List[str]:
    """
    Return a deduplicated list of source filenames currently indexed.
    Useful for the Streamlit sidebar to show what's been uploaded.
    """
    store = get_vector_store()
    try:
        collection = store._collection
        results = collection.get(include=["metadatas"])
        sources = {m.get("source", "unknown") for m in results["metadatas"]}
        return sorted(sources)
    except Exception:
        return []


def clear_collection() -> None:
    """Delete all documents in the ChromaDB collection."""
    store = get_vector_store()
    store.delete_collection()

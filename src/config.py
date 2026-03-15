"""
Configuration module — loads environment variables from Streamlit secrets.
"""

import os
import streamlit as st

# Load all Streamlit secrets into environment variables
for key, value in st.secrets.items():
    os.environ[key] = str(value)

# ── Google Gemini ─────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    raise EnvironmentError(
        "GOOGLE_API_KEY is not set. "
        "Add it to your Streamlit Cloud secrets."
    )

# Gemini model for chat/generation
LLM_MODEL = "gemini-2.5-flash"

# Gemini embedding model
EMBEDDING_MODEL = "gemini-embedding-001"

# ── LangSmith tracing (optional) ─────────────────────────────────────────────
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "self-rag-agent")

if LANGCHAIN_TRACING_V2 and LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

# ── Vector store ─────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
CHROMA_COLLECTION_NAME = "self_rag_docs"

# ── Retrieval parameters ──────────────────────────────────────────────────────
TOP_K = 5
CHUNK_SIZE = 500
CHUNK_OVERLAP = 200

# ── Agent limits ─────────────────────────────────────────────────────────────
MAX_RETRIES = 2
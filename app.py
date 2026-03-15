"""
Streamlit frontend for the Self-RAG Agent.

Layout:
  Sidebar  — PDF upload and list of indexed sources
  Main     — Question input, answer display, and pipeline diagnostics
"""

import os
import sys
import tempfile

import streamlit as st

# Make sure the project root is on the Python path
sys.path.insert(0, os.path.dirname(__file__))

from src.agent import index_pdf, list_indexed_sources, run_rag_query

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Self-RAG Agent",
    page_icon="🔍",
    layout="wide",
)

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🔍 Self-RAG Agent")
st.caption(
    "Upload PDF documents, ask questions, and let the agent retrieve, "
    "grade, generate, and verify answers automatically."
)

# ── Sidebar — PDF management ──────────────────────────────────────────────────
with st.sidebar:
    st.header("📄 Document Management")

    # PDF uploader
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files to add them to the knowledge base.",
    )

    if uploaded_files:
        if st.button("Index uploaded PDFs", type="primary"):
            for uploaded_file in uploaded_files:
                with st.spinner(f"Indexing {uploaded_file.name}…"):
                    # Save to a temp file so PyMuPDF can open it
                    with tempfile.NamedTemporaryFile(
                        suffix=".pdf", delete=False
                    ) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    result = index_pdf(tmp_path)
                    os.unlink(tmp_path)  # clean up temp file

                    if result["success"]:
                        st.success(
                            f"✅ {result['filename']} — "
                            f"{result['chunks_indexed']} chunks indexed"
                        )
                    else:
                        st.error(
                            f"❌ {result['filename']} — {result['error']}"
                        )

    # Show currently indexed documents
    st.divider()
    st.subheader("📚 Indexed Documents")
    sources = list_indexed_sources()
    if sources:
        for src in sources:
            st.markdown(f"- `{src}`")
    else:
        st.info("No documents indexed yet. Upload a PDF above.")

    # How the pipeline works
    st.divider()
    st.subheader("ℹ️ Pipeline")
    st.markdown(
        """
1. **Retrieve** — fetch top-5 chunks  
2. **Grade** — filter irrelevant chunks  
3. **Generate** — produce an answer  
4. **Hallucination check** — verify grounding  
5. **Rewrite & retry** — if quality fails  
        """
    )

# ── Main — Q&A interface ──────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])

with col1:
    question = st.text_area(
        "Ask a question about your documents",
        placeholder="e.g. What are the main findings of the paper?",
        height=100,
    )

with col2:
    st.write("")  # vertical spacer
    st.write("")
    run_button = st.button("🚀 Ask", type="primary", use_container_width=True)
    clear_button = st.button("🗑️ Clear", use_container_width=True)

if clear_button:
    st.rerun()

# ── Run pipeline ──────────────────────────────────────────────────────────────
if run_button and question.strip():

    # Verify there are documents to query
    if not list_indexed_sources():
        st.warning("⚠️ Please upload and index at least one PDF before asking a question.")
        st.stop()

    with st.spinner("Running Self-RAG pipeline…"):
        result = run_rag_query(question.strip())

    # ── Error handling ────────────────────────────────────────────────────────
    if result["error"]:
        st.error(f"Pipeline error: {result['error']}")
        st.stop()

    # ── Answer ────────────────────────────────────────────────────────────────
    st.subheader("💬 Answer")
    st.markdown(result["answer"])

    # ── Diagnostic badges ─────────────────────────────────────────────────────
    st.divider()
    diag_col1, diag_col2, diag_col3, diag_col4 = st.columns(4)

    with diag_col1:
        retries = result["retry_count"]
        st.metric("Query Rewrites", retries)

    with diag_col2:
        h_score = result["hallucination_score"]
        grounded = h_score == "yes"
        st.metric(
            "Grounded Answer",
            "✅ Yes" if grounded else "⚠️ No",
        )

    with diag_col3:
        a_grade = result["answer_grade"]
        addresses = a_grade == "yes"
        st.metric(
            "Addresses Question",
            "✅ Yes" if addresses else "⚠️ No",
        )

    with diag_col4:
        st.metric("Chunks Used", len(result["sources"]))

    # ── Rewrite notice ────────────────────────────────────────────────────────
    if result["retry_count"] > 0:
        with st.expander("🔄 Query was rewritten"):
            st.markdown(f"**Original:** {question.strip()}")
            st.markdown(f"**Final:** {result['final_question']}")

    # ── Source documents ──────────────────────────────────────────────────────
    if result["sources"]:
        st.divider()
        with st.expander(f"📄 Source chunks used ({len(result['sources'])})"):
            for i, doc in enumerate(result["sources"], 1):
                source = doc.metadata.get("source", "unknown")
                page = doc.metadata.get("page", "?")
                st.markdown(f"**Chunk {i}** — `{source}` page {page}")
                st.text(doc.page_content[:600] + ("…" if len(doc.page_content) > 600 else ""))
                st.divider()

elif run_button and not question.strip():
    st.warning("Please enter a question before clicking Ask.")

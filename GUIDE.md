# Self-RAG Agent — Project Guide

## Project Structure

```
artifacts/self-rag/
├── app.py                    ← Streamlit frontend entry point
├── requirements.txt          ← All Python dependencies with exact versions
├── .streamlit/
│   └── config.toml           ← Streamlit server config (port 5000, headless)
├── src/
│   ├── __init__.py
│   ├── config.py             ← Environment variable loading, all constants
│   ├── pdf_loader.py         ← PyMuPDF PDF parsing + LangChain text splitting
│   ├── vector_store.py       ← ChromaDB setup, indexing, retrieval helpers
│   ├── graph_state.py        ← LangGraph TypedDict state definition
│   ├── nodes.py              ← All 5 LangGraph node functions + LLM chains
│   ├── graph.py              ← StateGraph assembly, edges, conditional routing
│   └── agent.py              ← High-level API used by the Streamlit frontend
└── chroma_db/                ← Auto-created local ChromaDB storage
```

---

## LangGraph Flow

```
START
  │
  ▼
retrieve          → fetch top-5 chunks from ChromaDB
  │
  ▼
grade_documents   → LLM scores each chunk: relevant or not
  │
  ├─── relevant docs found ──────────────────────────────► generate
  │                                                            │
  └─── no relevant docs (retry < MAX) ──► rewrite_query       ▼
                                               │         check_hallucination
                                               │               │
                                               │   ┌── grounded & good ──► END
                                               │   │
                                               └───┴── not grounded (retry < MAX) ──► rewrite_query
                                                                                          │
                                                                               (exhausted) ▼
                                                                                          END
```

### Conditional routing logic

| From node | Condition | Next node |
|---|---|---|
| `grade_documents` | relevant docs found | `generate` |
| `grade_documents` | no relevant docs, retries remain | `rewrite_query` |
| `grade_documents` | no relevant docs, retries exhausted | `END` |
| `check_hallucination` | grounded (`yes`) AND answers question (`yes`) | `END` |
| `check_hallucination` | any check fails, retries remain | `rewrite_query` |
| `check_hallucination` | any check fails, retries exhausted | `END` |
| `rewrite_query` | always | `retrieve` |

---

## Setup Instructions

### 1. Clone / open the project
The project is already running on Replit. All dependencies are installed.

### 2. Set your OpenAI API key
The key is already stored as `OPENAI_API_KEY` in Replit Secrets.

For **local development**, create a `.env` file in `artifacts/self-rag/`:
```
OPENAI_API_KEY=sk-...your-key-here...
```

### 3. (Optional) Enable LangSmith tracing
Add these to Replit Secrets or your `.env`:
```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls-...your-key...
LANGCHAIN_PROJECT=self-rag-agent
```
Get a free API key at https://smith.langchain.com

### 4. Run the app
On Replit, it starts automatically. Locally:
```bash
cd artifacts/self-rag
pip install -r requirements.txt
streamlit run app.py --server.port 5000
```

### 5. Upload PDFs and ask questions
1. Use the sidebar to upload one or more PDF files
2. Click "Index uploaded PDFs"
3. Type your question in the main area
4. Click "Ask"

---

## How to Get an OpenAI API Key

1. Go to https://platform.openai.com/api-keys
2. Sign in or create an account
3. Click "Create new secret key"
4. Copy the key (it starts with `sk-`)
5. On Replit: paste it into the Secrets tab as `OPENAI_API_KEY`
6. Locally: add `OPENAI_API_KEY=sk-...` to a `.env` file in `artifacts/self-rag/`

---

## 3 Example Questions to Test With

1. **"What are the main conclusions of this document?"**
   — Tests general summarization across the full PDF.

2. **"What methodology was used in the research?"**
   — Tests specific factual retrieval; will trigger rewrite if vague.

3. **"What recommendations does the author make?"**
   — Tests extraction of structured insights; good for grounding/hallucination checks.

---

## Resume Bullet Points

- **Built a Self-RAG (Retrieval-Augmented Generation) agent** using LangGraph and LangChain, implementing a multi-node state machine with automated retrieval grading, hallucination detection, and adaptive query rewriting for high-accuracy document Q&A.
- **Engineered an end-to-end PDF intelligence pipeline** with PyMuPDF for text extraction, ChromaDB for local vector storage, and OpenAI GPT-4o-mini for relevance scoring, answer generation, and grounding verification — reducing hallucinations via self-correction loops.
- **Developed a Streamlit web interface** for the Self-RAG system, enabling users to upload PDF documents, ask natural language questions, and view real-time pipeline diagnostics including chunk grounding scores, query rewrite counts, and source attribution.

<div align="center">

<img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Books.png" alt="Books" width="80" />

# Self-RAG Agent

### *Retrieval-Augmented Generation with Self-Critique & Hallucination Detection*

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.1.2-FF6B35?style=flat-square)](https://langchain-ai.github.io/langgraph/)
[![LangChain](https://img.shields.io/badge/LangChain-1.2.12-1C3C3C?style=flat-square&logo=chainlink&logoColor=white)](https://langchain.com)
[![Gemini](https://img.shields.io/badge/Gemini_2.5_Flash-4285F4?style=flat-square&logo=google&logoColor=white)](https://deepmind.google/technologies/gemini/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5.5-FF4B4B?style=flat-square)](https://trychroma.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.55.0-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

<br/>

> Most RAG systems blindly trust what they retrieve.  
> **This one doesn't.**

<br/>

[Demo](#-demo) · [Features](#-features) · [Architecture](#-architecture) · [Quickstart](#-quickstart) · [Tech Stack](#-tech-stack)

---

</div>

## 📌 What is Self-RAG?

Traditional RAG pipelines follow a simple path: *retrieve → generate → done*. The problem? If the retrieved documents are irrelevant or the generated answer is fabricated, there's nothing to catch it.

**Self-RAG** introduces a critique loop at every stage:

- 🔍 **Retrieves** relevant chunks from your documents
- 🧑‍⚖️ **Grades** each chunk — keeps only the relevant ones
- ✍️ **Generates** an answer grounded in filtered context
- 🔬 **Checks** if the answer is hallucinated or unsupported
- 🔄 **Retries** with a rewritten query if anything fails

The result is an AI that knows when it doesn't know — and tries harder before giving up.

<br/>

## ✨ Features

- 📄 **PDF Upload** — drag and drop any PDF; chunks are stored in a local ChromaDB vector store
- 🤖 **Self-Critiquing Agent** — LangGraph state machine with 5 nodes and conditional routing
- 🛡️ **Hallucination Guard** — answers are verified to be grounded before being returned
- 🔁 **Query Rewriter** — if retrieval fails, the query is intelligently rewritten (up to 2 retries)
- 🌐 **Streamlit UI** — clean, interactive frontend with chat history
- 🔭 **LangSmith Tracing** — full pipeline observability (optional)
- 🔑 **Env-based config** — all secrets managed via `.env`, never hardcoded

<br/>

## 🏗️ Architecture

The agent is a **stateful cyclic graph** built with LangGraph. Here's how a query flows through it:

```
START
  │
  ▼
┌─────────────┐
│   retrieve  │  ← Fetches top-5 chunks from ChromaDB via semantic search
└─────────────┘
  │
  ▼
┌──────────────────┐
│  grade_documents │  ← LLM scores each chunk: relevant or not
└──────────────────┘
  │                    │
  │ relevant chunks    │ no relevant chunks
  ▼                    ▼
┌──────────┐     ┌───────────────┐
│ generate │     │ rewrite_query │  ← Rewrites the question
└──────────┘     └───────────────┘
  │                    │
  ▼                    └──→ back to retrieve (max 2 retries)
┌────────────────────┐
│ check_hallucination│  ← Is the answer grounded? Does it answer the question?
└────────────────────┘
  │                    │
  │ grounded +         │ failed checks
  │ answers question   │
  ▼                    ▼
 END           rewrite_query → retrieve (loop)
```

<br/>

## 🚀 Quickstart

### Prerequisites

- Python 3.11+
- A [Google AI Studio](https://aistudio.google.com) API key (free tier available)

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/self-rag-agent.git
cd self-rag-agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in:

```env
GOOGLE_API_KEY=your_google_api_key_here

# Optional — for LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=self-rag-agent
```

### 4. Run the app

```bash
streamlit run app.py
```

Visit `http://localhost:8501` — upload a PDF and start asking questions.

<br/>

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Agent Framework | LangGraph 1.1.2 | Stateful cyclic agent graph |
| LLM | Google Gemini 2.5 Flash | Grading, generation, hallucination checks |
| Embeddings | Google text-embedding-004 | Semantic vector representations |
| Orchestration | LangChain 1.2.12 | Chains, prompts, output parsers |
| Vector Store | ChromaDB 1.5.5 | Local persistent vector database |
| PDF Parsing | PyMuPDF 1.27.2 | Text extraction from PDFs |
| Text Splitting | LangChain Text Splitters | 1000-char chunks, 200 overlap |
| Frontend | Streamlit 1.55.0 | Interactive web UI |
| Tracing | LangSmith | Pipeline observability (optional) |
| Config | python-dotenv | Environment variable management |

<br/>

## 📁 Project Structure

```
self-rag-agent/
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
└── README.md
```

<br/>

## 💡 Example Questions to Try

Once you upload a PDF, try asking:

1. *"Summarize the key findings of this document."*
2. *"What methodology was used in the study?"* — tests retrieval precision
3. *"What does the author recommend for future work?"* — tests multi-chunk synthesis

<br/>

## 🗺️ Roadmap

- [ ] Multi-PDF support with source attribution
- [ ] Streaming response output in the UI
- [ ] Docker containerization
- [ ] Support for OpenAI / Groq as alternative LLM backends
- [ ] Evaluation dashboard using RAGAS metrics

<br/>

<div align="center">

Built with 🧠 and Python

*If this helped you, consider giving it a ⭐*

</div>

# 🏛️ Mysuru Civic Intelligence System (MCIS)

> **A fully local RAG-powered civic assistant that answers questions from Mysuru municipal documents — no cloud, no API costs, runs on a 16GB laptop.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-blue?style=flat)](https://github.com/facebookresearch/faiss)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black?style=flat)](https://ollama.ai)
[![sentence-transformers](https://img.shields.io/badge/SentenceTransformers-Embeddings-orange?style=flat)](https://sbert.net)

---

## 📌 What This Project Demonstrates

A **fully offline RAG pipeline** — from raw civic documents through to cited, natural-language answers — with a production-style FastAPI layer on top. Everything runs locally using open-source models.

| Capability | Implementation |
|---|---|
| **RAG Pipeline** | Document load → chunk → embed → FAISS index → retrieve → generate |
| **Local LLM Inference** | Mistral 7B (quantized) via Ollama — zero cloud dependency |
| **Vector Search** | FAISS similarity search with configurable top-k retrieval |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Source Citations** | Every answer references the source document chunks |
| **API Design** | FastAPI with chat, health, and admin ingest endpoints |
| **Multi-format Ingestion** | PDF, DOCX, and TXT document support |

---

## 🏗️ Architecture

```
 INGESTION PIPELINE
 ───────────────────────────────────────────────────────
 data/documents/  →  Load  →  Chunk  →  Embed  →  FAISS Index
 (PDF, DOCX, TXT)                                     │
                                                       │
 QUERY PIPELINE                                        │
 ───────────────────────────────────────────────────── │
 User Query  →  Embed  →  FAISS Search  ───────────────┘
                               │
                               ▼
                          Retriever (top-k chunks)
                               │
                               ▼
                     Mistral 7B via Ollama
                     (local inference, no cloud)
                               │
                               ▼
                    Answer + Source Citations
```

---

## ✨ Key Features

### 🔒 Fully Local & Private
No data leaves the machine. Ideal for civic or government document use cases where data privacy matters. Runs comfortably on a **16GB RAM laptop** using a quantized 7B model.

### 📄 Multi-format Document Ingestion
Drop any combination of PDF, DOCX, or TXT files into `data/documents/` and run the ingestion script. The pipeline handles loading, chunking, embedding, and indexing automatically.

### 🔎 Cited Answers
Every response includes references to the source document chunks used to generate it — enabling transparency and trust in the output.

### ⚙️ Configurable & Swappable
Model, chunk size, top-k retrieval, and document paths are all configurable via `settings.yaml` or `.env`. Switch from Mistral to any Ollama-compatible model in one line.

---

## 🧪 Example Usage

**Request:**
```json
POST /chat
{
  "query": "What are the property tax rates in Mysuru?",
  "top_k": 5,
  "include_sources": true
}
```

**Response:**
```json
{
  "answer": "Property tax in Mysuru is calculated based on...",
  "sources": [
    { "document": "mysuru_property_tax_2023.pdf", "chunk": "..." }
  ]
}
```

**Health Check:**
```json
GET /health
{
  "status": "ok",
  "index_loaded": true,
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "llm_model": "mistral:latest"
}
```

---

## 🗂️ Project Structure

```
├── app/
│   ├── main.py               # FastAPI entrypoint
│   ├── api/routes/           # Chat, health, admin ingest
│   ├── core/                 # Config, exceptions
│   ├── models/               # Pydantic schemas
│   ├── services/             # Loader, chunker, embeddings,
│   │                         # vector store, retriever, generator
│   └── ingestion/            # Ingestion pipeline
├── config/settings.yaml      # Main configuration
├── data/documents/           # Place source documents here
├── data/indices/             # FAISS index (auto-generated)
├── scripts/ingest.py         # Ingestion script
└── requirements.txt
```

---

## ⚙️ Configuration

| Variable | Default | Description |
|---|---|---|
| `DOCUMENTS_DIR` | `data/documents` | Source document folder |
| `INDEX_DIR` | `data/indices` | FAISS index output path |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `OLLAMA_MODEL` | `mistral:latest` | Local LLM (overridable) |
| `RETRIEVAL_TOP_K` | `5` | Chunks retrieved per query |

Configure via `config/settings.yaml` or override with a `.env` file.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai) installed and running

### 1. Install

```bash
cd "Mysuru Civic Intelligence System"
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 2. Pull the LLM

```bash
ollama pull mistral:latest
```

To use a lighter model on lower-spec hardware:
```bash
ollama pull llama3.2:3b
export OLLAMA_MODEL=llama3.2:3b
```

### 3. Add Documents

Place Mysuru municipal documents (PDF, DOCX, TXT) in `data/documents/`.

### 4. Ingest

```bash
python scripts/ingest.py
```

### 5. Run the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at:
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 🔭 Roadmap

- [ ] Streamlit or Gradio frontend for non-technical users
- [ ] Multilingual support (Kannada query handling)
- [ ] Hybrid search (BM25 + FAISS)
- [ ] Document update detection and incremental re-indexing
- [ ] Dockerised deployment for easy self-hosting

---

## 📄 License

MIT

---

## 👤 Author

**Niranjan G Shetty**
*AI Systems & Applied NLP Engineering*

> Built to demonstrate end-to-end RAG pipeline engineering — from document ingestion and vector indexing through to local LLM inference and a production-ready API layer.

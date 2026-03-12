# Mysuru Civic Intelligence System (MCIS)

A RAG-based civic assistant that answers questions using Mysuru municipal documents. Runs locally on a laptop (16GB RAM).

## Features

- **Document ingestion**: PDF, DOCX, TXT support
- **Vector search**: FAISS for fast similarity search
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **LLM**: 7B instruct model via Ollama (local inference)
- **Source citations**: Every answer includes document references
- **FastAPI**: REST API for chat and admin

## Architecture

```
data/documents/     →  Load  →  Chunk  →  Embed  →  FAISS index
                                                      ↓
Query  →  Embed  →  FAISS search  →  Retriever  →  LLM  →  Answer + citations
```

## Quick Start

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running

### 2. Install

```bash
cd "Mysuru Civic Intelligence System"
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 3. Pull the LLM model

By default the app uses `mistral:latest` (7B, quantized) which works on an 8GB RAM server:

```bash
ollama pull mistral:latest
```

To use a different model (e.g. a smaller one like `llama3.2:3b`), pull it and set:

```bash
export OLLAMA_MODEL=llama3.2:3b
```

or add `OLLAMA_MODEL=llama3.2:3b` to your `.env`.

### 4. Add documents

Place Mysuru municipal documents (PDF, DOCX, TXT) in `data/documents/`.

### 5. Ingest

```bash
python scripts/ingest.py
```

### 6. Run the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 7. Use the API

**Chat** (POST `/chat`):

```json
{
  "query": "What are the property tax rates in Mysuru?",
  "top_k": 5,
  "include_sources": true
}
```

**Health** (GET `/health`):

```json
{
  "status": "ok",
  "index_loaded": true,
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "llm_model": "mistral:latest"
}
```

**Ingest** (POST `/admin/ingest`): Re-run ingestion via API.

## Configuration

- `config/settings.yaml` – Main config
- `.env` – Environment overrides (copy from `.env.example`)

Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCUMENTS_DIR` | data/documents | Document folder |
| `INDEX_DIR` | data/indices | FAISS index output |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Embedding model |
| `OLLAMA_MODEL` | mistral:latest | LLM model (override `llm.model` in `settings.yaml`) |
| `RETRIEVAL_TOP_K` | 5 | Chunks per query |

## Project Structure

```
├── app/
│   ├── main.py           # FastAPI app
│   ├── api/routes/       # Chat, health, ingest
│   ├── core/             # Config, exceptions
│   ├── models/           # Pydantic schemas
│   ├── services/         # Loader, chunker, embeddings, vector store, retriever, generator
│   └── ingestion/        # Pipeline
├── config/settings.yaml
├── data/documents/       # Put documents here
├── data/indices/        # FAISS index (auto-generated)
├── scripts/ingest.py
└── requirements.txt
```

## API Docs

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## License

MIT

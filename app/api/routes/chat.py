"""Chat and health endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from app.core.config import get_settings
from app.core.exceptions import IndexNotFoundError, LLMError, RetrievalError
from app.models.schemas import ChatRequest, ChatResponse, HealthResponse, QueryRequest
from app.services.rag import RAGService
from app.services.vector_store import VectorStore

router = APIRouter()


def get_rag_service() -> RAGService:
    """Dependency for RAG service."""
    return RAGService()


@router.post("/query", response_model=ChatResponse)
def query(request: QueryRequest, rag: RAGService = Depends(get_rag_service)) -> ChatResponse:
    """Process a civic query (frontend-compatible: accepts 'question' field)."""
    return _chat_impl(
        query=request.question,
        history=request.history,
        top_k=None,
        include_sources=True,
        rag=rag,
    )


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, rag: RAGService = Depends(get_rag_service)) -> ChatResponse:
    """Process a civic query and return a grounded answer with citations."""
    return _chat_impl(
        query=request.query,
        history=request.history,
        top_k=request.top_k,
        include_sources=request.include_sources,
        rag=rag,
    )


def _chat_impl(
    query: str,
    history: list[str] | None,
    top_k: int | None,
    include_sources: bool,
    rag: RAGService,
) -> ChatResponse:
    """Shared chat logic for /chat and /query."""
    try:
        return rag.chat(
            query=query,
            history=history,
            top_k=top_k,
            include_sources=include_sources,
        )
    except IndexNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail="Search index not ready. Run ingestion first: python scripts/ingest.py",
        ) from e
    except RetrievalError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except LLMError as e:
        raise HTTPException(
            status_code=503,
            detail=f"LLM unavailable: {e}. Ensure Ollama is running.",
        ) from e


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check with component status."""
    settings = get_settings()
    index_loaded = False

    try:
        vs = VectorStore()
        if vs.index_path.exists():
            vs.load()
            index_loaded = vs.is_loaded()
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        index_loaded=index_loaded,
        embedding_model=settings.embeddings.model,
        llm_model=settings.llm.model,
    )

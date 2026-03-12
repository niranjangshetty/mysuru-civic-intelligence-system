"""RAG pipeline: retrieval + generation."""

from app.models.schemas import ChatResponse, SourceCitation
from app.services.generator import Generator
from app.services.retriever import Retriever


class RAGService:
    """Orchestrates retrieval and generation for chat."""

    def __init__(
        self,
        retriever: Retriever | None = None,
        generator: Generator | None = None,
    ):
        self.retriever = retriever or Retriever()
        self.generator = generator or Generator()

    def chat(
        self,
        query: str,
        top_k: int | None = None,
        include_sources: bool = True,
    ) -> ChatResponse:
        """Process a chat query and return answer with citations."""
        chunks_with_scores = self.retriever.retrieve(query=query, top_k=top_k)
        answer = self.generator.generate(query=query, chunks=chunks_with_scores)

        sources: list[SourceCitation] = []
        if include_sources:
            sources = [
                SourceCitation(
                    document_id=chunk.document_id,
                    document_name=chunk.document_name,
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    score=round(score, 4),
                )
                for chunk, score in chunks_with_scores
            ]

        return ChatResponse(
            answer=answer,
            sources=sources,
            query=query,
        )

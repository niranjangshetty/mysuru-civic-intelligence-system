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
        history: list[str] | None = None,
    ) -> ChatResponse:
        """Process a chat query and return answer with citations."""
        # Build a retrieval query that incorporates recent conversation history
        retrieval_query = self._build_retrieval_query(query=query, history=history)
        chunks_with_scores = self.retriever.retrieve(query=retrieval_query, top_k=top_k)
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

    def _build_retrieval_query(
        self,
        query: str,
        history: list[str] | None = None,
    ) -> str:
        """
        Combine recent conversation history with the current query to preserve context
        for retrieval.

        Example:
        history: ["how to get e-khata"]
        query: "what are the required documents"
        retrieval_query: "Regarding how to get e-khata, what are the required documents?"
        """
        if not history:
            return query

        # Use the last 2–3 messages for context
        recent = [h for h in history if h][-3:]
        if not recent:
            return query

        # Focus primarily on the most recent message to sharpen intent
        last = recent[-1]
        return f"Regarding {last}, {query}"

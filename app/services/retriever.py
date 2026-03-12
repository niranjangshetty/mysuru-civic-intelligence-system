"""RAG retrieval service."""

from typing import List

import numpy as np

from app.core.config import get_settings
from app.core.exceptions import IndexNotFoundError, RetrievalError
from app.models.schemas import DocumentChunk, SourceCitation
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore


class Retriever:
    """Retrieves relevant document chunks for a query."""

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_store: VectorStore | None = None,
    ):
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or VectorStore()
        self.settings = get_settings()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> List[tuple[DocumentChunk, float]]:
        """
        Retrieve relevant chunks for a query.
        Returns list of (chunk, score) sorted by relevance.
        """
        top_k = top_k or self.settings.retrieval.top_k
        score_threshold = score_threshold or self.settings.retrieval.score_threshold

        try:
            query_embedding = self.embedding_service.embed_single(query)
            query_vec = np.array([query_embedding], dtype=np.float32)
        except Exception as e:
            raise RetrievalError(f"Embedding failed: {e}") from e

        try:
            results = self.vector_store.search(
                query_embedding=query_vec[0],
                top_k=top_k,
                score_threshold=score_threshold,
            )
        except IndexNotFoundError:
            raise
        except Exception as e:
            raise RetrievalError(f"Vector search failed: {e}") from e

        return results

    def retrieve_with_citations(
        self,
        query: str,
        top_k: int | None = None,
    ) -> List[SourceCitation]:
        """Retrieve chunks and format as citations."""
        results = self.retrieve(query=query, top_k=top_k)
        return [
            SourceCitation(
                document_id=chunk.document_id,
                document_name=chunk.document_name,
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                score=round(score, 4),
            )
            for chunk, score in results
        ]

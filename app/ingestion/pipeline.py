"""Document ingestion pipeline: load -> chunk -> embed -> index."""

from typing import List

import numpy as np

from app.models.schemas import DocumentChunk
from app.services.chunker import TextChunker
from app.services.document_loader import DocumentLoader
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore


class IngestionPipeline:
    """End-to-end pipeline for ingesting documents into the vector store."""

    def __init__(
        self,
        loader: DocumentLoader | None = None,
        chunker: TextChunker | None = None,
        embedding_service: EmbeddingService | None = None,
        vector_store: VectorStore | None = None,
    ):
        self.loader = loader or DocumentLoader()
        self.chunker = chunker or TextChunker()
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or VectorStore()

    def run(self) -> dict:
        """
        Run full ingestion: load all documents, chunk, embed, and index.
        Returns stats: {documents_loaded, chunks_created, chunks_indexed}.
        """
        chunks: List[DocumentChunk] = []
        docs_loaded = 0

        for doc_id, doc_name, text in self.loader.load_all():
            docs_loaded += 1
            for chunk in self.chunker.chunk_document(doc_id, doc_name, text):
                chunks.append(chunk)

        if not chunks:
            return {
                "documents_loaded": docs_loaded,
                "chunks_created": 0,
                "chunks_indexed": 0,
            }

        # Embed in batches for efficiency
        texts = [c.content for c in chunks]
        embeddings = self.embedding_service.embed(texts)
        embeddings_np = np.array(embeddings, dtype=np.float32)

        # Create fresh index (overwrite existing)
        self.vector_store._index = None
        self.vector_store._metadata = []
        self.vector_store.add(embeddings_np, chunks)
        self.vector_store.save()

        return {
            "documents_loaded": docs_loaded,
            "chunks_created": len(chunks),
            "chunks_indexed": len(chunks),
        }

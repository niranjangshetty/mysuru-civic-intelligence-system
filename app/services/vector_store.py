"""FAISS vector store for document chunks."""

import json
from pathlib import Path
from typing import List

import faiss
import numpy as np

from app.core.config import get_settings
from app.core.exceptions import IndexNotFoundError
from app.models.schemas import DocumentChunk


class VectorStore:
    """FAISS-based vector store with metadata persistence."""

    INDEX_FILE = "faiss.index"
    METADATA_FILE = "metadata.json"

    def __init__(self, index_dir: str | None = None):
        settings = get_settings()
        self.index_dir = Path(index_dir or settings.paths.index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self._index = None
        self._metadata: List[dict] = []

    @property
    def index_path(self) -> Path:
        return self.index_dir / self.INDEX_FILE

    @property
    def metadata_path(self) -> Path:
        return self.index_dir / self.METADATA_FILE

    def _get_faiss_index(self, dimension: int):
        """Get or create FAISS index."""
        if self._index is not None:
            return self._index

        if self.index_path.exists():
            self._index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, encoding="utf-8") as f:
                self._metadata = json.load(f)
            return self._index

        # Create new index
        self._index = faiss.IndexFlatIP(dimension)  # Inner product (cosine with normalized)
        self._metadata = []
        return self._index

    def add(
        self,
        embeddings: np.ndarray,
        chunks: List[DocumentChunk],
    ) -> None:
        """
        Add embeddings and chunks to the index.
        Embeddings should be L2-normalized for cosine similarity via IndexFlatIP.
        """
        if len(embeddings) != len(chunks):
            raise ValueError("Embeddings and chunks count mismatch")

        if len(embeddings) == 0:
            return

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]

        if self._index is None and self.index_path.exists():
            self.load()

        index = self._get_faiss_index(dimension)
        index.add(embeddings)

        for chunk in chunks:
            self._metadata.append({
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "document_name": chunk.document_name,
                "content": chunk.content,
                "metadata": chunk.metadata,
            })

    def save(self) -> None:
        """Persist index and metadata to disk."""
        if self._index is None:
            return

        faiss.write_index(self._index, str(self.index_path))
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, indent=2, ensure_ascii=False)

    def load(self) -> bool:
        """Load index from disk. Returns True if loaded."""
        if not self.index_path.exists():
            raise IndexNotFoundError(f"Index not found at {self.index_dir}")

        self._index = faiss.read_index(str(self.index_path))
        with open(self.metadata_path, encoding="utf-8") as f:
            self._metadata = json.load(f)
        return True

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[tuple[DocumentChunk, float]]:
        """
        Search for similar chunks.
        query_embedding must be L2-normalized.
        Returns list of (chunk, score) sorted by relevance.
        """
        if self._index is None:
            if not self.index_path.exists():
                raise IndexNotFoundError(f"Index not found at {self.index_dir}")
            self.load()

        # Normalize query
        query = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query)

        scores, indices = self._index.search(query, min(top_k, len(self._metadata)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < score_threshold:
                continue
            meta = self._metadata[idx]
            chunk = DocumentChunk(
                chunk_id=meta["chunk_id"],
                document_id=meta["document_id"],
                document_name=meta["document_name"],
                content=meta["content"],
                metadata=meta.get("metadata", {}),
            )
            results.append((chunk, float(score)))

        return results

    def is_loaded(self) -> bool:
        """Check if index is loaded."""
        return self._index is not None

    def count(self) -> int:
        """Number of vectors in index."""
        if self._index is None:
            return 0
        return self._index.ntotal

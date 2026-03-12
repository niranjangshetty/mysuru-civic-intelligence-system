"""Embedding generation using sentence-transformers."""

from typing import List

from app.core.config import get_settings
from app.core.exceptions import EmbeddingError


class EmbeddingService:
    """Generates embeddings using sentence-transformers."""

    def __init__(self, model_name: str | None = None):
        settings = get_settings()
        self.model_name = model_name or settings.embeddings.model
        self._model = None

    @property
    def model(self):
        """Lazy load model to defer memory usage."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                raise EmbeddingError(f"Failed to load embedding model: {e}") from e
        return self._model

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        return self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        if not texts:
            return []
        try:
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 100,
            )
            return embeddings.tolist()
        except Exception as e:
            raise EmbeddingError(f"Embedding failed: {e}") from e

    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.embed([text])[0]

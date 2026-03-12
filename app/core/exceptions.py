"""Custom exceptions for MCIS."""


class MCISError(Exception):
    """Base exception for MCIS."""

    pass


class DocumentLoadError(MCISError):
    """Raised when document loading fails."""

    pass


class IndexNotFoundError(MCISError):
    """Raised when FAISS index is not found."""

    pass


class EmbeddingError(MCISError):
    """Raised when embedding generation fails."""

    pass


class LLMError(MCISError):
    """Raised when LLM inference fails."""

    pass


class RetrievalError(MCISError):
    """Raised when retrieval fails."""

    pass

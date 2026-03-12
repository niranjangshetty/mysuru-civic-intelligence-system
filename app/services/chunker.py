"""Text chunking for RAG."""

import hashlib
import re
from typing import Iterator

from app.core.config import get_settings
from app.models.schemas import DocumentChunk


class TextChunker:
    """Splits text into overlapping chunks suitable for retrieval."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunking.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunking.chunk_overlap

    def _chunk_id(self, doc_id: str, index: int) -> str:
        """Generate unique chunk ID."""
        return hashlib.sha256(f"{doc_id}:{index}".encode()).hexdigest()[:12]

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences, preserving structure."""
        # Simple sentence split - handles common cases
        text = re.sub(r"\s+", " ", text).strip()
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _chunk_by_tokens(self, text: str) -> Iterator[str]:
        """
        Chunk text by approximate token count (chars/4).
        Uses overlap for context continuity.
        """
        words = text.split()
        current_chunk: list[str] = []
        current_len = 0
        overlap_words: list[str] = []

        for word in words:
            current_chunk.append(word)
            current_len += len(word) // 4 + 1  # Approximate tokens

            if current_len >= self.chunk_size:
                chunk_text = " ".join(current_chunk)
                yield chunk_text

                # Prepare overlap for next chunk
                overlap_len = 0
                overlap_words = []
                for w in reversed(current_chunk):
                    overlap_words.insert(0, w)
                    overlap_len += len(w) // 4 + 1
                    if overlap_len >= self.chunk_overlap:
                        break

                current_chunk = overlap_words
                current_len = overlap_len

        if current_chunk:
            yield " ".join(current_chunk)

    def chunk_document(
        self,
        doc_id: str,
        doc_name: str,
        text: str,
    ) -> Iterator[DocumentChunk]:
        """Split document into chunks with metadata."""
        for i, content in enumerate(self._chunk_by_tokens(text)):
            if not content.strip():
                continue
            yield DocumentChunk(
                chunk_id=self._chunk_id(doc_id, i),
                document_id=doc_id,
                document_name=doc_name,
                content=content.strip(),
                metadata={"chunk_index": i},
            )

"""Document loading from various formats."""

import hashlib
from pathlib import Path
from typing import Iterator

from app.core.config import get_settings
from app.core.exceptions import DocumentLoadError
from app.models.schemas import DocumentChunk


class DocumentLoader:
    """Loads documents from supported formats (PDF, DOCX, TXT)."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}

    def __init__(self, documents_dir: str | None = None):
        settings = get_settings()
        self.documents_dir = Path(documents_dir or settings.paths.documents_dir)

    def _doc_id(self, path: Path) -> str:
        """Generate stable document ID from path."""
        return hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]

    def _load_pdf(self, path: Path) -> str:
        """Extract text from PDF."""
        try:
            from pypdf import PdfReader
        except ImportError:
            raise DocumentLoadError("pypdf required for PDF support. Install: pip install pypdf")

        reader = PdfReader(path)
        parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                parts.append(text)
        return "\n\n".join(parts)

    def _load_docx(self, path: Path) -> str:
        """Extract text from DOCX."""
        try:
            from docx import Document
        except ImportError:
            raise DocumentLoadError("python-docx required. Install: pip install python-docx")

        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    def _load_txt(self, path: Path) -> str:
        """Load plain text file."""
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()

    def load(self, path: Path) -> tuple[str, str]:
        """
        Load a single document. Returns (document_id, text).
        """
        path = Path(path).resolve()
        if not path.exists():
            raise DocumentLoadError(f"File not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".doc":
            suffix = ".docx"

        if suffix == ".pdf":
            text = self._load_pdf(path)
        elif suffix == ".docx":
            text = self._load_docx(path)
        elif suffix == ".txt":
            text = self._load_txt(path)
        else:
            raise DocumentLoadError(f"Unsupported format: {suffix}")

        if not text.strip():
            raise DocumentLoadError(f"Empty document: {path}")

        return self._doc_id(path), text

    def load_all(self) -> Iterator[tuple[str, str, str]]:
        """
        Load all supported documents from documents_dir.
        Yields (document_id, document_name, text).
        """
        if not self.documents_dir.exists():
            return

        for path in sorted(self.documents_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    doc_id, text = self.load(path)
                    yield doc_id, path.name, text
                except DocumentLoadError:
                    continue  # Skip problematic files

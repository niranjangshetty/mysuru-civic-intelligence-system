#!/usr/bin/env python3
"""Ingest Mysuru municipal documents into the vector store.

Usage:
    python scripts/ingest.py

Place PDF, DOCX, or TXT files in data/documents/ before running.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.ingestion.pipeline import IngestionPipeline


def main() -> int:
    """Run ingestion pipeline."""
    docs_dir = project_root / "data" / "documents"
    if not docs_dir.exists():
        docs_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created {docs_dir}. Add PDF, DOCX, or TXT files and run again.")
        return 0

    pipeline = IngestionPipeline()
    print("Starting ingestion...")
    stats = pipeline.run()

    print(f"Documents loaded: {stats['documents_loaded']}")
    print(f"Chunks created: {stats['chunks_created']}")
    print(f"Chunks indexed: {stats['chunks_indexed']}")

    if stats["chunks_indexed"] == 0:
        print("\nNo documents were ingested. Add files to data/documents/")
        return 1

    print("\nIngestion complete. Start the API with: uvicorn app.main:app --reload")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Ingestion API endpoint."""

from fastapi import APIRouter, HTTPException

from app.ingestion.pipeline import IngestionPipeline

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/ingest")
def ingest() -> dict:
    """Trigger document ingestion. Requires documents in data/documents/."""
    try:
        pipeline = IngestionPipeline()
        stats = pipeline.run()
        return {
            "status": "success",
            "message": "Ingestion completed",
            **stats,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

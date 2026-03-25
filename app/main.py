from typing import Tuple

from fastapi import Depends, FastAPI, HTTPException

from .config import Settings, get_settings
from .embedding import TextEmbedder
from .models import RebuildIndexResponse, SearchRequest, SearchResponse
from .qdrant_service import QdrantService


def get_qdrant_service(
    settings: Settings = Depends(get_settings),
) -> QdrantService:
    # A simple dependency factory; in a real system you'd typically manage
    # this as a singleton. For the purposes of this assessment, constructing
    # it on demand is sufficient and easy to test.
    embedder = TextEmbedder(dim=settings.embedding_dim)
    return QdrantService(settings=settings, embedder=embedder)


app = FastAPI(title="Healthcare Patient Semantic Search Service")


@app.on_event("startup")
def on_startup() -> None:
    """Initialize Qdrant collection and ingest seed data if available."""

    settings = get_settings()
    service = QdrantService(settings=settings, embedder=TextEmbedder(dim=settings.embedding_dim))

    # Ensure collection and index exist.
    service.ensure_collection()

    # Best-effort ingestion of seed data; failures during ingestion should
    # not prevent the API from starting, but they must be logged.
    try:
        service.ingest_seed_data()
    except Exception:
        import logging

        logging.getLogger(__name__).exception("Seed data ingestion failed during startup")


@app.post("/search", response_model=SearchResponse)
def search_notes(
    request: SearchRequest,
    service: QdrantService = Depends(get_qdrant_service),
) -> SearchResponse:
    """Semantic search endpoint over patient notes.

    Accepts a free-text query and returns relevant notes ordered by
    similarity. The underlying vector search is handled by Qdrant.
    """

    try:
        results = service.search(query=request.query, limit=request.limit)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Search backend error: {exc}")

    return SearchResponse(query=request.query, limit=request.limit, results=results)


@app.post("/admin/rebuild-index", response_model=RebuildIndexResponse)
def rebuild_index(
    service: QdrantService = Depends(get_qdrant_service),
) -> RebuildIndexResponse:
    """Trigger a manual index rebuild / retune.

    The operation is resilient to memory-related failures and returns a
    structured response indicating whether a fallback configuration was
    used.
    """

    try:
        success, used_fallback, message = service.rebuild_index()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Index rebuild failed: {exc}")

    return RebuildIndexResponse(success=success, used_fallback=used_fallback, message=message)


# Optional: allow running with `python -m app.main`
if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)

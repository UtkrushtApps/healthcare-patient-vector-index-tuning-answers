import json
import logging
import os
from typing import Iterable, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PointStruct,
    ScalarQuantizationConfig,
    ScalarType,
    SearchParams,
    VectorParams,
)

from .config import Settings
from .embedding import TextEmbedder
from .models import PatientNote, SearchResult

logger = logging.getLogger(__name__)


class QdrantService:
    """Encapsulates all interactions with Qdrant for patient embeddings.

    Responsibilities:
      - Ensure collection exists and is configured for 256-dim vectors.
      - Ingest and upsert patient notes with metadata payloads.
      - Perform semantic search over patient embeddings.
      - Rebuild / reconfigure the index with graceful handling of
        memory-related failures.
    """

    def __init__(self, settings: Settings, embedder: TextEmbedder) -> None:
        self._settings = settings
        self._embedder = embedder

        # Prefer full URL if provided, otherwise host/port/https.
        if settings.qdrant_url:
            self._client = QdrantClient(
                url=str(settings.qdrant_url),
                api_key=settings.qdrant_api_key or None,
            )
        else:
            self._client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                https=settings.qdrant_https,
                api_key=settings.qdrant_api_key or None,
            )

        self._collection = settings.qdrant_collection

    # ------------------------------------------------------------------
    # Collection and index management
    # ------------------------------------------------------------------

    def ensure_collection(self) -> None:
        """Create or validate the patient embeddings collection.

        If the collection exists but has the wrong vector size, it is
        re-created with the correct configuration.
        """

        logger.info("Ensuring Qdrant collection '%s' exists", self._collection)
        try:
            existing = {
                col.name for col in self._client.get_collections().collections
            }
        except Exception:
            logger.exception("Failed to list Qdrant collections")
            raise

        if self._collection in existing:
            try:
                info = self._client.get_collection(self._collection)
                vectors_config = info.vectors
                # Depending on qdrant-client version, this may be a dict-like
                # object or a VectorParams instance.
                size = getattr(vectors_config, "size", None)
                if size is None and isinstance(vectors_config, dict):
                    size = vectors_config.get("size")

                if size != self._settings.embedding_dim:
                    logger.warning(
                        "Existing collection '%s' has size=%s, expected %d. "
                        "Recreating collection.",
                        self._collection,
                        size,
                        self._settings.embedding_dim,
                    )
                    self._create_collection(force_recreate=True)
            except Exception:
                logger.exception("Failed to inspect existing Qdrant collection")
                raise
        else:
            self._create_collection(force_recreate=False)

        # Tune index configuration after ensuring collection exists.
        self._tune_collection_index()

    def _create_collection(self, force_recreate: bool) -> None:
        """Create or re-create the Qdrant collection with desired settings."""

        logger.info(
            "%s Qdrant collection '%s' with dim=%d",
            "Recreating" if force_recreate else "Creating",
            self._collection,
            self._settings.embedding_dim,
        )

        try:
            if force_recreate:
                self._client.recreate_collection(
                    collection_name=self._collection,
                    vectors_config=VectorParams(
                        size=self._settings.embedding_dim,
                        distance=Distance.COSINE,
                    ),
                )
            else:
                self._client.recreate_collection(
                    collection_name=self._collection,
                    vectors_config=VectorParams(
                        size=self._settings.embedding_dim,
                        distance=Distance.COSINE,
                    ),
                )
        except Exception:
            logger.exception("Failed to (re)create Qdrant collection '%s'", self._collection)
            raise

    def _tune_collection_index(self) -> None:
        """Apply a reasonably efficient index configuration for a 16 GB node.

        Uses HNSW with scalar quantization and conservative optimizer settings
        so that indexing remains memory-aware.
        """

        logger.info("Applying index tuning for collection '%s'", self._collection)

        try:
            self._client.update_collection(
                collection_name=self._collection,
                optimizers_config=OptimizersConfigDiff(
                    # Start using memmap when payloads get moderately large.
                    memmap_threshold=50000,
                    indexing_threshold=20000,
                ),
                hnsw_config=HnswConfigDiff(
                    m=16,
                    ef_construct=256,
                    full_scan_threshold=10000,
                    max_indexing_threads=0,  # let Qdrant decide based on CPUs
                ),
                quantization_config=ScalarQuantizationConfig(
                    scalar_type=ScalarType.INT8,
                    always_ram=False,
                ),
            )
        except (MemoryError, UnexpectedResponse) as exc:
            # Treat these as non-fatal; log and continue with default index.
            logger.warning(
                "Index tuning for collection '%s' failed due to resource "
                "constraints: %s. Proceeding with default configuration.",
                self._collection,
                exc,
            )
        except Exception:
            # Unexpected errors should be visible but must not crash service
            # initialization.
            logger.exception("Unexpected error while tuning collection index")

    def rebuild_index(self) -> Tuple[bool, bool, str]:
        """Attempt to rebuild / retune the collection index.

        Returns a tuple of (success, used_fallback, message).
        The method is resilient to memory-related failures; it logs issues
        and tries a lighter-weight fallback configuration.
        """

        logger.info("Rebuilding index for collection '%s'", self._collection)

        # Primary, more aggressive configuration.
        try:
            self._client.update_collection(
                collection_name=self._collection,
                hnsw_config=HnswConfigDiff(
                    m=16,
                    ef_construct=512,
                    full_scan_threshold=5000,
                    max_indexing_threads=0,
                ),
            )
            msg = "Index rebuilt with primary configuration."
            logger.info(msg)
            return True, False, msg
        except (MemoryError, UnexpectedResponse) as exc:
            logger.warning(
                "Primary index rebuild failed due to resource constraints: %s. "
                "Attempting fallback configuration.",
                exc,
            )
        except Exception:
            logger.exception("Unexpected error during primary index rebuild")
            return False, False, "Primary index rebuild failed due to unexpected error."

        # Fallback: more memory-frugal configuration (smaller M and ef_construct).
        try:
            self._client.update_collection(
                collection_name=self._collection,
                hnsw_config=HnswConfigDiff(
                    m=8,
                    ef_construct=128,
                    full_scan_threshold=20000,
                    max_indexing_threads=0,
                ),
            )
            msg = (
                "Index rebuilt with fallback configuration due to memory "
                "constraints. Search will still function but with potentially "
                "slower or slightly less accurate results."
            )
            logger.info(msg)
            return True, True, msg
        except Exception:
            logger.exception("Fallback index rebuild also failed")
            return (
                False,
                True,
                "Both primary and fallback index rebuild attempts failed.",
            )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_seed_data(self) -> int:
        """Load patient notes from the configured JSON file and upsert them.

        The seed file is expected to contain either:
          - a JSON list of objects representing PatientNote records, or
          - a JSON Lines file (one JSON object per line).

        Returns the number of notes ingested (0 if the file does not exist).
        """

        path = self._settings.seed_data_path
        if not path:
            logger.info("No seed data path configured; skipping ingestion")
            return 0

        if not os.path.exists(path):
            logger.info("Seed data file '%s' not found; skipping ingestion", path)
            return 0

        logger.info("Loading seed patient notes from '%s'", path)

        notes: List[PatientNote] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                first_char = f.read(1)
                f.seek(0)
                if first_char == "[":
                    raw = json.load(f)
                    if isinstance(raw, list):
                        for item in raw:
                            notes.append(PatientNote.parse_obj(item))
                else:
                    # JSON Lines
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        notes.append(PatientNote.parse_obj(obj))
        except Exception:
            logger.exception("Failed to read or parse seed data file '%s'", path)
            raise

        if not notes:
            logger.info("No notes found in seed data file '%s'", path)
            return 0

        logger.info("Ingesting %d patient notes into Qdrant", len(notes))
        self.upsert_patient_notes(notes)
        return len(notes)

    def _build_point_id(self, note: PatientNote, index: int) -> str:
        """Build a deterministic point ID to avoid duplicate inserts.

        If `note.id` is provided, it is used; otherwise we derive an ID
        from patient_id and a stable index.
        """

        if note.id:
            return str(note.id)
        return f"{note.patient_id}:{index}"

    def upsert_patient_notes(self, notes: Iterable[PatientNote]) -> None:
        """Upsert a collection of patient notes into Qdrant.

        Uses deterministic IDs and batch upserts to avoid duplicates and
        keep memory usage under control.
        """

        batch_size = max(1, self._settings.ingestion_batch_size)
        batch: List[PointStruct] = []
        total = 0

        for idx, note in enumerate(notes):
            vector = self._embedder.embed(note.note)
            point_id = self._build_point_id(note, idx)

            payload = note.dict()
            payload["id"] = point_id  # ensure payload ID is consistent

            batch.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

            if len(batch) >= batch_size:
                self._flush_batch(batch)
                total += len(batch)
                batch.clear()

        if batch:
            self._flush_batch(batch)
            total += len(batch)

        logger.info("Upserted %d patient notes into Qdrant", total)

    def _flush_batch(self, batch: List[PointStruct]) -> None:
        try:
            self._client.upsert(
                collection_name=self._collection,
                points=batch,
                wait=True,
            )
        except Exception:
            logger.exception("Failed to upsert batch of %d points", len(batch))
            raise

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, limit: int) -> List[SearchResult]:
        """Perform semantic search over patient notes.

        Returns a list of SearchResult objects sorted by similarity.
        """

        if not query or not query.strip():
            return []

        limit = max(1, min(limit, self._settings.max_search_results))
        vector = self._embedder.embed(query)

        try:
            hits = self._client.search(
                collection_name=self._collection,
                query_vector=vector,
                limit=limit,
                with_payload=True,
                with_vectors=False,
                search_params=SearchParams(
                    hnsw_ef=128,
                ),
            )
        except Exception:
            logger.exception("Qdrant search failed for query: %s", query)
            raise

        results: List[SearchResult] = []
        for hit in hits:
            payload = hit.payload or {}

            results.append(
                SearchResult(
                    id=str(payload.get("id", hit.id)),
                    patient_id=str(payload.get("patient_id", "")),
                    score=float(hit.score),
                    note=str(payload.get("note", "")),
                    diagnosis=payload.get("diagnosis"),
                    medications=payload.get("medications"),
                    timestamp=payload.get("timestamp"),
                )
            )

        return results

# Solution Steps

1. Create a configuration module to centralize settings.
- Add app/config.py with a Settings class derived from BaseSettings.
- Include fields for Qdrant connection (host, port, api key, url), collection name, embedding dimension, seed data path, batch size, and search result limits.
- Add a get_settings() function using @lru_cache to load and cache settings, and configure logging based on the configured log level.

2. Define Pydantic models for the domain and API contracts.
- In app/models.py, create a PatientNote model with fields: id (optional), patient_id, note, diagnosis (optional), medications (optional list of strings), and timestamp (optional datetime).
- Define SearchRequest with fields: query (string) and limit (int between 1 and 100, default 10).
- Define SearchResult with fields: id, patient_id, score (float), note, diagnosis, medications, and timestamp.
- Define SearchResponse with fields: query, limit, and a list of SearchResult.
- Add RebuildIndexResponse with fields: success (bool), used_fallback (bool), and message (string).

3. Implement a deterministic text embedding component.
- Create app/embedding.py and implement a TextEmbedder class.
- Constructor: accept dim (default 256) and validate that it is positive.
- Implement a private _tokenize method using a regex to split on word characters and lowercase tokens.
- Implement embed(text: str) to:
  - Tokenize the input (handle empty text by returning a unit vector with 1.0 in the first dimension).
  - For each token, compute an MD5 hash and map it to an index in [0, dim) and a sign (+1 or -1).
  - Accumulate token contributions into a float vector of length dim.
  - L2-normalize the vector before returning it.
- Log initialization details for observability.

4. Create a Qdrant service wrapper to encapsulate all database interactions.
- Add app/qdrant_service.py with a QdrantService class.
- In __init__, construct a QdrantClient using either a full URL or host/port/https from Settings, and store the collection name and TextEmbedder.
- Import required qdrant_client models: VectorParams, Distance, PointStruct, SearchParams, OptimizersConfigDiff, HnswConfigDiff, ScalarQuantizationConfig, ScalarType, and UnexpectedResponse for error handling.

5. Implement collection creation and index configuration.
- In QdrantService.ensure_collection():
  - List existing collections and check if the configured collection exists.
  - If it exists, fetch its info and compare the vector size with settings.embedding_dim; if mismatched, recreate the collection with VectorParams(size=embedding_dim, distance=Distance.COSINE).
  - If it does not exist, create it with the same VectorParams.
  - Call a private _tune_collection_index() method after ensuring the collection exists.
- In _create_collection(force_recreate: bool), log and call client.recreate_collection with the configured collection name and VectorParams (size=256, cosine distance).
- In _tune_collection_index(), call client.update_collection with:
  - OptimizersConfigDiff(memmap_threshold=50000, indexing_threshold=20000).
  - HnswConfigDiff(m=16, ef_construct=256, full_scan_threshold=10000, max_indexing_threads=0).
  - ScalarQuantizationConfig(scalar_type=INT8, always_ram=False).
- Catch MemoryError and UnexpectedResponse in _tune_collection_index(), log a warning about resource constraints, and continue without raising to keep startup resilient; log other exceptions as errors but do not crash the service.

6. Implement robust index rebuild / reconfiguration behavior.
- In QdrantService.rebuild_index():
  - Attempt a primary index update with HnswConfigDiff(m=16, ef_construct=512, full_scan_threshold=5000, max_indexing_threads=0).
  - On success, return (True, False, message) and log success.
  - If a MemoryError or UnexpectedResponse occurs, log a warning and attempt a fallback configuration with smaller HNSW parameters (e.g., m=8, ef_construct=128, full_scan_threshold=20000).
  - If the fallback succeeds, return (True, True, fallback message) and log it.
  - If both attempts fail, log an exception and return (False, True, failure message).
- This ensures that /admin/rebuild-index will never crash the service, even under memory pressure.

7. Implement ingestion of seeded patient notes with duplicate-safe upserts.
- In QdrantService.ingest_seed_data():
  - Read the seed_data_path from Settings; if empty or the file does not exist, log and return 0.
  - Open the file and detect whether it is a JSON array (first character is '[') or a JSON Lines file; parse accordingly into a list of PatientNote objects using PatientNote.parse_obj.
  - If no notes are found, log and return 0.
  - Call upsert_patient_notes(notes) and return the count.
- Implement a private _build_point_id(note, index) that returns note.id if provided, otherwise a deterministic ID such as f"{note.patient_id}:{index}".
- In upsert_patient_notes(notes):
  - Iterate over notes, compute embeddings via TextEmbedder, generate point IDs, and build PointStruct instances with id, vector, and note.dict() as payload; ensure the payload's "id" field matches the point ID.
  - Batch points according to Settings.ingestion_batch_size and call a private _flush_batch(batch) to upsert them with client.upsert(..., wait=True).
  - Log the total number of upserted notes and raise exceptions from _flush_batch so ingestion failures are visible in logs.

8. Implement semantic search backed by Qdrant.
- In QdrantService.search(query: str, limit: int):
  - Return an empty list if the query is empty or whitespace.
  - Clamp limit to [1, settings.max_search_results].
  - Embed the query using TextEmbedder.
  - Call client.search with collection_name, query_vector, limit, with_payload=True, with_vectors=False, and SearchParams(hnsw_ef=128).
  - Convert each hit into a SearchResult, pulling metadata fields (id, patient_id, note, diagnosis, medications, timestamp) from hit.payload, and using hit.score as the similarity score.
  - Handle and log any exceptions from Qdrant, then re-raise so the API layer can return an appropriate HTTP error.

9. Wire everything into a FastAPI application.
- Create app/main.py and instantiate a FastAPI app.
- Implement a dependency get_qdrant_service() that obtains Settings via get_settings(), constructs a TextEmbedder(dim=settings.embedding_dim), and returns a QdrantService.
- Add an on_startup() event handler that:
  - Creates a QdrantService with the global settings and a TextEmbedder.
  - Calls service.ensure_collection() to create/validate the patient embeddings collection.
  - Calls service.ingest_seed_data() inside a try/except block so that ingestion errors are logged but do not prevent the app from starting.
- Implement POST /search endpoint:
  - Accept a SearchRequest body.
  - Resolve QdrantService via dependency injection.
  - Call service.search(request.query, request.limit) and return a SearchResponse with the query, limit, and result list.
  - Wrap service.search in try/except and translate any exception into HTTP 503 with a descriptive message.
- Implement POST /admin/rebuild-index endpoint:
  - Resolve QdrantService via dependency injection.
  - Call service.rebuild_index() and map its (success, used_fallback, message) result into a RebuildIndexResponse.
  - Translate unexpected exceptions into HTTP 500.
- Optionally add a __main__ block to run uvicorn when executing the module directly.

10. Review and test behavior for robustness and correctness.
- Confirm that Settings.embedding_dim is 256 so the collection is configured for 256-dimensional vectors.
- Start the FastAPI app and verify that on startup it connects to Qdrant, ensures the collection exists, tunes the index, and ingests any available seed file without crashing on errors.
- Insert or verify seeded patient notes and ensure that repeated ingestions do not create duplicates because of deterministic point IDs.
- Call POST /search with a JSON body containing a clinical query and verify that the API returns ordered results with relevant note metadata.
- Call POST /admin/rebuild-index under normal conditions and with simulated resource constraints (e.g., by forcing exceptions in update_collection) to confirm that primary and fallback configurations are handled gracefully and that the endpoint always returns a structured, informative response.


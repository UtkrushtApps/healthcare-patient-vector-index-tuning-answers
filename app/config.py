import logging
from functools import lru_cache
from typing import Optional

from pydantic import BaseSettings, AnyUrl


class Settings(BaseSettings):
    """Application configuration loaded from environment variables.

    Prefix: APP_

    Example env vars:
      APP_QDRANT_HOST=localhost
      APP_QDRANT_PORT=6333
      APP_QDRANT_COLLECTION=patient_notes
      APP_SEED_DATA_PATH=data/patient_notes_seed.json
    """

    # Qdrant connection
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: Optional[str] = None
    qdrant_https: bool = False
    qdrant_url: Optional[AnyUrl] = None  # Optional full URL, overrides host/port/https

    # Collection settings
    qdrant_collection: str = "patient_notes"
    embedding_dim: int = 256

    # Ingestion
    seed_data_path: str = "data/patient_notes_seed.json"
    ingestion_batch_size: int = 128

    # Search
    max_search_results: int = 20
    default_search_results: int = 10

    # Logging
    log_level: str = "INFO"

    class Config:
        env_prefix = "APP_"
        env_file = ".env"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()

    # Configure root logger level once based on settings
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    return settings

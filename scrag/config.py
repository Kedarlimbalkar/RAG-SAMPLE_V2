"""
Configuration & Environment Management for Legal RAG Pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --------------------------------------------------------------------------
    # PostgreSQL & pgvector Database Settings
    # --------------------------------------------------------------------------
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_db: str = Field(default="scrag_legal", alias="POSTGRES_DB")
    postgres_user: str = Field(default="scrag_user", alias="POSTGRES_USER")
    postgres_password: str = Field(default="scrag_password", alias="POSTGRES_PASSWORD")
    postgres_max_connections: int = Field(default=20, alias="POSTGRES_MAX_CONNECTIONS")
    postgres_pool_timeout: int = Field(default=30, alias="POSTGRES_POOL_TIMEOUT")

    @property
    def database_url(self) -> str:
        """Construct standard SQLAlchemy connection string."""
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}@"
            f"{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # --------------------------------------------------------------------------
    # Embedding Settings (Local BAAI/bge-m3)
    # --------------------------------------------------------------------------
    embedding_model_name: str = Field(default="BAAI/bge-m3", alias="EMBEDDING_MODEL_NAME")
    embedding_device: str = Field(default="cpu", alias="EMBEDDING_DEVICE")
    embedding_batch_size: int = Field(default=16, alias="EMBEDDING_BATCH_SIZE")
    embedding_normalize: bool = Field(default=True, alias="EMBEDDING_NORMALIZE")

    # --------------------------------------------------------------------------
    # Storage & Local Snapshots
    # --------------------------------------------------------------------------
    data_dir: Path = Field(default=Path("./data"), alias="DATA_DIR")
    raw_pdf_dir: Path = Field(default=Path("./data/raw_pdfs"), alias="RAW_PDF_DIR")
    snapshot_dir: Path = Field(default=Path("./data/snapshots"), alias="PROCESSED_SNAPSHOT_DIR")
    export_format: str = Field(default="parquet", alias="EXPORT_FORMAT")

    # --------------------------------------------------------------------------
    # Google Cloud Infrastructure & V2 Storage Partitions
    # --------------------------------------------------------------------------
    gcp_project_id: str = Field(default="sc-legal-rag-prod", alias="GCP_PROJECT_ID")
    gcp_region: str = Field(default="asia-south1", alias="GCP_REGION")
    gcs_bucket_name: str = Field(default="sc-legal-rag-data-v2", alias="GCS_BUCKET_NAME")
    gcs_raw_prefix: str = Field(default="raw", alias="GCS_RAW_PREFIX")
    gcs_parsed_prefix: str = Field(default="parsed-v2", alias="GCS_PARSED_PREFIX")
    gcs_chunks_prefix: str = Field(default="chunks-v2", alias="GCS_CHUNKS_PREFIX")
    gcs_snapshots_prefix: str = Field(default="snapshots-v2", alias="GCS_SNAPSHOTS_PREFIX")

    # --------------------------------------------------------------------------
    # PDF Parser Tuning Parameters
    # --------------------------------------------------------------------------
    parser_header_band_ratio: float = Field(default=0.12, alias="PARSER_HEADER_BAND_RATIO")
    parser_footer_band_ratio: float = Field(default=0.88, alias="PARSER_FOOTER_BAND_RATIO")
    parser_header_footer_threshold: float = Field(
        default=0.30, alias="PARSER_HEADER_FOOTER_THRESHOLD"
    )
    parser_footnote_font_ratio: float = Field(default=0.85, alias="PARSER_FOOTNOTE_FONT_RATIO")
    parser_footnote_band_ratio: float = Field(default=0.75, alias="PARSER_FOOTNOTE_BAND_RATIO")
    parser_min_page_chars: int = Field(default=40, alias="PARSER_MIN_PAGE_CHARS")
    parser_max_garbage_ratio: float = Field(default=0.25, alias="PARSER_MAX_GARBAGE_RATIO")
    ocr_enabled: bool = Field(default=True, alias="OCR_ENABLED")
    tesseract_cmd: Optional[str] = Field(default=None, alias="TESSERACT_CMD")

    # --------------------------------------------------------------------------
    # Hierarchical Chunking Parameters
    # --------------------------------------------------------------------------
    chunk_parent_words: int = Field(default=1500, alias="CHUNK_PARENT_WORDS")
    chunk_parent_overlap: int = Field(default=150, alias="CHUNK_PARENT_OVERLAP")
    chunk_child_words: int = Field(default=350, alias="CHUNK_CHILD_WORDS")
    chunk_child_overlap: int = Field(default=50, alias="CHUNK_CHILD_OVERLAP")

    # --------------------------------------------------------------------------
    # Worker Queue & Pipeline Reliability
    # --------------------------------------------------------------------------
    max_retries: int = Field(default=3, alias="MAX_RETRIES")
    queue_poll_interval_sec: float = Field(default=2.0, alias="QUEUE_POLL_INTERVAL_SEC")
    worker_concurrency: int = Field(default=4, alias="WORKER_CONCURRENCY")

    def ensure_directories(self) -> None:
        """Create configured data directories if they do not exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_pdf_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)


# Global singleton settings instance
settings = Settings()


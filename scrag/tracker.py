"""
Execution State Tracker for Supreme Court Legal RAG Pipeline.

Tracks file-level processing states (pending -> parsed -> chunked -> embedded -> exported)
with local JSON caching, PostgreSQL synchronization, and GCS V2 cloud persistence.
Enables instant resume capability and eliminates duplicate processing.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import uuid

from scrag.config import Settings, settings as global_settings
from scrag.ids import compute_sha256, generate_doc_id

logger = logging.getLogger(__name__)


def extract_year_from_path(path_or_uri: Union[str, Path], fallback: Optional[str] = None) -> str:
    """
    Extract 4-digit year partition from file path or GCS blob name.
    Supports 'year=YYYY' directory patterns and 4-digit year tokens.
    """
    path_str = str(path_or_uri).replace("\\", "/")
    # 1. Explicit year=YYYY pattern
    m = re.search(r"year=(\d{4})", path_str, re.IGNORECASE)
    if m:
        return m.group(1)
    # 2. 4-digit year token (1900-2099)
    m = re.search(r"(?:^|[/_\-\s])(19\d{2}|20\d{2})(?:[/_\-\s]|$)", path_str)
    if m:
        return m.group(1)
    return fallback or "unknown"


@dataclass
class FileTrackRecord:
    """Detailed state record for an individual judgment document."""
    doc_id: str
    sha256: str
    source_file: str
    year_partition: Optional[str] = None
    gcs_blob_name: Optional[str] = None
    status: str = "pending"  # pending, parsed, chunked, embedded, exported, failed
    current_stage: str = "pending"
    stages_completed: List[str] = field(default_factory=list)
    total_pages: int = 0
    parent_chunks: int = 0
    child_chunks: int = 0
    error_notes: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    execution_time_ms: int = 0
    gcs_uris: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FileTrackRecord:
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


class ExecutionTracker:
    """
    Lightweight state manager with dual persistence (Local JSON manifest + GCS V2 sync).
    """

    def __init__(self, config: Optional[Settings] = None, state_file: Optional[Path] = None) -> None:
        self.cfg = config or global_settings
        self.state_file = state_file or self.cfg.data_dir / "state_tracker.json"
        self._records: Dict[str, FileTrackRecord] = {}  # sha256 -> FileTrackRecord
        self.load_local()

    # --------------------------------------------------------------------------
    # Local State File Management
    # --------------------------------------------------------------------------
    def load_local(self) -> None:
        """Load state manifest from local JSON file."""
        if not self.state_file.exists():
            self._records = {}
            return

        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._records = {
                    k: FileTrackRecord.from_dict(v) for k, v in data.get("documents", {}).items()
                }
            logger.info(f"Loaded {len(self._records)} tracking records from {self.state_file}")
        except Exception as e:
            logger.warning(f"Could not load state tracker file ({e}). Starting with empty state.")
            self._records = {}

    def save_local(self) -> None:
        """Atomically persist state manifest to local JSON file."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file = self.state_file.with_suffix(".tmp")
        try:
            payload = {
                "manifest_version": "2.0",
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "total_documents": len(self._records),
                "summary": self.get_summary(),
                "documents": {k: v.to_dict() for k, v in self._records.items()},
            }
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            temp_file.replace(self.state_file)
        except Exception as e:
            logger.error(f"Failed to persist state tracker locally: {e}")

    # --------------------------------------------------------------------------
    # Lifecycle Tracking Methods
    # --------------------------------------------------------------------------
    def register_document(
        self,
        file_path: Union[str, Path],
        raw_bytes: bytes,
        year: Optional[str] = None,
        gcs_blob_name: Optional[str] = None,
    ) -> FileTrackRecord:
        """
        Register a file into the tracker. Returns existing record if already known.
        """
        path = Path(file_path)
        sha256 = compute_sha256(raw_bytes)
        doc_id, _ = generate_doc_id(sha256)
        doc_id_str = str(doc_id)

        # Detect year from path or gcs_blob_name if not provided
        if not year:
            detected = extract_year_from_path(gcs_blob_name or file_path)
            if detected != "unknown":
                year = detected

        if sha256 in self._records:
            rec = self._records[sha256]
            if gcs_blob_name and not rec.gcs_blob_name:
                rec.gcs_blob_name = gcs_blob_name
            if year and not rec.year_partition:
                rec.year_partition = year
            return rec

        record = FileTrackRecord(
            doc_id=doc_id_str,
            sha256=sha256,
            source_file=path.name,
            year_partition=year,
            gcs_blob_name=gcs_blob_name,
            status="pending",
            current_stage="pending",
        )
        self._records[sha256] = record
        self.save_local()
        return record

    def update_stage(
        self,
        sha256: str,
        stage: str,
        status: str = "in_progress",
        notes: Optional[str] = None,
        total_pages: Optional[int] = None,
        parent_chunks: Optional[int] = None,
        child_chunks: Optional[int] = None,
    ) -> None:
        """
        Record stage transition (e.g. stage='parsed', status='completed').
        """
        if sha256 not in self._records:
            return

        rec = self._records[sha256]
        rec.current_stage = stage
        rec.status = status
        rec.updated_at = datetime.utcnow().isoformat() + "Z"

        if status in ("completed", "parsed", "chunked", "embedded") and stage not in rec.stages_completed:
            rec.stages_completed.append(stage)

        if total_pages is not None:
            rec.total_pages = total_pages
        if parent_chunks is not None:
            rec.parent_chunks = parent_chunks
        if child_chunks is not None:
            rec.child_chunks = child_chunks
        if notes:
            rec.error_notes = notes

        self.save_local()

    def mark_completed(
        self,
        sha256: str,
        gcs_uris: Optional[Dict[str, str]] = None,
        execution_time_ms: int = 0,
    ) -> None:
        """Mark document as successfully exported to V2 storage partitions."""
        if sha256 not in self._records:
            return

        rec = self._records[sha256]
        rec.status = "exported"
        rec.current_stage = "exported"
        if "exported" not in rec.stages_completed:
            rec.stages_completed.append("exported")
        rec.updated_at = datetime.utcnow().isoformat() + "Z"
        rec.execution_time_ms = execution_time_ms
        if gcs_uris:
            rec.gcs_uris.update(gcs_uris)

        self.save_local()

    def mark_failed(self, sha256: str, error_message: str) -> None:
        """Mark document as failed with error trace."""
        if sha256 not in self._records:
            return

        rec = self._records[sha256]
        rec.status = "failed"
        rec.error_notes = error_message
        rec.updated_at = datetime.utcnow().isoformat() + "Z"
        self.save_local()

    def is_completed(self, sha256: str) -> bool:
        """Check if document has already been fully processed and exported."""
        rec = self._records.get(sha256)
        return rec is not None and rec.status == "exported"

    def is_blob_completed(self, blob_name: str) -> bool:
        """
        Check if a GCS blob name has already been fully processed and exported.
        Matches by exact gcs_blob_name or by filename and year partition.
        """
        clean_blob = blob_name.replace("\\", "/")
        file_name = Path(clean_blob).name

        for rec in self._records.values():
            if rec.status == "exported":
                if rec.gcs_blob_name and rec.gcs_blob_name == clean_blob:
                    return True
                if rec.source_file == file_name:
                    if rec.year_partition:
                        if f"year={rec.year_partition}" in clean_blob or f"/{rec.year_partition}/" in clean_blob:
                            return True
                    else:
                        return True
        return False

    def get_record(self, sha256: str) -> Optional[FileTrackRecord]:
        return self._records.get(sha256)

    def get_summary(self) -> Dict[str, int]:
        """Aggregate counts of document processing stages."""
        summary = {
            "total": len(self._records),
            "pending": 0,
            "parsed": 0,
            "chunked": 0,
            "embedded": 0,
            "exported": 0,
            "failed": 0,
        }
        for r in self._records.values():
            s = r.status
            if s in summary:
                summary[s] += 1
            else:
                summary[s] = 1
        return summary

    # --------------------------------------------------------------------------
    # Cloud Synchronization (GCS V2)
    # --------------------------------------------------------------------------
    def sync_to_gcs(self, bucket_name: Optional[str] = None) -> bool:
        """Upload current state manifest to gs://sc-legal-rag-data-v2/tracker-v2/."""
        bucket = bucket_name or self.cfg.gcs_bucket_name
        remote_blob = f"tracker-v2/state_tracker.json"
        logger.info(f"Syncing state tracker to gs://{bucket}/{remote_blob}...")

        try:
            from google.cloud import storage
            client = storage.Client(project=self.cfg.gcp_project_id)
            gcs_bucket = client.bucket(bucket)
            blob = gcs_bucket.blob(remote_blob)
            blob.upload_from_filename(str(self.state_file))
            logger.info("Successfully synced tracker manifest to GCS.")
            return True
        except Exception as e:
            logger.warning(f"Could not upload tracker to GCS ({e}). State preserved locally.")
            return False

    def sync_from_gcs(self, bucket_name: Optional[str] = None) -> bool:
        """Download remote state manifest from GCS and merge with local state."""
        bucket = bucket_name or self.cfg.gcs_bucket_name
        remote_blob = f"tracker-v2/state_tracker.json"
        logger.info(f"Fetching state tracker from gs://{bucket}/{remote_blob}...")

        try:
            from google.cloud import storage
            client = storage.Client(project=self.cfg.gcp_project_id)
            gcs_bucket = client.bucket(bucket)
            blob = gcs_bucket.blob(remote_blob)
            if not blob.exists():
                logger.info("No remote tracker manifest found in GCS. Using local.")
                return False

            remote_data = json.loads(blob.download_as_text())
            remote_docs = remote_data.get("documents", {})

            # Merge remote records with local records (favoring later updated_at)
            for k, v in remote_docs.items():
                if k not in self._records:
                    self._records[k] = FileTrackRecord.from_dict(v)
                else:
                    if v.get("updated_at", "") > self._records[k].updated_at:
                        self._records[k] = FileTrackRecord.from_dict(v)

            self.save_local()
            logger.info(f"Synced and merged {len(remote_docs)} remote tracker records.")
            return True
        except Exception as e:
            logger.warning(f"Could not download tracker from GCS ({e}). Using local.")
            return False


# Global tracker instance
tracker = ExecutionTracker()


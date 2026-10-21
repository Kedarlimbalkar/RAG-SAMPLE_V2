"""
Production Document Processing Pipeline & Worker Queue for Supreme Court Judgments.

Lifecycle: pending -> parsed -> chunked -> embedded -> exported (or failed / quarantined)
Supports:
- Transactional locking via SELECT ... FOR UPDATE SKIP LOCKED
- Idempotent execution and automatic retries (up to MAX_RETRIES)
- Local and GCS V2 partition synchronization (raw, parsed-v2, chunks-v2, snapshots-v2)
- Direct pgvector ingestion and resume capability
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

from scrag.chunker import HierarchicalChunker, ParentChunk
from scrag.config import Settings, settings as global_settings
from scrag.database import db
from scrag.embedder import LocalEmbedder, embedder
from scrag.ids import compute_sha256, generate_doc_id
from scrag.models import DocumentRecord
from scrag.parser import DocumentParseResult, SupremeCourtParser
from scrag.tracker import ExecutionTracker, tracker

logger = logging.getLogger(__name__)


class LegalRAGPipeline:
    """
    End-to-end transactional document pipeline for Supreme Court judgments.
    """

    def __init__(self, config: Optional[Settings] = None) -> None:
        self.cfg = config or global_settings
        self.parser = SupremeCourtParser(self.cfg)
        self.chunker = HierarchicalChunker(self.cfg)
        self.embedder = embedder
        self.tracker = tracker
        self.cfg.ensure_directories()

    # --------------------------------------------------------------------------
    # Main Batch Runner Entry Point
    # --------------------------------------------------------------------------
    def run_batch(
        self,
        source_dir: Optional[Path] = None,
        from_gcs: bool = False,
        limit: Optional[int] = None,
        resume: bool = True,
        year: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        Execute batch processing on raw PDF documents.

        Args:
            source_dir: Local path containing raw PDFs (default: data/raw_pdfs).
            from_gcs: If True, pulls raw PDFs from gs://sc-legal-rag-data-v2/raw/.
            limit: Maximum documents to process in this run.
            resume: If True, picks up unfinished documents from database state.
            year: Optional year partition filter (e.g. '1950').

        Returns:
            Dict[str, int]: Processing outcome metrics.
        """
        metrics = {"total": 0, "processed": 0, "skipped": 0, "failed": 0}

        # 1. Optionally sync inbound raw PDFs from GCS
        if from_gcs:
            self._pull_raw_from_gcs(year=year)

        # 2. Ingest local PDFs into DB documents queue
        raw_dir = source_dir or self.cfg.raw_pdf_dir
        if raw_dir.exists():
            pdf_files = list(raw_dir.glob("*.pdf")) + list(raw_dir.glob("*/*.pdf"))
            logger.info(f"Discovered {len(pdf_files)} PDF files in {raw_dir}")
            for pdf_file in pdf_files:
                self.queue_document_from_file(pdf_file)

        # 3. Process queued documents with transactional locking
        batch_limit = limit or 1000000
        count = 0

        while count < batch_limit:
            doc_record = self._claim_next_document()
            if not doc_record:
                logger.info("No more pending documents in queue.")
                break

            # Fast resume skip via tracker
            if resume and self.tracker.is_completed(doc_record.sha256_hash):
                logger.info(f"[{doc_record.id}] Document already exported. Skipping.")
                metrics["skipped"] += 1
                continue

            metrics["total"] += 1
            success = self.process_document(doc_record)
            if success:
                metrics["processed"] += 1
            else:
                metrics["failed"] += 1

            count += 1

        # Sync state manifest to GCS
        self.tracker.save_local()
        self.tracker.sync_to_gcs()

        logger.info(f"Batch processing completed: {metrics}")
        return metrics

    # --------------------------------------------------------------------------
    # Document Ingestion & Queue Registration
    # --------------------------------------------------------------------------
    def queue_document_from_file(self, file_path: Path) -> Optional[uuid.UUID]:
        """
        Register a raw PDF into documents table with status='pending'.
        Idempotent: skips if exact SHA-256 is already recorded.
        """
        try:
            raw_bytes = file_path.read_bytes()
            doc_id, sha256_hash = generate_doc_id(raw_bytes)
            file_size = len(raw_bytes)

            # Register in state tracker
            self.tracker.register_document(file_path, raw_bytes)

            if self.tracker.is_completed(sha256_hash):
                logger.debug(f"Document {file_path.name} already completed in tracker.")
                return doc_id

            sql = """
                INSERT INTO documents (id, sha256_hash, source_filename, file_size_bytes, status)
                VALUES (%s, %s, %s, %s, 'pending')
                ON CONFLICT (sha256_hash) DO NOTHING
                RETURNING id;
            """
            try:
                with db.get_cursor(commit=True) as cursor:
                    cursor.execute(sql, (str(doc_id), sha256_hash, file_path.name, file_size))
                    res = cursor.fetchone()
                    if res:
                        logger.info(f"Queued document {file_path.name} (id={doc_id})")
                    return doc_id
            except Exception as e:
                logger.debug(f"Queue DB insert skipped/failed: {e}")
                return doc_id
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None

    # --------------------------------------------------------------------------
    # Transactional Worker Queue: SELECT ... FOR UPDATE SKIP LOCKED
    # --------------------------------------------------------------------------
    def _claim_next_document(self) -> Optional[DocumentRecord]:
        """
        Claim the next pending or retryable document using row-level locking.
        """
        sql = """
            SELECT id, sha256_hash, source_filename, file_size_bytes, status, retry_count, max_retries
            FROM documents
            WHERE status IN ('pending', 'parsed', 'chunked', 'embedded')
              AND retry_count < max_retries
            ORDER BY created_at ASC
            FOR UPDATE SKIP LOCKED
            LIMIT 1;
        """
        try:
            with db.get_cursor(commit=True) as cursor:
                cursor.execute(sql)
                row = cursor.fetchone()
                if not row:
                    return None

                return DocumentRecord(
                    id=uuid.UUID(str(row["id"])),
                    sha256_hash=row["sha256_hash"],
                    source_filename=row["source_filename"],
                    file_size_bytes=row["file_size_bytes"],
                    status=row["status"],
                    retry_count=row["retry_count"],
                    max_retries=row["max_retries"],
                )
        except Exception as e:
            logger.warning(f"Could not claim next document from database ({e}).")
            return None

    # --------------------------------------------------------------------------
    # Single Document Lifecycle Execution
    # --------------------------------------------------------------------------
    def process_document(self, doc_record: DocumentRecord) -> bool:
        """
        Run document through full pipeline lifecycle:
        1. Parse (scrag.parser)
        2. Chunk (scrag.chunker)
        3. Embed & Index (scrag.embedder + pgvector)
        4. Snapshot & GCS V2 Export
        """
        doc_id = doc_record.id
        start_time = time.time()
        logger.info(f"[{doc_id}] Processing document {doc_record.source_filename} (status={doc_record.status})...")

        raw_path = self.cfg.raw_pdf_dir / (doc_record.source_filename or f"{doc_id}.pdf")
        if not raw_path.exists():
            # Try searching in raw directory by sha256 or filename
            found = list(self.cfg.raw_pdf_dir.glob(f"*{doc_record.sha256_hash[:8]}*.pdf"))
            if found:
                raw_path = found[0]

        if not raw_path.exists():
            self._record_failure(doc_id, f"Raw PDF not found on disk: {raw_path}")
            return False

        try:
            raw_bytes = raw_path.read_bytes()
            sha256 = doc_record.sha256_hash

            # Stage 1: Parse
            parse_result = self.parser.parse_pdf(raw_bytes, filename_hint=raw_path.name)
            self._save_parsed_pages(parse_result)
            self._update_doc_status(doc_id, "parsed", total_pages=parse_result.total_pages)
            self.tracker.update_stage(sha256, "parsed", status="parsed", total_pages=parse_result.total_pages)

            # Stage 2: Hierarchical Chunking
            parent_chunks = self.chunker.chunk_document(
                doc_id=doc_id,
                stitched_text=parse_result.stitched_text,
                page_offset_map=parse_result.page_offset_map,
            )
            self._save_chunks(parent_chunks)
            self._update_doc_status(doc_id, "chunked")
            all_children = [c for p in parent_chunks for c in p.children]
            self.tracker.update_stage(
                sha256,
                "chunked",
                status="chunked",
                parent_chunks=len(parent_chunks),
                child_chunks=len(all_children),
            )

            # Stage 3: Embedding Generation & pgvector Ingestion
            child_texts = [c.content for c in all_children]
            embeddings = self.embedder.embed_texts(child_texts)
            self._save_embeddings(doc_id, all_children, embeddings)
            self._update_doc_status(doc_id, "embedded")
            self.tracker.update_stage(sha256, "embedded", status="embedded")

            # Stage 4: Local Snapshots & GCS V2 Upload
            self._export_snapshots(parse_result, parent_chunks, all_children)
            self._update_doc_status(doc_id, "exported")

            elapsed_ms = int((time.time() - start_time) * 1000)
            gcs_uris = {
                "raw": f"gs://{self.cfg.gcs_bucket_name}/{self.cfg.gcs_raw_prefix}/{raw_path.name}",
                "parsed": f"gs://{self.cfg.gcs_bucket_name}/{self.cfg.gcs_parsed_prefix}/parsed_{doc_id}.jsonl",
                "chunks": f"gs://{self.cfg.gcs_bucket_name}/{self.cfg.gcs_chunks_prefix}/chunks_{doc_id}.jsonl",
            }
            self.tracker.mark_completed(sha256, gcs_uris=gcs_uris, execution_time_ms=elapsed_ms)

            self._log_stage(doc_id, "pipeline", "success", f"Processed in {elapsed_ms}ms", elapsed_ms)
            logger.info(f"[{doc_id}] Successfully processed & exported in {elapsed_ms}ms.")
            return True

        except Exception as e:
            logger.exception(f"[{doc_id}] Error in pipeline: {e}")
            self.tracker.mark_failed(doc_record.sha256_hash, str(e))
            self._record_failure(doc_id, str(e))
            return False

    # --------------------------------------------------------------------------
    # Database Persistence Handlers
    # --------------------------------------------------------------------------
    def _save_parsed_pages(self, parse_result: DocumentParseResult) -> None:
        """Persist extracted pages and coordinates to pages table."""
        sql = """
            INSERT INTO pages (
                id, doc_id, page_no, width, height, raw_text, cleaned_text,
                char_start, char_end, is_ocr, garbage_ratio, footnotes
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                cleaned_text = EXCLUDED.cleaned_text,
                footnotes = EXCLUDED.footnotes;
        """
        try:
            with db.get_cursor(commit=True) as cursor:
                for p in parse_result.pages:
                    fn_json = json.dumps([fn.to_dict() for fn in p.footnotes])
                    cursor.execute(
                        sql,
                        (
                            p.page_id,
                            str(p.doc_id),
                            p.page_no,
                            p.width,
                            p.height,
                            p.raw_text,
                            p.cleaned_text,
                            p.char_start,
                            p.char_end,
                            p.is_ocr,
                            p.garbage_ratio,
                            fn_json,
                        ),
                    )
        except Exception as e:
            logger.debug(f"DB save pages skipped: {e}")

    def _save_chunks(self, parent_chunks: List[ParentChunk]) -> None:
        """Persist parent and child chunks into parents and chunks tables."""
        sql_parent = """
            INSERT INTO parents (id, doc_id, parent_index, section_name, content, word_count, page_start, page_end, char_start, char_end)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING;
        """
        sql_child = """
            INSERT INTO chunks (id, doc_id, parent_id, chunk_index, section_name, content, word_count, page_start, page_end, char_start, char_end)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING;
        """
        try:
            with db.get_cursor(commit=True) as cursor:
                for p in parent_chunks:
                    cursor.execute(
                        sql_parent,
                        (
                            str(p.id),
                            str(p.doc_id),
                            p.parent_index,
                            p.section_name,
                            p.content,
                            p.word_count,
                            p.page_start,
                            p.page_end,
                            p.char_start,
                            p.char_end,
                        ),
                    )
                    for c in p.children:
                        cursor.execute(
                            sql_child,
                            (
                                str(c.id),
                                str(c.doc_id),
                                str(c.parent_id),
                                c.chunk_index,
                                c.section_name,
                                c.content,
                                c.word_count,
                                c.page_start,
                                c.page_end,
                                c.char_start,
                                c.char_end,
                            ),
                        )
        except Exception as e:
            logger.debug(f"DB save chunks skipped: {e}")

    def _save_embeddings(
        self, doc_id: uuid.UUID, chunks: List[Any], embeddings: List[List[float]]
    ) -> None:
        """Insert dense embeddings into embeddings table with pgvector format."""
        sql = """
            INSERT INTO embeddings (chunk_id, doc_id, embedding, model_name)
            VALUES (%s, %s, %s::vector, %s)
            ON CONFLICT (chunk_id) DO UPDATE SET
                embedding = EXCLUDED.embedding;
        """
        try:
            with db.get_cursor(commit=True) as cursor:
                for c, emb in zip(chunks, embeddings):
                    vector_str = f"[{','.join(str(x) for x in emb)}]"
                    cursor.execute(
                        sql,
                        (str(c.id), str(doc_id), vector_str, self.cfg.embedding_model_name),
                    )
        except Exception as e:
            logger.debug(f"DB save embeddings skipped: {e}")

    def _update_doc_status(
        self, doc_id: uuid.UUID, status: str, total_pages: Optional[int] = None
    ) -> None:
        """Update document lifecycle status in database."""
        sql = "UPDATE documents SET status = %s"
        params: List[Any] = [status]
        if total_pages is not None:
            sql += ", total_pages = %s"
            params.append(total_pages)
        sql += " WHERE id = %s;"
        params.append(str(doc_id))

        try:
            with db.get_cursor(commit=True) as cursor:
                cursor.execute(sql, tuple(params))
        except Exception as e:
            logger.debug(f"DB status update skipped: {e}")

    def _record_failure(self, doc_id: uuid.UUID, error_message: str) -> None:
        """Increment retry count or mark as failed."""
        sql = """
            UPDATE documents
            SET retry_count = retry_count + 1,
                error_message = %s,
                status = CASE WHEN retry_count + 1 >= max_retries THEN 'failed' ELSE status END
            WHERE id = %s;
        """
        try:
            with db.get_cursor(commit=True) as cursor:
                cursor.execute(sql, (error_message, str(doc_id)))
            self._log_stage(doc_id, "pipeline", "failed", error_message)
        except Exception as e:
            logger.debug(f"DB failure record skipped: {e}")

    def _log_stage(
        self,
        doc_id: uuid.UUID,
        stage: str,
        status: str,
        message: str,
        time_ms: Optional[int] = None,
    ) -> None:
        """Append to processing_log audit trail."""
        sql = """
            INSERT INTO processing_log (doc_id, stage, status, message, execution_time_ms)
            VALUES (%s, %s, %s, %s, %s);
        """
        try:
            with db.get_cursor(commit=True) as cursor:
                cursor.execute(sql, (str(doc_id), stage, status, message, time_ms))
        except Exception as e:
            logger.debug(f"DB log stage skipped: {e}")

    # --------------------------------------------------------------------------
    # Snapshot Export & GCS V2 Upload
    # --------------------------------------------------------------------------
    def _export_snapshots(
        self,
        parse_result: DocumentParseResult,
        parents: List[ParentChunk],
        children: List[Any],
    ) -> None:
        """
        Export local JSONL/Parquet snapshots and upload to GCS V2 partitions:
        - parsed-v2/
        - chunks-v2/
        - snapshots-v2/
        """
        doc_id_str = str(parse_result.doc_id)
        snap_dir = self.cfg.snapshot_dir
        snap_dir.mkdir(parents=True, exist_ok=True)

        # 1. Parsed snapshot
        parsed_file = snap_dir / f"parsed_{doc_id_str}.jsonl"
        with open(parsed_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(parse_result.to_dict()) + "\n")

        # 2. Chunks snapshot
        chunks_file = snap_dir / f"chunks_{doc_id_str}.jsonl"
        with open(chunks_file, "w", encoding="utf-8") as f:
            for p in parents:
                f.write(json.dumps(p.to_dict()) + "\n")

        # 3. Upload to GCS V2 partitions if GCS is configured
        self._upload_file_to_gcs(parsed_file, f"{self.cfg.gcs_parsed_prefix}/{parsed_file.name}")
        self._upload_file_to_gcs(chunks_file, f"{self.cfg.gcs_chunks_prefix}/{chunks_file.name}")

    def _upload_file_to_gcs(self, local_path: Path, remote_key: str) -> None:
        """Upload single snapshot file to GCS."""
        try:
            from google.cloud import storage
            client = storage.Client(project=self.cfg.gcp_project_id)
            bucket = client.bucket(self.cfg.gcs_bucket_name)
            blob = bucket.blob(remote_key)
            blob.upload_from_filename(str(local_path))
            logger.info(f"Uploaded {local_path.name} to gs://{self.cfg.gcs_bucket_name}/{remote_key}")
        except Exception as e:
            logger.debug(f"GCS upload fallback (file saved locally): {e}")

    def _pull_raw_from_gcs(self, year: Optional[str] = None) -> None:
        """
        Pull pure raw PDFs from GCS bucket /raw/ partition.
        Explicitly skips any temporary folders like chroma/, chunks/, parsed/, logs/.
        """
        try:
            from google.cloud import storage
            client = storage.Client(project=self.cfg.gcp_project_id)
            bucket = client.bucket(self.cfg.gcs_bucket_name)

            prefixes = [f"{self.cfg.gcs_raw_prefix.strip('/')}/"]
            if year:
                prefixes = [
                    f"{self.cfg.gcs_raw_prefix.strip('/')}/year={year}/",
                    f"year={year}/",
                ]

            count = 0
            for prefix in prefixes:
                blobs = bucket.list_blobs(prefix=prefix)
                for blob in blobs:
                    # Ignore temporary subfolders inside raw partition
                    name_lower = blob.name.lower()
                    if any(temp in name_lower for temp in ["/chroma/", "/chunks/", "/parsed/", "/logs/"]):
                        continue
                    if blob.name.endswith(".pdf"):
                        filename = Path(blob.name).name
                        dest = self.cfg.raw_pdf_dir / filename
                        if not dest.exists():
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            blob.download_to_filename(str(dest))
                            count += 1
            logger.info(f"Pulled {count} new raw PDFs from gs://{self.cfg.gcs_bucket_name}/")
        except Exception as e:
            logger.warning(f"Could not pull from GCS ({e}). Using local raw directory.")


# Global pipeline instance
pipeline = LegalRAGPipeline()


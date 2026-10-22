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
import re
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid

from scrag.chunker import HierarchicalChunker, ParentChunk
from scrag.config import Settings, settings as global_settings
from scrag.database import db
from scrag.embedder import LocalEmbedder, embedder
from scrag.ids import compute_sha256, generate_doc_id
from scrag.models import DocumentRecord
from scrag.parser import DocumentParseResult, SupremeCourtParser
from scrag.tracker import ExecutionTracker, extract_year_from_path, tracker

logger = logging.getLogger(__name__)


def resolve_raw_prefixes(base_prefix: str, year: Optional[str] = None) -> List[str]:
    """
    Resolve recursive candidate prefixes for raw PDF traversal in GCS.
    Ensures 'raw/Files-for-test/' and all nested 'year=*' folders are covered.
    """
    clean = base_prefix.strip("/")
    candidates: List[str] = []

    if year:
        y_part = f"year={year}" if not str(year).startswith("year=") else str(year)
        if "Files-for-test" in clean:
            candidates.append(f"{clean}/{y_part}/")
        else:
            candidates.append(f"{clean}/Files-for-test/{y_part}/")
            candidates.append(f"{clean}/{y_part}/")
        candidates.append(f"raw/Files-for-test/{y_part}/")
        candidates.append(f"Files-for-test/{y_part}/")
        candidates.append(f"{y_part}/")
    else:
        if "Files-for-test" in clean:
            candidates.append(f"{clean}/")
        else:
            candidates.append(f"{clean}/Files-for-test/")
            candidates.append(f"{clean}/")
        candidates.append("raw/Files-for-test/")
        candidates.append("Files-for-test/")
        candidates.append("raw/")

    # Deduplicate while preserving order
    seen: Set[str] = set()
    result: List[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result


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
    # GCS Blob Discovery & Database Registration
    # --------------------------------------------------------------------------
    def list_gcs_raw_blobs(
        self, prefix: Optional[str] = None, year: Optional[str] = None
    ) -> List[Any]:
        """
        Recursively find all raw PDF blobs under target GCS prefix (e.g. raw/Files-for-test/).
        Filters out temporary subfolders (chroma/, chunks/, parsed/, logs/) and non-PDFs.
        """
        try:
            from google.cloud import storage
            client = storage.Client(project=self.cfg.gcp_project_id)
            bucket = client.bucket(self.cfg.gcs_bucket_name)

            base_prefix = prefix or self.cfg.gcs_raw_prefix
            prefixes = resolve_raw_prefixes(base_prefix, year=year)

            logger.info(f"Scanning GCS prefixes for raw PDFs in gs://{self.cfg.gcs_bucket_name}: {prefixes}")

            discovered: List[Any] = []
            seen_names: Set[str] = set()

            for pfx in prefixes:
                blobs = bucket.list_blobs(prefix=pfx)
                for blob in blobs:
                    b_name = blob.name
                    if b_name in seen_names:
                        continue
                    seen_names.add(b_name)

                    # Skip temporary subfolders
                    name_lower = b_name.lower()
                    if any(temp in name_lower for temp in ["/chroma/", "/chunks/", "/parsed/", "/logs/", "/tmp/"]):
                        continue

                    # Filter only valid PDFs
                    if name_lower.endswith(".pdf"):
                        # If year filter is active, verify year matches
                        if year:
                            blob_year = extract_year_from_path(b_name)
                            if str(year) != blob_year:
                                continue
                        discovered.append(blob)

            logger.info(f"Discovered {len(discovered)} valid raw PDF blobs in GCS.")
            return discovered
        except Exception as e:
            logger.warning(f"Could not list blobs from GCS ({e}).")
            return []

    def _register_doc_in_db(
        self, doc_id: uuid.UUID, sha256_hash: str, filename: str, file_size: int
    ) -> None:
        """Register document in PostgreSQL documents queue."""
        sql = """
            INSERT INTO documents (id, sha256_hash, source_filename, file_size_bytes, status)
            VALUES (%s, %s, %s, %s, 'pending')
            ON CONFLICT (sha256_hash) DO NOTHING;
        """
        try:
            with db.get_cursor(commit=True) as cursor:
                cursor.execute(sql, (str(doc_id), sha256_hash, filename, file_size))
        except Exception as e:
            logger.debug(f"DB insert document skipped/failed: {e}")

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
        gcs_prefix: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        Execute batch processing on raw PDF documents.
        Supports both direct recursive GCS streaming and local directory batches.

        Args:
            source_dir: Local path containing raw PDFs (default: data/raw_pdfs).
            from_gcs: If True, recursively pulls & processes PDFs from GCS.
            limit: Maximum documents to process in this run.
            resume: If True, skips documents already exported according to state tracker.
            year: Optional year partition filter (e.g. '1950').
            gcs_prefix: Optional custom GCS prefix to scan.

        Returns:
            Dict[str, int]: Processing outcome metrics.
        """
        metrics = {"total": 0, "processed": 0, "skipped": 0, "failed": 0}
        batch_limit = limit or 1000000
        count = 0

        # Sync remote tracker if resuming
        if resume and from_gcs:
            self.tracker.sync_from_gcs()

        if from_gcs:
            # ==================================================================
            # GCS Recursive Traversal & Direct Batch Processing
            # ==================================================================
            blobs = self.list_gcs_raw_blobs(prefix=gcs_prefix, year=year)
            logger.info(f"Processing GCS batch: {len(blobs)} candidates found.")

            for blob in blobs:
                if count >= batch_limit:
                    logger.info(f"Reached batch limit of {batch_limit}. Stopping.")
                    break

                blob_name = blob.name
                blob_year = extract_year_from_path(blob_name, fallback=year or "unknown")

                # Level 1 fast-resume check: Skip if already completed in tracker
                if resume and self.tracker.is_blob_completed(blob_name):
                    logger.info(f"Blob already completed according to tracker: {blob_name}. Skipping.")
                    metrics["skipped"] += 1
                    continue

                logger.info(f"Streaming {blob_name} (year={blob_year})...")
                try:
                    raw_bytes = blob.download_as_bytes()
                except Exception as e:
                    logger.error(f"Failed to download blob {blob_name}: {e}")
                    metrics["failed"] += 1
                    continue

                sha256 = compute_sha256(raw_bytes)
                doc_id, _ = generate_doc_id(sha256)

                # Level 2 fast-resume check: Skip if content SHA-256 already completed
                if resume and self.tracker.is_completed(sha256):
                    logger.info(f"[{doc_id}] Document SHA-256 already completed. Skipping.")
                    metrics["skipped"] += 1
                    continue

                # Local cache copy for audit / inspectability
                filename = Path(blob_name).name
                local_dir = self.cfg.raw_pdf_dir / f"year={blob_year}"
                local_dir.mkdir(parents=True, exist_ok=True)
                local_file = local_dir / filename
                if not local_file.exists():
                    try:
                        local_file.write_bytes(raw_bytes)
                    except Exception as e:
                        logger.debug(f"Local cache write skipped: {e}")

                # Register in tracker & database
                self.tracker.register_document(
                    local_file,
                    raw_bytes,
                    year=blob_year,
                    gcs_blob_name=blob_name,
                )
                self._register_doc_in_db(doc_id, sha256, filename, len(raw_bytes))

                doc_record = DocumentRecord(
                    id=doc_id,
                    sha256_hash=sha256,
                    source_filename=filename,
                    file_size_bytes=len(raw_bytes),
                    status="pending",
                )

                metrics["total"] += 1
                success = self.process_document(
                    doc_record,
                    raw_bytes=raw_bytes,
                    year=blob_year,
                    gcs_blob_name=blob_name,
                )
                if success:
                    metrics["processed"] += 1
                else:
                    metrics["failed"] += 1

                count += 1

                # Periodically sync tracker manifest to GCS (every 10 docs)
                if count % 10 == 0:
                    self.tracker.save_local()
                    self.tracker.sync_to_gcs()

        else:
            # ==================================================================
            # Local Filesystem Traversal
            # ==================================================================
            raw_dir = source_dir or self.cfg.raw_pdf_dir
            if raw_dir.exists():
                pdf_files = [p for p in raw_dir.rglob("*.pdf") if p.is_file()]
                logger.info(f"Discovered {len(pdf_files)} local PDF files under {raw_dir}")

                for pdf_file in pdf_files:
                    if count >= batch_limit:
                        break

                    file_year = extract_year_from_path(pdf_file, fallback=year or "unknown")
                    if year and str(year) != file_year:
                        continue

                    try:
                        raw_bytes = pdf_file.read_bytes()
                    except Exception as e:
                        logger.error(f"Failed to read local file {pdf_file}: {e}")
                        metrics["failed"] += 1
                        continue

                    sha256 = compute_sha256(raw_bytes)
                    doc_id, _ = generate_doc_id(sha256)

                    if resume and self.tracker.is_completed(sha256):
                        logger.info(f"[{doc_id}] Document {pdf_file.name} already exported. Skipping.")
                        metrics["skipped"] += 1
                        continue

                    self.tracker.register_document(pdf_file, raw_bytes, year=file_year)
                    self._register_doc_in_db(doc_id, sha256, pdf_file.name, len(raw_bytes))

                    doc_record = DocumentRecord(
                        id=doc_id,
                        sha256_hash=sha256,
                        source_filename=pdf_file.name,
                        file_size_bytes=len(raw_bytes),
                        status="pending",
                    )

                    metrics["total"] += 1
                    success = self.process_document(
                        doc_record,
                        raw_bytes=raw_bytes,
                        year=file_year,
                    )
                    if success:
                        metrics["processed"] += 1
                    else:
                        metrics["failed"] += 1

                    count += 1

        # Final state sync
        self.tracker.save_local()
        if from_gcs:
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
            file_year = extract_year_from_path(file_path)

            # Register in state tracker
            self.tracker.register_document(file_path, raw_bytes, year=file_year)

            if self.tracker.is_completed(sha256_hash):
                logger.debug(f"Document {file_path.name} already completed in tracker.")
                return doc_id

            self._register_doc_in_db(doc_id, sha256_hash, file_path.name, file_size)
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
    def process_document(
        self,
        doc_record: DocumentRecord,
        raw_bytes: Optional[bytes] = None,
        year: Optional[str] = None,
        gcs_blob_name: Optional[str] = None,
    ) -> bool:
        """
        Run document through full pipeline lifecycle:
        1. Parse (scrag.parser)
        2. Chunk (scrag.chunker)
        3. Embed & Index (scrag.embedder + pgvector)
        4. Snapshot & GCS V2 Structured Export (JSONL & Parquet under year=YYYY/)
        """
        doc_id = doc_record.id
        start_time = time.time()
        logger.info(f"[{doc_id}] Processing document {doc_record.source_filename} (status={doc_record.status})...")

        # Resolve raw bytes if not supplied
        if raw_bytes is None:
            raw_path = self.cfg.raw_pdf_dir / (doc_record.source_filename or f"{doc_id}.pdf")
            if not raw_path.exists():
                # Try recursive search under raw directory
                found = list(self.cfg.raw_pdf_dir.rglob(f"*{doc_record.source_filename}*"))
                if not found:
                    found = list(self.cfg.raw_pdf_dir.rglob(f"*{doc_record.sha256_hash[:8]}*.pdf"))
                if found:
                    raw_path = found[0]

            if not raw_path.exists():
                self._record_failure(doc_id, f"Raw PDF not found on disk: {raw_path}")
                return False

            try:
                raw_bytes = raw_path.read_bytes()
            except Exception as e:
                self._record_failure(doc_id, f"Could not read PDF bytes from disk: {e}")
                return False

        try:
            sha256 = doc_record.sha256_hash
            doc_year = year or extract_year_from_path(gcs_blob_name or doc_record.source_filename or "", fallback="unknown")

            # Stage 1: Parse
            parse_result = self.parser.parse_pdf(raw_bytes, filename_hint=doc_record.source_filename)
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

            # Stage 4: Local Snapshots & GCS V2 Structured Upload
            gcs_uris = self._export_snapshots(parse_result, parent_chunks, all_children, year=doc_year)
            self._update_doc_status(doc_id, "exported")

            elapsed_ms = int((time.time() - start_time) * 1000)
            if gcs_blob_name:
                gcs_uris["raw"] = f"gs://{self.cfg.gcs_bucket_name}/{gcs_blob_name}"

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
    # Snapshot Export & Structured GCS V2 Upload
    # --------------------------------------------------------------------------
    def _export_snapshots(
        self,
        parse_result: DocumentParseResult,
        parents: List[ParentChunk],
        children: List[Any],
        year: str = "unknown",
    ) -> Dict[str, str]:
        """
        Export JSONL and Parquet snapshots and upload to structured GCS V2 partitions:
        - gs://sc-legal-rag-data-v2/parsed-v2/year=YYYY/
        - gs://sc-legal-rag-data-v2/chunks-v2/year=YYYY/
        """
        doc_id_str = str(parse_result.doc_id)
        year_clean = str(year).strip()
        year_partition = f"year={year_clean}" if not year_clean.startswith("year=") else year_clean

        snap_dir = self.cfg.snapshot_dir
        snap_year_dir = snap_dir / year_partition
        snap_dir.mkdir(parents=True, exist_ok=True)
        snap_year_dir.mkdir(parents=True, exist_ok=True)

        gcs_uris: Dict[str, str] = {}

        # 1. Parsed snapshot (JSONL) - saved in both root snapshot_dir and snap_year_dir
        parsed_payload = json.dumps(parse_result.to_dict()) + "\n"
        parsed_file = snap_dir / f"parsed_{doc_id_str}.jsonl"
        parsed_year_file = snap_year_dir / f"parsed_{doc_id_str}.jsonl"
        with open(parsed_file, "w", encoding="utf-8") as f:
            f.write(parsed_payload)
        if parsed_year_file != parsed_file:
            with open(parsed_year_file, "w", encoding="utf-8") as f:
                f.write(parsed_payload)

        # 2. Parsed snapshot (Parquet)
        parsed_parquet = snap_year_dir / f"parsed_{doc_id_str}.parquet"
        try:
            import pandas as pd
            stitched = parse_result.stitched_text or ""
            parsed_row = {
                "doc_id": doc_id_str,
                "sha256_hash": parse_result.sha256_hash,
                "year": year_clean,
                "total_pages": parse_result.total_pages,
                "file_size_bytes": getattr(parse_result, "file_size_bytes", len(stitched)),
                "total_chars": len(stitched),
                "total_words": len(stitched.split()),
                "stitched_text": stitched,
                "page_offset_map": json.dumps({str(k): v for k, v in parse_result.page_offset_map.items()}),
                "detected_headers": json.dumps(getattr(parse_result, "detected_headers", [])),
                "detected_footers": json.dumps(getattr(parse_result, "detected_footers", [])),
            }
            pd.DataFrame([parsed_row]).to_parquet(parsed_parquet, engine="pyarrow", compression="snappy")
        except Exception as e:
            logger.warning(f"Parquet export for parsed failed: {e}")

        # 3. Chunks snapshot (JSONL) - saved in both root snapshot_dir and snap_year_dir
        chunks_payload = "".join(json.dumps(p.to_dict()) + "\n" for p in parents)
        chunks_file = snap_dir / f"chunks_{doc_id_str}.jsonl"
        chunks_year_file = snap_year_dir / f"chunks_{doc_id_str}.jsonl"
        with open(chunks_file, "w", encoding="utf-8") as f:
            f.write(chunks_payload)
        if chunks_year_file != chunks_file:
            with open(chunks_year_file, "w", encoding="utf-8") as f:
                f.write(chunks_payload)

        # 4. Chunks snapshot (Parquet)
        chunks_parquet = snap_year_dir / f"chunks_{doc_id_str}.parquet"
        try:
            import pandas as pd
            chunk_rows = []
            for c in children:
                chunk_rows.append({
                    "doc_id": str(c.doc_id),
                    "parent_id": str(c.parent_id),
                    "chunk_id": str(c.id),
                    "chunk_index": c.chunk_index,
                    "section_name": c.section_name or "",
                    "content": c.content or "",
                    "word_count": c.word_count,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "char_start": c.char_start,
                    "char_end": c.char_end,
                    "year": year_clean,
                })
            if chunk_rows:
                pd.DataFrame(chunk_rows).to_parquet(chunks_parquet, engine="pyarrow", compression="snappy")
        except Exception as e:
            logger.debug(f"Parquet export for chunks skipped: {e}")

        # 5. Upload to structured GCS partitions:
        # gs://sc-legal-rag-data-v2/parsed-v2/year=YYYY/
        # gs://sc-legal-rag-data-v2/chunks-v2/year=YYYY/
        parsed_key = f"{self.cfg.gcs_parsed_prefix}/{year_partition}/{parsed_file.name}"
        self._upload_file_to_gcs(parsed_file, parsed_key)
        gcs_uris["parsed_jsonl"] = f"gs://{self.cfg.gcs_bucket_name}/{parsed_key}"

        if parsed_parquet.exists():
            parsed_pq_key = f"{self.cfg.gcs_parsed_prefix}/{year_partition}/{parsed_parquet.name}"
            self._upload_file_to_gcs(parsed_parquet, parsed_pq_key)
            gcs_uris["parsed_parquet"] = f"gs://{self.cfg.gcs_bucket_name}/{parsed_pq_key}"

        chunks_key = f"{self.cfg.gcs_chunks_prefix}/{year_partition}/{chunks_file.name}"
        self._upload_file_to_gcs(chunks_file, chunks_key)
        gcs_uris["chunks_jsonl"] = f"gs://{self.cfg.gcs_bucket_name}/{chunks_key}"

        if chunks_parquet.exists():
            chunks_pq_key = f"{self.cfg.gcs_chunks_prefix}/{year_partition}/{chunks_parquet.name}"
            self._upload_file_to_gcs(chunks_parquet, chunks_pq_key)
            gcs_uris["chunks_parquet"] = f"gs://{self.cfg.gcs_bucket_name}/{chunks_pq_key}"

        return gcs_uris

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


"""
Unit tests for scrag.tracker (execution state tracker & resume capability).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

from scrag.tracker import ExecutionTracker, FileTrackRecord


class TestExecutionTracker:
    """Validate document state tracking, stage transitions, and persistence."""

    def test_document_registration_and_year_extraction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state_tracker.json"
            tracker = ExecutionTracker(state_file=state_file)

            raw_bytes = b"%PDF-1.4 sample case 1950 judgment"
            file_path = Path("raw/year=1950/judgment_001.pdf")

            record = tracker.register_document(file_path, raw_bytes)
            assert record.source_file == "judgment_001.pdf"
            assert record.year_partition == "1950"
            assert record.status == "pending"
            assert record.doc_id is not None
            assert record.sha256 is not None

    def test_stage_updates_and_completion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state_tracker.json"
            tracker = ExecutionTracker(state_file=state_file)

            raw_bytes = b"%PDF-1.4 sample case text"
            rec = tracker.register_document("test.pdf", raw_bytes)
            sha = rec.sha256

            # Stage 1: parsed
            tracker.update_stage(sha, "parsed", status="parsed", total_pages=12)
            assert tracker.get_record(sha).status == "parsed"
            assert "parsed" in tracker.get_record(sha).stages_completed
            assert tracker.get_record(sha).total_pages == 12

            # Stage 2: chunked
            tracker.update_stage(sha, "chunked", status="chunked", parent_chunks=4, child_chunks=16)
            assert tracker.get_record(sha).parent_chunks == 4
            assert tracker.get_record(sha).child_chunks == 16

            # Stage 3: completed / exported
            gcs_uris = {
                "parsed": "gs://sc-legal-rag-data-v2/parsed-v2/parsed_123.jsonl",
                "chunks": "gs://sc-legal-rag-data-v2/chunks-v2/chunks_123.jsonl",
            }
            tracker.mark_completed(sha, gcs_uris=gcs_uris, execution_time_ms=350)
            assert tracker.is_completed(sha) is True
            assert tracker.get_record(sha).status == "exported"
            assert tracker.get_record(sha).gcs_uris["parsed"] == gcs_uris["parsed"]

    def test_failure_recording(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state_tracker.json"
            tracker = ExecutionTracker(state_file=state_file)

            raw_bytes = b"%PDF corrupt bytes"
            rec = tracker.register_document("corrupt.pdf", raw_bytes)
            tracker.mark_failed(rec.sha256, "Corrupt PDF stream")

            assert tracker.get_record(rec.sha256).status == "failed"
            assert "Corrupt PDF stream" in tracker.get_record(rec.sha256).error_notes

    def test_atomic_save_and_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state_tracker.json"
            tracker1 = ExecutionTracker(state_file=state_file)

            rec = tracker1.register_document("persist.pdf", b"test content")
            tracker1.update_stage(rec.sha256, "parsed", status="parsed", total_pages=5)
            tracker1.save_local()

            # Create new tracker instance reading the same file
            tracker2 = ExecutionTracker(state_file=state_file)
            loaded_rec = tracker2.get_record(rec.sha256)
            assert loaded_rec is not None
            assert loaded_rec.status == "parsed"
            assert loaded_rec.total_pages == 5

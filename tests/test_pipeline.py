"""
Unit tests for scrag.pipeline (lifecycle, idempotency, and execution).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

from scrag.models import DocumentRecord
from scrag.pipeline import LegalRAGPipeline
from tests.test_parser import create_synthetic_pdf


class TestLegalRAGPipeline:
    """Validate document lifecycle, idempotency, and pipeline stages."""

    def test_queue_document_idempotency(self):
        pdf_bytes = create_synthetic_pdf(num_pages=2)
        pipeline = LegalRAGPipeline()

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "sample_judgment.pdf"
            file_path.write_bytes(pdf_bytes)

            doc_id_1 = pipeline.queue_document_from_file(file_path)
            doc_id_2 = pipeline.queue_document_from_file(file_path)

            assert doc_id_1 is not None
            assert doc_id_1 == doc_id_2

    def test_process_document_stages_local(self):
        pdf_bytes = create_synthetic_pdf(num_pages=3)
        pipeline = LegalRAGPipeline()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            pipeline.cfg.raw_pdf_dir = tmp_path
            pipeline.cfg.snapshot_dir = tmp_path / "snapshots"
            pipeline.cfg.snapshot_dir.mkdir(parents=True, exist_ok=True)

            pdf_file = tmp_path / "test_case.pdf"
            pdf_file.write_bytes(pdf_bytes)

            doc_id = pipeline.queue_document_from_file(pdf_file)
            assert doc_id is not None

            doc_record = DocumentRecord(
                id=doc_id,
                sha256_hash="dummy_hash",
                source_filename="test_case.pdf",
                file_size_bytes=len(pdf_bytes),
                status="pending",
            )

            success = pipeline.process_document(doc_record)
            assert success is True

            # Verify local snapshots were created
            snapshots = list(pipeline.cfg.snapshot_dir.glob("*.jsonl"))
            assert len(snapshots) >= 2  # parsed and chunks jsonl

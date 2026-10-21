"""
Unit tests for scrag.parser and scrag.ids.
"""

from __future__ import annotations

import fitz
import pytest

from scrag.ids import (
    compute_sha256,
    generate_chunk_id,
    generate_doc_id,
    generate_page_id,
    generate_parent_id,
)
from scrag.parser import (
    SupremeCourtParser,
    compute_garbage_ratio,
    dehyphenate_text,
    normalize_smart_quotes_and_symbols,
    resolve_char_span_to_pages,
)


def create_synthetic_pdf(num_pages: int = 4) -> bytes:
    """Helper to generate in-memory synthetic PDF with headers, footers, body, and footnotes."""
    doc = fitz.open()

    for i in range(num_pages):
        page = doc.new_page(width=595, height=842)
        page_num = i + 1

        # Repeating Header in top band (y <= 100 pt, band ratio 0.12 of 842 = 101 pt)
        page.insert_text((72, 45), "REPORTABLE", fontsize=10)
        page.insert_text((72, 60), "IN THE SUPREME COURT OF INDIA", fontsize=10)

        # Body text (middle region)
        page.insert_text(
            (72, 150),
            f"This is the authoritative body paragraph on page {page_num} dealing with constitutional law.",
            fontsize=12,
        )
        page.insert_text(
            (72, 180),
            "The doctrine of basic structure was examined in detail by the Constitution Bench.",
            fontsize=12,
        )

        # Footnote on page 1 only (lower region y >= 631 pt, small font 8pt)
        if page_num == 1:
            page.insert_text(
                (72, 700),
                "1 Kesavananda Bharati v. State of Kerala (1973) 4 SCC 225.",
                fontsize=8,
            )

        # Repeating Footer in bottom band (y >= 740 pt, band ratio 0.88 of 842 = 740 pt)
        page.insert_text((270, 800), f"Page {page_num} of {num_pages}", fontsize=9)

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class TestDeterministicIDs:
    """Validate RFC 4122 UUIDv5 deterministic ID architecture."""

    def test_doc_id_deterministic(self):
        raw_pdf_1 = b"%PDF-1.4 test document content A"
        raw_pdf_2 = b"%PDF-1.4 test document content B"

        doc_id_1a, hash_1a = generate_doc_id(raw_pdf_1)
        doc_id_1b, hash_1b = generate_doc_id(raw_pdf_1)
        doc_id_2, hash_2 = generate_doc_id(raw_pdf_2)

        # Same bytes must produce identical UUIDv5 and SHA-256
        assert doc_id_1a == doc_id_1b
        assert hash_1a == hash_1b
        assert hash_1a == compute_sha256(raw_pdf_1)

        # Different bytes must produce different IDs
        assert doc_id_1a != doc_id_2
        assert hash_1a != hash_2

    def test_page_id_format(self):
        doc_id, _ = generate_doc_id(b"sample bytes")
        page_id_1 = generate_page_id(doc_id, 1)
        page_id_5 = generate_page_id(doc_id, 5)

        assert page_id_1 == f"{doc_id}:1"
        assert page_id_5 == f"{doc_id}:5"

        with pytest.raises(ValueError):
            generate_page_id(doc_id, 0)

    def test_parent_and_child_id_deterministic(self):
        doc_id, _ = generate_doc_id(b"sample legal document")

        parent_0 = generate_parent_id(doc_id, 0)
        parent_0_again = generate_parent_id(doc_id, 0)
        parent_1 = generate_parent_id(doc_id, 1)

        assert parent_0 == parent_0_again
        assert parent_0 != parent_1

        child_0_0 = generate_chunk_id(doc_id, 0, 0)
        child_0_0_again = generate_chunk_id(doc_id, 0, 0)
        child_0_1 = generate_chunk_id(doc_id, 0, 1)

        assert child_0_0 == child_0_0_again
        assert child_0_0 != child_0_1


class TestNormalizationAndDehyphenation:
    """Validate Unicode normalization, smart quotes, and hyphenation logic."""

    def test_smart_quotes_and_symbols(self):
        text = '“Chief Justice” stated: ‘The law is clear’ — referring to Article 21…'
        cleaned = normalize_smart_quotes_and_symbols(text)

        assert '"Chief Justice"' in cleaned
        assert "'The law is clear'" in cleaned
        assert " - " in cleaned
        assert "..." in cleaned

    def test_dehyphenation_standard_words(self):
        text = "The juris-\nprudence of this Court is consid-\neration of justice."
        cleaned = dehyphenate_text(text)

        assert "jurisprudence" in cleaned
        assert "consideration" in cleaned

    def test_dehyphenation_legal_compounds_preserved(self):
        text = "This is a quasi-\njudicial authority acting under sub-\nsection (2)."
        cleaned = dehyphenate_text(text)

        assert "quasi-judicial" in cleaned
        assert "sub-section" in cleaned


class TestParserTwoPassZoning:
    """Validate PyMuPDF two-pass coordinate parsing on synthetic documents."""

    def test_header_footer_detection_and_safeguard(self):
        pdf_bytes = create_synthetic_pdf(num_pages=4)
        parser = SupremeCourtParser()
        result = parser.parse_pdf(pdf_bytes)

        assert result.total_pages == 4
        assert len(result.pages) == 4

        # Header lines should be detected in pass 1
        assert any("reportable" in h for h in result.detected_headers)
        assert any("supreme court" in h for h in result.detected_headers)

        # Footers should be detected
        assert any("page" in f for f in result.detected_footers)

        for p in result.pages:
            # Repeating header should be stripped from cleaned text
            assert "REPORTABLE" not in p.cleaned_text
            assert "IN THE SUPREME COURT OF INDIA" not in p.cleaned_text

            # Repeating footer should be stripped
            assert "Page" not in p.cleaned_text

            # Crucial safeguard: Body text must be preserved!
            assert "authoritative body paragraph" in p.cleaned_text
            assert "basic structure" in p.cleaned_text

    def test_footnote_separation(self):
        pdf_bytes = create_synthetic_pdf(num_pages=3)
        parser = SupremeCourtParser()
        result = parser.parse_pdf(pdf_bytes)

        page_1 = result.pages[0]
        # Footnote must be extracted into structured footnotes
        assert len(page_1.footnotes) == 1
        fn = page_1.footnotes[0]
        assert "Kesavananda Bharati" in fn.text
        assert fn.marker == "1"

        # Footnote text must NOT be present in page 1's cleaned body flow
        assert "Kesavananda Bharati" not in page_1.cleaned_text

    def test_stitched_text_and_offset_mapping(self):
        pdf_bytes = create_synthetic_pdf(num_pages=3)
        parser = SupremeCourtParser()
        result = parser.parse_pdf(pdf_bytes)

        stitched = result.stitched_text
        assert len(stitched) > 0
        assert len(result.page_offset_map) == 3

        # Validate offsets
        for page_no, (start, end) in result.page_offset_map.items():
            assert end >= start
            extracted_slice = stitched[start:end]
            assert extracted_slice == result.pages[page_no - 1].cleaned_text

        # Test resolving arbitrary character spans to pages
        p1_start, p1_end = result.page_offset_map[1]
        p2_start, p2_end = result.page_offset_map[2]

        # Span purely on page 1
        start_p, end_p = resolve_char_span_to_pages(
            result.page_offset_map, p1_start, p1_start + 10
        )
        assert start_p == 1 and end_p == 1

        # Span bridging page 1 and page 2
        start_p, end_p = resolve_char_span_to_pages(
            result.page_offset_map, p1_end - 5, p2_start + 5
        )
        assert start_p == 1 and end_p == 2

    def test_garbage_ratio_and_triage(self):
        clean_text = "The High Court committed a manifest error of law."
        garbage_text = "The\x00\x01\x02\ufffd\ufffd\ufffd\x03\x04\x05 Court"

        assert compute_garbage_ratio(clean_text) == 0.0
        assert compute_garbage_ratio(garbage_text) > 0.30


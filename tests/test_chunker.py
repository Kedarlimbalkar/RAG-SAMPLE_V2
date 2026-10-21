"""
Unit tests for scrag.chunker (hierarchical section-aware chunking).
"""

from __future__ import annotations

import uuid
import pytest

from scrag.chunker import HierarchicalChunker
from scrag.config import Settings


def make_legal_sample_text() -> tuple[str, dict[int, tuple[int, int]]]:
    """Generate sample legal text with explicit section boundaries."""
    text = (
        "IN THE SUPREME COURT OF INDIA\n\n"
        "BRIEF FACTS\n"
        "The appellant was convicted under Section 302 of the Indian Penal Code. "
        "The incident occurred on 15th August 2018 in Mumbai. The prosecution examined ten witnesses. "
        "The trial court found the evidence of eye-witnesses consistent and reliable.\n\n"
        "ISSUES\n"
        "Whether the circumstantial evidence is sufficient to establish guilt beyond reasonable doubt. "
        "Whether the recovery of the alleged weapon complies with Section 27 of the Evidence Act.\n\n"
        "ARGUMENTS\n"
        "Learned Senior Counsel for the appellant contended that the chain of circumstances is broken. "
        "On the other hand, learned counsel for the State supported the concurrent findings.\n\n"
        "ANALYSIS\n"
        "We have carefully examined the evidence on record. It is a well-settled principle of law "
        "that in cases resting on circumstantial evidence, the circumstances must be cogently established. "
        "The High Court failed to notice glaring contradictions in the testimony.\n\n"
        "ORDER\n"
        "The appeal is allowed. The judgment and order of the High Court is set aside. "
        "The appellant is acquitted of all charges and directed to be released forthwith."
    )
    # 2 pages offset map
    mid = len(text) // 2
    offset_map = {1: (0, mid), 2: (mid, len(text))}
    return text, offset_map


class TestHierarchicalChunker:
    """Validate hierarchical section-aware chunking."""

    def test_section_detection(self):
        text, offset_map = make_legal_sample_text()
        chunker = HierarchicalChunker()
        doc_id = uuid.uuid4()

        parents = chunker.chunk_document(doc_id, text, offset_map)
        assert len(parents) >= 4

        section_names = [p.section_name for p in parents]
        assert "Facts" in section_names
        assert "Issues" in section_names
        assert "Arguments" in section_names
        assert "Analysis" in section_names
        assert "Order/Judgement" in section_names

    def test_parent_child_hierarchy_and_ids(self):
        text, offset_map = make_legal_sample_text()
        chunker = HierarchicalChunker()
        doc_id = uuid.uuid4()

        parents = chunker.chunk_document(doc_id, text, offset_map)
        for p in parents:
            assert p.doc_id == doc_id
            assert p.word_count > 0
            assert len(p.children) > 0
            for c in p.children:
                assert c.parent_id == p.id
                assert c.doc_id == doc_id
                assert c.section_name == p.section_name
                assert c.page_start <= c.page_end

    def test_section_boundary_invariant(self):
        """Verify no chunk crosses section boundaries."""
        text, offset_map = make_legal_sample_text()
        chunker = HierarchicalChunker()
        doc_id = uuid.uuid4()

        parents = chunker.chunk_document(doc_id, text, offset_map)
        for p in parents:
            # A parent chunk in Facts must never contain Order text
            if p.section_name == "Facts":
                assert "appellant is acquitted" not in p.content
            if p.section_name == "Order/Judgement":
                assert "incident occurred on 15th August" not in p.content


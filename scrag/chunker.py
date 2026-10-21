"""
Section-Aware Hierarchical Chunking Engine for Supreme Court Judgments.

Implements coarse-grained parent chunks (~1500 words, 150 overlap) and
fine-grained child chunks (~350 words, 50 overlap).

Boundary Invariants:
- Never crosses case boundaries.
- Never crosses legal section boundaries (Facts, Issues, Arguments, Analysis, Order).
- Maps exact page_start and page_end via page_offset_map.
- Generates deterministic RFC 4122 UUIDv5 identifiers.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import uuid

from scrag.config import Settings, settings as global_settings
from scrag.ids import generate_chunk_id, generate_parent_id
from scrag.parser import resolve_char_span_to_pages

# Indian Supreme Court canonical section patterns
LEGAL_SECTION_PATTERNS: List[Tuple[str, re.Pattern]] = [
    (
        "Facts",
        re.compile(
            r"^[ \t]*(?:\d+[\.\)]\s*)?(?:BRIEF\s+FACTS|FACTS\s+OF\s+THE\s+CASE|FACTUAL\s+BACKGROUND|FACTUAL\s+MATRIX|BACKGROUND\s+FACTS|FACTS)[ \t]*$",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
    (
        "Issues",
        re.compile(
            r"^[ \t]*(?:\d+[\.\)]\s*)?(?:QUESTIONS?\s+FOR\s+CONSIDERATION|POINTS?\s+FOR\s+DETERMINATION|ISSUES?\s+FRAMED|ISSUES?)[ \t]*$",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
    (
        "Arguments",
        re.compile(
            r"^[ \t]*(?:\d+[\.\)]\s*)?(?:SUBMISSIONS?\s+ON\s+BEHALF\s+OF|ARGUMENTS?\s+OF|CONTENTIONS?\s+OF|SUBMISSIONS?|ARGUMENTS?|CONTENTIONS?)[ \t]*$",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
    (
        "Analysis",
        re.compile(
            r"^[ \t]*(?:\d+[\.\)]\s*)?(?:ANALYSIS|CONSIDERATION|DISCUSSION|REASONING|OUR\s+CONSIDERATION|FINDINGS?)[ \t]*$",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
    (
        "Order/Judgement",
        re.compile(
            r"^[ \t]*(?:\d+[\.\)]\s*)?(?:ORDER|JUDGMENT|CONCLUSION|DISPOSITION|OPERATIVE\s+ORDER|DECISION)[ \t]*$",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
]


@dataclass
class DocumentSection:
    """Logical legal section identified within stitched text."""
    section_name: str
    content: str
    char_start: int
    char_end: int
    heading: str = ""


@dataclass
class ChildChunk:
    """Fine-grained child chunk (~350 words, 50 overlap) for vector retrieval."""
    id: uuid.UUID
    parent_id: uuid.UUID
    doc_id: uuid.UUID
    chunk_index: int
    parent_index: int
    section_name: str
    content: str
    word_count: int
    page_start: int
    page_end: int
    char_start: int
    char_end: int

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["id"] = str(self.id)
        d["parent_id"] = str(self.parent_id)
        d["doc_id"] = str(self.doc_id)
        return d


@dataclass
class ParentChunk:
    """Coarse-grained parent chunk (~1500 words, 150 overlap) providing narrative context."""
    id: uuid.UUID
    doc_id: uuid.UUID
    parent_index: int
    section_name: str
    content: str
    word_count: int
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    children: List[ChildChunk] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["id"] = str(self.id)
        d["doc_id"] = str(self.doc_id)
        d["children"] = [c.to_dict() for c in self.children]
        return d


class HierarchicalChunker:
    """
    Splits stitched Supreme Court judgment texts into hierarchical parent-child
    chunks respecting legal section boundaries.
    """

    def __init__(self, config: Optional[Settings] = None) -> None:
        self.cfg = config or global_settings
        self.parent_words = self.cfg.chunk_parent_words
        self.parent_overlap = self.cfg.chunk_parent_overlap
        self.child_words = self.cfg.chunk_child_words
        self.child_overlap = self.cfg.chunk_child_overlap

    def chunk_document(
        self,
        doc_id: uuid.UUID,
        stitched_text: str,
        page_offset_map: Dict[int, Tuple[int, int]],
    ) -> List[ParentChunk]:
        """
        Partition document into section-bounded ParentChunks and ChildChunks.

        Args:
            doc_id: Deterministic document UUIDv5.
            stitched_text: Full cleaned text stream across all pages.
            page_offset_map: Mapping of page_no -> (char_start, char_end).

        Returns:
            List[ParentChunk]: Hierarchical chunks ready for embedding and indexing.
        """
        if not stitched_text.strip():
            return []

        # 1. Segment text into legal sections
        sections = self._detect_sections(stitched_text)

        parent_chunks: List[ParentChunk] = []
        global_parent_index = 0

        # 2. Process each section independently (NEVER crossing section boundaries)
        for sec in sections:
            sec_parents = self._chunk_section(
                doc_id=doc_id,
                section=sec,
                start_parent_index=global_parent_index,
                page_offset_map=page_offset_map,
            )
            parent_chunks.extend(sec_parents)
            global_parent_index += len(sec_parents)

        return parent_chunks

    # --------------------------------------------------------------------------
    # Section Detection Engine
    # --------------------------------------------------------------------------
    def _detect_sections(self, stitched_text: str) -> List[DocumentSection]:
        """
        Scan stitched text for legal section headers and partition text spans.
        """
        # Find all section header candidate matches
        matches: List[Tuple[int, int, str, str]] = []  # (start, end, section_name, heading)

        for sec_name, pattern in LEGAL_SECTION_PATTERNS:
            for m in pattern.finditer(stitched_text):
                matches.append((m.start(), m.end(), sec_name, m.group(0).strip()))

        # Sort matches by appearance in document
        matches.sort(key=lambda x: x[0])

        if not matches:
            # Fallback if no explicit headers found: entire document is "General"
            return [
                DocumentSection(
                    section_name="General",
                    content=stitched_text,
                    char_start=0,
                    char_end=len(stitched_text),
                    heading="JUDGMENT",
                )
            ]

        sections: List[DocumentSection] = []

        # Content before the first detected section (Preamble / General)
        if matches[0][0] > 0:
            preamble_text = stitched_text[: matches[0][0]]
            if preamble_text.strip():
                sections.append(
                    DocumentSection(
                        section_name="General",
                        content=preamble_text,
                        char_start=0,
                        char_end=matches[0][0],
                        heading="PREAMBLE",
                    )
                )

        # Slice between consecutive matched headings
        for i, (m_start, m_end, sec_name, heading) in enumerate(matches):
            next_start = matches[i + 1][0] if i + 1 < len(matches) else len(stitched_text)
            content = stitched_text[m_start:next_start]

            if content.strip():
                sections.append(
                    DocumentSection(
                        section_name=sec_name,
                        content=content,
                        char_start=m_start,
                        char_end=next_start,
                        heading=heading,
                    )
                )

        return sections

    # --------------------------------------------------------------------------
    # Hierarchical Splitting
    # --------------------------------------------------------------------------
    def _chunk_section(
        self,
        doc_id: uuid.UUID,
        section: DocumentSection,
        start_parent_index: int,
        page_offset_map: Dict[int, Tuple[int, int]],
    ) -> List[ParentChunk]:
        """
        Decompose a single legal section into parent chunks and their child chunks.
        """
        parent_chunks: List[ParentChunk] = []
        words_with_spans = self._tokenize_words_with_spans(
            section.content, section.char_start
        )

        if not words_with_spans:
            return []

        total_words = len(words_with_spans)
        p_step = max(1, self.parent_words - self.parent_overlap)
        p_idx = start_parent_index

        for p_start in range(0, total_words, p_step):
            p_end = min(total_words, p_start + self.parent_words)
            p_slice = words_with_spans[p_start:p_end]

            p_content = " ".join(w[0] for w in p_slice)
            p_char_start = p_slice[0][1]
            p_char_end = p_slice[-1][2]
            p_page_start, p_page_end = resolve_char_span_to_pages(
                page_offset_map, p_char_start, p_char_end
            )

            parent_id = generate_parent_id(doc_id, p_idx)

            parent_chunk = ParentChunk(
                id=parent_id,
                doc_id=doc_id,
                parent_index=p_idx,
                section_name=section.section_name,
                content=p_content,
                word_count=len(p_slice),
                page_start=p_page_start,
                page_end=p_page_end,
                char_start=p_char_start,
                char_end=p_char_end,
                children=[],
            )

            # Generate child chunks inside this parent window
            child_chunks = self._generate_child_chunks(
                doc_id=doc_id,
                parent_id=parent_id,
                parent_index=p_idx,
                section_name=section.section_name,
                parent_word_slice=p_slice,
                page_offset_map=page_offset_map,
            )
            parent_chunk.children = child_chunks
            parent_chunks.append(parent_chunk)

            p_idx += 1

            if p_end >= total_words:
                break

        return parent_chunks

    def _generate_child_chunks(
        self,
        doc_id: uuid.UUID,
        parent_id: uuid.UUID,
        parent_index: int,
        section_name: str,
        parent_word_slice: List[Tuple[str, int, int]],
        page_offset_map: Dict[int, Tuple[int, int]],
    ) -> List[ChildChunk]:
        """
        Subdivide a parent chunk's words into child chunks (~350 words, 50 overlap).
        """
        child_chunks: List[ChildChunk] = []
        total_words = len(parent_word_slice)
        c_step = max(1, self.child_words - self.child_overlap)
        c_idx = 0

        for c_start in range(0, total_words, c_step):
            c_end = min(total_words, c_start + self.child_words)
            c_slice = parent_word_slice[c_start:c_end]

            c_content = " ".join(w[0] for w in c_slice)
            c_char_start = c_slice[0][1]
            c_char_end = c_slice[-1][2]
            c_page_start, c_page_end = resolve_char_span_to_pages(
                page_offset_map, c_char_start, c_char_end
            )

            child_id = generate_chunk_id(doc_id, parent_index, c_idx)

            child_chunk = ChildChunk(
                id=child_id,
                parent_id=parent_id,
                doc_id=doc_id,
                chunk_index=c_idx,
                parent_index=parent_index,
                section_name=section_name,
                content=c_content,
                word_count=len(c_slice),
                page_start=c_page_start,
                page_end=c_page_end,
                char_start=c_char_start,
                char_end=c_char_end,
            )
            child_chunks.append(child_chunk)
            c_idx += 1

            if c_end >= total_words:
                break

        return child_chunks

    def _tokenize_words_with_spans(
        self, text: str, offset_base: int
    ) -> List[Tuple[str, int, int]]:
        """
        Tokenize text into words with absolute character offsets (word, start, end).
        """
        tokens: List[Tuple[str, int, int]] = []
        for match in re.finditer(r"\S+", text):
            tokens.append(
                (
                    match.group(0),
                    offset_base + match.start(),
                    offset_base + match.end(),
                )
            )
        return tokens


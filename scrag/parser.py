"""
Production-Grade PyMuPDF Coordinate Parser for Supreme Court of India Judgments.

Features:
- Two-pass PyMuPDF block & span extraction with exact (x0, y0, x1, y1) coordinates.
- Dynamic header/footer removal across >= 30% page threshold in top/bottom bands.
- Strict safeguard: Body text outside top/bottom bands is NEVER stripped.
- Footnote separation based on coordinate bands, relative font sizes, and indicators.
- Column-aware reading order (two-column detection with fallback to vertical flow).
- Per-page triage: Garbage ratio & minimal character threshold triggering OCR fallback.
- Text normalization: NFKC normalization, smart quotes, dashes, legal de-hyphenation.
- Stitched text stream with exact page_no -> (start_char, end_char) offset mapping.
- Deterministic UUIDv5 identifiers from raw PDF bytes.
"""

from __future__ import annotations

import io
import math
import re
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid

import fitz  # PyMuPDF

from scrag.config import Settings, settings as global_settings
from scrag.ids import (
    compute_sha256,
    generate_doc_id,
    generate_page_id,
)

# Known Indian legal compounds where hyphens must be retained during de-hyphenation
LEGAL_HYPHENATED_COMPOUNDS: Set[str] = {
    "quasi-judicial",
    "pre-existing",
    "pre-condition",
    "sub-section",
    "sub-clause",
    "sub-rule",
    "suo-motu",
    "cross-examination",
    "non-obstante",
    "inter-alia",
    "inter-state",
    "ultra-vires",
    "intra-vires",
    "bona-fide",
    "mala-fide",
    "prima-facie",
    "well-settled",
    "re-examination",
    "ex-parte",
    "post-facto",
    "by-law",
    "by-laws",
    "in-camera",
    "locus-standi",
    "res-judicata",
    "non-compliance",
    "non-speaking",
    "co-operative",
    "inter-se",
    "ante-dated",
}

# Standalone page number patterns in top/bottom bands
RE_PAGE_NUMBER = re.compile(
    r"^(?:page\s*)?(\d+|[ivxlcdm]+)(?:\s*(?:of|/|-)\s*(\d+|[ivxlcdm]+))?$",
    re.IGNORECASE,
)
RE_BRACKETED_PAGE = re.compile(r"^\[\s*\d+\s*\]$|^\-\s*\d+\s*\-$|^\(\s*\d+\s*\)$")

# Supreme Court judgment top header tokens that often repeat
RE_SC_HEADER_TOKENS = re.compile(
    r"\b(reportable|non-reportable|in the supreme court of india|"
    r"civil appellate jurisdiction|criminal appellate jurisdiction|"
    r"writ petition|civil appeal|criminal appeal|special leave petition)\b",
    re.IGNORECASE,
)

# Footnote indicator prefix (e.g. "1 ", "[1]", "* ", "† ")
RE_FOOTNOTE_MARKER = re.compile(
    r"^\s*(\d{1,2}|\[\d{1,2}\]|\*|†|‡)\s+(.+)",
    re.DOTALL,
)


@dataclass
class TextSpan:
    """Exact span-level coordinate and typographic telemetry."""
    bbox: Tuple[float, float, float, float]
    text: str
    font_size: float
    font_name: str
    flags: int


@dataclass
class TextBlock:
    """Block-level container preserving coordinates and classification."""
    bbox: Tuple[float, float, float, float]
    text: str
    font_size: float
    font_name: str
    lines: List[str] = field(default_factory=list)
    is_header_footer: bool = False
    is_footnote: bool = False


@dataclass
class FootnoteItem:
    """Structured legal footnote extracted cleanly from the body flow."""
    marker: str
    text: str
    bbox: Tuple[float, float, float, float]
    page_no: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PageParseResult:
    """Complete per-page parsing telemetry and cleaned text stream."""
    page_no: int
    page_id: str
    doc_id: uuid.UUID
    width: float
    height: float
    raw_text: str
    cleaned_text: str
    char_start: int = 0
    char_end: int = 0
    is_ocr: bool = False
    garbage_ratio: float = 0.0
    footnotes: List[FootnoteItem] = field(default_factory=list)
    blocks: List[TextBlock] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["doc_id"] = str(self.doc_id)
        d["footnotes"] = [fn.to_dict() for fn in self.footnotes]
        return d


@dataclass
class DocumentParseResult:
    """Document-level parse result with stitched text and offset mapping."""
    doc_id: uuid.UUID
    sha256_hash: str
    total_pages: int
    file_size_bytes: int
    pages: List[PageParseResult]
    stitched_text: str
    page_offset_map: Dict[int, Tuple[int, int]]
    detected_headers: List[str]
    detected_footers: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": str(self.doc_id),
            "sha256_hash": self.sha256_hash,
            "total_pages": self.total_pages,
            "file_size_bytes": self.file_size_bytes,
            "stitched_text_length": len(self.stitched_text),
            "page_offset_map": {str(k): v for k, v in self.page_offset_map.items()},
            "detected_headers": self.detected_headers,
            "detected_footers": self.detected_footers,
            "pages": [p.to_dict() for p in self.pages],
        }


# =============================================================================
# Text Normalization & Cleaning Utilities
# =============================================================================

def normalize_whitespace(text: str) -> str:
    """Standardize horizontal whitespace while preserving newlines."""
    # Collapse tabs and non-breaking spaces into standard space
    text = re.sub(r"[^\S\r\n]+", " ", text)
    # Collapse multiple consecutive newlines to maximum of 2 (\n\n)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_smart_quotes_and_symbols(text: str) -> str:
    """
    Standardize Unicode quotes, apostrophes, dashes, and bullet marks.
    """
    # 1. Unicode NFKC normalization
    text = unicodedata.normalize("NFKC", text)

    # 2. Smart double quotes -> standard ASCII double quote
    text = re.sub(r"[\u201C\u201D\u00AB\u00BB\u201E\u2033\u201F]", '"', text)

    # 3. Smart single quotes & backticks -> standard ASCII single quote
    text = re.sub(r"[\u2018\u2019\u201A\u201B\u0060\u00B4\u2032]", "'", text)

    # 4. Em-dashes, en-dashes, horizontal bars -> standard hyphen with spacing
    text = re.sub(r"[\u2014\u2015]", " - ", text)
    text = re.sub(r"[\u2012\u2013]", "-", text)

    # 5. Ellipsis normalization
    text = re.sub(r"\u2026", "...", text)

    return text


def dehyphenate_text(text: str) -> str:
    """
    De-hyphenate words broken across line breaks while preserving legitimate
    legal compound words (e.g. 'quasi-judicial', 'sub-section').

    Pattern: word- \n word
    """
    def _replace_hyphen(match: re.Match) -> str:
        part1 = match.group(1)
        part2 = match.group(2)
        candidate = f"{part1.lower()}-{part2.lower()}"

        # If it's a known legal compound word, keep hyphen
        if candidate in LEGAL_HYPHENATED_COMPOUNDS:
            return f"{part1}-{part2}"

        # If both parts are capitalized Proper Nouns (e.g. New-Delhi), keep hyphen
        if part1.istitle() and part2.istitle():
            return f"{part1}-{part2}"

        # If part2 starts with lowercase or continuation, merge without hyphen
        if part2[0].islower():
            return f"{part1}{part2}"

        # Default fallback: merge
        return f"{part1}{part2}"

    # Match word ending with hyphen, followed by line break, followed by continuation word
    return re.sub(r"(\b[a-zA-Z]+)-\s*\n\s*([a-zA-Z]+\b)", _replace_hyphen, text)


def compute_garbage_ratio(text: str) -> float:
    """
    Calculate ratio of unprintable or corrupt glyphs in extracted text.
    High ratio (>0.25) triggers OCR fallback.
    """
    if not text:
        return 1.0

    total_chars = len(text)
    garbage_chars = 0

    for char in text:
        code = ord(char)
        # Check for unprintable control characters (excluding newline, cr, tab)
        if code < 32 and char not in ("\n", "\r", "\t"):
            garbage_chars += 1
        # Unicode replacement character or private use area
        elif code in (0xFFFD, 0xFEFF) or (0xE000 <= code <= 0xF8FF):
            garbage_chars += 1
        # Unusual symbols that indicate font decoding corruption
        elif code > 126 and unicodedata.category(char) in ("Co", "Cs", "Cn"):
            garbage_chars += 1

    return garbage_chars / total_chars


def resolve_char_span_to_pages(
    page_offset_map: Dict[int, Tuple[int, int]],
    char_start: int,
    char_end: int,
) -> Tuple[int, int]:
    """
    Map an arbitrary character span in the stitched document back to its
    exact start page and end page (1-indexed).
    """
    if not page_offset_map:
        return 1, 1

    start_page: Optional[int] = None
    end_page: Optional[int] = None

    sorted_pages = sorted(page_offset_map.keys())

    for page_no in sorted_pages:
        p_start, p_end = page_offset_map[page_no]

        if start_page is None:
            if char_start <= p_end:
                start_page = page_no

        if char_end <= p_end:
            end_page = page_no
            break

    # If start_page or end_page boundary lies beyond last registered page
    if start_page is None:
        start_page = sorted_pages[-1]
    if end_page is None:
        end_page = sorted_pages[-1]

    return start_page, end_page


# =============================================================================
# PyMuPDF Coordinate Parser Implementation
# =============================================================================

class SupremeCourtParser:
    """
    Robust, production-grade legal document parser for Supreme Court judgments.
    """

    def __init__(self, config: Optional[Settings] = None) -> None:
        self.cfg = config or global_settings

    def parse_pdf(
        self,
        pdf_source: Union[str, Path, bytes],
        filename_hint: Optional[str] = None,
    ) -> DocumentParseResult:
        """
        Execute two-pass coordinate parsing on PDF bytes or file path.

        Args:
            pdf_source: File path (str/Path) or raw PDF bytes.
            filename_hint: Optional original filename for metadata tracking.

        Returns:
            DocumentParseResult: Structured result with page offsets & clean text.
        """
        raw_bytes: bytes
        if isinstance(pdf_source, (str, Path)):
            path = Path(pdf_source)
            if not path.is_file():
                raise FileNotFoundError(f"PDF file not found: {path}")
            raw_bytes = path.read_bytes()
            if filename_hint is None:
                filename_hint = path.name
        elif isinstance(pdf_source, bytes):
            raw_bytes = pdf_source
        else:
            raise TypeError(f"Unsupported pdf_source type: {type(pdf_source)}")

        # Deterministic document ID from SHA-256 of raw bytes
        doc_id, sha256_hash = generate_doc_id(raw_bytes)
        file_size_bytes = len(raw_bytes)

        # Open document with PyMuPDF
        doc = fitz.open(stream=raw_bytes, filetype="pdf")
        total_pages = len(doc)

        if total_pages == 0:
            doc.close()
            return DocumentParseResult(
                doc_id=doc_id,
                sha256_hash=sha256_hash,
                total_pages=0,
                file_size_bytes=file_size_bytes,
                pages=[],
                stitched_text="",
                page_offset_map={},
                detected_headers=[],
                detected_footers=[],
            )

        # ----------------------------------------------------------------------
        # Pass 1: Global Coordinate Zoning & Dynamic Header/Footer Detection
        # ----------------------------------------------------------------------
        header_patterns, footer_patterns = self._pass_1_detect_headers_footers(doc)

        # ----------------------------------------------------------------------
        # Pass 2: Reading Order, Footnote Extraction, and Page Cleaning
        # ----------------------------------------------------------------------
        pages: List[PageParseResult] = []
        for page_idx in range(total_pages):
            page_no = page_idx + 1
            page = doc[page_idx]
            page_result = self._pass_2_parse_page(
                page=page,
                page_no=page_no,
                doc_id=doc_id,
                header_patterns=header_patterns,
                footer_patterns=footer_patterns,
            )
            pages.append(page_result)

        doc.close()

        # ----------------------------------------------------------------------
        # Post-Processing: Cross-Page De-hyphenation & Document Stitching
        # ----------------------------------------------------------------------
        stitched_text, page_offset_map = self._stitch_document(pages)

        return DocumentParseResult(
            doc_id=doc_id,
            sha256_hash=sha256_hash,
            total_pages=total_pages,
            file_size_bytes=file_size_bytes,
            pages=pages,
            stitched_text=stitched_text,
            page_offset_map=page_offset_map,
            detected_headers=sorted(header_patterns),
            detected_footers=sorted(footer_patterns),
        )

    # --------------------------------------------------------------------------
    # Pass 1: Collect Header & Footer Repetition Candidates
    # --------------------------------------------------------------------------
    def _pass_1_detect_headers_footers(
        self, doc: fitz.Document
    ) -> Tuple[Set[str], Set[str]]:
        """
        Identify repeating lines located strictly in top and bottom bands across
        >= 30% of pages.
        """
        total_pages = len(doc)
        min_repeats = max(2, math.ceil(total_pages * self.cfg.parser_header_footer_threshold))

        top_line_occurrences: Dict[str, Set[int]] = defaultdict(set)
        bottom_line_occurrences: Dict[str, Set[int]] = defaultdict(set)

        for page_idx in range(total_pages):
            page_no = page_idx + 1
            page = doc[page_idx]
            height = page.rect.height

            top_boundary = height * self.cfg.parser_header_band_ratio
            bottom_boundary = height * self.cfg.parser_footer_band_ratio

            # Extract raw block dicts
            page_dict = page.get_text("dict", flags=fitz.TEXTFLAGS_SEARCH)
            blocks = page_dict.get("blocks", [])

            for block in blocks:
                if block.get("type", 0) != 0:  # 0 is text block
                    continue

                for line in block.get("lines", []):
                    line_bbox = line.get("bbox", (0, 0, 0, 0))
                    line_y0, line_y1 = line_bbox[1], line_bbox[3]

                    line_text = "".join(
                        span.get("text", "") for span in line.get("spans", [])
                    ).strip()

                    if not line_text:
                        continue

                    normalized_key = self._normalize_candidate_key(line_text)
                    if not normalized_key:
                        continue

                    # Top band check: line upper coordinate strictly within top ratio
                    if line_y1 <= top_boundary:
                        top_line_occurrences[normalized_key].add(page_no)

                    # Bottom band check: line lower coordinate strictly within bottom ratio
                    elif line_y0 >= bottom_boundary:
                        bottom_line_occurrences[normalized_key].add(page_no)

        # Filter by threshold >= 30% of pages
        detected_headers: Set[str] = {
            key
            for key, pages_seen in top_line_occurrences.items()
            if len(pages_seen) >= min_repeats
        }

        detected_footers: Set[str] = {
            key
            for key, pages_seen in bottom_line_occurrences.items()
            if len(pages_seen) >= min_repeats
        }

        return detected_headers, detected_footers

    def _normalize_candidate_key(self, text: str) -> str:
        """Create normalized lookup key for repeating header/footer line matching."""
        cleaned = unicodedata.normalize("NFKC", text).strip().lower()
        # Collapse whitespace
        cleaned = re.sub(r"\s+", " ", cleaned)
        # Strip variable standalone page numbers like "page 12 of 30", "12"
        cleaned_no_num = re.sub(r"\b\d+\b", "", cleaned).strip()
        return cleaned_no_num if cleaned_no_num else cleaned

    # --------------------------------------------------------------------------
    # Pass 2: Page Parsing, Zoning, Footnote Separation & Reading Order
    # --------------------------------------------------------------------------
    def _pass_2_parse_page(
        self,
        page: fitz.Page,
        page_no: int,
        doc_id: uuid.UUID,
        header_patterns: Set[str],
        footer_patterns: Set[str],
    ) -> PageParseResult:
        """
        Parse single page with coordinate-aware block classification.
        """
        page_id = generate_page_id(doc_id, page_no)
        width = float(page.rect.width)
        height = float(page.rect.height)

        top_boundary = height * self.cfg.parser_header_band_ratio
        bottom_boundary = height * self.cfg.parser_footer_band_ratio
        footnote_boundary = height * self.cfg.parser_footnote_band_ratio

        # Extract structured text blocks
        page_dict = page.get_text("dict", flags=fitz.TEXTFLAGS_SEARCH)
        raw_blocks = page_dict.get("blocks", [])

        extracted_text_blocks: List[TextBlock] = []
        body_font_sizes: List[float] = []

        # First collect all blocks and compute font statistics
        for b in raw_blocks:
            if b.get("type", 0) != 0:
                continue

            bbox = tuple(float(x) for x in b.get("bbox", (0, 0, 0, 0)))
            lines_data = b.get("lines", [])
            line_strings: List[str] = []
            block_font_sizes: List[float] = []
            block_font_names: List[str] = []

            for line in lines_data:
                spans = line.get("spans", [])
                line_str = "".join(s.get("text", "") for s in spans).strip()
                if line_str:
                    line_strings.append(line_str)
                for s in spans:
                    size = float(s.get("size", 0.0))
                    if size > 0:
                        block_font_sizes.append(size)
                    name = s.get("font", "")
                    if name:
                        block_font_names.append(name)

            if not line_strings:
                continue

            avg_font_size = (
                sum(block_font_sizes) / len(block_font_sizes) if block_font_sizes else 10.0
            )
            dominant_font_name = (
                max(set(block_font_names), key=block_font_names.count)
                if block_font_names
                else "default"
            )
            full_block_text = "\n".join(line_strings)

            # Collect font sizes outside header/footer zones to estimate body font
            if bbox[1] > top_boundary and bbox[3] < bottom_boundary:
                body_font_sizes.extend(block_font_sizes)

            extracted_text_blocks.append(
                TextBlock(
                    bbox=bbox,
                    text=full_block_text,
                    font_size=avg_font_size,
                    font_name=dominant_font_name,
                    lines=line_strings,
                )
            )

        # Determine median body font size
        median_body_font = (
            sorted(body_font_sizes)[len(body_font_sizes) // 2]
            if body_font_sizes
            else 10.5
        )
        footnote_font_threshold = median_body_font * self.cfg.parser_footnote_font_ratio

        # Classify and filter blocks
        filtered_body_blocks: List[TextBlock] = []
        footnotes: List[FootnoteItem] = []

        for blk in extracted_text_blocks:
            y0, y1 = blk.bbox[1], blk.bbox[3]
            block_text = blk.text.strip()
            norm_key = self._normalize_candidate_key(block_text)

            # 1. Header Check: ONLY in top band
            if y1 <= top_boundary:
                if (
                    norm_key in header_patterns
                    or RE_PAGE_NUMBER.match(block_text)
                    or RE_BRACKETED_PAGE.match(block_text)
                    or RE_SC_HEADER_TOKENS.search(block_text)
                ):
                    blk.is_header_footer = True
                    continue

            # 2. Footer Check: ONLY in bottom band
            elif y0 >= bottom_boundary:
                if (
                    norm_key in footer_patterns
                    or RE_PAGE_NUMBER.match(block_text)
                    or RE_BRACKETED_PAGE.match(block_text)
                ):
                    blk.is_header_footer = True
                    continue

            # 3. Footnote Check: in lower band and small font size or footnote marker
            if y0 >= footnote_boundary:
                marker_match = RE_FOOTNOTE_MARKER.match(block_text)
                is_small_font = blk.font_size <= footnote_font_threshold

                if is_small_font or marker_match:
                    blk.is_footnote = True
                    marker = marker_match.group(1) if marker_match else "*"
                    fn_content = marker_match.group(2) if marker_match else block_text
                    footnotes.append(
                        FootnoteItem(
                            marker=marker,
                            text=normalize_smart_quotes_and_symbols(fn_content.strip()),
                            bbox=blk.bbox,
                            page_no=page_no,
                        )
                    )
                    continue

            # SAFEGUARD: Body text outside bands is retained
            filtered_body_blocks.append(blk)

        # ----------------------------------------------------------------------
        # Strict Reading Order: Column Analysis and Flow Sorting
        # ----------------------------------------------------------------------
        sorted_body_blocks = self._sort_reading_order(filtered_body_blocks, width)

        # Stitch raw page text
        raw_text_parts = [b.text for b in sorted_body_blocks]
        raw_page_text = "\n\n".join(raw_text_parts)

        # ----------------------------------------------------------------------
        # Per-Page Triage: Minimum Characters & Garbage Ratio -> OCR Fallback
        # ----------------------------------------------------------------------
        garbage_ratio = compute_garbage_ratio(raw_page_text)
        is_ocr = False
        final_page_text = raw_page_text

        if (
            len(raw_page_text.strip()) < self.cfg.parser_min_page_chars
            or garbage_ratio > self.cfg.parser_max_garbage_ratio
        ):
            if self.cfg.ocr_enabled:
                ocr_text, ocr_success = self._run_ocr_fallback(page)
                if ocr_success and len(ocr_text.strip()) >= len(raw_page_text.strip()):
                    final_page_text = ocr_text
                    is_ocr = True
                    garbage_ratio = compute_garbage_ratio(final_page_text)
            else:
                is_ocr = True

        # ----------------------------------------------------------------------
        # Text Normalization & De-hyphenation for Current Page
        # ----------------------------------------------------------------------
        cleaned = normalize_smart_quotes_and_symbols(final_page_text)
        cleaned = dehyphenate_text(cleaned)
        cleaned = normalize_whitespace(cleaned)

        return PageParseResult(
            page_no=page_no,
            page_id=page_id,
            doc_id=doc_id,
            width=width,
            height=height,
            raw_text=raw_page_text,
            cleaned_text=cleaned,
            is_ocr=is_ocr,
            garbage_ratio=round(garbage_ratio, 4),
            footnotes=footnotes,
            blocks=sorted_body_blocks,
        )

    # --------------------------------------------------------------------------
    # Reading Order Sorting: Two-Column Detection & Vertical Ordering
    # --------------------------------------------------------------------------
    def _sort_reading_order(
        self, blocks: List[TextBlock], page_width: float
    ) -> List[TextBlock]:
        """
        Sort blocks into strict reading order. Detects two-column layouts
        (left column first, then right column), otherwise sorts vertically by y0.
        """
        if len(blocks) <= 1:
            return blocks

        mid_x = page_width / 2.0
        tolerance = 15.0

        left_blocks: List[TextBlock] = []
        right_blocks: List[TextBlock] = []
        spanning_blocks: List[TextBlock] = []

        for b in blocks:
            x0, _, x1, _ = b.bbox
            if x1 <= mid_x + tolerance and x0 < mid_x - tolerance:
                left_blocks.append(b)
            elif x0 >= mid_x - tolerance and x1 > mid_x + tolerance:
                right_blocks.append(b)
            else:
                spanning_blocks.append(b)

        # Detect if document has significant two-column layout
        is_two_column = (
            len(left_blocks) >= 2
            and len(right_blocks) >= 2
            and len(spanning_blocks) <= (len(left_blocks) + len(right_blocks)) * 0.3
        )

        if is_two_column:
            left_sorted = sorted(left_blocks, key=lambda b: (round(b.bbox[1], 1), b.bbox[0]))
            right_sorted = sorted(right_blocks, key=lambda b: (round(b.bbox[1], 1), b.bbox[0]))
            spanning_sorted = sorted(spanning_blocks, key=lambda b: (round(b.bbox[1], 1), b.bbox[0]))

            # Interleave spanning blocks by y position or append
            all_sorted: List[TextBlock] = []
            all_sorted.extend(spanning_sorted)
            all_sorted.extend(left_sorted)
            all_sorted.extend(right_sorted)
            # Re-sort preserving top spans before columns
            return sorted(all_sorted, key=lambda b: (
                0 if b in spanning_sorted and b.bbox[1] < page_width * 0.3 else 1,
                1 if b in right_sorted else 0,
                round(b.bbox[1], 1),
            ))

        # Single column standard flow: sort vertically by y0, snapping lines within 3pt
        return sorted(blocks, key=lambda b: (round(b.bbox[1] / 3.0) * 3.0, b.bbox[0]))

    # --------------------------------------------------------------------------
    # OCR Fallback Execution
    # --------------------------------------------------------------------------
    def _run_ocr_fallback(self, page: fitz.Page) -> Tuple[str, bool]:
        """
        Execute local OCR via PyMuPDF rendered pixmap and pytesseract.
        Fails gracefully if pytesseract or Tesseract binary is not installed.
        """
        try:
            import pytesseract
            from PIL import Image

            if self.cfg.tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = self.cfg.tesseract_cmd

            # Render high-resolution 300 DPI pixmap for OCR accuracy
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))

            ocr_text = pytesseract.image_to_string(img, config="--psm 1")
            return ocr_text, True
        except ImportError:
            # pytesseract or PIL is not installed
            return "", False
        except Exception:
            # Tesseract binary not configured or OCR failure
            return "", False

    # --------------------------------------------------------------------------
    # Document Stitching, Cross-Page De-hyphenation & Offset Mapping
    # --------------------------------------------------------------------------
    def _stitch_document(
        self, pages: List[PageParseResult]
    ) -> Tuple[str, Dict[int, Tuple[int, int]]]:
        """
        Stitch page cleaned text streams into unified document string.
        Resolves cross-page hyphen breaks and builds page_offset_map.
        """
        if not pages:
            return "", {}

        # 1. Resolve cross-page hyphen breaks (e.g. page N ends with 'appel-', page N+1 starts with 'lant')
        for i in range(len(pages) - 1):
            curr_page = pages[i]
            next_page = pages[i + 1]

            if not curr_page.cleaned_text or not next_page.cleaned_text:
                continue

            # Check if current page ends with a hyphenated word fragment
            match_curr = re.search(r"(\b[a-zA-Z]+)-\s*$", curr_page.cleaned_text)
            match_next = re.match(r"^\s*([a-zA-Z]+\b)", next_page.cleaned_text)

            if match_curr and match_next:
                part1 = match_curr.group(1)
                part2 = match_next.group(1)
                candidate = f"{part1.lower()}-{part2.lower()}"

                if candidate in LEGAL_HYPHENATED_COMPOUNDS:
                    merged = f"{part1}-{part2}"
                elif part2[0].islower():
                    merged = f"{part1}{part2}"
                else:
                    merged = f"{part1}-{part2}"

                # Update current page tail and next page head
                curr_page.cleaned_text = curr_page.cleaned_text[: match_curr.start()] + merged
                next_page.cleaned_text = next_page.cleaned_text[match_next.end():].lstrip()

        # 2. Build stitched text and exact character offset mapping
        stitched_parts: List[str] = []
        page_offset_map: Dict[int, Tuple[int, int]] = {}
        current_offset = 0
        delimiter = "\n\n"

        for idx, page in enumerate(pages):
            p_text = page.cleaned_text
            char_start = current_offset
            char_end = char_start + len(p_text)

            page.char_start = char_start
            page.char_end = char_end
            page_offset_map[page.page_no] = (char_start, char_end)

            stitched_parts.append(p_text)
            current_offset = char_end

            # Add delimiter between pages (not after the final page)
            if idx < len(pages) - 1:
                stitched_parts.append(delimiter)
                current_offset += len(delimiter)

        stitched_text = "".join(stitched_parts)
        return stitched_text, page_offset_map


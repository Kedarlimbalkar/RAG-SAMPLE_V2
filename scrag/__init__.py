"""
Supreme Court Legal RAG Pipeline (scrag).
"""

from scrag.config import Settings, settings
from scrag.ids import (
    compute_sha256,
    generate_chunk_id,
    generate_doc_id,
    generate_page_id,
    generate_parent_id,
)
from scrag.parser import (
    DocumentParseResult,
    FootnoteItem,
    PageParseResult,
    SupremeCourtParser,
    TextBlock,
    TextSpan,
    dehyphenate_text,
    normalize_smart_quotes_and_symbols,
    resolve_char_span_to_pages,
)

__version__ = "0.2.0"

__all__ = [
    "Settings",
    "settings",
    "generate_doc_id",
    "generate_page_id",
    "generate_parent_id",
    "generate_chunk_id",
    "compute_sha256",
    "SupremeCourtParser",
    "DocumentParseResult",
    "PageParseResult",
    "TextBlock",
    "TextSpan",
    "FootnoteItem",
    "dehyphenate_text",
    "normalize_smart_quotes_and_symbols",
    "resolve_char_span_to_pages",
]


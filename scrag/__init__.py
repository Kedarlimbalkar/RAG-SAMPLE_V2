"""
Supreme Court Legal RAG Pipeline (scrag).
"""

from scrag.chunker import ChildChunk, HierarchicalChunker, ParentChunk
from scrag.config import Settings, settings
from scrag.embedder import LocalEmbedder, embedder
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
from scrag.engine import LegalQueryResult, LegalRAGEngine, engine
from scrag.pipeline import LegalRAGPipeline, pipeline
from scrag.reranker import CrossEncoderReranker, reranker
from scrag.retriever import HybridRetriever, SearchResult, retriever
from scrag.router import QueryIntent, QueryRouter, RetrievalStrategy, router

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
    "HierarchicalChunker",
    "ParentChunk",
    "ChildChunk",
    "LocalEmbedder",
    "embedder",
    "LegalRAGPipeline",
    "pipeline",
    "HybridRetriever",
    "SearchResult",
    "retriever",
    "QueryIntent",
    "QueryRouter",
    "RetrievalStrategy",
    "router",
    "CrossEncoderReranker",
    "reranker",
    "LegalRAGEngine",
    "LegalQueryResult",
    "engine",
]


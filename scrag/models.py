"""
Data models and typed entities for Legal RAG Pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid


@dataclass
class DocumentRecord:
    id: uuid.UUID
    sha256_hash: str
    source_filename: Optional[str]
    file_size_bytes: int
    total_pages: int = 0
    status: str = "pending"  # pending, parsed, chunked, embedded, exported, failed, quarantined
    retry_count: int = 0
    max_retries: int = 3
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class PageRecord:
    id: str  # f"{doc_id}:{page_no}"
    doc_id: uuid.UUID
    page_no: int
    width: float
    height: float
    raw_text: str = ""
    cleaned_text: str = ""
    char_start: int = 0
    char_end: int = 0
    is_ocr: bool = False
    garbage_ratio: float = 0.0
    footnotes: List[Dict[str, Any]] = field(default_factory=list)
    created_at: Optional[datetime] = None


@dataclass
class MetadataRecord:
    doc_id: uuid.UUID
    case_title: Optional[str] = None
    case_number: Optional[str] = None
    petitioner: Optional[str] = None
    respondent: Optional[str] = None
    bench: Optional[List[str]] = None
    judgment_date: Optional[str] = None  # YYYY-MM-DD
    citation: Optional[str] = None
    raw_metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class ParentRecord:
    id: uuid.UUID  # UUIDv5(doc_id + parent_index)
    doc_id: uuid.UUID
    parent_index: int
    section_name: str
    content: str
    word_count: int
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    created_at: Optional[datetime] = None


@dataclass
class ChunkRecord:
    id: uuid.UUID  # UUIDv5(doc_id + parent_index + child_index)
    doc_id: uuid.UUID
    parent_id: uuid.UUID
    chunk_index: int
    section_name: str
    content: str
    word_count: int
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    created_at: Optional[datetime] = None


@dataclass
class EmbeddingRecord:
    chunk_id: uuid.UUID
    doc_id: uuid.UUID
    embedding: List[float]  # 1024-d float vector
    model_name: str = "BAAI/bge-m3"
    created_at: Optional[datetime] = None


@dataclass
class ProcessingLogRecord:
    id: Optional[int] = None
    doc_id: Optional[uuid.UUID] = None
    stage: str = "parsing"
    status: str = "started"  # started, success, retry, failed, quarantined
    message: Optional[str] = None
    execution_time_ms: Optional[int] = None
    created_at: Optional[datetime] = None


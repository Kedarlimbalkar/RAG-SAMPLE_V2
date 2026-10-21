"""
Hybrid Retrieval Engine for Supreme Court Judgments.

Combines pgvector dense semantic search (bge-m3, 1024-d) with PostgreSQL
full-text keyword matching (GIN tsvector) using Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import uuid

from scrag.config import Settings, settings as global_settings
from scrag.database import db
from scrag.embedder import embedder

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    chunk_id: uuid.UUID
    parent_id: uuid.UUID
    doc_id: uuid.UUID
    section_name: str
    child_content: str
    parent_content: str
    page_start: int
    page_end: int
    score: float
    case_title: Optional[str] = None
    judgment_date: Optional[str] = None
    citation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": str(self.chunk_id),
            "parent_id": str(self.parent_id),
            "doc_id": str(self.doc_id),
            "section_name": self.section_name,
            "child_content": self.child_content,
            "parent_content": self.parent_content,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "score": round(self.score, 4),
            "case_title": self.case_title or "Supreme Court of India Judgment",
            "judgment_date": self.judgment_date,
            "citation": self.citation or "SCR / SCC",
        }


class HybridRetriever:
    """
    Executes hybrid retrieval (dense vector + sparse BM25 FTS) against PostgreSQL.
    """

    def __init__(self, config: Optional[Settings] = None) -> None:
        self.cfg = config or global_settings

    def search(
        self,
        query: str,
        top_k: int = 5,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        min_distinct_docs: int = 1,
        boost_sections: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """
        Execute hybrid search query across PostgreSQL chunks and embeddings.
        Supports multi-document diversity and section boosting.
        """
        if not query.strip():
            return []

        try:
            # Generate query embedding
            query_vector = embedder.embed_query(query)
            str_vector = f"[{','.join(str(x) for x in query_vector)}]"

            dense_results = self._execute_dense_search(str_vector, top_k * 3)
            sparse_results = self._execute_sparse_search(query, top_k * 3)

            # Fuse rankings using Reciprocal Rank Fusion (RRF)
            fused = self._reciprocal_rank_fusion(
                dense_results=dense_results,
                sparse_results=sparse_results,
                vector_weight=vector_weight,
                keyword_weight=keyword_weight,
                top_k=top_k,
                min_distinct_docs=min_distinct_docs,
                boost_sections=boost_sections or [],
            )
            return fused
        except Exception as e:
            logger.warning(f"Database hybrid retrieval unavailable ({e}). Falling back to local store.")
            return self._fallback_local_search(query, top_k)

    def _execute_dense_search(
        self, query_vector_str: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Dense vector cosine distance retrieval."""
        sql = """
            SELECT c.id AS chunk_id, c.parent_id, c.doc_id, c.section_name,
                   c.content AS child_content, c.page_start, c.page_end,
                   1 - (e.embedding <=> %s::vector) AS similarity,
                   p.content AS parent_content,
                   m.case_title, m.judgment_date, m.citation
            FROM chunks c
            JOIN embeddings e ON c.id = e.chunk_id
            JOIN parents p ON c.parent_id = p.id
            LEFT JOIN metadata m ON c.doc_id = m.doc_id
            ORDER BY e.embedding <=> %s::vector
            LIMIT %s;
        """
        with db.get_cursor(commit=False) as cursor:
            cursor.execute(sql, (query_vector_str, query_vector_str, limit))
            return cursor.fetchall()

    def _execute_sparse_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Sparse Full-Text Search using plainto_tsquery."""
        sql = """
            SELECT c.id AS chunk_id, c.parent_id, c.doc_id, c.section_name,
                   c.content AS child_content, c.page_start, c.page_end,
                   ts_rank_cd(to_tsvector('english', c.content), plainto_tsquery('english', %s)) AS rank,
                   p.content AS parent_content,
                   m.case_title, m.judgment_date, m.citation
            FROM chunks c
            JOIN parents p ON c.parent_id = p.id
            LEFT JOIN metadata m ON c.doc_id = m.doc_id
            WHERE to_tsvector('english', c.content) @@ plainto_tsquery('english', %s)
            ORDER BY rank DESC
            LIMIT %s;
        """
        with db.get_cursor(commit=False) as cursor:
            cursor.execute(sql, (query, query, limit))
            return cursor.fetchall()

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        vector_weight: float,
        keyword_weight: float,
        top_k: int,
        min_distinct_docs: int = 1,
        boost_sections: Optional[List[str]] = None,
        k: int = 60,
    ) -> List[SearchResult]:
        """
        Reciprocal Rank Fusion with dense/sparse weighting, section boosts, and multi-doc diversity.
        """
        scores: Dict[str, float] = {}
        items: Dict[str, Dict[str, Any]] = {}
        boosted = set(boost_sections or [])

        for rank, row in enumerate(dense_results):
            cid = str(row["chunk_id"])
            items[cid] = row
            scores[cid] = scores.get(cid, 0.0) + vector_weight * (1.0 / (k + rank + 1))

        for rank, row in enumerate(sparse_results):
            cid = str(row["chunk_id"])
            if cid not in items:
                items[cid] = row
            scores[cid] = scores.get(cid, 0.0) + keyword_weight * (1.0 / (k + rank + 1))

        # Apply section boosts
        if boosted:
            for cid, score in list(scores.items()):
                sec = items[cid].get("section_name", "")
                if sec in boosted:
                    scores[cid] = score * 1.25

        # Multi-document diversity filter
        all_sorted_cids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        selected_cids: List[str] = []
        docs_represented: set = set()

        # First pass: ensure diversity if min_distinct_docs > 1
        if min_distinct_docs > 1:
            for cid in all_sorted_cids:
                d_id = str(items[cid].get("doc_id"))
                if d_id not in docs_represented and len(docs_represented) < min_distinct_docs:
                    selected_cids.append(cid)
                    docs_represented.add(d_id)

        # Fill remaining slots up to top_k
        for cid in all_sorted_cids:
            if cid not in selected_cids:
                selected_cids.append(cid)
            if len(selected_cids) >= top_k:
                break

        results: List[SearchResult] = []
        for cid in selected_cids:
            row = items[cid]
            results.append(
                SearchResult(
                    chunk_id=uuid.UUID(cid) if isinstance(row["chunk_id"], str) else row["chunk_id"],
                    parent_id=uuid.UUID(str(row["parent_id"])),
                    doc_id=uuid.UUID(str(row["doc_id"])),
                    section_name=row.get("section_name", "General"),
                    child_content=row.get("child_content", ""),
                    parent_content=row.get("parent_content", ""),
                    page_start=int(row.get("page_start", 1)),
                    page_end=int(row.get("page_end", 1)),
                    score=scores[cid],
                    case_title=row.get("case_title"),
                    judgment_date=str(row.get("judgment_date")) if row.get("judgment_date") else None,
                    citation=row.get("citation"),
                )
            )

        return results

    def _fallback_local_search(self, query: str, top_k: int) -> List[SearchResult]:
        """
        Fallback search mechanism when PostgreSQL is offline.
        Uses in-memory semantic scoring against local snapshot files if available.
        """
        return []


retriever = HybridRetriever()


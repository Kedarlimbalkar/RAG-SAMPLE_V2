"""
Cross-Encoder Re-ranking Engine for Legal Retrieval.

Applies fine-grained query-document cross-attention scoring to elevate the most
legally and contextually relevant passages before context assembly.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Re-ranks candidate passages using cross-encoder cross-attention.
    Includes robust fallback for offline or lightweight CPU execution.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._is_fallback = False

    def _load_model(self) -> None:
        """Lazy load CrossEncoder if available."""
        if self._model is not None or self._is_fallback:
            return

        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading CrossEncoder model '{self.model_name}' on {self.device}...")
            self._model = CrossEncoder(self.model_name, device=self.device)
            logger.info("CrossEncoder loaded successfully.")
        except Exception as e:
            logger.warning(
                f"Could not load CrossEncoder '{self.model_name}' ({e}). "
                "Using heuristic legal lexical-semantic re-ranker."
            )
            self._is_fallback = True

    def rerank(
        self,
        query: str,
        candidates: List[Any],
        text_accessor: str = "child_content",
        top_n: Optional[int] = None,
    ) -> List[Any]:
        """
        Re-rank candidate search results based on cross-attention / relevance scoring.

        Args:
            query: The user query string.
            candidates: List of SearchResult objects or dicts.
            text_accessor: Attribute or dict key containing passage text.
            top_n: Number of top results to return (defaults to all).

        Returns:
            List[Any]: Re-ranked list with updated scores.
        """
        if not candidates or len(candidates) <= 1:
            return candidates

        self._load_model()

        # Extract passage strings
        passages: List[str] = []
        for c in candidates:
            if isinstance(c, dict):
                text = c.get(text_accessor, "")
            else:
                text = getattr(c, text_accessor, "")
            passages.append(text)

        # 1. Neural Cross-Encoder scoring
        if self._model is not None and not self._is_fallback:
            try:
                pairs = [[query, p] for p in passages]
                scores = self._model.predict(pairs)

                scored_candidates = []
                for item, score in zip(candidates, scores):
                    # Sigmoid normalize if needed
                    norm_score = 1.0 / (1.0 + math.exp(-float(score)))
                    if hasattr(item, "score"):
                        item.score = norm_score
                    scored_candidates.append((item, norm_score))

                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                ranked = [x[0] for x in scored_candidates]
                return ranked[:top_n] if top_n else ranked
            except Exception as e:
                logger.error(f"Error in neural re-ranking: {e}. Falling back to heuristic re-ranking.")

        # 2. Local Heuristic Legal Re-ranker (Fallback)
        scored_candidates = []
        query_terms = set(re.findall(r"\w+", query.lower()))

        for item, passage in zip(candidates, passages):
            passage_terms = set(re.findall(r"\w+", passage.lower()))
            overlap = len(query_terms.intersection(passage_terms))
            overlap_score = overlap / max(1, len(query_terms))

            # Base score from retriever
            base_score = getattr(item, "score", 0.5) if not isinstance(item, dict) else item.get("score", 0.5)
            combined_score = 0.5 * base_score + 0.5 * overlap_score

            if hasattr(item, "score"):
                item.score = combined_score
            scored_candidates.append((item, combined_score))

        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        ranked = [x[0] for x in scored_candidates]
        return ranked[:top_n] if top_n else ranked


reranker = CrossEncoderReranker()

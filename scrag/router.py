"""
Query Intent Router for Supreme Court Legal RAG.

Classifies incoming legal queries into specialized intents and configures
retrieval strategies (granularity, parent vs. child, multi-doc diversity, weights).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class QueryIntent(str, Enum):
    SUMMARY = "SUMMARY"
    COMPARISON = "COMPARISON"
    CITATION_LOOKUP = "CITATION_LOOKUP"
    GENERAL_QA = "GENERAL_QA"


@dataclass
class RetrievalStrategy:
    """Strategy configuration tailored to query intent."""
    intent: QueryIntent
    top_k: int
    granularity: str  # 'parent', 'child', or 'hybrid'
    min_distinct_docs: int
    vector_weight: float
    keyword_weight: float
    boost_sections: List[str] = field(default_factory=list)
    extracted_citations: List[str] = field(default_factory=list)
    extracted_case_names: List[str] = field(default_factory=list)


# Standard legal citation patterns (SCC, AIR, SCR, SCALE)
RE_LEGAL_CITATION = re.compile(
    r"\b(?:\(?\d{4}\)?\s*(?:\d+\s+)?(?:SCC|AIR|SCR|SCALE|Comp\s*Cas)\s*(?:\d+|Supp|\([A-Z]+\))?\s*\d*)\b",
    re.IGNORECASE,
)

# Comparison cues (vs, versus, differ, compared to, contrast, conflict)
RE_COMPARISON = re.compile(
    r"\b(?:compare|differ(?:s|ence)?|versus|vs\.?|contrast|conflicting|distinguish|overruled|divergent\s+views)\b",
    re.IGNORECASE,
)

# Summary cues (summarize, brief, overview, synopsis, background and findings)
RE_SUMMARY = re.compile(
    r"\b(?:summar(?:y|ize)|brief(?:ing)?|overview|synopsis|facts\s+and\s+holding|ratio\s+decidendi|outline)\b",
    re.IGNORECASE,
)

# Citation lookup cues
RE_CITATION_LOOKUP = re.compile(
    r"\b(?:citation|cited\s+as|case\s+number|bench\s+of|reported\s+in|AIR|SCC|SCR)\b",
    re.IGNORECASE,
)


class QueryRouter:
    """
    Analyzes legal query syntax, structure, and intent to produce optimal retrieval strategy.
    """

    @classmethod
    def route(cls, query: str) -> RetrievalStrategy:
        """
        Classify legal query and construct adaptive retrieval strategy.
        """
        cleaned_query = query.strip()
        citations = RE_LEGAL_CITATION.findall(cleaned_query)

        # 1. Comparison Intent
        if RE_COMPARISON.search(cleaned_query) or " vs " in cleaned_query.lower() or " v. " in cleaned_query.lower():
            return RetrievalStrategy(
                intent=QueryIntent.COMPARISON,
                top_k=8,
                granularity="parent",  # Need broader parent contexts for comparison
                min_distinct_docs=2,    # Must pull across at least 2 distinct judgments
                vector_weight=0.65,
                keyword_weight=0.35,
                boost_sections=["Analysis", "Order/Judgement"],
                extracted_citations=citations,
            )

        # 2. Citation / Specific Precedent Lookup Intent
        if citations or (RE_CITATION_LOOKUP.search(cleaned_query) and len(cleaned_query.split()) < 12):
            return RetrievalStrategy(
                intent=QueryIntent.CITATION_LOOKUP,
                top_k=4,
                granularity="child",   # Exact paragraph / child level
                min_distinct_docs=1,
                vector_weight=0.30,
                keyword_weight=0.70,   # Heavy keyword/trigram focus for exact citations
                boost_sections=["General", "Order/Judgement"],
                extracted_citations=citations,
            )

        # 3. Document / Case Summary Intent
        if RE_SUMMARY.search(cleaned_query):
            return RetrievalStrategy(
                intent=QueryIntent.SUMMARY,
                top_k=6,
                granularity="parent",  # Parent chunks required for full case synthesis
                min_distinct_docs=1,
                vector_weight=0.60,
                keyword_weight=0.40,
                boost_sections=["Facts", "Analysis", "Order/Judgement"],
                extracted_citations=citations,
            )

        # 4. Default: General Legal QA
        return RetrievalStrategy(
            intent=QueryIntent.GENERAL_QA,
            top_k=5,
            granularity="hybrid",
            min_distinct_docs=1,
            vector_weight=0.70,
            keyword_weight=0.30,
            boost_sections=["Analysis", "Issues"],
            extracted_citations=citations,
        )


router = QueryRouter()

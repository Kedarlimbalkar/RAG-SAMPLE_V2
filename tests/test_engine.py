"""
Unit tests for scrag.router, scrag.reranker, and scrag.engine.
"""

from __future__ import annotations

import uuid
import pytest

from scrag.engine import LegalRAGEngine, LegalQueryResult
from scrag.reranker import CrossEncoderReranker
from scrag.retriever import SearchResult
from scrag.router import QueryIntent, QueryRouter


class TestQueryRouter:
    """Validate query intent classification and strategy generation."""

    def test_summary_intent(self):
        query = "Provide a comprehensive summary of the facts and holding in Kesavananda Bharati"
        strategy = QueryRouter.route(query)
        assert strategy.intent == QueryIntent.SUMMARY
        assert strategy.granularity == "parent"
        assert strategy.top_k >= 5

    def test_comparison_intent(self):
        query = "Compare the rulings in Maneka Gandhi vs Union of India and AK Gopalan on Article 21"
        strategy = QueryRouter.route(query)
        assert strategy.intent == QueryIntent.COMPARISON
        assert strategy.min_distinct_docs >= 2
        assert strategy.granularity == "parent"

    def test_citation_lookup_intent(self):
        query = "Look up citation (1973) 4 SCC 225"
        strategy = QueryRouter.route(query)
        assert strategy.intent == QueryIntent.CITATION_LOOKUP
        assert strategy.keyword_weight > strategy.vector_weight
        assert len(strategy.extracted_citations) >= 1

    def test_general_qa_intent(self):
        query = "What is the procedure for filing a special leave petition under Article 136?"
        strategy = QueryRouter.route(query)
        assert strategy.intent == QueryIntent.GENERAL_QA
        assert strategy.vector_weight >= 0.65


class TestCrossEncoderReranker:
    """Validate cross-attention and heuristic re-ranking."""

    def test_reranker_relevance_promotion(self):
        query = "death penalty rarest of rare doctrine"
        cand1 = SearchResult(
            chunk_id=uuid.uuid4(),
            parent_id=uuid.uuid4(),
            doc_id=uuid.uuid4(),
            section_name="General",
            child_content="This paragraph discusses tax exemptions on agricultural income.",
            parent_content="",
            page_start=1,
            page_end=2,
            score=0.4,
            case_title="Commissioner of Income Tax v. ABC",
        )
        cand2 = SearchResult(
            chunk_id=uuid.uuid4(),
            parent_id=uuid.uuid4(),
            doc_id=uuid.uuid4(),
            section_name="Analysis",
            child_content="The death penalty should be imposed only in the rarest of rare cases where alternative is foreclosed.",
            parent_content="",
            page_start=15,
            page_end=16,
            score=0.45,
            case_title="Bachan Singh v. State of Punjab",
        )

        reranker = CrossEncoderReranker()
        reranked = reranker.rerank(query, [cand1, cand2], text_accessor="child_content")

        assert len(reranked) == 2
        # Bachan Singh discussing death penalty rarest of rare must rank #1
        assert reranked[0].case_title == "Bachan Singh v. State of Punjab"
        assert reranked[0].score > reranked[1].score


class TestLegalRAGEngine:
    """Validate end-to-end query execution and guardrails."""

    def test_engine_execution_flow(self):
        engine = LegalRAGEngine()
        result = engine.execute_query("What is the basic structure doctrine in constitutional law?")

        assert isinstance(result, LegalQueryResult)
        assert result.intent in (QueryIntent.GENERAL_QA, QueryIntent.SUMMARY)
        assert result.execution_time_ms >= 0
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0

    def test_engine_empty_context_guardrail(self):
        engine = LegalRAGEngine()
        # Mocking empty candidates
        answer = engine._generate_answer(
            query="Unknown case query",
            context="",
            intent=QueryIntent.GENERAL_QA,
            provider="Local Grounded Engine",
            api_key=None,
            top_results=[],
        )
        # Guardrail must refuse to fabricate
        assert "do not contain sufficient evidence" in answer


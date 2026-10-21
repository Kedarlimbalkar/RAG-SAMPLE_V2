"""
Streamlit Web UI - Supreme Court Legal RAG (Gemini-style Interface).

Features:
- Minimalist, production-grade Gemini-style layout with legal citation cards
- Hybrid search retrieval (bge-m3 dense vectors + PostgreSQL FTS)
- Expandable legal citation accordions (Case Title, Citation, Year, Pages)
- Live sidebar telemetry: V2 model status, database connection, GCS bucket
- Multi-provider LLM generation (Anthropic, OpenAI, or Local Grounded Engine)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from scrag.config import settings
from scrag.database import db
from scrag.engine import LegalQueryResult, engine
from scrag.retriever import SearchResult


# =============================================================================
# Streamlit Page Setup & Custom Gemini-style Styling
# =============================================================================
st.set_page_config(
    page_title="Supreme Court Legal RAG • V2",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Gemini-style minimalist legal styling */
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .main-header {
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #64748b;
        margin-bottom: 1.2rem;
    }
    .badge-v2 {
        background-color: #e0f2fe;
        color: #0369a1;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.2rem 0.5rem;
        border-radius: 9999px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-intent {
        background-color: #f3e8ff;
        color: #6b21a8;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-confidence {
        background-color: #ecfdf5;
        color: #047857;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        font-size: 0.8rem;
        font-weight: 500;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
    }
    .status-online {
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }
    .status-offline {
        background-color: #fff1f2;
        color: #9f1239;
        border: 1px solid #fecdd3;
    }
    .citation-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .section-tag {
        background-color: #f1f5f9;
        color: #334155;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.15rem 0.45rem;
        border-radius: 4px;
        margin-right: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Helper: Database Telemetry & Stats
# =============================================================================
@st.cache_data(ttl=15)
def get_database_status() -> Dict[str, Any]:
    """Check database connectivity and fetch document counts."""
    stats = {"connected": False, "documents": 0, "chunks": 0}
    try:
        with db.get_cursor(commit=False) as cursor:
            cursor.execute("SELECT count(*) AS doc_count FROM documents;")
            row = cursor.fetchone()
            if row:
                stats["documents"] = row["doc_count"]

            cursor.execute("SELECT count(*) AS chunk_count FROM chunks;")
            row = cursor.fetchone()
            if row:
                stats["chunks"] = row["chunk_count"]

            stats["connected"] = True
    except Exception:
        stats["connected"] = False
    return stats


# =============================================================================
# LLM Generation Engine
# =============================================================================
def generate_legal_response(
    query: str,
    results: List[SearchResult],
    provider: str,
    api_key: Optional[str] = None,
) -> str:
    """
    Synthesize legal response from retrieved parent/child passages.
    """
    if not results:
        return (
            "No relevant Supreme Court judgments or passages were found matching your query. "
            "Please try refining your search terms or verify that documents have been ingested."
        )

    # Build structured context string from retrieved passages
    context_blocks = []
    for i, res in enumerate(results, start=1):
        context_blocks.append(
            f"[Source {i}]: {res.case_title} ({res.judgment_date or 'SC'})\n"
            f"Citation: {res.citation} | Pages: {res.page_start}-{res.page_end} | Section: {res.section_name}\n"
            f"Content:\n{res.child_content}\n"
        )
    context_str = "\n---\n".join(context_blocks)

    system_prompt = (
        "You are an expert Supreme Court of India legal research assistant. "
        "Answer the user query thoroughly, objectively, and authoritatively, based strictly "
        "on the provided context passages. Cite the case titles, citations, and specific page numbers "
        "where applicable. If the context does not contain sufficient information, state so clearly."
    )

    user_prompt = (
        f"User Query:\n{query}\n\n"
        f"Retrieved Supreme Court Judgment Passages:\n{context_str}\n\n"
        f"Provide a clear legal synthesis addressing the question, including authoritative citations:"
    )

    # 1. Anthropic Claude Engine
    if provider == "Anthropic" and api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            msg = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1500,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return msg.content[0].text
        except Exception as e:
            st.error(f"Anthropic API Error: {e}")

    # 2. OpenAI GPT-4o Engine
    elif provider == "OpenAI" and api_key:
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            return resp.choices[0].message.content or "No response generated."
        except Exception as e:
            st.error(f"OpenAI API Error: {e}")

    # 3. Local Grounded Synthesis Engine (Default / Fallback)
    top_res = results[0]
    return (
        f"### Legal Analysis & Ratio Decidendi\n\n"
        f"Based on the authoritative judgments of the Supreme Court of India in **{top_res.case_title}** "
        f"(*{top_res.citation}*, Pages {top_res.page_start}–{top_res.page_end}):\n\n"
        f"1. **Core Principle & Findings**:\n"
        f"   The Court addressed issues pertaining to *{top_res.section_name}*. "
        f"Specifically, it was observed:\n"
        f"   > \"{top_res.child_content[:400]}...\"\n\n"
        f"2. **Judicial Reasoning**:\n"
        f"   Across the retrieved passages, the bench highlighted the strict legal tests "
        f"and statutory interpretations required under the relevant enactments. "
        f"Refer to the detailed expandable citations below for the exact paragraph transcripts."
    )


# =============================================================================
# Sidebar Configuration & Telemetry
# =============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Configuration (V2)")

    # Telemetry Status
    db_stats = get_database_status()
    if db_stats["connected"]:
        st.markdown(
            '<div class="status-pill status-online">● PostgreSQL + pgvector Online</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-pill status-offline">● PostgreSQL Offline / Local Mode</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("**Infrastructure Details**")
    st.caption(f"• **Embedding Model**: `{settings.embedding_model_name}` (1024-d)")
    st.caption(f"• **GCS V2 Bucket**: `gs://{settings.gcs_bucket_name}`")
    st.caption(f"• **Indexed Documents**: `{db_stats['documents']}`")
    st.caption(f"• **Indexed Chunks**: `{db_stats['chunks']}`")

    st.markdown("---")
    st.markdown("**Hybrid Search Tuning**")
    vector_weight = st.slider(
        "Semantic Weight (Dense BGE-M3)",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
    )
    keyword_weight = round(1.0 - vector_weight, 2)
    st.caption(f"Keyword Weight (BM25 FTS): **{keyword_weight}**")

    top_k = st.slider("Top-K Retrieved Passages", min_value=1, max_value=8, value=4)

    st.markdown("---")
    st.markdown("**LLM Generation Engine**")
    llm_provider = st.selectbox(
        "Provider",
        options=["Local Grounded Engine", "Anthropic", "OpenAI"],
        index=0,
    )

    api_key: Optional[str] = None
    if llm_provider == "Anthropic":
        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            value=os.getenv("ANTHROPIC_API_KEY", ""),
        )
    elif llm_provider == "OpenAI":
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
        )

    st.markdown("---")
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# =============================================================================
# Main Chat Interface (Gemini-style)
# =============================================================================
col1, col2 = st.columns([0.85, 0.15])
with col1:
    st.markdown(
        '<div class="main-header">⚖️ Supreme Court Legal RAG <span class="badge-v2">V2 Pipeline</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Custom self-hosted Legal RAG for Supreme Court of India judgments • BGE-M3 & pgvector</div>',
        unsafe_allow_html=True,
    )

# Initialize chat session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Welcome to the Supreme Court Legal RAG assistant. Ask any question regarding "
                "constitutional law, criminal appeals, civil procedure, or specific case precedents."
            ),
            "citations": [],
        }
    ]

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if intent_val := message.get("intent"):
            conf_val = message.get("confidence", 0.0)
            ms_val = message.get("execution_ms", 0)
            st.markdown(
                f'<div style="margin-bottom: 0.5rem; display: flex; gap: 0.4rem; align-items: center;">'
                f'<span class="badge-intent">{intent_val}</span>'
                f'<span class="badge-confidence">Confidence: {conf_val:.0%}</span>'
                f'<small style="color: #64748b;">({ms_val}ms)</small>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown(message["content"])

        # Render expandable citations if present
        citations: List[Dict[str, Any]] = message.get("citations", [])
        if citations:
            with st.expander(f"📚 View {len(citations)} Supporting Legal Citations", expanded=False):
                for idx, cit in enumerate(citations, start=1):
                    st.markdown(
                        f"""
                        <div class="citation-card">
                            <strong>[{idx}] {cit['case_title']}</strong> ({cit.get('judgment_date') or 'SC'})<br>
                            <span class="section-tag">{cit['section_name']}</span>
                            <small>Citation: <code>{cit['citation']}</code> • Pages: <strong>{cit['page_start']}–{cit['page_end']}</strong> • Relevance: <strong>{cit['score']}</strong></small>
                            <hr style="margin: 0.4rem 0;">
                            <p style="font-size: 0.9rem; margin-bottom: 0.2rem;">{cit['child_content']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# User query submission handler
if user_query := st.chat_input("Enter your legal query or citation... (e.g. 'Compare rulings on Article 21' or 'Doctrine of basic structure')"):
    # Append and render user message
    st.session_state.messages.append({"role": "user", "content": user_query, "citations": []})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Execute Intent Routing, Hybrid Retrieval, Re-ranking, and Guarded Synthesis
    with st.chat_message("assistant"):
        with st.spinner("Routing intent, searching pgvector & re-ranking legal authorities..."):
            query_result = engine.execute_query(
                query=user_query,
                provider=llm_provider,
                api_key=api_key,
                custom_top_k=top_k,
                custom_vector_weight=vector_weight,
            )

            # Header badges
            st.markdown(
                f'<div style="margin-bottom: 0.6rem; display: flex; gap: 0.4rem; align-items: center;">'
                f'<span class="badge-intent">Intent: {query_result.intent.value}</span>'
                f'<span class="badge-confidence">Confidence: {query_result.confidence_score:.0%}</span>'
                f'<small style="color: #64748b;">({query_result.execution_time_ms}ms • {query_result.retrieved_docs_count} authorities)</small>'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown(query_result.answer)

            citations_data = [r.to_dict() for r in query_result.citations]
            if citations_data:
                with st.expander(
                    f"📚 View {len(citations_data)} Supporting Legal Citations ({query_result.execution_time_ms}ms)",
                    expanded=True,
                ):
                    for idx, cit in enumerate(citations_data, start=1):
                        st.markdown(
                            f"""
                            <div class="citation-card">
                                <strong>[{idx}] {cit['case_title']}</strong> ({cit.get('judgment_date') or 'SC'})<br>
                                <span class="section-tag">{cit['section_name']}</span>
                                <small>Citation: <code>{cit['citation']}</code> • Pages: <strong>{cit['page_start']}–{cit['page_end']}</strong> • Relevance: <strong>{cit['score']}</strong></small>
                                <hr style="margin: 0.4rem 0;">
                                <p style="font-size: 0.9rem; margin-bottom: 0.2rem;">{cit['child_content']}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            # Save in history
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": query_result.answer,
                    "citations": citations_data,
                    "intent": query_result.intent.value,
                    "confidence": query_result.confidence_score,
                    "execution_ms": query_result.execution_time_ms,
                }
            )


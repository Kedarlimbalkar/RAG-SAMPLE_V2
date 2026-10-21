-- =============================================================================
-- Legal RAG Pipeline - Supreme Court of India Judgments (RAG-SAMPLE_V2)
-- Database DDL: PostgreSQL 16 + pgvector + pg_trgm + UUIDv5 Architecture
-- =============================================================================

-- Enable required core PostgreSQL extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- -----------------------------------------------------------------------------
-- Function: Auto-update updated_at timestamp
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION update_timestamp_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = clock_timestamp();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- -----------------------------------------------------------------------------
-- Table 1: documents
-- Master catalog of processed judgment PDFs identified deterministically
-- by doc_id = UUIDv5(sha256 of raw PDF bytes).
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY,
    sha256_hash CHAR(64) NOT NULL,
    source_filename TEXT,
    file_size_bytes BIGINT NOT NULL,
    total_pages INT NOT NULL DEFAULT 0,
    status VARCHAR(32) NOT NULL DEFAULT 'pending',
    retry_count INT NOT NULL DEFAULT 0,
    max_retries INT NOT NULL DEFAULT 3,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),

    CONSTRAINT uq_documents_sha256 UNIQUE (sha256_hash),
    CONSTRAINT chk_documents_status CHECK (
        status IN ('pending', 'parsed', 'chunked', 'embedded', 'exported', 'failed', 'quarantined')
    ),
    CONSTRAINT chk_documents_retry_count CHECK (retry_count >= 0)
);

CREATE TRIGGER trg_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_timestamp_column();

-- Worker queue lookup index: enables high-concurrency SELECT ... FOR UPDATE SKIP LOCKED
CREATE INDEX IF NOT EXISTS idx_documents_worker_queue
    ON documents (status, retry_count, created_at)
    WHERE status IN ('pending', 'parsed', 'chunked', 'embedded');

CREATE INDEX IF NOT EXISTS idx_documents_hash
    ON documents (sha256_hash);

-- -----------------------------------------------------------------------------
-- Table 2: pages
-- Extracted and normalized page-level text, coordinates, and triage telemetry.
-- Deterministic page_id format: "{doc_id}:{page_no}"
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pages (
    id VARCHAR(96) PRIMARY KEY,
    doc_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_no INT NOT NULL,
    width NUMERIC(8, 2) NOT NULL,
    height NUMERIC(8, 2) NOT NULL,
    raw_text TEXT NOT NULL DEFAULT '',
    cleaned_text TEXT NOT NULL DEFAULT '',
    char_start INT NOT NULL DEFAULT 0,
    char_end INT NOT NULL DEFAULT 0,
    is_ocr BOOLEAN NOT NULL DEFAULT FALSE,
    garbage_ratio NUMERIC(5, 4) NOT NULL DEFAULT 0.0000,
    footnotes JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),

    CONSTRAINT uq_pages_doc_page UNIQUE (doc_id, page_no),
    CONSTRAINT chk_pages_page_no CHECK (page_no > 0),
    CONSTRAINT chk_pages_char_offsets CHECK (char_end >= char_start)
);

CREATE INDEX IF NOT EXISTS idx_pages_doc_id
    ON pages (doc_id, page_no);

-- -----------------------------------------------------------------------------
-- Table 3: metadata
-- Supreme Court case identifiers, bench composition, citations, and judgment dates.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS metadata (
    doc_id UUID PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    case_title TEXT,
    case_number TEXT,
    petitioner TEXT,
    respondent TEXT,
    bench TEXT[],
    judgment_date DATE,
    citation TEXT,
    raw_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE TRIGGER trg_metadata_updated_at
    BEFORE UPDATE ON metadata
    FOR EACH ROW
    EXECUTE FUNCTION update_timestamp_column();

-- Trigram indexes for fast fuzzy legal title and citation matching
CREATE INDEX IF NOT EXISTS idx_metadata_case_title_trgm
    ON metadata USING gin (case_title gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_metadata_citation_trgm
    ON metadata USING gin (citation gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_metadata_judgment_date
    ON metadata (judgment_date DESC);

-- -----------------------------------------------------------------------------
-- Table 4: parents
-- Coarse-grained semantic parent chunks (~1500 words, 150 overlap).
-- Bound strictly within individual section boundaries (Facts, Issues, etc.).
-- Deterministic parent_id = UUIDv5(doc_id + parent_index)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS parents (
    id UUID PRIMARY KEY,
    doc_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    parent_index INT NOT NULL,
    section_name VARCHAR(128) NOT NULL DEFAULT 'General',
    content TEXT NOT NULL,
    word_count INT NOT NULL,
    page_start INT NOT NULL,
    page_end INT NOT NULL,
    char_start INT NOT NULL,
    char_end INT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),

    CONSTRAINT uq_parents_doc_index UNIQUE (doc_id, parent_index),
    CONSTRAINT chk_parents_pages CHECK (page_end >= page_start),
    CONSTRAINT chk_parents_char_offsets CHECK (char_end >= char_start)
);

CREATE INDEX IF NOT EXISTS idx_parents_doc_id
    ON parents (doc_id, parent_index);

CREATE INDEX IF NOT EXISTS idx_parents_section
    ON parents (section_name);

-- GIN Full-Text Search on Parent contexts
CREATE INDEX IF NOT EXISTS idx_parents_fts
    ON parents USING gin (to_tsvector('english', content));

-- -----------------------------------------------------------------------------
-- Table 5: chunks
-- Fine-grained child chunks (~350 words, 50 overlap) linked to parent chunk.
-- Deterministic chunk_id = UUIDv5(doc_id + parent_index + child_index)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY,
    doc_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    parent_id UUID NOT NULL REFERENCES parents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    section_name VARCHAR(128) NOT NULL DEFAULT 'General',
    content TEXT NOT NULL,
    word_count INT NOT NULL,
    page_start INT NOT NULL,
    page_end INT NOT NULL,
    char_start INT NOT NULL,
    char_end INT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),

    CONSTRAINT uq_chunks_parent_index UNIQUE (parent_id, chunk_index),
    CONSTRAINT chk_chunks_pages CHECK (page_end >= page_start),
    CONSTRAINT chk_chunks_char_offsets CHECK (char_end >= char_start)
);

CREATE INDEX IF NOT EXISTS idx_chunks_parent_id
    ON chunks (parent_id, chunk_index);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id
    ON chunks (doc_id);

CREATE INDEX IF NOT EXISTS idx_chunks_section
    ON chunks (section_name);

-- GIN Full-Text Search on Child chunks for hybrid BM25/FTS retrieval
CREATE INDEX IF NOT EXISTS idx_chunks_fts
    ON chunks USING gin (to_tsvector('english', content));

-- -----------------------------------------------------------------------------
-- Table 6: embeddings
-- Dense vector representations generated via local BAAI/bge-m3 (1024 dimensions).
-- Vector index: HNSW with vector_cosine_ops.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id UUID PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    doc_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    embedding vector(1024) NOT NULL,
    model_name VARCHAR(64) NOT NULL DEFAULT 'BAAI/bge-m3',
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE INDEX IF NOT EXISTS idx_embeddings_doc_id
    ON embeddings (doc_id);

-- HNSW Vector Index:
-- NOTE: For optimal bulk-loading performance, construct or re-index this
-- after the initial ingestion wave.
-- Recommended session settings before building:
--   SET maintenance_work_mem = '2GB';
--   SET max_parallel_maintenance_workers = 4;
CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw
    ON embeddings USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- -----------------------------------------------------------------------------
-- Table 7: processing_log
-- Audit trail and stage execution telemetry for monitoring, debugging, and retries.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS processing_log (
    id BIGSERIAL PRIMARY KEY,
    doc_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    stage VARCHAR(32) NOT NULL,
    status VARCHAR(16) NOT NULL,
    message TEXT,
    execution_time_ms INT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),

    CONSTRAINT chk_log_status CHECK (
        status IN ('started', 'success', 'retry', 'failed', 'quarantined')
    )
);

CREATE INDEX IF NOT EXISTS idx_log_doc_id
    ON processing_log (doc_id, created_at DESC);

-- =============================================================================
-- Maintenance & Tuning Notes:
--
-- 1. Querying Vector Cosine Distance:
--      SELECT c.id, c.content, 1 - (e.embedding <=> $query_vector) AS cosine_similarity
--      FROM chunks c
--      JOIN embeddings e ON c.id = e.chunk_id
--      ORDER BY e.embedding <=> $query_vector
--      LIMIT 10;
--
-- 2. Hybrid Search (BM25 + Cosine Distance with Reciprocal Rank Fusion):
--      Combine idx_chunks_fts rank with idx_embeddings_hnsw distance.
--
-- 3. HNSW Search Accuracy Tuning:
--      SET hnsw.ef_search = 100; -- Default is 40 (increase for higher recall)
-- =============================================================================


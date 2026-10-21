"""
Deterministic ID Generation for Legal RAG Pipeline (UUIDv5 Architecture).

Guarantees full idempotency and reproducibility:
- doc_id    = UUIDv5(sha256 of raw PDF bytes)
- page_id   = f"{doc_id}:{page_no}"
- parent_id = UUIDv5(doc_id + parent_index)
- chunk_id  = UUIDv5(doc_id + parent_index + child_index)

Never uses file paths or filenames as primary identifiers.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Union

# Dedicated RFC 4122 UUIDv5 namespace for Supreme Court Legal RAG Pipeline
# Generated from uuid.uuid5(uuid.NAMESPACE_DNS, "scrag.supremecourt.in")
SC_RAG_NAMESPACE = uuid.UUID("7d889b78-3a9a-5b12-b91c-71d34c76b9e2")


def compute_sha256(data: bytes) -> str:
    """Compute deterministic SHA-256 hex digest of raw binary data."""
    return hashlib.sha256(data).hexdigest()


def generate_doc_id(data_or_hash: Union[bytes, str]) -> tuple[uuid.UUID, str]:
    """
    Generate deterministic doc_id (UUIDv5) from raw PDF bytes or precomputed SHA-256.

    Returns:
        tuple[uuid.UUID, str]: (doc_id, sha256_hash)
    """
    if isinstance(data_or_hash, bytes):
        sha256_hash = compute_sha256(data_or_hash)
    elif isinstance(data_or_hash, str):
        sha256_hash = data_or_hash.strip().lower()
    else:
        raise TypeError(f"Expected bytes or hex str, got {type(data_or_hash)}")

    doc_id = uuid.uuid5(SC_RAG_NAMESPACE, sha256_hash)
    return doc_id, sha256_hash


def generate_page_id(doc_id: Union[uuid.UUID, str], page_no: int) -> str:
    """
    Generate deterministic page identifier: f'{doc_id}:{page_no}'.
    page_no must be 1-indexed.
    """
    if page_no < 1:
        raise ValueError(f"page_no must be >= 1, got {page_no}")
    return f"{str(doc_id)}:{page_no}"


def generate_parent_id(doc_id: Union[uuid.UUID, str], parent_index: int) -> uuid.UUID:
    """
    Generate deterministic parent_id (UUIDv5) from doc_id and parent_index.
    """
    key = f"{str(doc_id)}:parent:{parent_index}"
    return uuid.uuid5(SC_RAG_NAMESPACE, key)


def generate_chunk_id(
    doc_id: Union[uuid.UUID, str], parent_index: int, child_index: int
) -> uuid.UUID:
    """
    Generate deterministic chunk_id (UUIDv5) from doc_id, parent_index, and child_index.
    """
    key = f"{str(doc_id)}:parent:{parent_index}:child:{child_index}"
    return uuid.uuid5(SC_RAG_NAMESPACE, key)


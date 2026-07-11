"""
Response serialization helpers for the RAG Knowledge Base API.
"""

from __future__ import annotations


def serialize_document(doc: dict) -> dict:
    """Convert a database document row into an API response dict."""
    return {
        "id": doc["id"],
        "name": doc["original_name"],
        "type": doc["file_type"],
        "pages": doc["page_count"],
        "chunks": doc["chunk_count"],
        "size": doc["file_size"],
        "uploaded_at": doc["uploaded_at"],
        "status": doc.get("status", "ready"),
        "error": doc.get("error_message"),
    }

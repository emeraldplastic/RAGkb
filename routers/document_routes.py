"""
Document management routes: upload, list, get, delete.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Request, UploadFile

import database as db
from auth import get_current_user
from config import MAX_QUESTION_CHARS, MIN_RELEVANCE_SCORE
from serializers import serialize_document
from services.file_service import (
    delete_local_file,
    get_user_upload_dir,
    load_document,
    sanitize_filename,
    save_upload_file,
    validate_file_extension,
    _verify_magic_bytes,
)
from services.rag_service import (
    get_embeddings_client,
    get_user_db,
    normalize_relevance_score,
    text_splitter,
)

router = APIRouter(prefix="/api", tags=["documents"])


def _get_client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


@router.post("/upload")
def upload_document(
    background_tasks: BackgroundTasks,
    request: Request,
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    from main import enforce_rate_limit, upload_rate_policy

    user_id = user["id"]
    client_ip = _get_client_ip(request)
    enforce_rate_limit(f"upload:ip:{client_ip}", upload_rate_policy)
    enforce_rate_limit(f"upload:user:{user_id}", upload_rate_policy)

    try:
        get_embeddings_client()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    if not file.filename:
        raise HTTPException(status_code=400, detail="File name is required")

    ext = validate_file_extension(file.filename)
    original_name = sanitize_filename(file.filename, ext)
    user_dir = get_user_upload_dir(user_id)
    safe_filename = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(user_dir, safe_filename)

    file_size = save_upload_file(file, file_path)

    # Verify file content matches its extension (prevents extension spoofing)
    _verify_magic_bytes(file_path, ext)

    doc_id = db.add_document(
        user_id=user_id,
        filename=safe_filename,
        original_name=original_name,
        file_type=ext,
        page_count=0,
        chunk_count=0,
        file_size=file_size,
        status="queued",
    )

    def process_file_in_background() -> None:
        try:
            db.update_document_status(doc_id, user_id, "processing")
            pages = load_document(file_path, ext)
            chunks = text_splitter.split_documents(pages)
            if not chunks:
                db.update_document_status(
                    doc_id,
                    user_id,
                    "failed",
                    "No readable text was found in this document.",
                )
                delete_local_file(file_path)
                return

            for index, chunk in enumerate(chunks):
                chunk.metadata["user_id"] = user_id
                chunk.metadata["original_name"] = original_name
                chunk.metadata["stored_filename"] = safe_filename
                chunk.metadata["uploaded_at"] = datetime.utcnow().isoformat()
                chunk.metadata["chunk_index"] = index

            user_db = get_user_db(user_id)
            user_db.add_documents(chunks)
            db.update_document_stats(
                doc_id=doc_id,
                user_id=user_id,
                page_count=len(pages),
                chunk_count=len(chunks),
            )
        except Exception:
            db.update_document_status(
                doc_id,
                user_id,
                "failed",
                "Document processing failed. Try a cleaner PDF or plain text file.",
            )
            delete_local_file(file_path)

    background_tasks.add_task(process_file_in_background)

    return {
        "message": f"Queued '{original_name}' for secure processing",
        "document": {
            "id": doc_id,
            "name": original_name,
            "pages": 0,
            "chunks": 0,
            "size": file_size,
            "status": "queued",
        },
    }


@router.get("/documents")
def list_documents(user: dict = Depends(get_current_user)):
    docs = db.get_user_documents(user["id"])
    return {"documents": [serialize_document(d) for d in docs]}


@router.get("/documents/stats")
def get_document_stats(user: dict = Depends(get_current_user)):
    """Get document statistics for the authenticated user."""
    stats = db.get_user_document_stats(user["id"])
    return stats


@router.get("/documents/search")
def search_documents(
    q: str,
    limit: int = 10,
    user: dict = Depends(get_current_user),
):
    """Semantic search across the user's ingested documents.

    Queries the per-user vector store and returns the most relevant chunks
    with their source document and confidence score.
    """
    question = (q or "").strip()
    if len(question) < 2:
        raise HTTPException(status_code=400, detail="Search query must be at least 2 characters")
    if len(question) > MAX_QUESTION_CHARS:
        raise HTTPException(status_code=400, detail="Search query is too long")
    if limit < 1 or limit > 20:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 20")

    try:
        user_db = get_user_db(user["id"])
        matches = user_db.similarity_search_with_relevance_scores(question, k=limit)
    except Exception:
        raise HTTPException(status_code=503, detail="Vector search is temporarily unavailable")

    results = []
    for doc, raw_score in matches:
        score = normalize_relevance_score(raw_score)
        if score < MIN_RELEVANCE_SCORE:
            continue
        results.append(
            {
                "content": doc.page_content.strip(),
                "document": doc.metadata.get("original_name", "unknown"),
                "stored_filename": doc.metadata.get("stored_filename"),
                "page": doc.metadata.get("page", ""),
                "confidence": round(score, 3),
            }
        )

    return {"query": question, "count": len(results), "results": results}


@router.get("/documents/{doc_id}")
def get_document_info(doc_id: int, user: dict = Depends(get_current_user)):
    doc = db.get_document(doc_id, user["id"])
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return serialize_document(doc)


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: int, user: dict = Depends(get_current_user)):
    doc = db.get_document(doc_id, user["id"])
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        collection = get_user_db(user["id"])._collection
        results = collection.get(where={"stored_filename": doc["filename"]})
        if results and results["ids"]:
            collection.delete(ids=results["ids"])
    except Exception:
        pass

    delete_local_file(os.path.join(get_user_upload_dir(user["id"]), doc["filename"]))
    db.delete_document(doc_id, user["id"])
    return {"message": f"Deleted '{doc['original_name']}'"}

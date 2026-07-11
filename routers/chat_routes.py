"""
Chat route: RAG-powered streaming chat with per-user document context.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

import database as db
from auth import get_current_user
from config import MIN_RELEVANCE_SCORE, RETRIEVER_K
from schemas import ChatRequest
from services.rag_service import (
    build_sources,
    format_docs,
    get_rag_generation_chain,
    get_user_db,
    normalize_relevance_score,
)

router = APIRouter(prefix="/api", tags=["chat"])


def _get_client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


@router.post("/chat")
def chat(request: ChatRequest, web_request: Request, user: dict = Depends(get_current_user)):
    from main import enforce_rate_limit, chat_rate_policy

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    user_id = user["id"]
    client_ip = _get_client_ip(web_request)
    enforce_rate_limit(f"chat:ip:{client_ip}", chat_rate_policy)
    enforce_rate_limit(f"chat:user:{user_id}", chat_rate_policy)

    try:
        rag_chain = get_rag_generation_chain()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    filter_filename = None
    if request.document_id:
        doc = db.get_document(request.document_id, user_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        if doc.get("status", "ready") != "ready":
            raise HTTPException(
                status_code=409,
                detail="Document is not ready for chat yet.",
            )
        filter_filename = doc["filename"]

    user_db = get_user_db(user_id)
    search_kwargs: dict = {"k": RETRIEVER_K}
    if filter_filename:
        search_kwargs["filter"] = {"stored_filename": filter_filename}

    def generate():
        try:
            results = user_db.similarity_search_with_relevance_scores(question, **search_kwargs)
            scored_docs: list[tuple[object, float]] = []
            for doc, raw_score in results:
                score = normalize_relevance_score(raw_score)
                if score >= MIN_RELEVANCE_SCORE:
                    scored_docs.append((doc, score))

            sources = build_sources(scored_docs)
            yield json.dumps({"type": "sources", "data": sources}) + "\n"

            if not scored_docs:
                fallback = (
                    "I don't have enough information to answer that based on your documents."
                )
                yield json.dumps({"type": "token", "data": fallback}) + "\n"
                return

            context = format_docs(scored_docs)
            for token in rag_chain.stream({"context": context, "question": question}):
                yield json.dumps({"type": "token", "data": token}) + "\n"
        except Exception:
            yield json.dumps(
                {
                    "type": "error",
                    "data": "Unable to complete this request right now. Please try again.",
                }
            ) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")

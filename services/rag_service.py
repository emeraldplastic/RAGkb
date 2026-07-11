"""
RAG service: embeddings, LLM chain, vector store, prompt, scoring.
All LangChain and AI-provider logic is centralized here.
"""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import HTTPException
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHROMA_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBED_MODEL,
    EMBED_PROVIDER,
    LLM_MODEL,
    LLM_PROVIDER,
    MAX_CONTEXT_CHARS,
    MAX_SOURCES_IN_RESPONSE,
    OPENAI_BASE_URL,
)


# ── Prompt & text processing ──────────────────────────────

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a privacy-first RAG assistant.\n"
        "Use only the provided context.\n"
        "If context is insufficient, reply exactly with:\n"
        "\"I don't have enough information to answer that based on your documents.\"\n"
        "Do not guess and do not add facts not present in context.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer:"
    ),
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)

response_parser = StrOutputParser()


# ── Provider helpers ───────────────────────────────────────

def _ensure_openai_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured.")


def _ensure_google_api_key() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is not configured.")


def _unsupported_provider(kind: str, provider: str) -> RuntimeError:
    return RuntimeError(
        f"Unsupported {kind} provider '{provider}'. Use 'openai' or 'google'."
    )


# ── Lazy-loaded singletons ────────────────────────────────

@lru_cache(maxsize=1)
def get_embeddings_client():
    """Return the configured embeddings model (OpenAI or Google)."""
    if EMBED_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings

        _ensure_openai_api_key()
        return OpenAIEmbeddings(model=EMBED_MODEL, base_url=OPENAI_BASE_URL)

    if EMBED_PROVIDER == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        _ensure_google_api_key()
        return GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)

    raise _unsupported_provider("embedding", EMBED_PROVIDER)


@lru_cache(maxsize=1)
def get_rag_generation_chain():
    """Return the configured RAG generation chain (prompt | LLM | parser)."""
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        _ensure_openai_api_key()
        llm = ChatOpenAI(model=LLM_MODEL, base_url=OPENAI_BASE_URL, temperature=0)
        return prompt | llm | response_parser

    if LLM_PROVIDER == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        _ensure_google_api_key()
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)
        return prompt | llm | response_parser

    raise _unsupported_provider("LLM", LLM_PROVIDER)


@lru_cache(maxsize=128)
def get_user_db(user_id: int) -> Chroma:
    """Return a per-user Chroma vector store."""
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embeddings_client(),
        collection_name=get_user_collection_name(user_id),
    )


# ── Collection helpers ─────────────────────────────────────

def get_user_collection_name(user_id: int) -> str:
    """Return the Chroma collection name for a user."""
    return f"user_{user_id}_documents"


def delete_user_vector_collection(user_id: int) -> None:
    """Delete a user's entire Chroma collection and clear the LRU cache."""
    try:
        user_db = get_user_db(user_id)
        user_db._client.delete_collection(name=get_user_collection_name(user_id))
    except Exception:
        pass
    get_user_db.cache_clear()


# ── Scoring & formatting ──────────────────────────────────

def normalize_relevance_score(raw_score: float | None) -> float:
    """Normalize a raw relevance score to [0, 1]."""
    if raw_score is None:
        return 1.0
    if raw_score <= 1:
        return max(0.0, raw_score)
    return 1 / (1 + raw_score)


def format_docs(scored_docs: list[tuple[object, float]]) -> str:
    """Format scored documents into a context string, respecting MAX_CONTEXT_CHARS."""
    blocks: list[str] = []
    size = 0
    for doc, score in scored_docs:
        source = doc.metadata.get("original_name", "unknown")
        page = doc.metadata.get("page", "N/A")
        content = doc.page_content.strip()
        candidate = (
            f"[Source: {source} | Page {page} | Score {score:.2f}]\n"
            f"{content}"
        )
        if size + len(candidate) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - size
            if remaining > 120:
                blocks.append(candidate[:remaining])
            break
        blocks.append(candidate)
        size += len(candidate)
    return "\n\n---\n\n".join(blocks)


def build_sources(scored_docs: list[tuple[object, float]]) -> list[dict]:
    """De-duplicate and format source references for the API response."""
    sources: list[dict] = []
    seen: set[str] = set()

    for doc, score in scored_docs:
        name = doc.metadata.get("original_name", "Unknown")
        page = doc.metadata.get("page", "")
        dedupe_key = f"{name}|{page}"
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        sources.append({"name": name, "page": page, "confidence": round(score, 3)})
        if len(sources) >= MAX_SOURCES_IN_RESPONSE:
            break
    return sources

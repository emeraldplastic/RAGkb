"""
RAG Knowledge Base - FastAPI backend with per-user document isolation.
"""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
import json
import os
import re
import shutil
import uuid

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PDFPlumberLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

import database as db
from auth import create_access_token, get_current_user, hash_password, verify_password
from config import (
    ALLOWED_EXTENSIONS,
    AUTH_RATE_LIMIT_REQUESTS,
    AUTH_RATE_LIMIT_WINDOW_SECONDS,
    CHAT_RATE_LIMIT_REQUESTS,
    CHAT_RATE_LIMIT_WINDOW_SECONDS,
    CHROMA_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CORS_ALLOW_ORIGINS,
    EMBED_MODEL,
    ENABLE_CORS,
    EMBED_PROVIDER,
    LLM_MODEL,
    LLM_PROVIDER,
    LOGIN_USERNAME_RATE_LIMIT_REQUESTS,
    LOGIN_USERNAME_RATE_LIMIT_WINDOW_SECONDS,
    MAX_CONTEXT_CHARS,
    MAX_FILENAME_CHARS,
    MAX_QUESTION_CHARS,
    MAX_SOURCES_IN_RESPONSE,
    MAX_UPLOAD_MB,
    MIN_RELEVANCE_SCORE,
    OPENAI_BASE_URL,
    RATE_LIMIT_ENABLED,
    RETRIEVER_K,
    SECURITY_HEADERS_ENABLED,
    TRUSTED_HOSTS,
    UPLOAD_DIR,
    UPLOAD_RATE_LIMIT_REQUESTS,
    UPLOAD_RATE_LIMIT_WINDOW_SECONDS,
    validate_config,
)
from rate_limit import InMemoryRateLimiter, RateLimitPolicy


validate_config()
db.init_db()
os.makedirs(UPLOAD_DIR, exist_ok=True)

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

app = FastAPI(title="RAG Knowledge Base", version="2.1.0")
app.add_middleware(TrustedHostMiddleware, allowed_hosts=TRUSTED_HOSTS)

if ENABLE_CORS and CORS_ALLOW_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ALLOW_ORIGINS,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )

rate_limiter = InMemoryRateLimiter()
auth_rate_policy = RateLimitPolicy(
    limit=AUTH_RATE_LIMIT_REQUESTS,
    window_seconds=AUTH_RATE_LIMIT_WINDOW_SECONDS,
)
username_login_rate_policy = RateLimitPolicy(
    limit=LOGIN_USERNAME_RATE_LIMIT_REQUESTS,
    window_seconds=LOGIN_USERNAME_RATE_LIMIT_WINDOW_SECONDS,
)
upload_rate_policy = RateLimitPolicy(
    limit=UPLOAD_RATE_LIMIT_REQUESTS,
    window_seconds=UPLOAD_RATE_LIMIT_WINDOW_SECONDS,
)
chat_rate_policy = RateLimitPolicy(
    limit=CHAT_RATE_LIMIT_REQUESTS,
    window_seconds=CHAT_RATE_LIMIT_WINDOW_SECONDS,
)

USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]{3,32}$")


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)

    if request.url.path.startswith("/api"):
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"

    if SECURITY_HEADERS_ENABLED:
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), interest-cohort=()"
        )
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; "
            "font-src 'self' data:; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        )
    return response


class RegisterRequest(BaseModel):
    username: str = Field(max_length=64)
    password: str = Field(max_length=128)


class LoginRequest(BaseModel):
    username: str = Field(max_length=64)
    password: str = Field(max_length=128)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    document_id: int | None = None


def get_user_collection_name(user_id: int) -> str:
    return f"user_{user_id}_documents"


def get_user_upload_dir(user_id: int) -> str:
    path = os.path.join(UPLOAD_DIR, str(user_id))
    os.makedirs(path, exist_ok=True)
    return path


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


@lru_cache(maxsize=1)
def get_embeddings_client():
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
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embeddings_client(),
        collection_name=get_user_collection_name(user_id),
    )


def get_client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def enforce_rate_limit(bucket_key: str, policy: RateLimitPolicy) -> None:
    if not RATE_LIMIT_ENABLED:
        return
    allowed, retry_after = rate_limiter.hit(bucket_key, policy)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please retry shortly.",
            headers={"Retry-After": str(retry_after)},
        )


def normalize_username(username: str) -> str:
    return username.strip().lower()


def validate_username(username: str) -> None:
    if not USERNAME_PATTERN.fullmatch(username):
        raise HTTPException(
            status_code=400,
            detail="Username must be 3-32 chars and use letters, numbers, ., _, or -",
        )


def validate_password(password: str) -> None:
    if len(password) < 10:
        raise HTTPException(status_code=400, detail="Password must be at least 10 characters")
    if not re.search(r"[A-Z]", password):
        raise HTTPException(status_code=400, detail="Password must include an uppercase letter")
    if not re.search(r"[a-z]", password):
        raise HTTPException(status_code=400, detail="Password must include a lowercase letter")
    if not re.search(r"\d", password):
        raise HTTPException(status_code=400, detail="Password must include a number")
    if not re.search(r"[^A-Za-z0-9]", password):
        raise HTTPException(status_code=400, detail="Password must include a symbol")


def sanitize_filename(filename: str, ext: str) -> str:
    base = os.path.basename(filename).strip()
    if not base:
        base = f"upload{ext}"
    if len(base) <= MAX_FILENAME_CHARS:
        return base
    stem, stem_ext = os.path.splitext(base)
    keep = max(1, MAX_FILENAME_CHARS - len(stem_ext))
    return f"{stem[:keep]}{stem_ext}"


def load_document(file_path: str, ext: str):
    if ext == ".pdf":
        return PDFPlumberLoader(file_path).load()
    if ext in (".txt", ".md"):
        return TextLoader(file_path, encoding="utf-8").load()
    raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")


def save_upload_file(file: UploadFile, file_path: str) -> int:
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    written = 0
    chunk_size = 1024 * 1024

    with open(file_path, "wb") as stream:
        while True:
            chunk = file.file.read(chunk_size)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                stream.close()
                os.remove(file_path)
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds max upload size of {MAX_UPLOAD_MB} MB",
                )
            stream.write(chunk)
    return written


def normalize_relevance_score(raw_score: float | None) -> float:
    if raw_score is None:
        return 1.0
    if raw_score <= 1:
        return max(0.0, raw_score)
    return 1 / (1 + raw_score)


def format_docs(scored_docs: list[tuple[object, float]]) -> str:
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


def serialize_document(doc: dict) -> dict:
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


def delete_local_file(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)


def delete_user_vector_collection(user_id: int) -> None:
    try:
        user_db = get_user_db(user_id)
        user_db._client.delete_collection(name=get_user_collection_name(user_id))
    except Exception:
        pass
    get_user_db.cache_clear()


@app.post("/api/register")
def register(req: RegisterRequest, request: Request):
    client_ip = get_client_ip(request)
    enforce_rate_limit(f"register:ip:{client_ip}", auth_rate_policy)

    username = normalize_username(req.username)
    validate_username(username)
    validate_password(req.password)

    existing = db.get_user_by_username(username)
    if existing:
        raise HTTPException(status_code=409, detail="Username already taken")

    hashed = hash_password(req.password)
    user_id = db.create_user(username, hashed)
    token = create_access_token({"sub": str(user_id)})

    return {
        "message": "Account created successfully",
        "token": token,
        "user": {"id": user_id, "username": username},
    }


@app.post("/api/login")
def login(req: LoginRequest, request: Request):
    client_ip = get_client_ip(request)
    enforce_rate_limit(f"login:ip:{client_ip}", auth_rate_policy)

    username = normalize_username(req.username)
    validate_username(username)
    enforce_rate_limit(f"login:user:{username}", username_login_rate_policy)

    user = db.get_user_by_username(username)
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = create_access_token({"sub": str(user["id"])})
    return {
        "token": token,
        "user": {"id": user["id"], "username": user["username"]},
    }


@app.get("/api/me")
def get_profile(user: dict = Depends(get_current_user)):
    return {"id": user["id"], "username": user["username"], "created_at": user["created_at"]}


@app.delete("/api/me")
def delete_profile(user: dict = Depends(get_current_user)):
    user_id = user["id"]
    docs = db.get_user_documents(user_id)

    for doc in docs:
        file_path = os.path.join(get_user_upload_dir(user_id), doc["filename"])
        delete_local_file(file_path)

    delete_user_vector_collection(user_id)
    shutil.rmtree(get_user_upload_dir(user_id), ignore_errors=True)
    deleted = db.delete_user(user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="User not found")

    return {"message": "Account and documents deleted"}


@app.post("/api/upload")
def upload_document(
    background_tasks: BackgroundTasks,
    request: Request,
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    user_id = user["id"]
    client_ip = get_client_ip(request)
    enforce_rate_limit(f"upload:ip:{client_ip}", upload_rate_policy)
    enforce_rate_limit(f"upload:user:{user_id}", upload_rate_policy)
    try:
        get_embeddings_client()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    if not file.filename:
        raise HTTPException(status_code=400, detail="File name is required")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    original_name = sanitize_filename(file.filename, ext)
    user_dir = get_user_upload_dir(user_id)
    safe_filename = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(user_dir, safe_filename)

    file_size = save_upload_file(file, file_path)
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


@app.get("/api/documents")
def list_documents(user: dict = Depends(get_current_user)):
    docs = db.get_user_documents(user["id"])
    return {"documents": [serialize_document(d) for d in docs]}


@app.get("/api/documents/{doc_id}")
def get_document_info(doc_id: int, user: dict = Depends(get_current_user)):
    doc = db.get_document(doc_id, user["id"])
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return serialize_document(doc)


@app.delete("/api/documents/{doc_id}")
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


@app.post("/api/chat")
def chat(request: ChatRequest, web_request: Request, user: dict = Depends(get_current_user)):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if len(question) > MAX_QUESTION_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Question is too long. Max length is {MAX_QUESTION_CHARS} characters.",
        )

    user_id = user["id"]
    client_ip = get_client_ip(web_request)
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


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "2.1.0"}


BUILD_DIR = os.path.join(os.path.dirname(__file__), "frontend", "build")
if os.path.isdir(BUILD_DIR):
    app.mount("/static", StaticFiles(directory=os.path.join(BUILD_DIR, "static")), name="static")

    @app.get("/{full_path:path}")
    async def serve_react(full_path: str):
        file_path = os.path.join(BUILD_DIR, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(BUILD_DIR, "index.html"))

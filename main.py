"""
RAG Knowledge Base — FastAPI Backend
Provides authenticated, per-user document upload, search, and chat.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
import json
from functools import lru_cache
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PDFPlumberLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import shutil
import uuid
from datetime import datetime

from config import (
    CHROMA_PATH, EMBED_MODEL, UPLOAD_DIR, LLM_MODEL,
    OPENAI_BASE_URL, CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVER_K
)
from auth import hash_password, verify_password, create_access_token, get_current_user
import database as db

# ── Init ──────────────────────────────────────────────────

db.init_db()
os.makedirs(UPLOAD_DIR, exist_ok=True)

embeddings = OpenAIEmbeddings(
    model=EMBED_MODEL,
    base_url=OPENAI_BASE_URL,
)
llm = ChatOpenAI(
    model=LLM_MODEL,
    base_url=OPENAI_BASE_URL,
    temperature=0,
)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an intelligent assistant. Use ONLY the provided context to answer the question.
If the answer is not in the context, say "I don't have enough information to answer that based on your documents."
Be precise, helpful, and cite specific details from the context when possible.

Context:
{context}

Question:
{question}

Answer:""",
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)
response_parser = StrOutputParser()
rag_generation_chain = prompt | llm | response_parser

app = FastAPI(title="RAG Knowledge Base", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}


# ── Pydantic models ──────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    question: str
    document_id: int | None = None


# ── Helpers ───────────────────────────────────────────────

def get_user_collection_name(user_id: int) -> str:
    """Each user gets their own ChromaDB collection."""
    return f"user_{user_id}_documents"


def get_user_upload_dir(user_id: int) -> str:
    """Each user gets their own upload directory."""
    path = os.path.join(UPLOAD_DIR, str(user_id))
    os.makedirs(path, exist_ok=True)
    return path


@lru_cache(maxsize=64)
def get_user_db(user_id: int) -> Chroma:
    """Get a ChromaDB instance scoped to a specific user."""
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name=get_user_collection_name(user_id),
    )


def format_docs(docs) -> str:
    """Format retrieved docs into a compact context block for the model."""
    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('original_name', 'unknown')} | "
        f"Page {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
        for doc in docs
    )


def load_document(file_path: str, ext: str):
    """Load a document based on its file extension."""
    if ext == ".pdf":
        loader = PDFPlumberLoader(file_path)
        return loader.load()
    elif ext in (".txt", ".md"):
        loader = TextLoader(file_path, encoding="utf-8")
        return loader.load()
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")


# ══════════════════════════════════════════════════════════
#  AUTH ROUTES
# ══════════════════════════════════════════════════════════

@app.post("/api/register")
def register(req: RegisterRequest):
    if len(req.username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    existing = db.get_user_by_username(req.username)
    if existing:
        raise HTTPException(status_code=409, detail="Username already taken")

    hashed = hash_password(req.password)
    user_id = db.create_user(req.username, hashed)
    token = create_access_token({"sub": str(user_id)})

    return {
        "message": "Account created successfully",
        "token": token,
        "user": {"id": user_id, "username": req.username},
    }


@app.post("/api/login")
def login(req: LoginRequest):
    user = db.get_user_by_username(req.username)
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


# ══════════════════════════════════════════════════════════
#  DOCUMENT ROUTES (all protected)
# ══════════════════════════════════════════════════════════

@app.post("/api/upload")
def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    user_id = user["id"]
    original_name = file.filename
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Save file to user's upload directory immediately
    user_dir = get_user_upload_dir(user_id)
    safe_filename = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(user_dir, safe_filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    file_size = os.path.getsize(file_path)

    # Insert a placeholder record in SQLite so the UI can list it immediately
    doc_id = db.add_document(
        user_id=user_id,
        filename=safe_filename,
        original_name=original_name,
        file_type=ext,
        page_count=0,
        chunk_count=0,
        file_size=file_size,
    )

    # Define the background worker
    def process_file_in_background():
        try:
            pages = load_document(file_path, ext)
        except Exception:
            return  # Could log error here

        chunks = text_splitter.split_documents(pages)

        if not chunks:
            return

        for chunk in chunks:
            chunk.metadata["user_id"] = user_id
            chunk.metadata["original_name"] = original_name
            chunk.metadata["stored_filename"] = safe_filename
            chunk.metadata["uploaded_at"] = datetime.utcnow().isoformat()

        try:
            user_db = get_user_db(user_id)
            user_db.add_documents(chunks)
            db.update_document_stats(
                doc_id=doc_id,
                user_id=user_id,
                page_count=len(pages),
                chunk_count=len(chunks),
            )
        except Exception:
            return

    # Queue the heavy LangChain work
    background_tasks.add_task(process_file_in_background)

    return {
        "message": f"Successfully queued '{original_name}' for processing",
        "document": {
            "id": doc_id,
            "name": original_name,
            "pages": 0,
            "chunks": 0,
            "size": file_size,
        },
    }


@app.get("/api/documents")
def list_documents(user: dict = Depends(get_current_user)):
    docs = db.get_user_documents(user["id"])
    return {
        "documents": [
            {
                "id": d["id"],
                "name": d["original_name"],
                "type": d["file_type"],
                "pages": d["page_count"],
                "chunks": d["chunk_count"],
                "size": d["file_size"],
                "uploaded_at": d["uploaded_at"],
            }
            for d in docs
        ]
    }


@app.get("/api/documents/{doc_id}")
def get_document_info(doc_id: int, user: dict = Depends(get_current_user)):
    doc = db.get_document(doc_id, user["id"])
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "id": doc["id"],
        "name": doc["original_name"],
        "type": doc["file_type"],
        "pages": doc["page_count"],
        "chunks": doc["chunk_count"],
        "size": doc["file_size"],
        "uploaded_at": doc["uploaded_at"],
    }


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: int, user: dict = Depends(get_current_user)):
    doc = db.get_document(doc_id, user["id"])
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Remove from ChromaDB — delete chunks that match this filename
    user_db = get_user_db(user["id"])
    try:
        collection = user_db._collection
        # Get all IDs matching this document
        results = collection.get(where={"stored_filename": doc["filename"]})
        if results and results["ids"]:
            collection.delete(ids=results["ids"])
    except Exception:
        pass  # ChromaDB collection might not exist yet

    # Remove physical file
    file_path = os.path.join(get_user_upload_dir(user["id"]), doc["filename"])
    if os.path.exists(file_path):
        os.remove(file_path)

    # Remove from SQLite
    db.delete_document(doc_id, user["id"])

    return {"message": f"Deleted '{doc['original_name']}'"}


# ══════════════════════════════════════════════════════════
#  CHAT ROUTE (protected)
# ══════════════════════════════════════════════════════════

@app.post("/api/chat")
def chat(request: ChatRequest, user: dict = Depends(get_current_user)):
    user_db = get_user_db(user["id"])

    search_kwargs = {"k": RETRIEVER_K}

    if request.document_id:
        doc = db.get_document(request.document_id, user["id"])
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        search_kwargs["filter"] = {"stored_filename": doc["filename"]}

    retriever = user_db.as_retriever(search_kwargs=search_kwargs)

    def generate():
        try:
            # Fetch once and reuse for both source chips and model context.
            source_docs = retriever.invoke(request.question)
            sources = []
            seen = set()
            for doc in source_docs:
                name = doc.metadata.get("original_name", "Unknown")
                page = doc.metadata.get("page", "")
                key = f"{name}|{page}"
                if key not in seen:
                    seen.add(key)
                    sources.append({"name": name, "page": page})

            yield json.dumps({"type": "sources", "data": sources}) + "\n"

            context = format_docs(source_docs)
            for chunk in rag_generation_chain.stream(
                {"context": context, "question": request.question}
            ):
                yield json.dumps({"type": "token", "data": chunk}) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "data": str(e)}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


# ══════════════════════════════════════════════════════════
#  HEALTH
# ══════════════════════════════════════════════════════════

@app.get("/api/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


# ── Serve React build in production ──────────────────────
# If a built frontend exists, serve it as static files
BUILD_DIR = os.path.join(os.path.dirname(__file__), "frontend", "build")
if os.path.isdir(BUILD_DIR):
    from fastapi.responses import FileResponse

    app.mount("/static", StaticFiles(directory=os.path.join(BUILD_DIR, "static")), name="static")

    @app.get("/{full_path:path}")
    async def serve_react(full_path: str):
        """Catch-all route to serve the React SPA."""
        file_path = os.path.join(BUILD_DIR, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(BUILD_DIR, "index.html"))

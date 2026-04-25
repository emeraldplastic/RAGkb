"""
Centralized configuration for the RAG Knowledge Base.
Loads environment variables from .env with secure defaults.
"""

from __future__ import annotations

import os

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # Optional in minimal runtime/test environments.
    def load_dotenv(*_args, **_kwargs):  # type: ignore[no-redef]
        return False

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_csv(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
IS_VERCEL = os.getenv("VERCEL", "").strip() == "1"
DATA_ROOT = "/tmp/ragkb_data" if IS_VERCEL else "."

# Security
SECRET_KEY = os.getenv("SECRET_KEY", "dev-only-secret-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "480"))
JWT_ISSUER = os.getenv("JWT_ISSUER", "ragkb")
JWT_AUDIENCE = os.getenv("JWT_AUDIENCE", "ragkb-users")

# API security and privacy
SECURITY_HEADERS_ENABLED = _env_bool("SECURITY_HEADERS_ENABLED", True)
TRUSTED_HOSTS = _env_csv(
    "TRUSTED_HOSTS",
    "localhost,127.0.0.1,testserver,*.vercel.app",
)
CORS_ALLOW_ORIGINS = _env_csv(
    "CORS_ALLOW_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
)
ENABLE_CORS = _env_bool("ENABLE_CORS", True)

# Rate limiting
RATE_LIMIT_ENABLED = _env_bool("RATE_LIMIT_ENABLED", True)
AUTH_RATE_LIMIT_REQUESTS = int(os.getenv("AUTH_RATE_LIMIT_REQUESTS", "12"))
AUTH_RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("AUTH_RATE_LIMIT_WINDOW_SECONDS", "60"))
LOGIN_USERNAME_RATE_LIMIT_REQUESTS = int(os.getenv("LOGIN_USERNAME_RATE_LIMIT_REQUESTS", "8"))
LOGIN_USERNAME_RATE_LIMIT_WINDOW_SECONDS = int(
    os.getenv("LOGIN_USERNAME_RATE_LIMIT_WINDOW_SECONDS", "300")
)
UPLOAD_RATE_LIMIT_REQUESTS = int(os.getenv("UPLOAD_RATE_LIMIT_REQUESTS", "20"))
UPLOAD_RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("UPLOAD_RATE_LIMIT_WINDOW_SECONDS", "300"))
CHAT_RATE_LIMIT_REQUESTS = int(os.getenv("CHAT_RATE_LIMIT_REQUESTS", "45"))
CHAT_RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("CHAT_RATE_LIMIT_WINDOW_SECONDS", "60"))

# Database and storage
DATABASE_PATH = os.getenv(
    "DATABASE_PATH",
    os.path.join(DATA_ROOT, "ragkb.db") if IS_VERCEL else "./ragkb.db",
)
CHROMA_PATH = os.getenv(
    "CHROMA_PATH",
    os.path.join(DATA_ROOT, "chroma_db") if IS_VERCEL else "./chroma_db",
)
UPLOAD_DIR = os.getenv(
    "UPLOAD_DIR",
    os.path.join(DATA_ROOT, "uploaded_docs") if IS_VERCEL else "./uploaded_docs",
)
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
MAX_FILENAME_CHARS = int(os.getenv("MAX_FILENAME_CHARS", "180"))
ALLOWED_EXTENSIONS = {ext.lower() for ext in _env_csv("ALLOWED_EXTENSIONS", ".pdf,.txt,.md")}

# AI provider and model settings
AI_PROVIDER = os.getenv("AI_PROVIDER", "openai").strip().lower()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", AI_PROVIDER).strip().lower()
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", AI_PROVIDER).strip().lower()

EMBED_MODEL = os.getenv(
    "EMBED_MODEL",
    "models/text-embedding-004" if EMBED_PROVIDER == "google" else "text-embedding-3-small",
)
LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "gemini-1.5-flash" if LLM_PROVIDER == "google" else "gpt-4o-mini",
)
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# RAG behavior
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "6"))
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.2"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "22000"))
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "2000"))
MAX_SOURCES_IN_RESPONSE = int(os.getenv("MAX_SOURCES_IN_RESPONSE", "8"))


def validate_config() -> None:
    """Fail fast for unsafe production defaults."""
    if APP_ENV == "production" and SECRET_KEY == "dev-only-secret-change-me":
        raise RuntimeError("SECRET_KEY must be set in production.")

    if MAX_UPLOAD_MB < 1:
        raise RuntimeError("MAX_UPLOAD_MB must be >= 1.")

    if RETRIEVER_K < 1:
        raise RuntimeError("RETRIEVER_K must be >= 1.")

    if not (0 <= MIN_RELEVANCE_SCORE <= 1):
        raise RuntimeError("MIN_RELEVANCE_SCORE must be between 0 and 1.")

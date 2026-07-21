"""
RAG Knowledge Base - FastAPI backend with per-user document isolation.
Main application factory and routing entrypoint.
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

import database as db
from config import (
    AUTH_RATE_LIMIT_REQUESTS,
    AUTH_RATE_LIMIT_WINDOW_SECONDS,
    CHAT_RATE_LIMIT_REQUESTS,
    CHAT_RATE_LIMIT_WINDOW_SECONDS,
    CORS_ALLOW_ORIGINS,
    ENABLE_CORS,
    LOGIN_USERNAME_RATE_LIMIT_REQUESTS,
    LOGIN_USERNAME_RATE_LIMIT_WINDOW_SECONDS,
    RATE_LIMIT_ENABLED,
    TRUSTED_HOSTS,
    UPLOAD_DIR,
    UPLOAD_RATE_LIMIT_REQUESTS,
    UPLOAD_RATE_LIMIT_WINDOW_SECONDS,
    validate_config,
)
from middleware import SecurityHeadersMiddleware
from rate_limit import InMemoryRateLimiter, RateLimitPolicy
from routers import auth_routes, chat_routes, document_routes


# ── Initialization ────────────────────────────────────────

validate_config()
db.init_db()
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="RAG Knowledge Base", version="2.1.0")

# ── Middleware ──────────────────────────────────────────

app.add_middleware(TrustedHostMiddleware, allowed_hosts=TRUSTED_HOSTS)
app.add_middleware(SecurityHeadersMiddleware)

if ENABLE_CORS and CORS_ALLOW_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ALLOW_ORIGINS,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )

# ── Rate Limiting ───────────────────────────────────────

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


def enforce_rate_limit(bucket_key: str, policy: RateLimitPolicy) -> None:
    """Enforce rate limits and raise 429 if exceeded, otherwise return silently."""
    if not RATE_LIMIT_ENABLED:
        return
    allowed, retry_after = rate_limiter.hit(bucket_key, policy)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please retry shortly.",
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(policy.limit),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(retry_after),
            },
        )


# ── Routers ─────────────────────────────────────────────

app.include_router(auth_routes.router)
app.include_router(document_routes.router)
app.include_router(chat_routes.router)


@app.get("/api/health", tags=["system"])
def health():
    return {"status": "ok", "version": "2.1.0"}


@app.get("/api/stats", tags=["system"])
def stats():
    """Return usage statistics for monitoring and analytics."""
    return {
        "rate_limit_stats": {
            "enabled": RATE_LIMIT_ENABLED,
            "active_clients": len(rate_limiter._requests) if hasattr(rate_limiter, '_requests') else 0,
            "policies": {
                "auth": {"limit": AUTH_RATE_LIMIT_REQUESTS, "window": AUTH_RATE_LIMIT_WINDOW_SECONDS},
                "username_login": {"limit": LOGIN_USERNAME_RATE_LIMIT_REQUESTS, "window": LOGIN_USERNAME_RATE_LIMIT_WINDOW_SECONDS},
                "upload": {"limit": UPLOAD_RATE_LIMIT_REQUESTS, "window": UPLOAD_RATE_LIMIT_WINDOW_SECONDS},
                "chat": {"limit": CHAT_RATE_LIMIT_REQUESTS, "window": CHAT_RATE_LIMIT_WINDOW_SECONDS},
            }
        },
        "config": {
            "cors_enabled": ENABLE_CORS,
            "trusted_hosts": TRUSTED_HOSTS,
            "upload_dir": UPLOAD_DIR,
            "version": "2.1.0"
        }
    }


# ── Static React Frontend (if built) ────────────────────

BUILD_DIR = os.path.join(os.path.dirname(__file__), "frontend", "build")
if os.path.isdir(BUILD_DIR):
    app.mount("/static", StaticFiles(directory=os.path.join(BUILD_DIR, "static")), name="static")

    @app.get("/{full_path:path}")
    async def serve_react(full_path: str):
        file_path = os.path.join(BUILD_DIR, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(BUILD_DIR, "index.html"))

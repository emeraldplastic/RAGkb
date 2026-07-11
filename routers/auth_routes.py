"""
Authentication routes: register, login, profile management.
"""

from __future__ import annotations

import os
import shutil

from fastapi import APIRouter, Depends, HTTPException, Request

import database as db
from auth import create_access_token, get_current_user, hash_password, verify_password
from schemas import LoginRequest, RegisterRequest
from serializers import serialize_document
from services.file_service import delete_local_file, get_user_upload_dir
from services.rag_service import delete_user_vector_collection
from services.validators import normalize_username, validate_password, validate_username

router = APIRouter(prefix="/api", tags=["auth"])


def _get_client_ip(request: Request) -> str:
    """Extract the real client IP from X-Forwarded-For or the connection."""
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


@router.post("/register")
def register(req: RegisterRequest, request: Request):
    from main import enforce_rate_limit, auth_rate_policy

    client_ip = _get_client_ip(request)
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


@router.post("/login")
def login(req: LoginRequest, request: Request):
    from main import enforce_rate_limit, auth_rate_policy, username_login_rate_policy

    client_ip = _get_client_ip(request)
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


@router.get("/me")
def get_profile(user: dict = Depends(get_current_user)):
    return {"id": user["id"], "username": user["username"], "created_at": user["created_at"]}


@router.delete("/me")
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

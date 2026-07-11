"""
Pydantic request/response models for the RAG Knowledge Base API.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from config import MAX_QUESTION_CHARS


class RegisterRequest(BaseModel):
    username: str = Field(max_length=64)
    password: str = Field(max_length=128)


class LoginRequest(BaseModel):
    username: str = Field(max_length=64)
    password: str = Field(max_length=128)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    document_id: int | None = None

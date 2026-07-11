"""
Input validation helpers for usernames and passwords.
"""

from __future__ import annotations

import re

from fastapi import HTTPException


USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]{3,32}$")


def normalize_username(username: str) -> str:
    """Strip whitespace and lowercase a username."""
    return username.strip().lower()


def validate_username(username: str) -> None:
    """Raise 400 if username does not meet policy."""
    if not USERNAME_PATTERN.fullmatch(username):
        raise HTTPException(
            status_code=400,
            detail="Username must be 3-32 chars and use letters, numbers, ., _, or -",
        )


def validate_password(password: str) -> None:
    """Raise 400 if password does not meet strength policy."""
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

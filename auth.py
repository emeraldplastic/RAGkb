"""
Authentication module: JWT tokens + bcrypt password hashing.
Provides a FastAPI dependency to protect routes.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import os
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

import database as db
from config import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    ALGORITHM,
    JWT_AUDIENCE,
    JWT_ISSUER,
    SECRET_KEY,
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/login")

PBKDF2_ROUNDS = 260_000

try:
    import bcrypt  # type: ignore
except ModuleNotFoundError:
    bcrypt = None  # type: ignore


def hash_password(password: str) -> str:
    pwd_bytes = password.encode("utf-8")
    if bcrypt is not None:
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(pwd_bytes, salt).decode("utf-8")

    salt = os.urandom(16)
    derived = hashlib.pbkdf2_hmac("sha256", pwd_bytes, salt, PBKDF2_ROUNDS)
    return (
        "pbkdf2$"
        f"{PBKDF2_ROUNDS}$"
        f"{salt.hex()}$"
        f"{derived.hex()}"
    )


def verify_password(plain: str, hashed: str) -> bool:
    plain_bytes = plain.encode("utf-8")
    if hashed.startswith("pbkdf2$"):
        try:
            _, rounds, salt_hex, digest_hex = hashed.split("$", 3)
            rounds_int = int(rounds)
            salt = bytes.fromhex(salt_hex)
            expected = bytes.fromhex(digest_hex)
        except (ValueError, TypeError):
            return False
        actual = hashlib.pbkdf2_hmac("sha256", plain_bytes, salt, rounds_int)
        return hmac.compare_digest(actual, expected)

    if bcrypt is None:
        return False
    return bcrypt.checkpw(plain_bytes, hashed.encode("utf-8"))


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))

    to_encode = data.copy()
    to_encode.update(
        {
            "iat": now,
            "nbf": now,
            "exp": expire,
            "iss": JWT_ISSUER,
            "aud": JWT_AUDIENCE,
        }
    )
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """
    Decode the JWT, look up the user, and return their profile.
    Raises 401 if the token is invalid or user not found.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
            audience=JWT_AUDIENCE,
            issuer=JWT_ISSUER,
            options={"verify_aud": True, "verify_signature": True},
        )
        user_id = payload.get("sub")
        if not user_id:
            raise credentials_exception
        user_id_int = int(user_id)
    except (JWTError, ValueError):
        raise credentials_exception

    user = db.get_user_by_id(user_id_int)
    if user is None:
        raise credentials_exception
    return user

"""
File handling services: upload, load, sanitize, and delete.
Includes magic-byte validation for upload security.
"""

from __future__ import annotations

import os

from fastapi import HTTPException, UploadFile
from langchain_community.document_loaders import PDFPlumberLoader, TextLoader

from config import ALLOWED_EXTENSIONS, MAX_FILENAME_CHARS, MAX_UPLOAD_MB, UPLOAD_DIR


# Magic bytes for file type verification
_MAGIC_SIGNATURES: dict[str, list[bytes]] = {
    ".pdf": [b"%PDF"],
}


def get_user_upload_dir(user_id: int) -> str:
    """Return (and create) the per-user upload directory."""
    path = os.path.join(UPLOAD_DIR, str(user_id))
    os.makedirs(path, exist_ok=True)
    return path


def sanitize_filename(filename: str, ext: str) -> str:
    """Return a safe, length-limited filename."""
    base = os.path.basename(filename).strip()
    if not base:
        base = f"upload{ext}"
    if len(base) <= MAX_FILENAME_CHARS:
        return base
    stem, stem_ext = os.path.splitext(base)
    keep = max(1, MAX_FILENAME_CHARS - len(stem_ext))
    return f"{stem[:keep]}{stem_ext}"


def validate_file_extension(filename: str) -> str:
    """Validate file extension against allowlist. Returns the lowercase extension."""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )
    return ext


def _verify_magic_bytes(file_path: str, ext: str) -> None:
    """Verify that file content matches the expected magic bytes for its extension.

    Prevents extension spoofing (e.g. renaming an .exe to .pdf).
    Only enforced for extensions with known signatures; text formats are skipped.
    """
    signatures = _MAGIC_SIGNATURES.get(ext)
    if not signatures:
        return  # No signature check for plain-text formats

    try:
        with open(file_path, "rb") as f:
            header = f.read(8)
        if not any(header.startswith(sig) for sig in signatures):
            os.remove(file_path)
            raise HTTPException(
                status_code=400,
                detail=f"File content does not match the expected {ext} format.",
            )
    except HTTPException:
        raise
    except Exception:
        pass  # If we can't read headers, let the document loader handle it


def save_upload_file(file: UploadFile, file_path: str) -> int:
    """Stream an upload to disk with size-limit enforcement. Returns bytes written."""
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


def load_document(file_path: str, ext: str):
    """Load a document using the appropriate LangChain loader."""
    if ext == ".pdf":
        return PDFPlumberLoader(file_path).load()
    if ext in (".txt", ".md"):
        return TextLoader(file_path, encoding="utf-8").load()
    raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")


def delete_local_file(path: str) -> None:
    """Remove a file from disk if it exists."""
    if os.path.exists(path):
        os.remove(path)

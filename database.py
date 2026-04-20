"""
SQLite database setup for user management and document metadata.
Users can only see and query their own documents.
"""

import sqlite3
from config import DATABASE_PATH


def get_connection():
    """Get a SQLite connection with row_factory for dict-like access."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            original_name TEXT NOT NULL,
            file_type TEXT NOT NULL,
            page_count INTEGER DEFAULT 0,
            chunk_count INTEGER DEFAULT 0,
            file_size INTEGER DEFAULT 0,
            uploaded_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_documents_user_uploaded
        ON documents(user_id, uploaded_at DESC)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_documents_user_filename
        ON documents(user_id, filename)
    """)

    conn.commit()
    conn.close()


# ── User operations ──────────────────────────────────────

def create_user(username: str, password_hash: str) -> int:
    """Create a new user and return their ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO users (username, password_hash) VALUES (?, ?)",
        (username, password_hash)
    )
    conn.commit()
    user_id = cursor.lastrowid
    conn.close()
    return user_id


def get_user_by_username(username: str) -> dict | None:
    """Fetch a user by username."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_id(user_id: int) -> dict | None:
    """Fetch a user by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, created_at FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


# ── Document operations ──────────────────────────────────

def add_document(user_id: int, filename: str, original_name: str,
                 file_type: str, page_count: int, chunk_count: int,
                 file_size: int) -> int:
    """Record a document upload for a specific user."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO documents
           (user_id, filename, original_name, file_type, page_count, chunk_count, file_size)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (user_id, filename, original_name, file_type, page_count, chunk_count, file_size)
    )
    conn.commit()
    doc_id = cursor.lastrowid
    conn.close()
    return doc_id


def update_document_stats(doc_id: int, user_id: int, page_count: int, chunk_count: int) -> bool:
    """Update processed page/chunk counts for a document that belongs to the user."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE documents
        SET page_count = ?, chunk_count = ?
        WHERE id = ? AND user_id = ?
        """,
        (page_count, chunk_count, doc_id, user_id),
    )
    updated = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return updated


def get_user_documents(user_id: int) -> list[dict]:
    """Get all documents belonging to a specific user."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM documents WHERE user_id = ? ORDER BY uploaded_at DESC",
        (user_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_document(doc_id: int, user_id: int) -> dict | None:
    """Get a specific document, only if it belongs to the user."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM documents WHERE id = ? AND user_id = ?",
        (doc_id, user_id)
    )
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_document_by_filename(filename: str, user_id: int) -> dict | None:
    """Get a document by filename for a specific user."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM documents WHERE filename = ? AND user_id = ?",
        (filename, user_id)
    )
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def delete_document(doc_id: int, user_id: int) -> bool:
    """Delete a document record, only if it belongs to the user."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM documents WHERE id = ? AND user_id = ?",
        (doc_id, user_id)
    )
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted

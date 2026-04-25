"""
Test fixtures: temp database, FastAPI test client, auth helpers.
Mocks heavy dependencies (embeddings) for fast, lightweight tests.
"""

import os
import sys
import pytest

# Ensure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class FakeEmbeddings:
    """Lightweight mock embedding model."""
    def __init__(self, **kwargs):
        pass

    def embed_documents(self, texts):
        return [[0.1] * 384 for _ in texts]

    def embed_query(self, text):
        return [0.1] * 384


class FakeRagChain:
    """Lightweight chain mock used in tests."""

    def stream(self, _payload):
        yield "Mocked answer"


@pytest.fixture(autouse=True)
def setup_test_env(tmp_path, monkeypatch):
    """Configure all paths to use temporary directories for isolation."""
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-for-testing-only")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "0")


@pytest.fixture
def client(setup_test_env):
    """Create a fresh FastAPI test client with mocked embeddings.

    Reload order matters: config → database → auth → main.
    Each module imports from the previous ones, so they must be
    reloaded in dependency order to pick up the test env vars.
    """
    import importlib

    # 1. Reload config first — everything depends on it
    import config
    importlib.reload(config)

    # 2. Reload database — it imports DATABASE_PATH from config
    import database
    importlib.reload(database)

    # 3. Reload auth — it imports SECRET_KEY etc. from config & database
    import auth
    importlib.reload(auth)

    # 4. Reload main and replace external model clients with fakes
    import main
    importlib.reload(main)
    main.get_embeddings_client = lambda: FakeEmbeddings()
    main.get_rag_generation_chain = lambda: FakeRagChain()
    main.get_user_db.cache_clear()

    from starlette.testclient import TestClient
    yield TestClient(main.app)


@pytest.fixture
def auth_headers(client):
    """Register a test user and return auth headers."""
    res = client.post("/api/register", json={
        "username": "testuser",
        "password": "Testpass123!",
    })
    assert res.status_code == 200, f"Register failed: {res.json()}"
    token = res.json()["token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def second_user_headers(client):
    """Register a second test user and return auth headers."""
    res = client.post("/api/register", json={
        "username": "otheruser",
        "password": "Otherpass123!",
    })
    assert res.status_code == 200, f"Register failed: {res.json()}"
    token = res.json()["token"]
    return {"Authorization": f"Bearer {token}"}

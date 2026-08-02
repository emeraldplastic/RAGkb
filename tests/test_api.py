"""
API tests for the RAG Knowledge Base backend.
Tests authentication, document management, and access isolation.
"""

import io
import os


class TestAuth:
    """Authentication endpoint tests."""

    def test_register_success(self, client):
        res = client.post("/api/register", json={
            "username": "newuser",
            "password": "Password123!",
        })
        assert res.status_code == 200
        data = res.json()
        assert "token" in data
        assert data["user"]["username"] == "newuser"
        assert "id" in data["user"]

    def test_register_duplicate_username(self, client):
        client.post("/api/register", json={
            "username": "dupuser",
            "password": "Password123!",
        })
        res = client.post("/api/register", json={
            "username": "dupuser",
            "password": "Different123!",
        })
        assert res.status_code == 409
        assert "already taken" in res.json()["detail"].lower()

    def test_register_short_username(self, client):
        res = client.post("/api/register", json={
            "username": "ab",
            "password": "password123",
        })
        assert res.status_code == 400

    def test_register_short_password(self, client):
        res = client.post("/api/register", json={
            "username": "validuser",
            "password": "weak",
        })
        assert res.status_code == 400

    def test_login_success(self, client):
        # Register first
        client.post("/api/register", json={
            "username": "loginuser",
            "password": "Password123!",
        })
        # Then login
        res = client.post("/api/login", json={
            "username": "loginuser",
            "password": "Password123!",
        })
        assert res.status_code == 200
        assert "token" in res.json()

    def test_login_wrong_password(self, client):
        client.post("/api/register", json={
            "username": "loginuser2",
            "password": "Password123!",
        })
        res = client.post("/api/login", json={
            "username": "loginuser2",
            "password": "Wrongpassword1!",
        })
        assert res.status_code == 401

    def test_login_nonexistent_user(self, client):
        res = client.post("/api/login", json={
            "username": "nonexistent",
            "password": "Password123!",
        })
        assert res.status_code == 401


class TestProtectedRoutes:
    """Verify that all routes require authentication."""

    def test_documents_without_token(self, client):
        res = client.get("/api/documents")
        assert res.status_code == 401

    def test_upload_without_token(self, client):
        res = client.post("/api/upload")
        assert res.status_code == 401

    def test_chat_without_token(self, client):
        res = client.post("/api/chat", json={"question": "test"})
        assert res.status_code == 401

    def test_me_without_token(self, client):
        res = client.get("/api/me")
        assert res.status_code == 401

    def test_invalid_token(self, client):
        res = client.get("/api/documents", headers={
            "Authorization": "Bearer invalidtoken123"
        })
        assert res.status_code == 401


class TestDocuments:
    """Document management tests."""

    def test_list_documents_empty(self, client, auth_headers):
        res = client.get("/api/documents", headers=auth_headers)
        assert res.status_code == 200
        assert res.json()["documents"] == []

    def test_upload_pdf(self, client, auth_headers):
        # Create a minimal valid PDF
        pdf_content = b"%PDF-1.0\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\nxref\n0 3\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \ntrailer\n<< /Size 3 /Root 1 0 R >>\nstartxref\n115\n%%EOF"
        files = {"file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")}
        res = client.post("/api/upload", headers=auth_headers, files=files)
        # Even if PDF parsing fails on minimal PDF, the endpoint should handle it
        assert res.status_code in (200, 400)

    def test_upload_txt(self, client, auth_headers):
        content = b"This is a test document.\nIt has multiple lines.\nFor testing RAG search."
        files = {"file": ("test.txt", io.BytesIO(content), "text/plain")}
        res = client.post("/api/upload", headers=auth_headers, files=files)
        assert res.status_code == 200
        data = res.json()
        assert data["document"]["name"] == "test.txt"
        assert data["document"]["chunks"] == 0
        assert data["document"]["status"] == "queued"

    def test_upload_unsupported_type(self, client, auth_headers):
        files = {"file": ("test.exe", io.BytesIO(b"binary"), "application/octet-stream")}
        res = client.post("/api/upload", headers=auth_headers, files=files)
        assert res.status_code == 400

    def test_list_documents_after_upload(self, client, auth_headers):
        # Upload a file
        content = b"Test content for listing."
        files = {"file": ("list_test.txt", io.BytesIO(content), "text/plain")}
        client.post("/api/upload", headers=auth_headers, files=files)

        res = client.get("/api/documents", headers=auth_headers)
        assert res.status_code == 200
        docs = res.json()["documents"]
        assert len(docs) == 1
        assert docs[0]["name"] == "list_test.txt"
        assert docs[0]["status"] in {"queued", "processing", "ready", "failed"}

    def test_delete_document(self, client, auth_headers):
        # Upload
        content = b"Delete me."
        files = {"file": ("deleteme.txt", io.BytesIO(content), "text/plain")}
        client.post("/api/upload", headers=auth_headers, files=files)

        # Get the doc ID
        docs = client.get("/api/documents", headers=auth_headers).json()["documents"]
        doc_id = docs[0]["id"]

        # Delete
        res = client.delete(f"/api/documents/{doc_id}", headers=auth_headers)
        assert res.status_code == 200

        # Confirm deletion
        docs_after = client.get("/api/documents", headers=auth_headers).json()["documents"]
        assert len(docs_after) == 0

    def test_delete_nonexistent_document(self, client, auth_headers):
        res = client.delete("/api/documents/9999", headers=auth_headers)
        assert res.status_code == 404


class TestUserIsolation:
    """Verify that users cannot see each other's documents."""

    def test_documents_isolated_between_users(self, client, auth_headers, second_user_headers):
        # User 1 uploads a document
        content1 = b"User 1 secret document."
        files1 = {"file": ("user1_doc.txt", io.BytesIO(content1), "text/plain")}
        client.post("/api/upload", headers=auth_headers, files=files1)

        # User 2 uploads a different document
        content2 = b"User 2 private file."
        files2 = {"file": ("user2_doc.txt", io.BytesIO(content2), "text/plain")}
        client.post("/api/upload", headers=second_user_headers, files=files2)

        # User 1 should only see their document
        docs1 = client.get("/api/documents", headers=auth_headers).json()["documents"]
        assert len(docs1) == 1
        assert docs1[0]["name"] == "user1_doc.txt"

        # User 2 should only see their document
        docs2 = client.get("/api/documents", headers=second_user_headers).json()["documents"]
        assert len(docs2) == 1
        assert docs2[0]["name"] == "user2_doc.txt"

    def test_cannot_delete_other_users_document(self, client, auth_headers, second_user_headers):
        # User 1 uploads
        content = b"User 1 data."
        files = {"file": ("protected.txt", io.BytesIO(content), "text/plain")}
        client.post("/api/upload", headers=auth_headers, files=files)

        doc_id = client.get("/api/documents", headers=auth_headers).json()["documents"][0]["id"]

        # User 2 tries to delete User 1's document
        res = client.delete(f"/api/documents/{doc_id}", headers=second_user_headers)
        assert res.status_code == 404  # Not found (because it's not theirs)

        # User 1's document should still exist
        docs = client.get("/api/documents", headers=auth_headers).json()["documents"]
        assert len(docs) == 1


class TestProfile:
    """Profile endpoint tests."""

    def test_get_profile(self, client, auth_headers):
        res = client.get("/api/me", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        assert data["username"] == "testuser"
        assert "id" in data


class TestHealth:
    """Health check tests."""

    def test_health_endpoint(self, client):
        res = client.get("/api/health")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"


class TestChatExport:
    """Chat export endpoint tests."""

    def test_chat_export_markdown(self, client, auth_headers):
        res = client.get("/api/chat/export?format=markdown", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        assert data["format"] == "markdown"
        assert "# RAGkb Export Report" in data["content"]

    def test_chat_export_json(self, client, auth_headers):
        res = client.get("/api/chat/export?format=json", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        assert "ready_documents_count" in data
        assert "documents" in data


class TestDocumentStats:
    """Document statistics endpoint tests."""

    def test_get_document_stats_empty(self, client, auth_headers):
        res = client.get("/api/documents/stats", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        assert data["total_documents"] == 0
        assert "status_counts" in data
        assert data["status_counts"]["ready"] == 0
        assert data["status_counts"]["processing"] == 0
        assert data["status_counts"]["failed"] == 0
        assert data["status_counts"]["queued"] == 0
        assert data["total_storage_bytes"] == 0

    def test_get_document_stats_after_upload(self, client, auth_headers):
        # Upload a document
        content = b"Test document for stats."
        files = {"file": ("stats_test.txt", io.BytesIO(content), "text/plain")}
        client.post("/api/upload", headers=auth_headers, files=files)

        # Get stats
        res = client.get("/api/documents/stats", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        assert data["total_documents"] >= 1
        assert "status_counts" in data
        assert data["total_storage_bytes"] > 0

    def test_get_document_stats_without_auth(self, client):
        res = client.get("/api/documents/stats")
        assert res.status_code == 401


class TestDocumentSearch:
    """Document semantic search endpoint tests."""

    def test_search_requires_auth(self, client):
        res = client.get("/api/documents/search", params={"q": "python"})
        assert res.status_code == 401

    def test_search_short_query_rejected(self, client, auth_headers):
        res = client.get("/api/documents/search", params={"q": "a"}, headers=auth_headers)
        assert res.status_code == 400

    def test_search_invalid_limit_rejected(self, client, auth_headers):
        res = client.get(
            "/api/documents/search",
            params={"q": "python", "limit": 50},
            headers=auth_headers,
        )
        assert res.status_code == 400

    def test_search_empty_without_documents(self, client, auth_headers):
        res = client.get("/api/documents/search", params={"q": "python"}, headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        assert data["query"] == "python"
        assert data["count"] == 0
        assert data["results"] == []

    def test_search_returns_matches_after_upload(self, client, auth_headers):
        content = b"Python is a programming language. React is a UI library. SQL is used for databases."
        files = {"file": ("knowledge.txt", io.BytesIO(content), "text/plain")}
        res = client.post("/api/upload", headers=auth_headers, files=files)
        assert res.status_code == 200

        res = client.get(
            "/api/documents/search",
            params={"q": "python"},
            headers=auth_headers,
        )
        assert res.status_code == 200
        data = res.json()
        assert data["count"] >= 1
        hit = data["results"][0]
        assert "content" in hit
        assert "confidence" in hit
        assert hit["document"] == "knowledge.txt"

    def test_search_is_scoped_to_current_user(self, client, auth_headers, second_user_headers):
        content = b"Private notes about the RAG architecture."
        files = {"file": ("private.txt", io.BytesIO(content), "text/plain")}
        client.post("/api/upload", headers=auth_headers, files=files)

        res = client.get(
            "/api/documents/search",
            params={"q": "architecture"},
            headers=second_user_headers,
        )
        assert res.status_code == 200
        assert res.json()["count"] == 0


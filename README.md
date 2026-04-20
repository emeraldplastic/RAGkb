# RAGkb Deployment Guide

This app runs as one FastAPI service and serves the built React frontend.

## 1) Deploy online on Render (recommended)

### Prerequisites
- A GitHub repo containing this project.
- An OpenAI API key.

### Steps
1. Push this project to GitHub.
2. In Render, create a new **Blueprint** and select your repo.
3. Render will detect [`render.yaml`](./render.yaml).
4. Set `OPENAI_API_KEY` in Render environment variables.
5. (Optional) Set `OPENAI_BASE_URL` only if you use an OpenAI-compatible gateway.
6. Deploy.
7. Open: `https://<your-service>.onrender.com`

### Default health endpoint
- `GET /api/health`

## 2) Important env vars

- `SECRET_KEY`: JWT signing key.
- `DATABASE_PATH`: SQLite path (use mounted disk path in cloud).
- `CHROMA_PATH`: Chroma directory path.
- `UPLOAD_DIR`: uploaded files path.
- `OPENAI_API_KEY`: required API key.
- `OPENAI_BASE_URL`: optional override for compatible providers.
- `LLM_MODEL`: chat model, default `gpt-4o-mini`.
- `EMBED_MODEL`: embedding model, default `text-embedding-3-small`.

## 3) Local Docker test

```bash
docker compose up --build
```

Then open `http://localhost:8000`.

## Notes

- Frontend API base behavior:
  - `http://127.0.0.1:8000` on localhost.
  - same-origin API in production.
- The container respects cloud `PORT` automatically.

# RAGkb

RAGkb is a private document Q&A app:
- FastAPI backend (auth, upload, retrieval, streaming chat)
- React frontend
- per-user document isolation

## Security and privacy highlights

- Strict security headers + no-store API responses
- Configurable trusted hosts and CORS allowlist
- JWT issuer/audience validation
- Strong password policy
- Endpoint rate limiting for login/register/upload/chat
- Upload size and filename limits
- Relevance-threshold retrieval before generation
- Account-level data deletion endpoint (`DELETE /api/me`)
- Session-scoped auth token storage in the frontend

## Local development

1. Backend dependencies:

```bash
pip install -r requirements.txt
```

2. Frontend dependencies:

```bash
cd frontend
npm install
```

3. Copy env template:

```bash
cp .env.example .env
```

4. Run backend:

```bash
uvicorn main:app --reload
```

5. Run frontend (in another terminal):

```bash
cd frontend
npm start
```

## Test backend

```bash
pytest -q
```

## Deploy to Vercel

This repo is configured for Vercel using `vercel.json` and `api/index.py`.

1. Build the frontend so `frontend/build` exists:

```bash
cd frontend
npm run build
cd ..
```

2. Deploy:

```bash
vercel
```

3. Production deploy:

```bash
vercel --prod
```

## Key environment variables

See `.env.example` for full list. Most important in production:
- `APP_ENV=production`
- `SECRET_KEY` (required, long random value)
- `AI_PROVIDER` (`openai` or `google`)
- `OPENAI_API_KEY` for OpenAI/OpenAI-compatible APIs
- `GOOGLE_API_KEY` for Gemini
- `TRUSTED_HOSTS`
- `CORS_ALLOW_ORIGINS`

### Using Gemini instead of OpenAI

Set these Vercel environment variables:

```bash
AI_PROVIDER=google
LLM_PROVIDER=google
EMBED_PROVIDER=google
GOOGLE_API_KEY=<your-gemini-api-key>
```

Optional model overrides:

```bash
LLM_MODEL=gemini-1.5-flash
EMBED_MODEL=models/text-embedding-004
```

For OpenAI-compatible gateways, keep `AI_PROVIDER=openai`, set `OPENAI_API_KEY`,
and set `OPENAI_BASE_URL` when the provider requires a custom base URL. Make sure
the provider supports embeddings, not only chat.

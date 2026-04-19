# ── Build stage: React frontend ───────────────────
FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ── Runtime stage: Python backend ─────────────────
FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY config.py database.py auth.py main.py ./

# Copy built frontend
COPY --from=frontend-build /app/frontend/build ./frontend/build

# Create directories for data persistence
RUN mkdir -p /app/chroma_db /app/uploaded_docs

# Expose port
EXPOSE 8000

# Run with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

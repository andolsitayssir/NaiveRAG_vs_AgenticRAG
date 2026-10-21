#!/usr/bin/env bash
set -e

echo "[entrypoint] Activiity Chat server starting..."

# Ingest baseline index if not present
if [ ! -f /app/baseline/.index/docstore.json ]; then
    echo "[entrypoint] Naive RAG index not found -- running ingestion..."
    python -m baseline.cli ingest
    echo "[entrypoint] Ingestion done."
else
    echo "[entrypoint] Naive RAG index already exists -- skipping ingestion."
fi

# Agentic Ingestion (requires Qdrant and LLM)
echo "[entrypoint] Running Agentic Ingestion (if needed)..."
python -m activiity.ingest.cli

echo "[entrypoint] Starting uvicorn..."
exec uvicorn eval.webapp.chat_server:app --host 0.0.0.0 --port "${PORT:-8787}"
#!/usr/bin/env bash
set -euo pipefail

SKIP_INGEST=0
for arg in "$@"; do
  if [[ "$arg" == "--skip-ingest" ]]; then
    SKIP_INGEST=1
  fi
done

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required. Install Docker and try again." >&2
  exit 1
fi

echo "Starting containers..."
docker compose up -d --build

if [[ $SKIP_INGEST -eq 0 ]]; then
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY not set. Ingestion may fail unless you switch embeddings to Ollama." >&2
  fi

  echo "Running agentic ingestion..."
  docker compose run --rm chat python -m activiity.ingest.cli --force

  echo "Running naive ingestion..."
  docker compose run --rm chat python -m baseline.cli ingest --force
fi

echo "Ready:"
echo "- http://localhost:8787/chat/naive"
echo "- http://localhost:8787/chat/agentic"

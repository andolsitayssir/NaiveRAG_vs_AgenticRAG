param(
  [switch]$SkipIngest
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
  Write-Host "Docker is required. Install Docker Desktop and try again." -ForegroundColor Red
  exit 1
}

Write-Host "Starting containers..." -ForegroundColor Cyan
docker compose up -d --build

if (-not $SkipIngest) {
  if (-not $env:OPENAI_API_KEY) {
    Write-Host "OPENAI_API_KEY not set. Ingestion may fail unless you switch embeddings to Ollama." -ForegroundColor Yellow
  }

  Write-Host "Running agentic ingestion..." -ForegroundColor Cyan
  docker compose run --rm chat python -m activiity.ingest.cli --force

  Write-Host "Running naive ingestion..." -ForegroundColor Cyan
  docker compose run --rm chat python -m baseline.cli ingest --force
}

Write-Host "Ready:" -ForegroundColor Green
Write-Host "- http://localhost:8787/chat/naive"
Write-Host "- http://localhost:8787/chat/agentic"

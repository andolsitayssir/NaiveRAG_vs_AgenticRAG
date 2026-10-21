# Chatbot guide (naive + agentic)

## 1) What already exists in the repo

- FastAPI chat server: [eval/webapp/chat_server.py](eval/webapp/chat_server.py)
- Chat pages: [eval/webapp/static/chat_naive.html](eval/webapp/static/chat_naive.html) and [eval/webapp/static/chat_agentic.html](eval/webapp/static/chat_agentic.html)
- Frontend logic and API calls: [eval/webapp/static/chat.js](eval/webapp/static/chat.js)
- Chat routes: http://127.0.0.1:8787/chat/naive and http://127.0.0.1:8787/chat/agentic

## 2) Prerequisites

- Python 3.11+ and pip
- OpenRouter + OpenAI credentials
- Qdrant reachable at `QDRANT_URL` (required for agentic chat). The naive chatbot does not need Qdrant.

## 3) Clone and install dependencies

```bash
git clone https://github.com/andolsitayssir/activiity-agentic-naive-chatbots.git
cd activiity-agentic-naive-chatbots
```

Install dependencies:

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Bash:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4) Configure environment (OpenRouter Qwen3 14B)

Set these in your shell or in a `.env` file at the repo root:

```
OPENROUTER_API_KEY=sk-or-v1-...
OPENAI_API_KEY=sk-...
LLM_PROVIDER=openrouter
LLM_MODEL=qwen/qwen3-14b
QDRANT_URL=http://localhost:6333
```

If you use OpenAI embeddings, set this once on the settings page (http://127.0.0.1:8787):

- Embedding provider: `openai`
- Embedding model: `text-embedding-3-small`

If you prefer HuggingFace embeddings, set `HF_TOKEN` instead.

## 5) Build the indexes

Naive baseline index (no Qdrant required):

```bash
python -m baseline.cli ingest
```

Agentic index (requires Qdrant):

```bash
python -m activiity.ingest.cli
```

If you only want the naive chatbot, you can skip the agentic ingest step.

## 6) Run the chatbot UI

Start the local chat server:

PowerShell:

```powershell
python -m uvicorn eval.webapp.chat_server:app --reload --port 8787
```

Bash:

```bash
python -m uvicorn eval.webapp.chat_server:app --reload --port 8787
```

Open:
- http://127.0.0.1:8787/chat/naive
- http://127.0.0.1:8787/chat/agentic

# Activiity — RAG Eval Control Panel (Web App)

A single-page web app that controls the RAG evaluation program:

- **Settings**: configure the target RAG (OpenRouter / Ollama LLM, OpenAI / Ollama embeddings), the judge LLM for Ragas, API keys, budget cap.
- **Run**: pick a tier (reference / medium / advanced / all), choose flags (mock, ragas, BERTScore, SBERT, limit, workers), start a run, watch the live console stream.
- **Dashboard**: for the selected run, live KPI cards with pass/fail badges, per-class F1 bar charts, metric radars — one block per tier.
- **History**: all past runs, click to open in the dashboard, delete.

## Stack

- **Backend**: FastAPI (`eval/webapp/server.py`) — runs `eval.run_eval` as a subprocess and streams stdout via Server-Sent Events.
- **Frontend**: Single HTML file (`static/index.html`) using Tailwind CSS CDN, Alpine.js for reactivity, Chart.js for charts. No build step.
- **Persistence**: `eval/webapp/config.json` for settings; `eval/webapp/runs/<run-id>/` for each run's report, traces, HTML, and metadata.

## Run

From the project root:

```bash
# install the extra deps (already used by the eval program, but just in case):
pip install fastapi uvicorn httpx

# start the control panel
uvicorn eval.webapp.chat_server:app --reload --port 8787

# open
xdg-open http://localhost:8787   # or just paste the URL in the browser
```

On Windows:

```powershell
$env:PYTHONIOENCODING="utf-8"
python -m uvicorn eval.webapp.server:app --reload --port 8787
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/api/config` | current config (secrets masked) |
| `PUT`  | `/api/config` | update config (send raw keys to rotate) |
| `POST` | `/api/config/test-openrouter` | ping OpenRouter with the stored key |
| `POST` | `/api/config/test-ollama` | ping Ollama base URL |
| `GET`  | `/api/datasets` | count pairs per tier and bucket |
| `GET`  | `/api/thresholds` | release-gate thresholds |
| `GET`  | `/api/runs` | list all runs (meta) |
| `POST` | `/api/runs` | start a new run (returns run meta) |
| `GET`  | `/api/runs/{id}` | run meta + full report JSON |
| `GET`  | `/api/runs/{id}/stream` | SSE: live stdout lines and status transitions |
| `GET`  | `/api/runs/{id}/traces.csv` | per-pair trace CSV download |
| `GET`  | `/api/runs/{id}/report.html` | full HTML report (the one written by the runner) |
| `DELETE` | `/api/runs/{id}` | delete run and its files |
| `GET`  | `/api/health` | ok + tiers list |

## Security notes

This panel is designed for **local / trusted network** use. Before exposing it:

- add authentication (FastAPI dependency, NextAuth, or a reverse-proxy gate),
- put it behind HTTPS,
- consider moving secrets out of `config.json` into a secret store (AWS Secrets Manager, HashiCorp Vault, etc.).

The API keys in `config.json` are **never** returned by `GET /api/config` — they are masked (`sk-o…abcd`). Only the real key sent in a `PUT` body overwrites them.

## Screenshot map

```
┌─ Header ─────────────────────────────────────────────┐
│ Activiity — RAG Eval Control Panel   [⚙][▶][📊][🗂] │
├──────────────────────────────────────────────────────┤
│                                                      │
│  ⚙ Settings                                          │
│  ┌─ Target RAG ─┐ ┌─ Judge LLM ──┐ ┌─ API Keys ────┐│
│  │ provider +   │ │ judge model + │ │ OpenRouter +  ││
│  │ model pickers│ │ temperature   │ │ OpenAI + test ││
│  └──────────────┘ └───────────────┘ └───────────────┘│
│                                                      │
│  ▶ Run                                               │
│  tier • workers • limit • mock/ragas/bertscore/sbert │
│  [▶ Start]  →  live SSE console                      │
│                                                      │
│  📊 Dashboard                                        │
│  [run selector]                                      │
│  ✓/✗ gate banner                                     │
│  per-tier:  KPI cards  •  F1 bars  •  metric radar   │
│                                                      │
│  🗂 History                                          │
│  table of past runs  •  open / delete                │
└──────────────────────────────────────────────────────┘
```

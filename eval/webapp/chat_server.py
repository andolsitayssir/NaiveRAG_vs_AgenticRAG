"""FastAPI control panel for the Activiity RAG evaluation program.

Run:
    uvicorn eval.webapp.chat_server:app --reload --port 8787

Then open http://localhost:8787
"""
from __future__ import annotations
import asyncio, json, os, sys, time, uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import (
    StreamingResponse, FileResponse, JSONResponse, HTMLResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import httpx

from eval.config import TIERS, THRESHOLDS, TEST_DIR
from eval.parser import load_tier

# Per-config service cache to avoid reinit on repeated requests
# Keyed by (key, model, reranker, synth_model) tuple → service instance
_NAIVE_SVC: dict[tuple, Any] = {}
_AGENTIC_SVC: dict[tuple, Any] = {}

_WARM_EMB = None  # Embeddings don't need API key (local or HF)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
RUNS_DIR = ROOT / "runs"
CONFIG_FILE = ROOT / "config.json"
RUNS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Activiity RAG Eval — Control Panel")

# Robust static mount
print(f"DEBUG: Static directory: {STATIC_DIR}")
if STATIC_DIR.exists():
    print("DEBUG: Mounting /static")
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
else:
    print(f"WARNING: Static directory not found at {STATIC_DIR}")


DEFAULT_CONFIG: dict[str, Any] = {
    # Target program (the Agentic RAG being evaluated)
    "target_llm_provider": "openrouter",     # openrouter | ollama
    "target_openrouter_model": "qwen/qwen3-14b",
    "target_ollama_base_url": "http://localhost:11434",
    "target_ollama_model": "qwen2.5:14b-instruct",
    "target_embed_provider": "hf",       # openai | ollama | hf
    "target_embed_model": "ibm-granite/granite-embedding-311m-multilingual-r2",

    # Judge (for Ragas)
    "judge_model": "openai/gpt-4o-mini",
    "judge_temperature": 0.0,

    # Secrets — stored masked in API responses
    "openrouter_api_key": "",
    "openai_api_key": "",

    # Budget cap for a single run (USD); 0 = disabled
    "budget_usd": 0.0,
}

SECRET_KEYS = ("openrouter_api_key", "openai_api_key")
SECRETS: dict[str, str] = {
    "openrouter_api_key": os.environ.get("OPENROUTER_API_KEY", ""),
    "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
}


def load_config() -> dict:
    cfg = DEFAULT_CONFIG.copy()
    if CONFIG_FILE.exists():
        cfg.update(json.loads(CONFIG_FILE.read_text(encoding="utf-8")))
    for k in SECRET_KEYS:
        if k == "openrouter_api_key":
            cfg[k] = SECRETS.get(k, "")
            continue
        env_val = os.environ.get("OPENAI_API_KEY", "")
        cfg[k] = env_val or SECRETS.get(k, "")
    return cfg


def save_config(cfg: dict):
    safe_cfg = {k: v for k, v in cfg.items() if k not in SECRET_KEYS}
    CONFIG_FILE.write_text(json.dumps(safe_cfg, indent=2), encoding="utf-8")


def mask_secret(s: str) -> str:
    if not s:
        return ""
    if len(s) < 8:
        return "***"
    return f"{s[:4]}…{s[-4:]}"


def public_config(cfg: dict) -> dict:
    return {**cfg,
            "openrouter_api_key": mask_secret(cfg.get("openrouter_api_key", "")),
            "openai_api_key": mask_secret(cfg.get("openai_api_key", ""))}


# ---------------------------------------------------------------------------
# Run registry — in-memory + on-disk
# ---------------------------------------------------------------------------

class RunMeta(BaseModel):
    id: str
    status: str                   # pending | running | done | failed | cancelled
    tier: str                     # reference | medium | advanced | all
    ragas: bool
    mock: bool
    started_at: str
    finished_at: str | None = None
    progress: dict = Field(default_factory=dict)   # tier -> {done, total}
    gate: str | None = None       # PASS | FAIL
    error: str | None = None


RUNS: dict[str, RunMeta] = {}
RUN_LOGS: dict[str, list[str]] = {}   # run_id -> lines
RUN_QUEUES: dict[str, asyncio.Queue] = {}
RUN_PROCS: dict[str, Any] = {}        # run_id -> asyncio.subprocess.Process (for cancel)


def run_dir(run_id: str) -> Path:
    return RUNS_DIR / run_id


def _load_persisted_runs():
    for d in RUNS_DIR.iterdir():
        if not d.is_dir():
            continue
        meta = d / "meta.json"
        if meta.exists():
            try:
                m = RunMeta.model_validate(json.loads(meta.read_text(encoding="utf-8")))
                RUNS[m.id] = m
            except Exception:
                pass


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Activiity RAG Eval — Control Panel")

# Robust static mount
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
async def _on_startup():
    _load_persisted_runs()


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


def _chat_page() -> HTMLResponse:
    from fastapi.responses import Response
    content = (STATIC_DIR / "chat_unified.html").read_text(encoding="utf-8")
    return Response(
        content=content,
        media_type="text/html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/chat", response_class=HTMLResponse)
def chat_unified():
    return _chat_page()


@app.get("/chat/naive", response_class=HTMLResponse)
def chat_naive():
    return _chat_page()


@app.get("/chat/agentic", response_class=HTMLResponse)
def chat_agentic():
    return _chat_page()


@app.get("/styles.css", include_in_schema=False)
def get_styles_fallback():
    from fastapi.responses import Response
    content = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")
    # Force a unique ETag or cache header to break browser caching
    return Response(
        content=content, 
        media_type="text/css", 
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )


# ---------------------------- Config ---------------------------------------

@app.get("/api/config")
def get_config():
    return public_config(load_config())


class ConfigUpdate(BaseModel):
    target_llm_provider: str | None = None
    target_openrouter_model: str | None = None
    target_ollama_base_url: str | None = None
    target_ollama_model: str | None = None
    target_embed_provider: str | None = None
    target_embed_model: str | None = None
    judge_model: str | None = None
    judge_temperature: float | None = None
    openrouter_api_key: str | None = None       # pass the raw key to update
    openai_api_key: str | None = None
    budget_usd: float | None = None


class OpenRouterKeyRequest(BaseModel):
    openrouter_api_key: str | None = None


@app.put("/api/config")
def put_config(update: ConfigUpdate):
    cfg = load_config()
    data = update.model_dump(exclude_unset=True)
    # refuse to overwrite a secret with the masked string from GET
    for k in ("openrouter_api_key", "openai_api_key"):
        if k in data and "…" in (data[k] or ""):
            data.pop(k)
    for k in SECRET_KEYS:
        if k in data:
            SECRETS[k] = data.pop(k) or ""
    cfg.update(data)
    save_config(cfg)
    return public_config(cfg)


@app.post("/api/config/test-openrouter")
async def test_openrouter(payload: OpenRouterKeyRequest | None = None):
    cfg = load_config()
    key = (payload.openrouter_api_key if payload else None) or cfg.get("openrouter_api_key", "")
    if not key:
        raise HTTPException(400, "No OpenRouter key set")
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get("https://openrouter.ai/api/v1/models",
                        headers={"Authorization": f"Bearer {key}"})
    if r.status_code == 200:
        models = [m["id"] for m in r.json().get("data", [])][:50]
        return {"ok": True, "status": r.status_code, "n_models": len(models),
                "sample_models": models[:20]}
    return {"ok": False, "status": r.status_code, "body": r.text[:500]}


@app.post("/api/config/test-ollama")
async def test_ollama():
    cfg = load_config()
    url = cfg.get("target_ollama_base_url") or "http://localhost:11434"
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{url}/api/tags")
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            return {"ok": True, "n_models": len(models), "models": models}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}
    return {"ok": False, "status": r.status_code}


# ---------------------------- Datasets -------------------------------------

@app.get("/api/datasets")
def get_datasets():
    out = {}
    for t in TIERS:
        try:
            pairs = load_tier(t)
            by_bucket: dict[str, int] = {}
            for p in pairs:
                # Exclusion de l'UA-2 pour le filtrage UI (demande utilisateur)
                if p.bucket == "UA-2":
                    continue
                by_bucket[p.bucket] = by_bucket.get(p.bucket, 0) + 1
            out[t] = {"total": sum(by_bucket.values()), "by_bucket": by_bucket}
        except Exception as e:  # noqa: BLE001
            out[t] = {"error": str(e)}
    return out


@app.get("/api/thresholds")
def get_thresholds():
    return {t: th.__dict__ for t, th in THRESHOLDS.items()}


# ---------------------------- Chat / Demo -----------------------------------

def _compute_single_ragas(question: str, answer: str, contexts: list[str],
                          ground_truth: str | None, api_key: str) -> dict[str, float | None]:
    """Run the 5 RAGAS metrics on a single Q/A sample. Returns a dict of scores.
    If anything fails, return an empty dict so the caller can still respond.
    
    api_key can be user-provided or the server's configured key."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            Faithfulness, AnswerRelevancy,
            ContextPrecision, ContextRecall, AnswerCorrectness,
        )
        from ragas.run_config import RunConfig
        from eval.metrics.ragas_bridge import _build_judge, _build_judge_embeddings

        ds = Dataset.from_list([{
            "question": question,
            "answer": answer,
            "contexts": contexts or [""],
            "ground_truth": ground_truth or "",
        }])

        judge = _build_judge(api_key)
        emb = _WARM_EMB or _build_judge_embeddings()

        m_faith = Faithfulness(llm=judge)
        try:
            m_rel = AnswerRelevancy(llm=judge, embeddings=emb, strictness=1)
        except TypeError:
            m_rel = AnswerRelevancy(llm=judge, embeddings=emb)
        m_prec = None
        m_rec = None
        m_corr = None
        if ground_truth:
            m_prec = ContextPrecision(llm=judge)
            m_rec = ContextRecall(llm=judge)
            try:
                m_corr = AnswerCorrectness(llm=judge, embeddings=emb, strictness=1)
            except TypeError:
                m_corr = AnswerCorrectness(llm=judge, embeddings=emb)

        metrics = [m_faith, m_rel]
        if m_prec:
            metrics.append(m_prec)
        if m_rec:
            metrics.append(m_rec)
        if m_corr:
            metrics.append(m_corr)

        for m in metrics:
            if hasattr(m, "n"):
                m.n = 1

        # Use multiple workers for the 5 metrics even for a single sample
        results = evaluate(
            ds,
            metrics=metrics,
            llm=judge, embeddings=emb,
            run_config=RunConfig(max_workers=4, timeout=120),
        )
        df = results.to_pandas()
        row = df.iloc[0].to_dict()

        def _val(col):
            v = row.get(col)
            if v is None:
                return None
            try:
                f = float(v)
                return None if f != f else f  # NaN check
            except Exception:
                return None

        return {
            "faithfulness": _val("faithfulness"),
            "answer_relevancy": _val("answer_relevancy"),
            "context_precision": _val("context_precision"),
            "context_recall": _val("context_recall") if ground_truth else None,
            "answer_correctness": _val("answer_correctness") if ground_truth else None,
        }
    except Exception as e:
        print(f"[chat] RAGAS scoring failed: {e}", flush=True)
        return {}


@app.get("/api/chat/pairs")
def get_chat_pairs():
    """Get all question-answer pairs for chat demo (all tiers)."""
    from eval.parser import load_tier
    from activiity.prompts.defaults_fr import UA_TITLES
    out = []
    for tier in TIERS:
        try:
            pairs = load_tier(tier)
            for p in pairs:
                # Exclusion de l'UA-2 pour le filtrage UI (demande utilisateur)
                if p.bucket == "UA-2":
                    continue

                out.append({
                    "id": p.id,
                    "question": p.question,
                    "ground_truth": p.ground_truth,
                    "bucket": p.bucket,
                    "unit_name": UA_TITLES.get(p.bucket, "Global"),
                    "tier": p.tier,
                })
        except Exception as e:
            pass
    return out


class ChatRequest(BaseModel):
    question: str
    target: str = "agentic"  # naive or agentic
    model: str = "qwen/qwen3-14b"
    synth_model: str | None = None
    use_reranker: bool = False
    compute_ragas: bool = False          # if True, run single-sample RAGAS inline
    ground_truth: str | None = None      # needed for RAGAS scoring
    openrouter_api_key: str | None = None


class ChatMultiRequest(BaseModel):
    question: str
    target: str = "agentic"
    model: str = "qwen/qwen3-14b"         # router model
    synth_models: list[str] = Field(default_factory=list)
    use_reranker: bool = True
    openrouter_api_key: str | None = None


def _extract_chunks(result) -> list[str]:
    """Extract text chunks from retrieved nodes (works for both Naive & Agentic)."""
    chunks = []
    for n in getattr(result, "retrieved_nodes", []):
        txt = getattr(n, "text", None)
        if isinstance(txt, str):
            chunks.append(txt)
            continue
        node = getattr(n, "node", None)
        if node is not None:
            txt2 = getattr(node, "text", None)
            if isinstance(txt2, str):
                chunks.append(txt2)
    return chunks


@app.post("/api/chat/ask")
async def chat_ask(req: ChatRequest):
    """Ask a question via Naive or Agentic RAG.
    Always returns the answer + contexts immediately.
    RAGAS is NOT computed here — use /api/chat/ragas for that.

    If the user provides an OpenRouter API key it is used;
    otherwise the server's configured key is used as fallback."""
    global _NAIVE_SVC, _AGENTIC_SVC
    cfg = load_config()
    actual_model = req.model if req.model else "qwen/qwen3-14b"

    user_key = (req.openrouter_api_key or "").strip()
    effective_key = user_key or cfg.get("openrouter_api_key", "") or ""

    try:
        if req.target == "naive":
            from baseline.config import CFG as BaselineCFG
            object.__setattr__(BaselineCFG, "openrouter_model", actual_model)
            object.__setattr__(BaselineCFG, "openrouter_api_key", effective_key)

            from baseline.service import NaiveRagService
            cfg_key = (effective_key, actual_model)
            svc = _NAIVE_SVC.get(cfg_key)
            if svc is None:
                _NAIVE_SVC[cfg_key] = svc = NaiveRagService()

            result = await svc.achat_with_trace(req.question)
            chunks = _extract_chunks(result)

            return {
                "answer": result.answer,
                "model": result.model_name,
                "contexts": chunks,
                "retrieved_nodes": [
                    {"text": getattr(n, "text", "")[:200], "score": getattr(n, "score", None)}
                    for n in result.retrieved_nodes[:3]
                ],
            }

        else:
            from activiity.config import CFG as AgenticCFG
            object.__setattr__(AgenticCFG, "openrouter_model", actual_model)
            object.__setattr__(AgenticCFG, "openrouter_api_key", effective_key)
            object.__setattr__(AgenticCFG, "use_reranker", req.use_reranker)
            if req.synth_model:
                object.__setattr__(AgenticCFG, "use_synth_custom", True)
                object.__setattr__(AgenticCFG, "synth_model", req.synth_model)
            else:
                object.__setattr__(AgenticCFG, "use_synth_custom", False)

            from activiity.rag.service import AgenticRagService
            cfg_key = (effective_key, actual_model, req.use_reranker, req.synth_model)
            svc = _AGENTIC_SVC.get(cfg_key)
            if svc is None:
                _AGENTIC_SVC[cfg_key] = svc = AgenticRagService()

            result = await svc.achat_with_trace(req.question)
            chunks = _extract_chunks(result)

            return {
                "answer": result.answer,
                "model": result.model_name,
                "contexts": chunks,
                "tools_called": result.tools_called,
                "units_called": result.units_called,
                "iterations": result.iterations,
            }

    except Exception as e:
        return {"error": str(e)}


@app.post("/api/chat/ask_multi")
async def chat_ask_multi(req: ChatMultiRequest):
    """Agentic-only: run routing + retrieval once, then synthesize per model."""
    if req.target != "agentic":
        raise HTTPException(status_code=400, detail="ask_multi supports agentic only")

    global _AGENTIC_SVC
    cfg = load_config()
    actual_model = req.model if req.model else "qwen/qwen3-14b"

    user_key = (req.openrouter_api_key or "").strip()
    effective_key = user_key or cfg.get("openrouter_api_key", "") or ""

    try:
        from activiity.config import CFG as AgenticCFG
        object.__setattr__(AgenticCFG, "openrouter_model", actual_model)
        object.__setattr__(AgenticCFG, "openrouter_api_key", effective_key)
        object.__setattr__(AgenticCFG, "use_reranker", req.use_reranker)
        object.__setattr__(AgenticCFG, "use_synth_custom", False)

        from activiity.rag.service import AgenticRagService
        cfg_key = (effective_key, actual_model, req.use_reranker, "multi")
        svc = _AGENTIC_SVC.get(cfg_key)
        if svc is None:
            _AGENTIC_SVC[cfg_key] = svc = AgenticRagService()

        result = await svc.achat_with_trace(req.question)
        chunks = _extract_chunks(result)

        async def _one(model: str) -> dict:
            try:
                answer = await svc.synthesize_with_nodes(
                    req.question,
                    result.retrieved_nodes,
                    model,
                )
                return {
                    "model": model,
                    "answer": answer,
                    "contexts": chunks,
                }
            except Exception as e:  # noqa: BLE001
                return {
                    "model": model,
                    "error": f"{type(e).__name__}: {e}",
                    "contexts": chunks,
                }

        items = await asyncio.gather(*[_one(m) for m in req.synth_models])

        return {
            "items": items,
            "tools_called": result.tools_called,
            "units_called": result.units_called,
            "iterations": result.iterations,
        }
    except Exception as e:
        return {"error": str(e)}


class RagasRequest(BaseModel):
    question: str
    answer: str
    contexts: list[str] = []
    ground_truth: str | None = None
    openrouter_api_key: str | None = None


@app.post("/api/chat/ragas")
async def chat_ragas(req: RagasRequest):
    """Compute RAGAS metrics on a previously-obtained answer.
    This is called separately so the UI can show the answer first.

    Uses the user-provided key, or falls back to the server's configured key."""
    cfg = load_config()
    user_key = (req.openrouter_api_key or "").strip()
    effective_key = user_key or cfg.get("openrouter_api_key", "") or ""

    try:
        ragas = await asyncio.to_thread(
            _compute_single_ragas,
            req.question, req.answer, req.contexts, req.ground_truth,
            effective_key,
        )
        return {"ragas": ragas}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------- Runs -----------------------------------------

class RunRequest(BaseModel):
    tier: str = "reference"          # reference | medium | advanced | all
    target: str = "naive"            # naive | agentic | (mock override below)
    ragas: bool = False
    mock: bool = True                # default to mock so the panel is usable w/o providers
    no_bertscore: bool = True
    no_sbert: bool = True
    limit: int | None = None
    workers: int = 6


@app.get("/api/runs")
def list_runs():
    return [m.model_dump() for m in
            sorted(RUNS.values(), key=lambda m: m.started_at, reverse=True)]


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    m = RUNS.get(run_id)
    if not m:
        raise HTTPException(404, "run not found")
    rep = run_dir(run_id) / "report.json"
    data = {"meta": m.model_dump()}
    if rep.exists():
        data["report"] = json.loads(rep.read_text(encoding="utf-8"))
    return data


@app.get("/api/runs/{run_id}/traces.csv")
def get_run_csv(run_id: str):
    f = run_dir(run_id) / "traces.csv"
    if not f.exists():
        raise HTTPException(404, "traces not found")
    return FileResponse(f, media_type="text/csv", filename=f"traces_{run_id}.csv")


@app.get("/api/runs/{run_id}/report.html")
def get_run_html(run_id: str):
    f = run_dir(run_id) / "report.html"
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(f, media_type="text/html")


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: str):
    RUNS.pop(run_id, None)
    RUN_LOGS.pop(run_id, None)
    RUN_QUEUES.pop(run_id, None)
    d = run_dir(run_id)
    if d.exists():
        import shutil; shutil.rmtree(d, ignore_errors=True)
    return {"ok": True}


@app.post("/api/runs")
async def start_run(req: RunRequest, background: BackgroundTasks):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    m = RunMeta(
        id=run_id, status="pending",
        tier=req.tier, ragas=req.ragas, mock=req.mock,
        started_at=datetime.now().isoformat(timespec="seconds"),
    )
    RUNS[run_id] = m
    RUN_LOGS[run_id] = []
    RUN_QUEUES[run_id] = asyncio.Queue()
    d = run_dir(run_id); d.mkdir(exist_ok=True)
    _persist_meta(m)

    background.add_task(_execute_run, run_id, req)
    return m.model_dump()


@app.get("/api/runs/{run_id}/stream")
async def stream_run(run_id: str):
    """SSE stream of log lines + status transitions."""
    if run_id not in RUNS:
        raise HTTPException(404)
    q = RUN_QUEUES.setdefault(run_id, asyncio.Queue())

    async def gen():
        # replay any buffered lines first
        for line in RUN_LOGS.get(run_id, []):
            yield f"data: {json.dumps({'line': line})}\n\n"
        while True:
            try:
                item = await asyncio.wait_for(q.get(), timeout=30)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                if RUNS[run_id].status in ("done", "failed", "cancelled"):
                    break
                continue
            if item is None:
                yield f"event: done\ndata: {json.dumps(RUNS[run_id].model_dump())}\n\n"
                break
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


# ---------------------------- Run execution -------------------------------

def _persist_meta(m: RunMeta):
    (run_dir(m.id) / "meta.json").write_text(
        json.dumps(m.model_dump(), indent=2), encoding="utf-8",
    )


async def _push(run_id: str, **kw):
    q = RUN_QUEUES.get(run_id)
    if q:
        await q.put(kw)
    if "line" in kw:
        RUN_LOGS.setdefault(run_id, []).append(kw["line"])


async def _execute_run(run_id: str, req: RunRequest):
    m = RUNS[run_id]
    m.status = "running"; _persist_meta(m)

    # propagate current config to env so the eval reads it
    cfg = load_config()
    env = os.environ.copy()
    if cfg.get("openrouter_api_key"):
        env["OPENROUTER_API_KEY"] = cfg["openrouter_api_key"]
    if cfg.get("openai_api_key"):
        env["OPENAI_API_KEY"] = cfg["openai_api_key"]
    if cfg.get("judge_model"):
        env["ACTIVIITY_JUDGE_MODEL"] = cfg["judge_model"]
    env["ACTIVIITY_JUDGE_TEMPERATURE"] = str(cfg.get("judge_temperature", 0.0))
    env["ACTIVIITY_EVAL_BUDGET_USD"] = str(cfg.get("budget_usd", 0.0))
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"   # CRITICAL: makes print() flush line-by-line so
                                    # the SSE stream shows real-time progress instead
                                    # of waiting for the 8 KB pipe buffer to fill.

    cmd = [
        sys.executable, "-u",       # same idea: `-u` forces unbuffered stdout/stderr
        "-m", "eval.run_eval",
        "--tier", req.tier,
        "--workers", str(req.workers),
        "--out-dir", str(run_dir(run_id)),
    ]
    if req.ragas: cmd.append("--ragas")
    if req.mock: cmd.append("--mock")
    else: cmd += ["--target", req.target]
    if req.no_bertscore: cmd.append("--no-bertscore")
    if req.no_sbert: cmd.append("--no-sbert")
    if req.limit: cmd += ["--limit", str(req.limit)]

    # find project root (two levels up from this file: eval/webapp/server.py)
    cwd = Path(__file__).resolve().parents[2]
    await _push(run_id, line=f"$ {' '.join(cmd)}")

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=str(cwd), env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        RUN_PROCS[run_id] = proc
        async for raw in proc.stdout:  # type: ignore[union-attr]
            line = raw.decode("utf-8", errors="replace").rstrip()
            await _push(run_id, line=line)
            # parse tier progress from the runner's output
            if line.startswith(">>> Running tier="):
                import re as _re
                mt = _re.search(r"tier=(\S+)\s+n=(\d+)", line)
                if mt:
                    m.progress[mt.group(1)] = {"done": 0, "total": int(mt.group(2))}
        code = await proc.wait()
        m.finished_at = datetime.now().isoformat(timespec="seconds")

        rep_path = run_dir(run_id) / "report.json"
        if code == 0:
            m.status = "done"
            m.gate = "PASS"
        elif code == 1:
            m.status = "done"
            m.gate = "FAIL"
        else:
            m.status = "failed"
            m.error = f"exit code {code}"
        if rep_path.exists():
            await _push(run_id, line=f"report at {rep_path}")
    except Exception as e:  # noqa: BLE001
        m.status = "failed"
        m.error = f"{type(e).__name__}: {e}"
        m.finished_at = datetime.now().isoformat(timespec="seconds")
        await _push(run_id, line=f"FATAL: {m.error}")
    finally:
        RUN_PROCS.pop(run_id, None)
        _persist_meta(m)
        await _push(run_id, line=f"[{m.status}] gate={m.gate}")
        q = RUN_QUEUES.get(run_id)
        if q:
            await q.put(None)   # terminate SSE


@app.post("/api/runs/{run_id}/cancel")
async def cancel_run(run_id: str):
    """Kill a running or stuck eval subprocess."""
    m = RUNS.get(run_id)
    if not m:
        raise HTTPException(404)
    proc = RUN_PROCS.get(run_id)
    if proc and proc.returncode is None:
        try:
            proc.kill()
            await _push(run_id, line="[cancel] subprocess killed by user")
        except Exception as e:   # noqa: BLE001
            await _push(run_id, line=f"[cancel] failed: {e}")
    m.status = "cancelled"
    m.finished_at = datetime.now().isoformat(timespec="seconds")
    _persist_meta(m)
    return m.model_dump()


# ---------------------------- Health ---------------------------------------

@app.get("/api/health")
def health():
    return {"ok": True, "tiers": list(TIERS), "n_runs": len(RUNS)}


@app.get("/api/preflight")
def preflight():
    """Check all prerequisites for a real (non-mock) run: provider packages,
    API keys, data dir. Returns a list of issues the UI can display."""
    issues: list[str] = []
    cfg = load_config()

    # 1. Provider Python packages
    checks = [
        ("llama_index.llms.openrouter", "pip install llama-index-llms-openrouter",
         cfg.get("target_llm_provider") == "openrouter"),
        ("llama_index.llms.ollama", "pip install llama-index-llms-ollama",
         cfg.get("target_llm_provider") == "ollama"),
        ("llama_index.embeddings.openai", "pip install llama-index-embeddings-openai",
         cfg.get("target_embed_provider") == "openai"),
        ("llama_index.embeddings.ollama", "pip install llama-index-embeddings-ollama",
         cfg.get("target_embed_provider") == "ollama"),
        ("huggingface_hub", "pip install huggingface-hub",
         cfg.get("target_embed_provider") in ("hf", "hf-inference")),
    ]
    import importlib
    for mod, hint, needed in checks:
        if not needed:
            continue
        try:
            importlib.import_module(mod)
        except ImportError:
            issues.append(f"Missing package `{mod}` — install with: {hint}")

    # 2. API keys for cloud providers
    if cfg.get("target_llm_provider") == "openrouter" and not cfg.get("openrouter_api_key"):
        issues.append("OpenRouter API key is not set (Settings → API Keys).")
    if cfg.get("target_embed_provider") == "openai" and not cfg.get("openai_api_key"):
        issues.append("OpenAI API key (for embeddings) is not set (Settings → API Keys).")

    # 3. Data directory must exist
    from eval.config import TEST_DIR as _T
    data_dir = Path(__file__).resolve().parents[2] / "data"
    if not data_dir.exists():
        issues.append(f"Corpus directory missing: {data_dir}")

    return {"ok": len(issues) == 0, "issues": issues}
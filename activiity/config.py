"""Env-driven config for the Agentic RAG (matches naive baseline knobs)."""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("ACTIVIITY_DATA_DIR", str(ROOT / "data")))


@dataclass(frozen=True)
class Cfg:
    # LLM (used by the ReAct agent and synthesis)
    llm_provider: str = os.getenv("LLM_PROVIDER", "openrouter")
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_model: str = os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4-6")
    ollama_base_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_llm_model: str = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:14b-instruct")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    # The router only emits tool calls (design C) — a hard token cap makes
    # every round-trip faster without hurting routing.
    router_max_tokens: int = int(os.getenv("ROUTER_MAX_TOKENS", "800"))
    # Real context window of the configured models. llama-index's OpenRouter
    # wrapper hard-codes 3900 tokens, which (a) truncates the router's system
    # prompt + tool results (hurting instruction-following) and (b) makes the
    # compact synthesizer repack the retrieved nodes into several tiny chunks,
    # forcing the REFINE path instead of one QA_FR pass.
    context_window: int = int(os.getenv("CONTEXT_WINDOW", "128000"))

    # Embeddings
    embed_provider: str = os.getenv("EMBED_PROVIDER", "openai")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_embed_model: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    ollama_embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3:latest")
    st_embed_model: str = os.getenv(
        "ST_EMBED_MODEL",
        "ibm-granite/granite-embedding-311m-multilingual-r2",
    )
    hf_token: str = os.getenv("HF_TOKEN", "")

    # Qdrant
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_collection_base: str = os.getenv("QDRANT_COLLECTION", "activiity_kb")

    # Retrieval / agent
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "64"))
    top_k_per_ua: int = int(os.getenv("TOP_K_PER_UA", "8"))
    top_k_global: int = int(os.getenv("TOP_K_GLOBAL", "8"))
    # The router drives the ReAct loop. In v6 the escalation is AUTOMATIC
    # (inside the service, zero extra LLM round-trips): query_by_ua retries
    # internally on empty retrieval then falls back to query_global. The
    # router only needs: observe -> query_by_ua -> "Answer:".
    # early_stopping_method="generate" turns a cap trip into a clean stop.
    max_iterations: int = int(os.getenv("AGENT_MAX_ITER", "4"))

    # --- Automatic empty-retrieval escalation (v6) ---------------------
    # verify_context tool removed. query_by_ua re-queries up to RETRY_EMPTY
    # times when it returns zero nodes (handles transient embed/network
    # failures), then falls back to query_global — all inside the service.
    retry_empty: int = int(os.getenv("RETRY_EMPTY", "2"))

    # --- Post-run verdict (informational only) -------------------------
    # verify_min_nodes / verify_min_chars are used by nodes_verdict() purely
    # for the [agentic] debug log after the run; they no longer gate routing.
    verify_min_nodes: int = int(os.getenv("VERIFY_MIN_NODES", "1"))
    verify_min_chars: int = int(os.getenv("VERIFY_MIN_CHARS", "50"))

    # --- Reranker (improvement #4) — optional, controlled by env var ----
    # Per-unit quotas: RERANK_TOP_K candidates are retrieved per UA, the
    # reranker scores all of them in one batch and RERANK_KEEP are kept per
    # UA (never a global top_n, so multi-UA retrievals keep full coverage).
    use_reranker: bool = os.getenv("USE_RERANKER", "1") == "1"
    reranker_provider: str = "openrouter"
    reranker_model: str = str(os.getenv("RERANKER_MODEL"))
    rerank_top_k: int = int(os.getenv("RERANK_TOP_K", "10"))   # candidates / UA before rerank
    rerank_keep: int = int(os.getenv("RERANK_KEEP", "5"))      # kept / UA after rerank
    rerank_max_total: int = int(os.getenv("RERANK_MAX_TOTAL", "100"))  # safety cap on total candidates

    # --- Two-LLM split (separate optimization) — Sonnet for routing, Haiku for synthesis --
    # When ON, the ReAct agent reasons with Sonnet 4.6 (good at tool choice)
    # but each query engine's synthesizer uses Haiku (fast + cheap).
    use_haiku_synth: bool = os.getenv("USE_HAIKU_SYNTH", "1") == "1"
    haiku_model: str = os.getenv("HAIKU_MODEL", "anthropic/claude-haiku-4-5")

    # --- Structure-aware splitter (improvement #5) ---------------------
    # When ON, ingestion uses StructureAwareSplitter to chunk on
    # Fiche Protocole markers (principles, À retenir, Astuces, ...)
    # rather than fixed-size sentence windows. Stored in a separate
    # Qdrant collection so v2 (sentence) and v3 (structure) coexist.
    use_structure_splitter: bool = os.getenv("USE_STRUCTURE_SPLITTER", "1") == "1"
    # Three-threshold block packing (characters):
    #   target — pack chunks toward this size
    #   min    — below this, merge with neighbouring content (never drop)
    #   max    — above this, split at list-item / sentence boundaries
    structure_target_chars: int = int(os.getenv("STRUCTURE_TARGET_CHARS", "1100"))
    structure_min_chars: int = int(os.getenv("STRUCTURE_MIN_CHARS", "300"))
    structure_max_chars: int = int(os.getenv("STRUCTURE_MAX_CHARS", "2000"))

    # --- Context compression (improvement #6) --------------------------
    # When ON, retrieved nodes are compressed to keep only sentences
    # most relevant to the query before synthesis. Reduces noise, improves
    # faithfulness. Controlled via env var.
    use_context_compression: bool = os.getenv("USE_CONTEXT_COMPRESSION", "0") == "0"

    # --- Logging --------------------------------------------------------
    # Gates the [agentic] debug prints (routing, nodes, fallback). Off by
    # default so eval runs show only progress; enable with ACTIVIITY_VERBOSE=1.
    verbose: bool = os.getenv("ACTIVIITY_VERBOSE", "0") == "1"


CFG = Cfg()
UA_IDS = [f"UA-{i}" for i in range(1, 11)]


def collection_name(embed_model: str) -> str:
    """Per-embed-model + per-splitter collection so vectors don't mix
    when we change either the embedding model OR the chunking strategy.
    """
    safe = embed_model.replace("/", "_")
    suffix = "__structure" if CFG.use_structure_splitter else ""
    return f"{CFG.qdrant_collection_base}__{safe}{suffix}"


def embed_model_name() -> str:
    if CFG.embed_provider == "openai":
        return CFG.openai_embed_model
    if CFG.embed_provider == "ollama":
        return CFG.ollama_embed_model
    return CFG.st_embed_model

# Naive RAG baseline

A deliberately minimal baseline: **one flat index over all UA-1..UA-10**, top-k retrieval, single QA prompt, no router, no agent, no reranker.

Used to:
- smoke-test the evaluation program against a real LLM (not the mock),
- establish a lower bound that the Agentic RAG must beat.

## What's "naive" vs. the Agentic version

| Aspect | Naive baseline | Agentic RAG |
|---|---|---|
| Index | 1 flat store, all UAs mixed | 1 store with per-UA metadata filter, 10 per-UA query engines |
| Routing | None (post-hoc metadata vote on top-k) | ReAct agent picks per-UA tool or global |
| Retrieval | Dense top-k only | Dense + optional BM25 + optional reranker |
| Synthesis | Single `compact` call | Compact/Refine with configurable FR prompts |
| Prompts | Hardcoded French QA | Runtime-editable registry |
| Iterations | 1 | up to 4 (ReAct) |

## Layout

```
baseline/
├── config.py       # env-driven (LLM_PROVIDER, LLM_MODEL, TOP_K, etc.)
├── providers.py    # OpenRouter/Ollama LLM + OpenAI/Ollama embeddings
├── loader.py       # docx + pdf loader, same UA metadata as agentic
├── index.py        # build_or_load over data/, persists to .index/
├── service.py      # NaiveRagService.achat_with_trace (eval-runner shape)
└── cli.py          # `python -m baseline.cli ingest|ask`
```

## One-shot quickstart

```bash
# deps (only if not already installed)
pip install llama-index-core llama-index-llms-openrouter \
            llama-index-embeddings-openai llama-index-readers-file \
            python-docx pymupdf

# pick providers (defaults = cloud)
export OPENROUTER_API_KEY=sk-or-v1-...
export OPENAI_API_KEY=sk-...

# build the flat index once (≈30s on cloud)
python -m baseline.cli ingest

# ad-hoc Q&A
python -m baseline.cli ask "Quels sont les cinq principes d'une délégation efficace ?"
```

## Swap to on-prem in one line

```bash
LLM_PROVIDER=ollama EMBED_PROVIDER=ollama \
  OLLAMA_URL=http://localhost:11434 \
  OLLAMA_LLM_MODEL=qwen2.5:14b-instruct \
  OLLAMA_EMBED_MODEL=bge-m3 \
  python -m baseline.cli ask "..."
```

Changing the embedding model invalidates the index — delete `.index/` and re-ingest.

## Using with the eval program

Point the eval runner at the naive baseline by starting its subprocess with
`ACTIVIITY_TARGET=naive`. The runner imports `baseline.service.NaiveRagService`
via a small shim in `eval/targets.py`. Expected behaviour vs. the agentic:

- **Routing accuracy** lower (metadata vote ≠ real routing decisions)
- **Retrieval** slightly lower (no per-UA scoping = harder top-k)
- **Synthesis** broadly similar when the right chunks are retrieved
- **Latency** lower (1 LLM call vs. up to 4 agent iterations)
- **Cost** lower per call; but higher error rate may require re-asks

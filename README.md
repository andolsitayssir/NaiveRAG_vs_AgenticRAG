# Activiity RAG

Activiity RAG is a complete experimentation and evaluation workspace for
retrieval-augmented generation over a French management and coaching
knowledge base. The repository contains two independent RAG architectures:

1. **Simple baseline RAG** - a deliberately small, flat retrieval pipeline
   that provides a transparent reference implementation.
2. **Agentic ReAct RAG** - a routed architecture in which a ReAct agent
   selects targeted knowledge-base searches before a synthesis model produces
   the final response.

The repository also includes document ingestion, evaluation datasets and
metrics, a browser-based evaluation control panel, Docker services, and tests.
It is intended for studying the design trade-offs between straightforward
retrieval and tool-using agentic retrieval—not as a production-ready public
service.

## What the repository contains

### 1. Shared knowledge base

The source corpus is stored under [`data/`](C:/Users/lando/Documents/Activiitykdb/data/)
and consists of French PDF and Word documents organized into learning units
(`UA-1` through `UA-10`). The material covers subjects such as delegation,
motivation, communication, workplace habits, commercial skills, and
management.

The two RAG systems use the same source documents and the same general
metadata conventions so that their behaviour can be compared through the
same evaluation program.

### 2. Simple baseline RAG

The baseline implementation lives in [`baseline/`](C:/Users/lando/Documents/Activiitykdb/baseline/).
It intentionally keeps the retrieval path simple:

```text
Question
   │
   ▼
Flat vector index containing the full corpus
   │
   ▼
Top-k dense retrieval
   │
   ▼
Single French answer-synthesis prompt
   │
   ▼
Answer and trace
```

The baseline has no agent, no tool routing, and no reranker. It is useful as a
readable reference pipeline and as a controlled comparison point for the
agentic architecture.

Main components:

- [`baseline/loader.py`](C:/Users/lando/Documents/Activiitykdb/baseline/loader.py)
  loads PDF and Word documents.
- [`baseline/index.py`](C:/Users/lando/Documents/Activiitykdb/baseline/index.py)
  builds or loads the persistent flat vector index.
- [`baseline/service.py`](C:/Users/lando/Documents/Activiitykdb/baseline/service.py)
  performs retrieval and answer synthesis.
- [`baseline/config.py`](C:/Users/lando/Documents/Activiitykdb/baseline/config.py)
  defines environment-driven provider and retrieval settings.
- [`baseline/cli.py`](C:/Users/lando/Documents/Activiitykdb/baseline/cli.py)
  exposes ingestion and one-shot question-answering commands.

### 3. Agentic ReAct RAG

The agentic implementation is located in [`activiity/`](C:/Users/lando/Documents/Activiitykdb/activiity/).
It adds structured ingestion, metadata-aware retrieval, tool routing, optional
reranking, and trace collection:

```text
Question
   │
   ▼
ReAct router
   │
   ├── targeted UA retrieval tools
   └── global corpus retrieval
          │
          ▼
   Retrieved nodes and metadata
          │
          ▼
   Optional reranking and post-processing
          │
          ▼
   French synthesis prompt
          │
          ▼
   Answer, sources, tool calls, and trace
```

The router can select knowledge-base searches associated with individual
learning units or use a global search for cross-topic questions. Retrieved
nodes retain metadata such as the learning unit, source document, section
type, page, and chunk position.

Main components:

- [`activiity/ingest/pipeline.py`](C:/Users/lando/Documents/Activiitykdb/activiity/ingest/pipeline.py)
  builds the agentic vector index.
- [`activiity/ingest/structure_splitter.py`](C:/Users/lando/Documents/Activiitykdb/activiity/ingest/structure_splitter.py)
  splits documents using semantic and pedagogical sections rather than only
  fixed-size windows.
- [`activiity/rag/service.py`](C:/Users/lando/Documents/Activiitykdb/activiity/rag/service.py)
  implements retrieval, ReAct orchestration, synthesis, and tracing.
- [`activiity/providers/factory.py`](C:/Users/lando/Documents/Activiitykdb/activiity/providers/factory.py)
  creates the configured LLM, embedding, and retrieval providers.
- [`activiity/prompts/defaults_fr.py`](C:/Users/lando/Documents/Activiitykdb/activiity/prompts/defaults_fr.py)
  contains the French routing and answer-generation prompts.

## Ingestion and retrieval

The ingestion pipeline extracts text from PDF and Word files, associates each
document with its learning unit, and creates vector representations for
retrieval. The agentic pipeline additionally recognizes structures such as:

- principles and numbered lists,
- key takeaways,
- checklists,
- tips,
- examples,
- pitfalls,
- page boundaries and parent sections.

These structures are stored as chunk metadata and can be used during retrieval,
filtering, post-processing, and trace inspection.

The project supports several provider combinations through environment
configuration:

- OpenRouter or Ollama for language models,
- OpenAI, Ollama, or Hugging Face-compatible embeddings,
- Qdrant for the agentic vector store,
- a local persisted vector store for the baseline,
- optional reranking through a hosted or local backend.

## Evaluation framework

The evaluation code is in [`eval/`](C:/Users/lando/Documents/Activiitykdb/eval/).
It runs the same question sets against either target architecture through a
common service interface. The target can be selected without changing the
evaluation logic.

The framework includes:

- dataset parsing and tier selection,
- routing classification metrics,
- refusal detection metrics,
- retrieval precision, recall, MRR, and citation checks,
- answer similarity metrics,
- optional judge-based Ragas metrics,
- latency, token, cost, and iteration instrumentation,
- JSON, CSV, trace, and HTML report writers,
- release-gate checks configured in [`eval/config.py`](C:/Users/lando/Documents/Activiitykdb/eval/config.py).

Evaluation inputs are organized under [`test/`](C:/Users/lando/Documents/Activiitykdb/test/)
into reference, medium, and advanced tiers. The runner can execute a real RAG
target or a mock target for testing the evaluation machinery itself.

The main entry point is:

```bash
python -m eval.run_eval --target agentic --tier reference
```

Other commonly used options include:

```bash
python -m eval.run_eval --target naive --tier reference --limit 5
python -m eval.run_eval --target agentic --tier all --ragas
python -m eval.run_eval --target agentic --tier reference --mock
```

Generated indexes, run directories, caches, and reports are local artifacts
and are excluded from source publication by the repository ignore rules.

## Web evaluation control panel

[`eval/webapp/`](C:/Users/lando/Documents/Activiitykdb/eval/webapp/) contains a
FastAPI application and a browser interface for operating the evaluation
pipeline. It provides:

- provider and model configuration,
- dataset-tier and evaluation-option selection,
- live run output through Server-Sent Events,
- run history,
- report and trace access,
- deletion of local run artifacts.

The frontend is a static page with Alpine.js and Chart.js; it does not require
a separate frontend build step. The backend starts evaluation runs as
subprocesses and stores their local metadata and generated artifacts.

This panel is intended for local or trusted-network use. It is not configured
for unauthenticated internet exposure. Before deploying it publicly, add
authentication and authorization, protect Qdrant, use HTTPS, restrict
outbound upstream URLs, and protect reports and traces.

## Quick start

### Option A: Docker Compose

Create a local `.env` file from [`.env.example`](C:/Users/lando/Documents/Activiitykdb/.env.example)
and provide only the credentials required by the providers you select. Then
run:

```powershell
.\quickstart.ps1
```

The script starts the Qdrant and chat services, ingests the corpus into both
architectures, and makes the local panel available at:

- `http://localhost:8787/chat/naive`
- `http://localhost:8787/chat/agentic`

To start the services without rebuilding the indexes:

```powershell
.\quickstart.ps1 -SkipIngest
```

### Option B: Local Python

Install the dependencies:

```powershell
pip install -r requirements.txt
```

Build the agentic index:

```powershell
python -m activiity.ingest.cli --force
```

Build the baseline index:

```powershell
python -m baseline.cli ingest --force
```

Ask a question with either implementation:

```powershell
python -m baseline.cli ask "Quels sont les principes d'une délégation efficace ?"
```

Start the local evaluation panel:

```powershell
python -m uvicorn eval.webapp.chat_server:app --reload --port 8787
```

Provider selection, model names, retrieval limits, and optional features are
configured through environment variables. See the configuration modules and
the provider documentation before changing defaults.

## Repository layout

```text
.
├── activiity/                 Agentic ReAct RAG and structured ingestion
├── baseline/                  Simple flat-index RAG implementation
├── data/                      French source documents
├── eval/                      Evaluation runner, metrics, reports, and webapp
├── test/                      Evaluation question sets by tier
├── Dockerfile                 Container image definition
├── docker-compose.yml         Local Qdrant and webapp services
├── entrypoint.sh              Container startup command
├── quickstart.ps1             Windows setup and ingestion helper
├── quickstart.sh              Shell setup and ingestion helper
├── requirements.txt           Python dependencies
└── .env.example               Example environment configuration
```

Additional design and usage documentation is available in [`docs/`](C:/Users/lando/Documents/Activiitykdb/docs/)
and [`guide.md`](C:/Users/lando/Documents/Activiitykdb/guide.md).

## Configuration and secrets

Do not commit real API keys or credentials. Store them in a local `.env` file
or in the environment used to run the application. The repository ignores
`.env`, `.env.local`, runtime configuration, indexes, caches, reports, and
other generated artifacts.

Before publishing changes, inspect the exact files that will be staged:

```powershell
git status --short
git add -n .
git diff --cached --name-only
```

## License and source material

The repository contains a project-specific management and coaching corpus under
[`data/`](C:/Users/lando/Documents/Activiitykdb/data/). Confirm that you have
the right to redistribute those source documents before making the repository
public.

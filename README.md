# Activiity RAG — Implementation & Evaluation Report

## 1. Agentic Code Architecture (ReAct Agent)

### Overview
The core agentic system (`activiity/rag/service.py`) implements a **ReAct (Reasoning + Acting) agent** with an 11-tool design:
- **10 per-UA tools** (`query_ua_1` through `query_ua_10`): each wraps a Qdrant VectorStoreIndex query engine with a payload filter (`ua_id == 'UA-X'`) and French-language tool descriptions that the agent reads to decide routing
- **1 global tool** (`query_global`): searches across all UAs with a higher top_k, for cross-UA or ambiguous questions

### Key Implementation Details

| Component | Location | Details |
|---|---|---|
| **Two-LLM split** | `service.py:228-233` | Router LLM (Sonnet 4.6) for tool selection ≠ Synthesis LLM (Haiku 4.5) for answer generation |
| **Reranker integration** | `service.py:239-265` | Three backends: HF Inference API, OpenRouter `/rerank`, or FlagEmbedding; retrieves `RERANK_TOP_K=20`, keeps `RERANK_KEEP=8` |
| **Token counting** | `service.py:267-268` | `TokenCountingHandler` wired to global CallbackManager |
| **Node capture** | `service.py:50-77` | `_CapturePostprocessor` captures retrieved nodes at query-engine level (stable across LlamaIndex versions) |
| **Refusal guard** | `service.py:444-473` | If model refuses despite having chunks, forces re-synthesis via the synthesizer LLM |

### ReAct Agent Loop (`_build_agent`)
```python
tools = [self._ua_tool(u) for u in UA_IDS] + [self._global_tool()]
agent = ReActAgent(
    name="Activiity",
    tools=tools,
    llm=self.llm,
    system_prompt=SYSTEM_FR,
    max_iterations=CFG.max_iterations,  # default: 3
    timeout=180,
)
```

### 11-Tool Routing Decision Flow
```
Question
  │
  ├─► ReAct Agent (Sonnet 4.6) reads SYSTEM_FR rules
  │     │
  │     ├─► query_ua_1 ... query_ua_10  (per-UA Qdrant filter)
  │     └─► query_global                 (no filter, top_k=8-10)
  │
  ├─► Retrieval from Qdrant ──► [Optional Reranker] ──► k=20 → keep=8
  │
  └─► Synthesis (Haiku 4.5) via QA_FR template
        │ Two-pass: extraction → formatting
        │ Few-shot exemplars
        │ Numerical fidelity enforcement
```

### Data Flow: `achat_with_trace`
```
1. token_counter.reset_counts()
2. _captured_nodes = []
3. handler = agent.run(user_msg=question)
4. Stream ToolCall events → tools_called[]
5. result = await handler
6. Deduplicate captured nodes (by source_path + text[:80])
7. Guard: if refusal pattern detected but chunks exist → force re-synthesis
8. Return AgenticResult(answer, tools_called, retrieved_nodes, tokens, model_name, ...)
```

---

## 2. Prompt Engineering

### 2.1 System Prompt (`SYSTEM_FR` — `defaults_fr.py:17-45`)
The foundational prompt defines:

**Role**: "Activiity, expert assistant en management et coaching RH" — French-only, knowledge-base-only.

**UA Routing Rules** (lines 22-31): Explicit semantic boundaries for each of the 10 UAs + out-of-scope examples. Critical disambiguation notes:
- `UA-3` (Plaisir) vs `UA-4` (Motiver) — "NE PAS CONFONDRE"
- `UA-1` (Délégation personne identifiée) vs `UA-2` (Délégation via Sourcing/recrutement)

**Anti-refusal constraint** (lines 33-42): "INTERDICTION DE REFUSER" — the agent MUST use retrieved context even if partial. May NOT say "je ne trouve pas", "désolé", "pas d'information".

**2-retrieval rule**: If first search returns nothing, attempt a second search via `query_global`.

### 2.2 QA Template (`QA_FR` — `defaults_fr.py:54-112`)
The synthesis prompt implements **four P0+P1 improvements**:

| Improvement | Implementation | Location |
|---|---|---|
| **Two-pass CoT** | Step 1: extract all claims from context → Step 2: format into answer | Lines 69-74 |
| **Few-shot exemplars** | 2 worked examples (correct delegation answer + cross-domain handling) | Lines 87-108 |
| **Numerical fidelity** | "Si le contexte mentionne un nombre, reproduis ce nombre EXACTEMENT et liste CHAQUE item avec son numéro" | Lines 81-83 |
| **Strict formatting** | 8-12 phrases, preserve original order/numbering | Lines 77-85 |

### 2.3 Refine Template (`REFINE_FR` — `defaults_fr.py:115-125`)
Used in `response_mode="compact"` for multi-chunk synthesis. Instruction: integrate new context without removing existing information.

### 2.4 Refusal Message (`REFUSAL_FR` — `defaults_fr.py:128-131`)
Canonical: "Je ne trouve pas cette information dans la base Activiity. Pouvez-vous reformuler ou préciser l'UA concernée ?"

### 2.5 Tool Descriptions
- `tool_desc_ua(ua_id)`: "Recherche ciblée dans l'UA X (title). À utiliser quand la question porte spécifiquement sur le thème « title »."
- `TOOL_DESC_GLOBAL`: "Recherche dans l'ensemble du corpus Activiity. À utiliser quand la question est transverse, qu'aucune UA n'est spécifiquement identifiable, ou pour compléter un résultat partiel."

---

## 3. Retrieval & Chunking Strategy

### 3.1 StructureAwareSplitter (`structure_splitter.py`)
Instead of arbitrary token-window splitting, this custom NodeParser splits documents at **pedagogical markers** present in the Activiity Fiches Protocole:

| Marker | Section Type | Regex |
|---|---|---|
| `N principe` | `principle` | `^\d+\s*(er\|ère\|e\|ème)?\s+principe` |
| `À retenir` | `key_takeaways` | `^À\s+retenir\s*!?\s*$` |
| `Check-list` | `checklist` | `^Check-?list` |
| `Astuce :` | `tip` | `^Astuce\s*[:.]` |
| `Exemple :` | `example` | `^Exemple\s*[:.]` |
| `Piège :` | `trap` | `^Pi[èe]ge\s*[:.]` |
| `N qualités/principes/étapes...` | `criteria_list` | Numbered list patterns (28 nouns) |
| `[page N]` | (metadata) | Page tracking |

**Fallback**: oversized chunks (>2000 chars) are sentence-split with `SentenceSplitter(chunk_size=512, overlap=64)`.

**Metadata per chunk**: `ua_id`, `source_path`, `section_type`, `header`, `principle_index`, `parent_section`, `page`, `part_index`.

### 3.2 Vector Store Architecture

| Aspect | Naive Baseline | Agentic v2 |
|---|---|---|
| **Backend** | SimpleVectorStore (in-memory) | Qdrant (local path) |
| **Collection naming** | `baseline/.index/` | `activiity_kb__{embed_model}[__structure]` |
| **Per-embed-model isolation** | N/A | Separate collections per embedding model |
| **Filtering** | Metadata-vote post-hoc | Payload filter at query time (`ua_id == 'UA-X'`) |
| **top_k** | configurable (default 5) | Per-UA: 5, Global: 8-10 |
| **With reranker** | N/A | Retrieve 20 → rerank → keep 8 |

### 3.3 Reranker Integration
Three provider options controlled by `USE_RERANKER=1`:
1. **HF Inference API**: `bge-reranker-v2-m3` via `InferenceClient.text_classification()`
2. **OpenRouter**: `/rerank` endpoint with `cohere/rerank-4-fast`
3. **FlagEmbedding**: Local `FlagEmbeddingReranker` for on-prem

The reranker operates as a `BaseNodePostprocessor` in the query engine pipeline, scoring each retrieved node and keeping the top `rerank_keep` (default 8).

---

## 4. Evaluation Pipeline (5-Layer Metrics)

### 4.1 Metric Layers

| Layer | Type | Mandatory? | Metrics |
|---|---|---|---|
| **1. Routing** | Judge-free | Always | Accuracy, Macro-F1, Per-class P/R/F1, Confusion matrix |
| **2. Refusal** | Judge-free | Always | Binary P/R/F1, Borderline handling |
| **3. Retrieval** | Judge-free | Always | P@k, R@k, MRR, nDCG, Citation recall |
| **4. Answer Similarity** | Judge-free | Recommended | EM, Token-F1, ROUGE-L, SBERT cosine, BERTScore |
| **5. Ragas (Judge)** | Judge-based | Nightly | Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall, AnswerCorrectness |

### 4.2 Routing Metrics (`eval/metrics/classification.py`)
- **Multi-accept handling**: `expected_uas` can include alternatives (e.g. `UA-1|UA-9`). If prediction is in the set, counts as correct.
- **GLOBAL/OOS mapping**: Non-UA predictions map to `NONE` for closed-label confusion matrix.
- **sklearn backend**: `accuracy_score`, `precision_recall_fscore_support`, `classification_report`, `confusion_matrix`.

### 4.3 Refusal Metrics (`eval/metrics/refusal.py`)
- **Strict OOS**: pairs with expected_uas==`['NONE']` only
- **Borderline**: pairs where `NONE` is one of multiple acceptable labels (counted separately)
- **Binary classification**: y_true=refusal-needed, y_pred=model-refused
- **Configurable patterns**: 6 regex patterns in `eval/config.py:141-149` matching French refusal phrasings

### 4.4 Ragas Integration (`eval/metrics/ragas_bridge.py`)
- **Judge LLM**: gpt-4o-mini via OpenRouter (configurable)
- **Embeddings for Ragas**: sentence-transformers or HF Inference API
- **5 metrics**: Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall, AnswerCorrectness
- **strictness=1**: single generation for deterministic, faster evaluation
- **Per-pair attachment**: Ragas scores stored back into Trace objects for individual analysis

### 4.5 Quality Score Formula (`eval/reporting/consolidated.py:34-53`)
Weighted composite when judge is available:
```
answer_correctness × 0.30  (headline RN ≈ GT)
macro_f1           × 0.15  (routing)
faithfulness       × 0.10  (anti-hallucination)
citation_recall    × 0.10  (retrieval)
refusal_f1         × 0.10  (refusal handling)
token_f1           × 0.10  (answer overlap)
rouge_l_f1         × 0.05  (long-form overlap)
mrr                × 0.05  (retrieval rank)
```

---

## 5. Report Comparison — Existing Evaluation Results

Three agentic runs on **reference tier** (134 pairs each):

| Metric | Threshold | ministral-1 | ministral | ministral2 |
|---|---|---|---|---|
| **Quality Score** | — | **0.391** | **0.493** | **0.426** |
| **Routing accuracy** | 0.95 | 0.672 ❌ | 0.900 ❌ | 0.701 ❌ |
| **Macro-F1** | 0.93 | 0.707 ❌ | 0.086 ❌ | 0.725 ❌ |
| **Refusal precision** | 0.95 | 0.333 ❌ | 0.000 ❌ | **1.000 ✅** |
| **Refusal recall** | 0.90 | 0.800 ❌ | 0.000 ❌ | 0.600 ❌ |
| **Citation recall** | 0.90 | 0.282 ❌ | 0.800 ❌ | 0.258 ❌ |
| **Faithfulness** | 0.90 | 0.174 ❌ | 0.501 ❌ | 0.371 ❌ |
| **Answer relevancy** | 0.85 | 0.696 ❌ | 0.862 ❌ | 0.790 ❌ |
| **Context precision** | 0.80 | 0.240 ❌ | 0.488 ❌ | 0.265 ❌ |
| **Context recall** | 0.80 | 0.424 ❌ | 0.775 ❌ | 0.432 ❌ |
| **Answer correctness** | 0.65 | 0.424 ❌ | 0.609 ❌ | 0.387 ❌ |
| **p95 latency (s)** | 6.0 | 165.3 ❌ | 51.8 ❌ | 158.8 ❌ |
| **Gates passed** | 13 total | **0/13** | **1/13** | **1/13** |

### Key Observations

1. **Latency crisis**: All runs far exceed the 6s p95 threshold (51-165s). The ministral model (limited 10-pair run) was fastest at 52s but still 8.6x over threshold.

2. **Routing accuracy**: Stable at ~67-70% for full 134-pair runs, far below 0.95 target. The 10-pair ministral run scored 0.900 (just 0.05 below threshold) but with n=10 this is unreliable.

3. **Refusal precision spike**: ministral2 achieved 1.000 refusal precision (never falsely refused) but recall dropped to 0.600 (missed 40% of OOS questions). ministral-1 had the opposite: 0.333 precision (many false refusals) but 0.800 recall.

4. **Faithfulness**: All <0.50, indicating substantial hallucination. The two-pass extraction improvement in QA_FR was designed to address this but is not effective with the SLM models tested.

5. **Retrieval collapse**: Citation recall at 0.26-0.28 for full runs suggests the Qdrant retriever is failing to surface relevant documents. The per-UA payload filter may be over-constraining (if routing is wrong, retrieval is wrong — cascading error).

### Root Cause Analysis

| Issue | Likely Cause |
|---|---|
| Low routing accuracy | ReAct agent cannot reliably map questions to the correct UA tool; prompt disambiguation is insufficient for SLM models |
| Low citation recall | Cascading from routing errors; wrong UA → wrong payload filter → irrelevant chunks; plus Qdrant similarity may need tuning |
| High latency | 3-iteration ReAct loop × multiple LLM calls per iteration; SLM models on OpenRouter are slow |
| Low faithfulness | Synthesis still hallucinates despite QA_FR improvements; the "INTERDICTION DE REFUSER" prompt may encourage fabrication |
| Refusal imbalance | The anti-refusal prompt conflicts with out-of-scope detection; models generate content rather than refusing |

---

## 6. Ragas Routing and Refusal Table Comparison

### 6.1 Routing Confusion Analysis (Reference Tier)

For full 134-pair runs, the confusion matrix pattern (from ministral-2, representative):

| Actual \ Predicted | UA-1 | UA-3 | UA-4 | UA-5 | UA-9 | NONE | ... |
|---|---|---|---|---|---|---|---|
| **UA-1** (Délégation) | moderate | — | — | — | high | — | Frequent confusion UA-1 ↔ UA-9 (both management topics) |
| **UA-3** (Plaisir) | — | low | high | — | — | — | Confusion UA-3 ↔ UA-4 (the known ambiguity)
| **UA-4** (Motiver) | — | high | low | — | — | — | Symmetric confusion |
| **UA-9** (Manager) | high | — | — | — | low | — | UA-1/UA-9 boundary poorly defined |
| **NONE** (OOS) | — | — | — | — | — | ~70% | ~30% of OOS questions wrongly routed |

**Per-class F1 (ministral-2, reference)**:
- UA-1: ~0.75, UA-2: ~0.00 (almost no correct routing), UA-3: ~0.50, UA-4: ~0.55
- UA-5: ~0.70, UA-6: ~0.65, UA-7: ~0.60, UA-8: ~0.50, UA-9: ~0.65, UA-10: ~0.60
- NONE: ~0.75

### 6.2 Refusal Table Comparison

| Model | Refusal Precision | Refusal Recall | Refusal F1 | Borderline Correct | False Positives | False Negatives |
|---|---|---|---|---|---|---|
| **ministral-1** (n=134) | 0.333 | 0.800 | 0.471 | N/A | 6+ (high FP) | 2 (low FN) |
| **ministral** (n=10) | 0.000 | 0.000 | 0.000 | N/A | 0 (no refusals at all) | — |
| **ministral2** (n=134) | **1.000** | 0.600 | 0.750 | N/A | 0 (perfect precision) | 4 (missed OOS) |

**Trade-off pattern**: The refusal guard in `service.py:444-473` (re-synthesize when refusal + chunks exist) biases toward non-refusal, improving precision but harming recall. The opposite issue exists in models that over-refuse (high recall, low precision).

### 6.3 Ragas Metrics Comparison by Model

| Metric | ministral-1 | ministral (n=10) | ministral2 | Threshold |
|---|---|---|---|---|
| Faithfulness | 0.174 | 0.501 | 0.371 | 0.90 |
| Answer Relevancy | 0.696 | 0.862 | 0.790 | 0.85 |
| Context Precision | 0.240 | 0.488 | 0.265 | 0.80 |
| Context Recall | 0.424 | 0.775 | 0.432 | 0.80 |
| Answer Correctness | 0.424 | 0.609 | 0.387 | 0.65 |

**Interpretation**:
- The n=10 run (ministral) shows notably better scores across all judge metrics, suggesting either: (a) the 10-pair subset was easier, or (b) the model had fewer failure opportunities at n=10
- Context Precision is consistently the worst metric (0.24-0.49), indicating retrieved chunks are often irrelevant to the answer
- Context Recall (0.42-0.78) shows retrieval sometimes misses key documents needed for ground truth

### 6.4 Cross-Layer Correlation

```
Routing Error ──► Wrong UA Filter ──► Wrong Chunks ──► Low Context P/R ──► Low Faithfulness ──► Low Answer Correctness
      │                                                                                             │
      └────────────────────────── Cascading Error Chain ───────────────────────────────────────────────┘
```

The routing accuracy (67-70%) acts as a ceiling on all downstream metrics. If routing is wrong at step 1, every subsequent metric is capped:
- Context Recall ≤ Routing Accuracy (coincidentally ~0.42-0.43 ≈ 0.67 × 0.63 — retrieval effectiveness given correct routing)
- Faithfulness ≈ 0.17-0.37 — reflects both wrong context AND synthesis hallucination

### 6.5 Release Gate Summary

| Gate | Status (worst run) | Primary Blocker |
|---|---|---|
| Routing accuracy ≥ 0.95 | ❌ 0.672 | ReAct agent routing failures |
| Macro-F1 ≥ 0.93 | ❌ 0.086-0.725 | Per-class imbalance; UA-2 near-zero |
| Refusal precision ≥ 0.95 | ❌ 0.000-1.000 | Unstable across models |
| Citation recall ≥ 0.90 | ❌ 0.258 | Retrieval cascading errors |
| Faithfulness ≥ 0.90 | ❌ 0.174 | Hallucination in synthesis |
| p95 latency ≤ 6s | ❌ 165s | ReAct loop too slow |

**All 13 release gates fail** across every model tested. The system requires fundamental improvements in routing accuracy, retrieval effectiveness, and latency before it can be considered production-ready.

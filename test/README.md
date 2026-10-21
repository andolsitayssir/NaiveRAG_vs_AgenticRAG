# Activiity — Test Datasets (Golden Sets)

Three complexity tiers, same bucket sizing. Each tier targets a specific
failure mode of the agent and lets us track progress independently.

```
test/
├── reference/   ← Reference tier — direct, factual, single-source
├── medium/      ← Medium tier — indirect phrasing, scenarios, compound
└── advanced/    ← Advanced tier — cross-UA, adversarial, synthesis
```

## Why three tiers?

A single golden set that mixes easy and hard questions hides where the agent
actually fails. Splitting by difficulty gives us three signals that regress
independently: routing accuracy, retrieval depth, and synthesis quality.

| Tier | What it measures | Typical failure if it regresses |
|------|------------------|---------------------------------|
| **Reference** | Basic retrieval + synthesis on direct questions. | Ingestion broken, chunking too small, wrong embedding model. |
| **Medium** | Intent inference from paraphrased / indirect questions, application of principles to concrete scenarios, compound queries touching two facets of one UA. | Query rewriter / HyDE under-performs, synthesis too literal, tool description too narrow. |
| **Advanced** | Cross-UA reasoning, handling counter-intuitive conseils, adversarial phrasing, case-based diagnostic questions, tight paraphrases where a wrong chunk looks right. | Router too eager (picks one UA instead of combining), reranker missing, tree_summarize not enabled, synthesis hallucinates from training data. |

## Sizing

Same X per bucket across the three tiers — 134 pairs per tier, **402 total**.

| Bucket | Ref | Med | Adv |
|--------|----:|----:|----:|
| UA-1 Délégation | 12 | 12 | 12 |
| UA-2 Délégation via Sourcing | 6 | 6 | 6 |
| UA-3 Nourrir le plaisir | 20 | 20 | 20 |
| UA-4 Motiver | 16 | 16 | 16 |
| UA-5 Mauvaise ambiance | 12 | 12 | 12 |
| UA-6 Écouter et Dialoguer | 8 | 8 | 8 |
| UA-7 Changer les comportements | 10 | 10 | 10 |
| UA-8 Agir sur les habitudes | 5 | 5 | 5 |
| UA-9 Manager | 25 | 25 | 25 |
| UA-10 Compétences commerciales | 10 | 10 | 10 |
| Out-of-scope / refusal | 10 | 10 | 10 |
| **Total** | **134** | **134** | **134** |

## Pair schema (unchanged)

| Field | Meaning |
|-------|---------|
| `id` | Stable identifier (e.g. `UA-1-M-03` for Medium, `-A-` for Advanced). |
| `question` | User question in French. |
| `expected_ua` | UA the router *should* pick. `GLOBAL` = global tool is acceptable. `UA-X\|UA-Y` means a multi-UA answer is expected. `NONE` = out-of-scope (refusal). |
| `must_cite` | File(s) that should appear in the citations (at least one, unless refusal). |
| `ground_truth` | Canonical answer — short, factual, French. |
| `tags` | Difficulty tags: `indirect`, `scenario`, `compound`, `cross_ua`, `counter_intuitive`, `adversarial`, `case`, `paraphrase`, `synthesis`. |

## Release-gate thresholds per tier

The baseline gates (§8.4 of the design doc) relax as difficulty rises:

| Metric | Reference | Medium | Advanced |
|--------|----------:|-------:|---------:|
| faithfulness | 0.90 | 0.85 | 0.80 |
| answer_relevancy | 0.85 | 0.80 | 0.75 |
| context_precision | 0.80 | 0.70 | 0.60 |
| context_recall | 0.80 | 0.75 | 0.65 |
| UA routing accuracy | 0.95 | 0.90 | 0.75 |
| Refusal precision | 0.95 | 0.95 | 0.95 |

Reference is the smoke test; Medium the acceptance test; Advanced the
aspirational bar. A new prompt / retrieval change must not regress
**any metric in any tier by more than 2 points**.

# Judge LLM — recommendation & setup

## Does the evaluator need a judge?

**Only for the optional, nightly layer.** The per-PR path runs fully judge-free:

| Layer | Needs judge? | Cost | When |
|---|:---:|---|---|
| Routing (Accuracy/P/R/F1), Refusal, Retrieval (P@k/R@k/MRR/nDCG), Citation recall, System | **No** | $0 | every PR |
| **Answer similarity** (EM, Token-F1, ROUGE-L, SBERT cosine, BERTScore) | **No** | $0 (local SBERT/BERTScore models) | every PR |
| Ragas (Faithfulness, Answer-Relevancy, Ctx-Precision, Ctx-Recall) | **Yes** | ~$0.20 per 100 pairs | nightly |

The judge-free layer catches ~70 % of what Ragas catches. Ragas stays valuable for nuanced hallucination detection that token-overlap can't see.

## Recommended judge LLM (OpenRouter)

Ranked by suitability for **French Q&A evaluation on a management/coaching corpus**. These are empirical recommendations based on public benchmarks and community evaluations; the last word is your own spot-checks on 20-30 hand-labelled pairs.

### 1. ⭐ **`anthropic/claude-sonnet-4-6`** — **primary recommendation**

- **Why:** best-in-class French fluency, nuanced judgement, strong calibration on "does this answer stick to this context?" (the Ragas core questions). Low refusal rate for benign content, high faithfulness scoring agreement with humans.
- **Cost:** ~$3/M input, $15/M output. For 100-pair full-Ragas run: ~$0.15-0.30.
- **Latency:** 1-3 s per metric/pair.
- **Pin this as the baseline judge.**

### 2. `openai/gpt-4o` (2024-08-06 or later snapshot)

- **Why:** second-best on French nuance, widely used in Ragas community → many published references to calibrate against.
- **Cost:** similar to Sonnet 4.6.
- **Use as:** cross-check to detect if Claude has a systematic bias on your corpus. Run on 10-20 % of pairs periodically; if agreement with Claude drops below ~0.85, investigate.

### 3. `openai/gpt-4o-mini`

- **Why:** 10× cheaper than Sonnet/GPT-4o, French OK, noisier on faithfulness (tends to be more lenient).
- **Cost:** ~$0.15/M input, $0.60/M output. Full-Ragas run: ~$0.02.
- **Use as:** CI bulk judge if budget matters. Accept ~3-5 pts noise on absolute scores; still reliable for *regression detection*.

### 4. `qwen/qwen-2.5-72b-instruct`

- **Why:** open-weight, strong multilingual (trained with heavy Chinese+English, reasonable French). Good for hybrid cloud/on-prem setups where the eval must also run offline.
- **Cost:** ~$0.40/M on OpenRouter; ~$0 if self-hosted via Ollama.
- **Use as:** fall-back when the on-prem Ollama provider is selected for the RAG service itself (keeps the judge inside the same perimeter).

### Not recommended as judge (despite being fine for the RAG itself)

- **Small local LLMs (`qwen2.5:14b`, `llama-3.1-8b`)** — they under-penalise hallucinations and over-penalise paraphrases. Fine for production, too noisy as judge.
- **Models with short contexts (<32 k)** — Ragas needs to see (question + answer + ground_truth + all retrieved chunks) in a single prompt; short contexts force truncation that biases scores.
- **Any model with `temperature > 0`** — kills reproducibility. We pin `temperature=0.0`.

## Integration with your OpenRouter key

1. Put your key in `backend/.env` (or export it):

   ```
   OPENROUTER_API_KEY=sk-or-v1-...
   ```

2. The judge is already wired in [`eval/config.py`](config.py):

   ```python
   JUDGE = JudgeConfig(
       provider="openrouter",
       model="anthropic/claude-sonnet-4-6",
       temperature=0.0,
       max_retries=2,
   )
   ```

   Override via env var if you want to A/B two judges without editing code:

   ```bash
   ACTIVIITY_JUDGE_MODEL="openai/gpt-4o-mini" \
     python -m eval.run_eval --tier reference --ragas
   ```

3. Run:

   ```bash
   # cheap path: judge-free only
   python -m eval.run_eval --tier all

   # full path: judge-augmented (nightly)
   python -m eval.run_eval --tier all --ragas
   ```

## Reproducibility & cost guard-rails

- **Temperature pinned to 0.0** → same input yields same judgement (up to vendor-side stochasticity, which is low at T=0).
- **Model snapshot pinned** → use an explicit snapshot string when available (`openai/gpt-4o-2024-08-06` rather than the floating `openai/gpt-4o` alias) so results don't drift when the vendor rotates the endpoint.
- **Budget cap**: the runner prints `total cost` at the end of each tier. Set an env var `ACTIVIITY_EVAL_BUDGET_USD=1.00` to abort if the estimated cost exceeds this.
- **Cache on disk**: Ragas supports a response cache. Enable it for CI to avoid re-judging pairs whose (question, answer, contexts, ground_truth) tuple is unchanged:

   ```python
   # in eval/metrics/ragas_bridge.py
   from ragas.cache import DiskCacheBackend
   evaluate(..., cache=DiskCacheBackend(path=".ragas_cache/"))
   ```

## Spot-check protocol (recommended before trusting a new judge)

1. Pick 25 pairs — 10 reference, 10 medium, 5 advanced (including 2 refusal cases).
2. Manually score each on 4 dimensions (faithful/relevant/cP/cR) as `0 / 0.5 / 1`.
3. Run Ragas with the candidate judge.
4. Compute Pearson correlation and MAE between your scores and the judge's.
5. Accept the judge if Pearson ≥ 0.75 and MAE ≤ 0.15 on each of the four metrics.

This matters because the judge is the only LLM in the loop during evaluation — its biases become your blind spots. Claude Sonnet 4.6 clears this bar easily on this corpus; GPT-4o-mini is borderline on `context_precision` (too lenient).

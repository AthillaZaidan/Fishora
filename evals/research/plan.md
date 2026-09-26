# Fishora RAG — analysis and implementation plan

Locked 2026-09-26 09:30, before any baseline was measured. Hard deadline 12:10.

> **Scope change 09:28 (user):** this phase delivers findings only — the test-suite framework,
> the first run with baseline scores, and the weakness list. No RAG code is modified. The fixes
> below are *proposed* and are carried into [findings.md](findings.md) (W1–W17); they are applied
> in the modification phase, which ends with `scripts.quality --label current` and the
> before/after `reports/comparison.json`.

## Current architecture (as found)

```
verify ──> BackgroundTasks: run_graph (orchestrator.py)
             researcher: E5 embed CARD_QUERY -> pgvector (species-filtered, limit 36)
                         -> category-first selection (6)  [+1 LLM sub-query if a category is missing]
             4 experts (parallel threads, 1 LLM call each, chunk text cut to 300 chars)
             critic: lexical token overlap claim<->cited chunk  [+1 LLM downgrade call]
             writer: supported claims only [+1 LLM "polish" call] -> KnowledgeGenerator.build_card
GET /predictions/{id}/knowledge -> 202 while job runs, card when done
                                 -> sync KnowledgeService (1 LLM call) when no job exists
```

Corpus: 49 candidate chunks, 11 species, 6 categories, all written in **English**. The prompts,
expert outputs and cards are in **Indonesian**. No approved corpus, no tests and no `.env` exist in
this checkout; Docker/Postgres is not running.

## Main flaws, ranked by impact

| # | Flaw | Effect | Fix | Cost |
|---|------|--------|-----|------|
| F1 | The critic grounds claims by lexical token overlap, but claims are Indonesian and evidence is English | True claims are graded `unsupported`, the writer finds no grounded claim, and the job fails or the card comes back empty | Cross-lingual grounding: E5 symmetric similarity (`query:` on both sides) OR lexical overlap, with the threshold calibrated on a dev split | M |
| F2 | 4 sequential LLM rounds on the critical path (sub-query, experts, critic, polish) | Card latency is roughly 4× one LLM call; polish can drift facts after grading | Drop the sub-query (species filter makes it redundant: the same rows come back) and the post-grading polish; 2 rounds remain | S |
| F3 | No cache: every verification regenerates the same card for the same species and corpus | Repeated LLM cost and latency for identical evidence | Cache the card keyed by species id plus an evidence fingerprint (chunk ids + content hash) | S |
| F4 | E5 loads lazily on the first request | The first knowledge request pays the model-load time | Warm the embedder in a background thread at startup | S |
| F5 | The chunker allows 600 tokens; E5 truncates at 512 | Tails of long chunks are silently not embedded | `max_tokens=480` (leaves room for the prefix and special tokens) | XS |
| F6 | Expert JSON is parsed with a bare `json.loads` | A markdown-fenced answer means the expert silently returns nothing | Tolerant JSON extraction | XS |
| F7 | Expert evidence is cut to 300 characters | Facts late in a chunk are invisible to the expert but still citable | Pass the whole chunk (chunks are < 120 tokens) | XS |
| F8 | No tests, no evaluation, no quality visibility | Regressions such as F1 went unnoticed | Eval harness, test suite, dashboard | L |
| F9 | The verify route swallows every exception silently | Failures to schedule a job are invisible | Log them | XS |

Keep: the species-scoped, verification-gated retrieval; category-first selection; the fail-closed
`KnowledgeGenerator.build_card` citation gate; relational taxonomy guardrails; the signed
approval/ingestion pipeline; pgvector (exact scan is correct at this size — no ANN index needed
below ~10k rows). Remove: the LLM sub-query and polish passes. Modify: critic grounding, chunk
size, JSON parsing. Add: a card cache, embedder warmup, logging.

## Evaluation design (what the dashboard shows)

Adapted from the RAGAS split (context precision/recall vs faithfulness), restricted to what can be
measured reproducibly without an LLM key:

1. **Retrieval** — templated Indonesian and English gold queries per (species, category) cell,
   run through the real `VerifiedRetriever` + real E5 over an in-memory store with the same filter
   invariants as `search_verified`. Metrics: Recall@1/3/6, MRR, nDCG@6, species purity; card-query
   category coverage; global (unfiltered) species hit@1 as an embedding-discrimination signal;
   query latency p50/p95.
2. **Grounding (faithfulness proxy)** — Indonesian paraphrases of each chunk (supported) plus hard
   negatives (another species' claim in the same category). The critic's grader classifies them.
   Metrics: precision, recall, F1, accuracy, false-support rate (hallucination leakage) on a held-out
   test split.
3. **Pipeline** — the full orchestrator graph with a scripted, latency-simulating LLM whose experts
   emit real and hallucinated claims. Metrics: job success rate, supported-claim retention,
   hallucination leakage, citation validity, LLM calls per card, sequential LLM rounds, critical-path
   latency, cache-hit latency.
4. **Tests** — pytest JUnit XML + coverage JSON, shown per suite and per module.

Generation quality with the real OpenCode Go model is **not measurable here** (no key). The pipeline
eval measures the guardrails, not the LLM's prose, and the dashboard states that.

## Hypotheses and predictions

- H1 (F1): the lexical grader has recall < 0.3 on Indonesian claims against English evidence; E5
  similarity raises test-split F1 above 0.8 while keeping false-support ≤ 0.2.
- H2 (F2, F3): LLM rounds on the critical path drop from 4 to 2 and a cache hit completes in < 50 ms.
- H3: with the species filter, Recall@6 is ≈ 1.0 for every cell (the retriever returns almost the
  whole species slice); global species hit@1 is the informative retrieval-quality number.

## Time budget

09:30–09:50 datasets · 09:50–10:20 eval harness + baseline · 10:20–10:50 RAG fixes + re-eval ·
10:50–11:20 tests + quality runner · 11:20–11:45 dashboard · 11:45–12:05 end-to-end verification.

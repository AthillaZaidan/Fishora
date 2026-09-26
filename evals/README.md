# Fishora RAG test suite and evaluation (kerangka test suite)

An offline, reproducible harness for the knowledge-card RAG path. It runs against the **real
candidate corpus** (49 chunks, 11 species) and the **real local E5 model**, with no Postgres and no
LLM key. The same code produces the test results and the evaluation artifact.

```
evals/
├── datasets/
│   ├── retrieval_queries.json   gold query templates, Indonesian + English, per category
│   ├── grounding_claims.json    one Indonesian card-style claim per chunk (98 graded pairs)
│   ├── grounding_traps.json     14 negation/scope traps (iteration 1)
│   └── grounding_real_claims.json  345 real expert claim atoms + 23 fresh traps (iteration 2)
├── corpus.py                    corpus loader + InMemoryKnowledgeRepository (same filters as pgvector)
├── fakes.py                     in-memory ports + ScriptedLLM (deterministic, latency-simulating)
├── retrieval_eval.py            recall@k, MRR, nDCG, species purity, card coverage, latency
├── grounding_eval.py            critic accept/reject vs gold labels, dev/test split, E5 score analysis
├── pipeline_eval.py             production run_graph end to end: success, retention, leakage, rounds
├── run.py                       python -m evals.run --label <name>  -> reports/<name>/rag_eval.json
├── tests/
│   ├── conftest.py              fixtures: E5 (session), store, scripted LLM, app factory
│   ├── unit/                    chunking, selection, generation guards, graph nodes (no model)
│   ├── retrieval/               retrieval quality gates
│   ├── generation/              grounding + card-pipeline quality gates
│   ├── integration/             FastAPI app, verification gate, job lifecycle, warmup  (marker: api)
│   └── e2e/                     photo -> identify -> verify -> card; quality dashboard
├── results/baseline/            the first recorded run (tracked copy of reports/baseline)
└── research/                    plan, baseline report, findings
```

## Run

```bash
PY=.venv/Scripts/python.exe        # Windows; .venv/bin/python elsewhere
HF_HUB_OFFLINE=1 $PY -m pytest evals/tests -q           # the suite alone (~40 s)
HF_HUB_OFFLINE=1 $PY -m pytest evals/tests -m unit -q   # no model load (~1 s)
HF_HUB_OFFLINE=1 $PY -m scripts.quality --label current # suite + evals -> reports/current/
```

`scripts.quality` writes `junit.xml`, `coverage.json`, `tests.json`, `pytest.log` and
`rag_eval.json` under `reports/<label>/`. Once both `reports/baseline/` and `reports/current/`
exist it also writes `reports/comparison.json`, the before/after artifact for the fix phase.
`reports/` is generated and gitignored. The baseline is also kept under `evals/results/baseline/`.

## What each layer asserts

| Layer | Asserts | Needs |
|---|---|---|
| unit | chunk limits and overlap, category-first selection, query-vector validation, fail-closed citation gate, taxonomy guardrail, expert JSON parsing, expert evidence visibility, critic/writer rules | nothing |
| retrieval | species purity = 1, scoped recall@6 ≥ 0.95, card-query coverage = 1, species hit@1 without filter ≥ 0.9, warm p95 < 150 ms, unverified rows never returned | E5 |
| generation | Indonesian claim grounded in English evidence; shared-token borrowed claim rejected; wrong numbers rejected; grounding F1 ≥ 0.85, false support ≤ 0.15; every job completes; retention ≥ 0.8; leakage ≤ 0.1; citation validity = 1; ≤ 2.5 LLM rounds; repeat card < 50 ms; fenced JSON still completes; LLM outage fails cleanly | E5 |
| integration | health, 409 before verification, verify → job completes, correction retrieves the corrected species, embedder warmed at startup | E5 |
| e2e | photo → grounded, cited tuna card with genus guardrail; quality dashboard served | E5 |

Failing tests are the specification the fix phase has to meet. Each failure maps to a weakness in
[research/findings.md](research/findings.md).

## Honest limits of these measurements

- **No real LLM.** `ScriptedLLM` shows what the orchestration and guardrails do with a given model
  output. It says nothing about the quality of OpenCode Go's prose. With a key, the same
  `run_graph` path can be driven by the real model.
- **Candidates are treated as verified** for evaluation only. Production ingestion still requires
  the signed human approval manifest, and no approved corpus exists yet.
- **In-memory store instead of pgvector.** It applies the same four filters and ordering, and it is
  exact cosine like pgvector without an ANN index, so rankings are identical. SQL behaviour itself
  (advisory lock, upsert) is not covered here.
- **The gold data is small and hand-made:** 88 templated retrieval queries and 98 grounding pairs.
  The grounding claims were written by the same agent that built the harness. Treat the numbers as
  regression signals with wide error bars, not as benchmarks.
- **Synthetic claims flattered the verifier.** On them the shipped verifier scored perfectly; on the
  real expert claims of `grounding_real_claims.json` it both drops true claims and keeps false ones
  (finding W29, `experiment_verifier.py`). Those real claims came from the production model, but
  their labels are an AI assistant's and have not been reviewed by a person yet.

# Baseline — first execution of the test suite (hasil eksekusi pertama + skor baseline)

Run 2026-09-26 09:26–09:28 on `main` (`da6f8e9`) plus the uncommitted harness. The RAG code is
unmodified; only the harness and the tests were added. Environment: Python
3.12.8, CPU torch 2.14, `intfloat/multilingual-e5-base` from the local cache, scripted LLM
(0.2 s per call), in-memory store. Raw artifacts are in [`../results/baseline/`](../results/baseline/).

Reproduce: `HF_HUB_OFFLINE=1 .venv/Scripts/python.exe -m scripts.quality --label baseline`

## Test suite: 35 / 50 passed (70.0 %)

| Layer | Passed | Failed | Pass rate |
|---|---:|---:|---:|
| unit | 20 | 3 | 87 % |
| retrieval | 7 | 0 | 100 % |
| generation | 6 | 7 | 46 % |
| integration (api) | 2 | 3 | 40 % |
| e2e | 0 | 2 | 0 % |
| **total** | **35** | **15** | **70 %** |

Duration 48 s. No errors, no skips. Coverage of the RAG modules is **80.5 %**; all of
`apps/main_api` is 60.6 %, since commerce and auth are out of scope.

| Failing test | Observed | Weakness |
|---|---|---|
| unit · test_default_chunk_limit_fits_the_e5_window | 600 + 4 > 512 | W10 |
| unit · test_expert_parses_markdown_fenced_json | expert returned `error`, no claim | W4 |
| unit · test_expert_sees_the_whole_chunk | chunk cut at 300 chars | W14 |
| generation · test_indonesian_claim_is_grounded_in_english_evidence | `unsupported` | W1 |
| generation · test_shared_token_does_not_ground_a_borrowed_claim | `supported` (token "indo") | W1 |
| generation · test_grounding_test_split_f1 | 0.791 < 0.85 | W1 |
| generation · test_true_claims_survive_into_the_card | 0.56 < 0.8 | W1 |
| generation · test_card_needs_at_most_two_sequential_llm_rounds | 5.4 > 2.5 (at 0.05 s/call) | W5 |
| generation · test_repeat_card_is_served_from_cache | 270 ms > 50 ms | W6 |
| generation · test_markdown_fenced_llm_output_still_completes | 0 / 11 jobs completed | W4 |
| integration · test_verification_schedules_a_job_that_completes | job `failed` | W2 |
| integration · test_correction_retrieves_for_the_corrected_species | 502, no card | W2 |
| integration · test_embedder_is_warmed_at_startup | never warmed | W7 |
| e2e · test_photo_to_grounded_card | 502 "knowledge generation failed" | W2 |
| e2e · test_quality_dashboard_is_served | 404 | planned feature, not a weakness |

## Baseline scores (skor baseline)

### Retrieval — 88 gold queries (44 cells × {id, en}), k = 6

| Metric | Species-scoped (production) | Global (no filter, probe) |
|---|---:|---:|
| Recall@1 | 0.224 | 0.349 |
| Recall@3 | 0.678 | 0.513 |
| Recall@6 | **0.989** | 0.693 |
| MRR | **0.509** | 0.478 |
| nDCG@6 | 0.620 | 0.526 |
| Species purity / species hit@1 | **1.000** | 0.932 |
| Card-query category coverage | **1.000** | — |
| Warm latency p50 / p95 | 35.5 / 38.5 ms | — |

Scoped MRR per category: identity 1.00, physical 0.50, taste 0.37, processing 0.30, commercial
**0.24**. Global species hit@1 is 0.91 for Indonesian queries and 0.95 for English ones.

### Grounding (faithfulness proxy) — the shipped lexical critic

| Split | n | Precision | Recall | F1 | False-support rate |
|---|---:|---:|---:|---:|---:|
| test | 48 | 0.895 | 0.708 | **0.791** | 0.083 |
| dev | 50 | 0.800 | 0.480 | 0.600 | 0.120 |
| all | 98 | 0.853 | **0.592** | 0.699 | **0.102** |

E5 cross-lingual similarity for the same pairs, analysis only (the code does not use it): supported
mean 0.856, borrowed mean 0.793, supported min 0.811, borrowed max 0.830. The best dev threshold,
0.82, gives F1 0.875 with false-support 0.08. Adding a numbers-must-match check at 0.81 gives dev
F1 0.96 with false-support 0.08. That is a candidate for the fix phase and has not been validated
on the test split.

### Card pipeline — production `run_graph`, 11 species, scripted LLM

| Metric | Plain JSON | Fenced JSON |
|---|---:|---:|
| Job success rate | 1.00 | **0.00** |
| True-claim retention | **0.56** | 0.00 |
| Hallucination leakage | 0.00 * | — |
| Citation validity | 1.00 | — |
| LLM calls per card | **6.82** | 4.82 |
| Sequential LLM rounds | **4.2** | — |
| Card latency, mean (0.2 s/call) | 839 ms | 73 ms |
| Repeat-card latency | **833 ms** (no cache) | — |

\* Optimistic. The scripted borrowed claims happened to share no tokens with their cited chunks.
The grounding eval shows a 10.2 % false-support rate on the same grader, so leakage does occur.

### Platform

| Metric | Value |
|---|---:|
| Embedder cold load (first request) | **10.97 s** |
| Warm query embedding + search | 35 ms |
| Chunker `max_tokens` default vs E5 window | 600 vs 512 (does not fit) |
| Corpus | 49 chunks, max 89 tokens, 0 over the E5 limit |
| Category chunks | identity 14, physical 9, taste 5, processing 10, commercial 11, **substitutes 0** |
| Worst-case card latency (4 rounds × 60 s timeout) | 240 s |

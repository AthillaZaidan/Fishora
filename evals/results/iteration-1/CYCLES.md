# iteration-1: three complete iteration cycles

Checkpoint requirement: *Progres perbaikan pada Evaluation Track dan Product Track. Minimal satu siklus iterasi lengkap terdokumentasi: temuan, perbaikan, lalu hasil uji ulang.*

Each cycle below runs finding (temuan) -> fix (perbaikan) -> re-test (hasil uji ulang), with changes on both the product track (`apps/main_api`) and the evaluation track (`evals/`). These are the three largest RAG cycles of this iteration; [`REPORT.md`](REPORT.md) lists every fix.

Generated 2026-09-26T07:51:15+00:00 by `python -m evals.iteration` from `reports/baseline` (baseline, tag `checkpoint-1-baseline`) and `reports/current` (after the fixes). Every number is read from those run artifacts, which are archived in this folder.

| Cycle | Starts from | Fixes | Headline re-test | Findings now |
|---|---|---|---|---|
| [C1](#c1) The agent card path works against the real LLM | W20, W21, W4, W3 | F1, F2, F12, E2 | Agent card success, real LLM: 0 -> 1 | W20 resolved, W21 resolved, W4 resolved, W3 resolved |
| [C2](#c2) Claims are checked against their evidence, across languages | W1, W13, R2 | F3, F18, E3, E4 | Grounding F1, held-out test split: 0.7907 -> 1 | W1 resolved, W13 partial, R2 partial |
| [C3](#c3) Each card sees all of its evidence, in fewer LLM rounds | R1, W9, W14, W5, W6 | F5, F4, F7, F8, E1 | Card evidence completeness, min over species: n/a -> 1 | R1 resolved, W9 resolved, W14 resolved, W5 resolved, W6 resolved |

Whole suite: 35/50 tests passing at baseline, 71/71 after the fixes.

## C1

### The agent card path works against the real LLM

The card path the app ships (researcher, experts, critic, writer) had never completed a card on OpenCode Go. The gateway rejected calls without a session header, and the experts could not read Responses-API content blocks or JSON wrapped in code fences, so every expert failed and the spend was wasted.

#### 1. Finding (temuan, baseline run)

- **W20** (high): OpenCode Go rejects every call: no x-opencode-session header.
  - client sends x-opencode-session: False; sets a user agent: False
  - without it the gateway answers 400 MissingSessionID
- **W21** (high): Agent path cannot parse Responses-API content blocks.
  - agent path as shipped, real LLM: 0% cards completed of 22
  - same run, content read as text (eval-only): 82% completed
  - spend on the failing path: $0.00100 per card, all wasted
- **W4** (high): Markdown-fenced LLM JSON fails every expert.
  - cards completed when the LLM wraps JSON in ``` fences: 0%
  - unit test: failed
- **W3** (high): Sync knowledge fallback returns an unhandled 500 when the key is blank.
  - GET /predictions/{id}/knowledge, no job, blank key -> HTTP 500

#### 2. Fix (perbaikan)

| Fix | Track | Source | Change | Files |
|---|---|---|---|---|
| F1 OpenCode Go session header and user agent | product | W20 | make_opencode_go_llm sends x-opencode-session (one id per card) and a descriptive User-Agent. | `apps/main_api/services/generation.py` |
| F2 Read Responses-API content blocks and fenced JSON | product | W21, W4 | llm_output.reply_text/reply_json read every reply; experts, critic and judge use them. | `apps/main_api/services/llm_output.py`<br>`apps/main_api/services/orchestrator.py` |
| F12 A blank key is a provider outage, not a crash | product | W3 | KnowledgeGenerator maps client-construction ValueError to OpenCodeUnavailable (HTTP 502). | `apps/main_api/services/generation.py` |
| E2 Real-LLM runs use the production client | evaluation | W20 | cost_eval drives make_opencode_go_llm instead of adding the header itself. | `evals/cost_eval.py` |

#### 3. Re-test (hasil uji ulang)

| Metric | Baseline | After | Artifact |
|---|---:|---:|---|
| Agent card success, real LLM | 0 | 1 | `cost_eval.summary.by_path.agent.success_rate` |
| Fenced-JSON job success (scripted) | 0 | 1 | `rag_eval.pipeline.fenced.job_success_rate` |
| Published card success, real LLM | 0.9545 | 1 | `cost_eval.summary.by_path.published.success_rate` |
| Spend per agent card, real LLM (USD) | 0.001 | 0.0009 | `cost_eval.summary.by_path.agent.cost_usd_per_card.mean` |

| Test | Baseline | After |
|---|---|---|
| `test_provider_client::test_client_sends_session_and_user_agent` | absent | passed |
| `test_provider_client::test_client_without_session_gets_a_fresh_one` | absent | passed |
| `test_llm_output::test_responses_api_content_blocks` | absent | passed |
| `test_orchestrator_nodes::test_expert_parses_markdown_fenced_json` | failed | passed |
| `test_orchestrator_iteration::test_expert_reads_responses_api_blocks` | absent | passed |
| `test_provider_client::test_blank_key_is_a_provider_outage_not_a_crash` | absent | passed |

| Finding | Baseline | After | Evidence after |
|---|---|---|---|
| W20 | open | resolved | client sends x-opencode-session: True; sets a user agent: True; without it the gateway answers 400 MissingSessionID |
| W21 | open | resolved | agent path as shipped, real LLM: 100% cards completed of 11; spend: $0.00090 per completed card |
| W4 | open | resolved | cards completed when the LLM wraps JSON in ``` fences: 100%; unit test: passed |
| W3 | open | resolved | GET /predictions/{id}/knowledge, no job, blank key -> HTTP 502 |

**Reading the result, and what remains.** Nothing in this cycle is open. Read the spend row with care: at baseline it bought no card. The evaluation fix (E2) mattered as much as the product fixes: the cost run used to add the session header itself, so it could not see that the app never sent it. Whether OpenCode Go may carry production traffic is a separate open question (W23).

## C2

### Claims are checked against their evidence, across languages

The critic kept a claim when enough words overlapped with the cited chunk. The claims are Indonesian and the evidence English, so it dropped most true claims and still passed some claims borrowed from other species. The review (R2) also asked whether it catches negated or inverted claims.

#### 1. Finding (temuan, baseline run)

- **W1** (high): Lexical grounding across languages: Indonesian claims vs English evidence.
  - grader: lexical
  - test-split F1 0.791, recall 0.708
  - all 98 pairs: recall 0.592, false support 0.102
  - scripted pipeline claim retention 56%
  - test_indonesian_claim_is_grounded_in_english_evidence: failed
  - test_shared_token_does_not_ground_a_borrowed_claim: failed
- **W13** (medium): Embedding similarity alone separates supported from borrowed claims poorly.
  - E5 claim-evidence cosine: supported mean 0.8562, borrowed mean 0.7931
  - supported min 0.8111 vs borrowed max 0.8299 (overlap: True)
- **R2** (high): Grounding must catch negated and inverted claims, not only borrowed ones.
  - not measured at baseline: this finding came from the review

#### 2. Fix (perbaikan)

| Fix | Track | Source | Change | Files |
|---|---|---|---|---|
| F3 Cross-lingual claim verifier replaces lexical overlap | product | W1, W13, R2, W26 | Critic grades each claim with E5 symmetric cosine >= tau (0.805, chosen on the dev split) AND every number and taxon (family names, supported binomials) in the claim appearing in the cited chunk (the LLM judge pass is off, see F18). | `apps/main_api/services/orchestrator.py`<br>`evals/calibrate_grounding.py` |
| F18 Critic's LLM judge pass removed | product | R2, W5 | USE_LLM_JUDGE = False: measured with gpt-5.6-luna, the downgrade-only judge halved held-out recall (1.0 -> 0.5) and still accepted 3 of 14 traps; the verifier alone scores higher on the dev split. | `apps/main_api/services/orchestrator.py` |
| E3 Threshold calibration on the dev split | evaluation | W1, W13 | calibrate_grounding sweeps tau for the shipped rule and reports the held-out split. | `evals/calibrate_grounding.py` |
| E4 Negation/scope trap set and verifier comparison | evaluation | R2 | 14 traps; E5+exact vs multilingual NLI vs the production LLM judge, pre-registered selection rule. | `evals/datasets/grounding_traps.json`<br>`evals/experiment_nli_grounding.py` |

#### 3. Re-test (hasil uji ulang)

| Metric | Baseline | After | Artifact |
|---|---:|---:|---|
| Grounding F1, held-out test split | 0.7907 | 1 | `rag_eval.grounding.test.f1` |
| Grounding precision, test split | 0.8947 | 1 | `rag_eval.grounding.test.precision` |
| Grounding recall, test split | 0.7083 | 1 | `rag_eval.grounding.test.recall` |
| False support rate, all pairs | 0.102 | 0.0204 | `rag_eval.grounding.all.false_support_rate` |
| True-claim retention in cards (scripted) | 0.56 | 1 | `rag_eval.pipeline.plain.claim_retention` |
| Hallucinated claims reaching cards (scripted) | 0 | 0 | `rag_eval.pipeline.plain.hallucination_leakage` |
| Negation/scope traps accepted (of 14) | n/a | 7 | `experiment_nli_grounding.variants.A_e5_exact.traps_accepted` |

| Test | Baseline | After |
|---|---|---|
| `test_grounding_and_pipeline::test_indonesian_claim_is_grounded_in_english_evidence` | failed | passed |
| `test_grounding_and_pipeline::test_shared_token_does_not_ground_a_borrowed_claim` | failed | passed |
| `test_grounding_and_pipeline::test_grounding_test_split_f1` | failed | passed |
| `test_orchestrator_iteration::test_llm_judge_pass_is_off_in_production` | absent | passed |

| Finding | Baseline | After | Evidence after |
|---|---|---|---|
| W1 | open | resolved | grader: E5 cosine + numbers; test-split F1 1.000, recall 1.000; all 98 pairs: recall 1.000, false support 0.020; scripted pipeline claim retention 100%; test_indonesian_claim_is_grounded_in_english_evidence: passed; test_shared_token_does_not_ground_a_borrowed_claim: passed |
| W13 | open | partial | E5 claim-evidence cosine: supported mean 0.8562, borrowed mean 0.7931; supported min 0.8111 vs borrowed max 0.8299 (overlap: True) |
| R2 | not_measured | partial | shipped verifier (E5 + exact checks): held-out F1 1.0, false support 0.0, traps accepted 7/14; with the LLM judge (removed): held-out F1 0.6667, traps accepted 3/14; NLI + exact (candidate): held-out F1 0.9362, traps accepted 0/14 |

**Reading the result, and what remains.** The threshold was chosen on the dev split and is reported on the held-out test split, which is small, so its perfect score has a wide interval. The false support left on the full set is a claim borrowed from another species' processing chunk (`rag_eval.grounding.errors`). R2 stays partial: the shipped verifier still accepts part of the negation/scope traps. The LLM judge was measured and removed (F18) because it cut recall; multilingual NLI rejects every trap but is too slow on CPU, so it is the next candidate, to be confirmed on a fresh trap set. W13 stays partial because the number and taxon checks mitigate the embedding overlap rather than remove it.

## C3

### Each card sees all of its evidence, in fewer LLM rounds

The card took the ranked top 6 of a species slice of at most 7 chunks, so ranking errors decided what the model saw, and the experts read only the first 300 characters of each chunk. A card took several sequential LLM rounds and the same evidence was regenerated every time.

#### 1. Finding (temuan, baseline run)

- **R1** (medium): Retrieval is unnecessary at this corpus size and loses evidence.
  - the card takes the ranked top 6 of a 1-7 chunk slice (baseline design)
  - scoped MRR 0.5087 decides which evidence the model sees
- **W9** (medium): Fixed 6-chunk cap drops evidence.
  - species with more chunks than the 6-chunk cap: {'tuna': 7}
  - scoped recall@6 0.989
- **W14** (low): Expert evidence truncated at 300 characters.
  - expert evidence cut at 300 chars; unit test: failed
- **W5** (medium): Too many sequential LLM rounds per card.
  - LLM rounds per card (scripted): 4.2; wall time / call delay 4.2
  - real LLM: agent card p50 11.6s with 6.5 calls vs published card p50 5.2s with 1 call
- **W6** (medium): No card cache for identical evidence.
  - repeat card 833 ms vs first card 839 ms (no cache)

#### 2. Fix (perbaikan)

| Fix | Track | Source | Change | Files |
|---|---|---|---|---|
| F5 The whole species slice is the card's evidence | product | R1, W9, W8 | VerifiedRetriever.card_evidence returns every verified chunk of the species when there are at most 20, ordered by category; ranked category-first selection only above that. | `apps/main_api/services/retrieval.py`<br>`apps/main_api/services/knowledge.py`<br>`apps/main_api/services/orchestrator.py` |
| F4 Experts and the critic judge read whole chunks | product | W14, R3 | Removed both 300-character cuts (expert payload and critic LLM pass). | `apps/main_api/services/orchestrator.py` |
| F7 Two LLM rounds per card | product | W5 | Removed the researcher's LLM sub-query and the writer's post-verification polish. | `apps/main_api/services/orchestrator.py` |
| F8 Card cache keyed by the evidence | product | W6 | Species + evidence ids and contents + model + pipeline version; invalidates itself on corpus change. | `apps/main_api/services/orchestrator.py` |
| E1 LLM rounds counted from call timestamps | evaluation | W5 | The wall-time/delay proxy counted verifier compute as rounds; overlapping calls now form one round. The old proxy is still reported. | `evals/fakes.py`<br>`evals/pipeline_eval.py` |

#### 3. Re-test (hasil uji ulang)

| Metric | Baseline | After | Artifact |
|---|---:|---:|---|
| Card evidence completeness, min over species | n/a | 1 | `rag_eval.retrieval.card_evidence_completeness.min` |
| LLM calls per card (scripted) | 6.82 | 3.27 | `rag_eval.pipeline.plain.llm_calls_per_card` |
| LLM rounds per card (call timestamps) | n/a | 1 | `rag_eval.pipeline.plain.llm_rounds` |
| Repeat card latency (ms) | 832.7 | 11.3 | `rag_eval.pipeline.plain.repeat_latency_ms_mean` |
| Agent card p50 wall time, real LLM (s) | 11.55 | 4.03 | `cost_eval.agent.wall_s_per_card.p50` |
| Agent card p95 wall time, real LLM (s) | 14.2 | 66.63 | `cost_eval.agent.wall_s_per_card.p95` |
| Published card p95 wall time, real LLM (s), one LLM call | 7.14 | 67.32 | `cost_eval.summary.by_path.published.wall_s_per_card.p95` |
| Cost per agent card, real LLM (USD) | 0.0014 | 0.0009 | `cost_eval.agent.cost_usd_per_card.mean` |

| Test | Baseline | After |
|---|---|---|
| `test_orchestrator_nodes::test_expert_sees_the_whole_chunk` | failed | passed |
| `test_orchestrator_iteration::test_critic_llm_pass_reads_the_whole_chunk` | absent | passed |
| `test_grounding_and_pipeline::test_card_needs_at_most_two_sequential_llm_rounds` | failed | passed |
| `test_grounding_and_pipeline::test_repeat_card_is_served_from_cache` | failed | passed |
| `test_orchestrator_iteration::test_cache_key_follows_the_evidence` | absent | passed |

| Finding | Baseline | After | Evidence after |
|---|---|---|---|
| R1 | open | resolved | cards receive the whole species slice: completeness mean 100%, min 100%; semantic ranking applies only above 20 chunks per species |
| W9 | open | resolved | share of each species' verified chunks that reach its card: mean 100%, min 100%; largest species slice 7 chunks |
| W14 | open | resolved | experts receive whole chunks; unit test: passed |
| W5 | open | resolved | LLM rounds per card (scripted): 1.0; wall time / call delay 1.4; real LLM: agent card p50 4.0s with 3.2727 calls vs published card p50 8.9s with 1 call |
| W6 | open | resolved | repeat card 11 ms vs first card 280 ms (served from the card cache) |

**Reading the result, and what remains.** The shipped agent path completed no card at baseline (C1), so its real-LLM baseline latency and cost come from the eval-only run that read the replies correctly. Cache hits are excluded from the real-LLM statistics. The p95 got worse. The published path, a single LLM call, shows the same p95 in the same run, which points to provider stalls rather than the pipeline; a repeat cost run would confirm it.

## Reproduce

```bash
HF_HUB_OFFLINE=1 .venv/Scripts/python.exe -m scripts.quality --label current            # tests, evals, findings, dashboard
HF_HUB_OFFLINE=1 .venv/Scripts/python.exe -m evals.cost_eval --label current --repeat 2   # real LLM, needs OPENCODE_GO_API_KEY
HF_HUB_OFFLINE=1 .venv/Scripts/python.exe -m evals.calibrate_grounding --label current
HF_HUB_OFFLINE=1 .venv/Scripts/python.exe -m evals.experiment_nli_grounding --label current --with-llm
.venv/Scripts/python.exe -m evals.iteration --name iteration-1                              # this file and REPORT.md
```

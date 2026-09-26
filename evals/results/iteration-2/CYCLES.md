# iteration-2: 3 complete iteration cycles

Checkpoint requirement: *Progres perbaikan pada Evaluation Track dan Product Track. Minimal satu siklus iterasi lengkap terdokumentasi: temuan, perbaikan, lalu hasil uji ulang.*

Each cycle below runs finding (temuan) -> fix (perbaikan) -> re-test (hasil uji ulang), with changes on both the product track (`apps/main_api`) and the evaluation track (`evals/`). These are the largest RAG cycles of this iteration; [`REPORT.md`](REPORT.md) lists every fix.

Generated 2026-09-26T09:27:01+00:00 by `python -m evals.iteration` from `reports/iteration-1` (iteration 1, archived in `evals/results/iteration-1`) and `reports/current` (after the fixes). Every number is read from those run artifacts, which are archived in this folder.

| Cycle | Starts from | Fixes | Headline re-test | Findings now |
|---|---|---|---|---|
| [C1](#c1) Every card that reaches a buyer has passed the claim critic | W19, W3 | F19, E8, E10 | Sync path grades claims (probe): False -> True | W19 resolved, W3 resolved |
| [C2](#c2) A card run can be inspected after the fact | W12 | F20, E9 | Silent broad exception handlers: 5 -> 0 | W12 resolved |
| [C3](#c3) The claim verifier, re-tested on real claims | W26, W29, R2, W13 | E11, E12, E13 | Real-claims recall, verifier in use: n/a -> 0.6792 | W26 open, W29 open, R2 partial, W13 partial |

Whole suite: 71/71 tests passing at baseline, 84/84 after the fixes.

## C1

### Every card that reaches a buyer has passed the claim critic

Publication froze the graded job card when one existed, but a manually declared catch has no job, and for it the card, the lot snapshot and the QR page came from a one-call generator with no per-claim critic. The finding was partial: the path buyers see most could still publish ungraded claims.

#### 1. Finding (temuan, baseline run)

- **W19** (high): The published card (QR page, buyer matching) bypasses the per-claim critic.
  - publication freezes the graded job card when one exists: True
  - lot publication uses the sync KnowledgeService: True
  - sync path has a per-claim critic: False
  - buyer matching reads snapshot fields: commercial_uses, physical_characteristics, processing_methods, taste, texture
- **W3** (high): Sync knowledge fallback returns an unhandled 500 when the key is blank.
  - GET /predictions/{id}/knowledge, no job, blank key -> HTTP 502

#### 2. Fix (perbaikan)

| Fix | Track | Source | Change | Files |
|---|---|---|---|---|
| F19 The synchronous card path runs the graded graph | product | W19, W3 | KnowledgeService (a manually declared catch, the publication fallback) calls orchestrator.grade_card, the same researcher-experts-critic-writer graph as the background job, so no card reaches a buyer without the per-claim critic. A blank key stays a mapped provider outage. The one-call generator wiring (deps.retriever, deps.generator) is removed from the app. | `apps/main_api/services/knowledge.py`<br>`apps/main_api/services/orchestrator.py`<br>`apps/main_api/services/card_llm.py`<br>`apps/main_api/api/fish.py`<br>`apps/main_api/api/lots.py`<br>`apps/main_api/main.py`<br>`apps/main_api/ports.py` |
| E8 The W19 probe reads the syntax tree | evaluation | W19 | sync_path_has_claim_critic is true only when knowledge.py's code calls grade_card or critic_node; the former token check would have passed on a comment. | `evals/probes.py` |
| E10 Cost runs measure the card the product publishes | evaluation | W19, W25, W26 | The one-call generator becomes an eval-only reference path (one_call); 'published' metrics resolve to the path the product publishes; agent cards keep their trace and expert outputs. | `evals/cost_eval.py`<br>`evals/findings.py`<br>`evals/dashboard.py`<br>`evals/model_compare.py` |

#### 3. Re-test (hasil uji ulang)

| Metric | Baseline | After | Artifact |
|---|---:|---:|---|
| Sync path grades claims (probe) | False | True | `probes.publication_path.sync_path_has_claim_critic` |
| Blank key on the sync path (HTTP status) | 502 | 502 | `probes.sync_path_blank_key.status_code` |
| Published card success, real LLM | 1 | 1 | `cost_eval.published.success_rate` |
| Cost per published card (USD) | 0.0007 | 0.0009 | `cost_eval.published.cost_usd_per_card.mean` |
| Published card p50 wall time, real LLM (s) | 8.86 | 4.45 | `cost_eval.published.wall_s_per_card.p50` |

| Test | Baseline | After |
|---|---|---|
| `test_knowledge_api::test_manual_catch_card_is_graded_by_the_critic` | absent | passed |
| `test_knowledge_api::test_manual_catch_without_a_key_is_a_provider_outage` | absent | passed |
| `test_lot_publication::test_published_snapshot_is_the_graded_job_card` | passed | passed |

| Finding | Baseline | After | Evidence after |
|---|---|---|---|
| W19 | partial | resolved | publication freezes the graded job card when one exists: True; lot publication uses the sync KnowledgeService: True; sync path has a per-claim critic: True; buyer matching reads snapshot fields: commercial_uses, physical_characteristics, processing_methods, taste, texture |
| W3 | resolved | resolved | GET /predictions/{id}/knowledge, no job, blank key -> HTTP 502 |

**Reading the result, and what remains.** The published card is now the graded card, so its cost and time are the agent path's (several parallel LLM calls instead of one); the baseline column is the one-call card it replaces. The evaluation fixes matter as much: the probe now reads code rather than words, and the cost run no longer calls an ungraded path 'published'.

## C2

### A card run can be inspected after the fact

Handlers on the card path caught everything and returned quietly, and a completed job kept only the card: nothing recorded which evidence each expert saw, what the critic decided, or where the time went.

#### 1. Finding (temuan, baseline run)

- **W12** (medium): Broad exception handlers swallow errors silently.
  - 5 broad except handlers that neither log nor re-raise: {'fish.py': 2, 'orchestrator.py': 1, 'lots.py': 2}
  - stage trace kept per card job: test absent

#### 2. Fix (perbaikan)

| Fix | Track | Source | Change | Files |
|---|---|---|---|---|
| F20 Every card run leaves a stage trace; no handler fails silently | product | W12 | grade_card records evidence ids and distances, each expert's chunk ids, prompt hash, latency and tokens, the claim verdicts and per-stage timings; run_graph stores it on the job (knowledge_jobs.trace). The silent broad handlers log; the langgraph branch, never installed and never run, is deleted. | `apps/main_api/services/orchestrator.py`<br>`apps/main_api/api/fish.py`<br>`apps/main_api/services/lots.py`<br>`apps/main_api/db/sql_repositories.py`<br>`apps/main_api/db/models.py`<br>`apps/main_api/contracts.py`<br>`alembic/versions/0007_knowledge_job_trace.py` |
| E9 W12 requires a stage trace | evaluation | W12 | Resolved needs no silent handler and the stage-trace test passing (judge methodology 6.1). | `evals/findings.py` |

#### 3. Re-test (hasil uji ulang)

| Metric | Baseline | After | Artifact |
|---|---:|---:|---|
| Silent broad exception handlers | 5 | 0 | `probes.silent_excepts.count` |

| Test | Baseline | After |
|---|---|---|
| `test_knowledge_api::test_card_job_records_a_stage_trace` | absent | passed |

| Finding | Baseline | After | Evidence after |
|---|---|---|---|
| W12 | open | resolved | 0 broad except handlers that neither log nor re-raise: none; stage trace kept per card job: test passed |

**Reading the result, and what remains.** W12 now needs both halves: no silent handler, and a stage trace on every job, checked by a test.

## C3

### The claim verifier, re-tested on real claims

W26 said graded cards are emptier than ungraded ones. Measured fairly, only where evidence exists, it held. Reading the real claims showed why: the verifier had been chosen on synthetic claims, and real expert claims are short translations that it misses, while it keeps facts from a sibling chunk.

#### 1. Finding (temuan, baseline run)

- **W26** (medium): Critic-guarded agent cards are emptier than published cards.
  - fields where the graded card is emptier than the one-call card on cells with evidence: 4/6
  - physical_characteristics: graded 73% (n=22) vs one-call 91% (n=22)
  - taste: graded 20% (n=10) vs one-call 80% (n=10)
  - texture: graded 40% (n=10) vs one-call 70% (n=10)
  - processing_methods: graded 89% (n=18) vs one-call 100% (n=18)
  - commercial_uses: graded 90% (n=20) vs one-call 80% (n=20)
  - similar_or_substitute_species: no species has evidence for it
  - potential_buyer_segments: graded 80% (n=20) vs one-call 15% (n=20)
  - unrestricted fill, graded/one-call: physical_characteristics 73%/91%, taste 9%/36%, texture 18%/32%, processing_methods 73%/82%, commercial_uses 82%/73%, similar_or_substitute_species 0%/9%, potential_buyer_segments 73%/14%
- **W29** (high): The claim verifier was validated on synthetic claims only; on real expert claims it drops true ones and keeps false ones.
  - not measured at baseline: this finding came from the review
- **R2** (high): Grounding must catch negated and inverted claims, not only borrowed ones.
  - shipped verifier (E5 + exact checks): held-out F1 1.0, false support 0.0, traps accepted 7/14
  - with the LLM judge (removed): held-out F1 0.6667, traps accepted 3/14
  - NLI + exact (candidate): held-out F1 0.9362, traps accepted 0/14
- **W13** (medium): Embedding similarity alone separates supported from borrowed claims poorly.
  - E5 claim-evidence cosine: supported mean 0.8562, borrowed mean 0.7931
  - supported min 0.8111 vs borrowed max 0.8299 (overlap: True)

#### 2. Fix (perbaikan)

| Fix | Track | Source | Change | Files |
|---|---|---|---|---|
| E11 W26 counts only cells with evidence | evaluation | W26 | An empty field with no evidence is the abstention R4 requires; the graded card is compared with the one-call card only where the species has evidence for the field. The rule is otherwise unchanged and the unrestricted rates stay in the evidence. | `evals/findings.py` |
| E12 Real-claims grounding set and fresh traps | evaluation | W26, R2, W29 | 345 atom-level (claim, cited chunk) pairs from three real-LLM agent runs, labelled with a written rubric by an AI assistant and pending human review, split by species; 23 fresh negation, antonym and scope traps used only as a test set. | `evals/datasets/grounding_real_claims.json`<br>`evals/claim_atoms.py` |
| E13 Verifier experiment on real claims | evaluation | W26, R2, W29, W13 | E5, E5 re-tuned, and NLI (mDeBERTa and two compact multilingual models) as verifier, veto and E5-or-NLI, under a pre-registered rule with a CPU budget per card. The rule's winner was not confirmed on the held-out test, so the shipped verifier stays. Grading atom by atom was measured and rejected. | `evals/experiment_verifier.py` |

#### 3. Re-test (hasil uji ulang)

| Metric | Baseline | After | Artifact |
|---|---:|---:|---|
| Real-claims recall, verifier in use | n/a | 0.6792 | `experiment_verifier.variants.A_e5_exact.test_real.recall` |
| Real-claims false support, verifier in use | n/a | 0.2295 | `experiment_verifier.variants.A_e5_exact.test_real.false_support_rate` |
| Fresh traps accepted, verifier in use | n/a | 11 | `experiment_verifier.variants.A_e5_exact.test_traps_accepted` |
| Fresh traps accepted, mDeBERTa NLI | n/a | 2 | `experiment_verifier.variants.C_mdeberta_exact.test_traps_accepted` |
| mDeBERTa NLI added CPU time per card (s) | n/a | 25.04 | `experiment_verifier.variants.C_mdeberta_exact.cpu_added_s_per_card_p95` |
| Card recall, whole-claim grading | n/a | 0.8686 | `experiment_verifier.card_grading_policy.whole_recall` |
| Card recall, atomic grading | n/a | 0.5705 | `experiment_verifier.card_grading_policy.atomic_recall` |
| Grounding F1, synthetic test split | 1 | 1 | `rag_eval.grounding.test.f1` |

| Finding | Baseline | After | Evidence after |
|---|---|---|---|
| W26 | open | open | fields where the graded card is emptier than the one-call card on cells with evidence: 4/6; physical_characteristics: graded 73% (n=22) vs one-call 91% (n=22); taste: graded 20% (n=10) vs one-call 80% (n=10); texture: graded 20% (n=10) vs one-call 70% (n=10); processing_methods: graded 78% (n=18) vs one-call 100% (n=18); commercial_uses: graded 90% (n=20) vs one-call 90% (n=20); similar_or_substitute_species: no species has evidence for it; potential_buyer_segments: graded 90% (n=20) vs one-call 15% (n=20); unrestricted fill, graded/one-call: physical_characteristics 73%/91%, taste 9%/36%, texture 9%/32%, processing_methods 64%/82%, commercial_uses 82%/82%, similar_or_substitute_species 0%/4%, potential_buyer_segments 82%/14% |
| W29 | not_measured | open | real expert claim atoms, held-out species (n=167): recall 0.6792, false support 0.2295 (14 of 61 unsupported atoms kept); synthetic test claims (n=48): recall 1.0, false support 0.0; labels: evals/datasets/grounding_real_claims.json, by an AI assistant, pending human review |
| R2 | partial | partial | verifier in use (A_e5_exact): fresh traps accepted 11/23, iteration-1 traps 7/14; fewest fresh traps: C_mdeberta_exact 2/23, +25.04 s per card on CPU (over budget) |
| W13 | partial | partial | E5 claim-evidence cosine: supported mean 0.8562, borrowed mean 0.7931; supported min 0.8111 vs borrowed max 0.8299 (overlap: True) |

**Reading the result, and what remains.** This cycle changed what we know, not the verifier. No variant within the CPU budget improved the held-out test without keeping more false claims, so E5 stays and the owner declined the trade. W29 is the new, measured statement of the problem. The labels are an AI assistant's and need a human pass before these numbers are cited outside the team. Next: NLI on a GPU, or a verifier trained on real claim atoms.

## Reproduce

```bash
HF_HUB_OFFLINE=1 .venv/Scripts/python.exe -m scripts.quality --label current            # tests, evals, findings, dashboard
HF_HUB_OFFLINE=1 .venv/Scripts/python.exe -m evals.cost_eval --label current --repeat 2   # real LLM, needs OPENCODE_GO_API_KEY
HF_HUB_OFFLINE=1 .venv/Scripts/python.exe -m evals.calibrate_grounding --label current
HF_HUB_OFFLINE=1 .venv/Scripts/python.exe -m evals.experiment_nli_grounding --label current --with-llm
HF_HUB_OFFLINE=1 .venv/Scripts/python.exe -m evals.experiment_verifier --label current   # needs the iter2-* claim runs
.venv/Scripts/python.exe -m evals.iteration --name iteration-2 --baseline iteration-1 --current current
```

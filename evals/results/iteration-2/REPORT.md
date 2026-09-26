# iteration-2: findings, fixes, re-test

Generated 2026-09-26T09:27:01+00:00 by `python -m evals.iteration` from `reports/iteration-1` and `reports/current`. Every number below comes from those run artifacts.

## Summary

- Tests: 71/71 passing at baseline, 84/84 now; RAG-module coverage 83.4% -> 86.5%.
- Findings at baseline: {'open': 9, 'partial': 3, 'resolved': 16, 'not_measured': 2, 'not_observed': 1, 'info': 2}; now: {'open': 9, 'partial': 2, 'resolved': 18, 'not_measured': 1, 'not_observed': 1, 'info': 2}.

## Fixes

### F19 (product): The synchronous card path runs the graded graph
Source: W19, W3. KnowledgeService (a manually declared catch, the publication fallback) calls orchestrator.grade_card, the same researcher-experts-critic-writer graph as the background job, so no card reaches a buyer without the per-claim critic. A blank key stays a mapped provider outage. The one-call generator wiring (deps.retriever, deps.generator) is removed from the app.
Files: `apps/main_api/services/knowledge.py`, `apps/main_api/services/orchestrator.py`, `apps/main_api/services/card_llm.py`, `apps/main_api/api/fish.py`, `apps/main_api/api/lots.py`, `apps/main_api/main.py`, `apps/main_api/ports.py`.
Findings: W19 partial -> resolved; W3 resolved -> resolved.
Metrics: Sync path grades claims (probe) False -> True; Blank key on the sync path (HTTP status) 502 -> 502.
Tests: `test_knowledge_api::test_manual_catch_card_is_graded_by_the_critic` absent -> passed; `test_knowledge_api::test_manual_catch_without_a_key_is_a_provider_outage` absent -> passed; `test_lot_publication::test_published_snapshot_is_the_graded_job_card` passed -> passed.

### F20 (product): Every card run leaves a stage trace; no handler fails silently
Source: W12. grade_card records evidence ids and distances, each expert's chunk ids, prompt hash, latency and tokens, the claim verdicts and per-stage timings; run_graph stores it on the job (knowledge_jobs.trace). The silent broad handlers log; the langgraph branch, never installed and never run, is deleted.
Files: `apps/main_api/services/orchestrator.py`, `apps/main_api/api/fish.py`, `apps/main_api/services/lots.py`, `apps/main_api/db/sql_repositories.py`, `apps/main_api/db/models.py`, `apps/main_api/contracts.py`, `alembic/versions/0007_knowledge_job_trace.py`.
Findings: W12 open -> resolved.
Metrics: Silent broad exception handlers 5 -> 0.
Tests: `test_knowledge_api::test_card_job_records_a_stage_trace` absent -> passed.

### E8 (evaluation): The W19 probe reads the syntax tree
Source: W19. sync_path_has_claim_critic is true only when knowledge.py's code calls grade_card or critic_node; the former token check would have passed on a comment.
Files: `evals/probes.py`.
Findings: W19 partial -> resolved.

### E9 (evaluation): W12 requires a stage trace
Source: W12. Resolved needs no silent handler and the stage-trace test passing (judge methodology 6.1).
Files: `evals/findings.py`.
Findings: W12 open -> resolved.

### E10 (evaluation): Cost runs measure the card the product publishes
Source: W19, W25, W26. The one-call generator becomes an eval-only reference path (one_call); 'published' metrics resolve to the path the product publishes; agent cards keep their trace and expert outputs.
Files: `evals/cost_eval.py`, `evals/findings.py`, `evals/dashboard.py`, `evals/model_compare.py`.
Findings: W19 partial -> resolved; W25 resolved -> resolved; W26 open -> open.
Metrics: Published card success, real LLM 1 -> 1; Cost per published card (USD) 0.0007 -> 0.0009.

### E11 (evaluation): W26 counts only cells with evidence
Source: W26. An empty field with no evidence is the abstention R4 requires; the graded card is compared with the one-call card only where the species has evidence for the field. The rule is otherwise unchanged and the unrestricted rates stay in the evidence.
Files: `evals/findings.py`.
Findings: W26 open -> open.

### E12 (evaluation): Real-claims grounding set and fresh traps
Source: W26, R2, W29. 345 atom-level (claim, cited chunk) pairs from three real-LLM agent runs, labelled with a written rubric by an AI assistant and pending human review, split by species; 23 fresh negation, antonym and scope traps used only as a test set.
Files: `evals/datasets/grounding_real_claims.json`, `evals/claim_atoms.py`.
Findings: W26 open -> open; R2 partial -> partial; W29 not_measured -> open.

### E13 (evaluation): Verifier experiment on real claims
Source: W26, R2, W29, W13. E5, E5 re-tuned, and NLI (mDeBERTa and two compact multilingual models) as verifier, veto and E5-or-NLI, under a pre-registered rule with a CPU budget per card. The rule's winner was not confirmed on the held-out test, so the shipped verifier stays. Grading atom by atom was measured and rejected.
Files: `evals/experiment_verifier.py`.
Findings: W26 open -> open; R2 partial -> partial; W29 not_measured -> open; W13 partial -> partial.
Metrics: Real-claims recall, verifier in use n/a -> 0.6792; Real-claims false support, verifier in use n/a -> 0.2295; Fresh traps accepted, verifier in use n/a -> 11.

## Target metrics

| Metric | Baseline | Current | Target |
|---|---|---|---|
| Test suite pass rate | 1 | 1 | 1 |
| Grounding F1 (test split) | 1 | 1 | 0.85 |
| Grounding false support | 0.0204 | 0.0204 | 0.1 |
| True-claim retention (scripted) | 1 | 1 | 0.8 |
| Fenced-JSON job success | 1 | 1 | 1 |
| LLM rounds / card | 1 | 1 | 2.5 |
| Repeat card latency (ms) | 11.3 | 10.8 | 50 |
| Embedder cold load (s) | 15.38 | 14.72 | n/a |
| Scoped retrieval MRR | 0.5087 | 0.5087 | 0.7 |
| Scoped recall@6 | 0.9886 | 0.9886 | 1 |
| Agent card success, real LLM | 1 | 1 | 0.9 |
| Published card success, real LLM | 1 | 1 | 1 |
| Agent card p50 wall time (s) | 4.03 | 4.45 | 6 |
| Cost per published card (USD) | 0.0007 | 0.0009 | n/a |
| Cost per agent card (USD) | 0.0009 | 0.0009 | n/a |
| Species ID: field accuracy | 0.8178 | 0.8178 | 0.9 |
| Species ID: wrong and confident (field) | 0.1512 | 0.1512 | 0.05 |
| Species ID: OOD images accepted | 1 | 1 | 0.1 |
| Species ID: background shortcut rate | 0.4245 | 0.4245 | 0.18 |

## Experiments

- **Passage prefix (W8):** hypothesis supported: False. Per-category MRR deltas {'identity': 0.0, 'physical_characteristics': 0.0, 'taste_texture': 0.0, 'processing_methods': 0.0, 'commercial_uses': 0.0}.
- **Calibration (W1, W13):** dev-chosen tau 0.805, shipped 0.805; held-out F1 1.0, false support 0.0.
- **Verifier comparison (R2):** rule: highest dev F1 at dev FSR <= 0.10; within 0.02, fewer traps accepted. Winner A_e5_exact. NLI costs 1165.8 ms per pair on CPU.
  - A_e5_exact: held-out F1 1.0, false support 0.0, traps accepted 7/14
  - B_nli: held-out F1 0.9167, false support 0.0833, traps accepted 0/14
  - C_nli_exact: held-out F1 0.9362, false support 0.0417, traps accepted 0/14
- **Verifier on real claims (W26, R2, W29):** rule: eligible (CPU <= 5 s/card p95); highest dev F1 at dev FSR <= 0.10; within 0.02, fewer dev traps accepted, then no NLI. Winner H_minilm_l12_e5_or_nli_veto; confirmed on test: False; in use: A_e5_exact.
  - A_e5_exact: test F1 0.8 (real recall 0.6792, real false support 0.2295), fresh traps 11/23, +0.0 s/card CPU
  - A2_e5_exact_retuned: test F1 0.8 (real recall 0.6792, real false support 0.2295), fresh traps 11/23, +0.0 s/card CPU
  - C_mdeberta_exact: test F1 0.8627 (real recall 0.8396, real false support 0.2295), fresh traps 2/23, +25.04 s/card CPU
  - V_mdeberta_veto: test F1 0.7848 (real recall 0.6509, real false support 0.2295), fresh traps 3/23, +25.04 s/card CPU
  - H_mdeberta_e5_or_nli_veto: test F1 0.8873 (real recall 0.9245, real false support 0.3607), fresh traps 2/23, +25.04 s/card CPU
  - C_minilm_l6_exact: test F1 0.6108 (real recall 0.434, real false support 0.1803), fresh traps 6/23, +0.24 s/card CPU
  - V_minilm_l6_veto: test F1 0.7027 (real recall 0.5472, real false support 0.2295), fresh traps 9/23, +0.24 s/card CPU
  - H_minilm_l6_e5_or_nli_veto: test F1 0.7355 (real recall 0.6509, real false support 0.377), fresh traps 10/23, +0.24 s/card CPU
  - C_minilm_l12_exact: test F1 0.8101 (real recall 0.6887, real false support 0.1639), fresh traps 6/23, +0.47 s/card CPU
  - V_minilm_l12_veto: test F1 0.7639 (real recall 0.6226, real false support 0.2295), fresh traps 8/23, +0.47 s/card CPU
  - H_minilm_l12_e5_or_nli_veto: test F1 0.8095 (real recall 0.7453, real false support 0.3115), fresh traps 8/23, +0.47 s/card CPU
  - card grading: whole-claim precision 0.9313, recall 0.8686; atomic precision 0.9368, recall 0.5705

## Corpus gaps (R4)

22 of 66 species x category cells have no evidence: {'substitutes': 11, 'taste_texture': 6, 'physical_characteristics': 2, 'processing_methods': 2, 'commercial_uses': 1}. Filling them: research agents -> candidate chunk with a verbatim source quote -> scripts.corpus_pipeline collect -> human review -> approve (signed) -> ingest.

## Still open

| ID | Severity | Status | Worked on | Finding |
|---|---|---|---|---|
| R2 | high | partial | yes | Grounding must catch negated and inverted claims, not only borrowed ones |
| W13 | medium | partial | yes | Embedding similarity alone separates supported from borrowed claims poorly |
| CV1 | high | open | no | Field photos are misidentified far more often than clean photos |
| CV2 | high | open | no | Confidently wrong on field photos (confidence >= 0.9) |
| CV3 | high | open | no | Abstain threshold 0.0 accepts every unknown fish and non-fish image |
| W29 | high | open | yes | The claim verifier was validated on synthetic claims only; on real expert claims it drops true ones and keeps false ones |
| CV4 | medium | open | no | Background shortcut: species predicted with the fish erased |
| CV5 | medium | open | no | Accuracy drop under image corruption |
| W8 | medium | open | no | Ranking follows the species name, not the requested category |
| W11 | medium | open | no | Categories with no evidence at all |
| W26 | medium | open | yes | Critic-guarded agent cards are emptier than published cards |

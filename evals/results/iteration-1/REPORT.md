# iteration-1: findings, fixes, re-test

Generated 2026-09-26T07:51:15+00:00 by `python -m evals.iteration` from `reports/baseline` and `reports/current`. Every number below comes from those run artifacts.

## Summary

- Tests: 35/50 passing at baseline, 71/71 now; RAG-module coverage 80.5% -> 83.4%.
- Findings at baseline: {'open': 28, 'not_measured': 1, 'not_observed': 1, 'info': 2}; now: {'open': 9, 'partial': 3, 'resolved': 16, 'not_measured': 1, 'not_observed': 1, 'info': 2}.

## Fixes

### F1 (product): OpenCode Go session header and user agent
Source: W20. make_opencode_go_llm sends x-opencode-session (one id per card) and a descriptive User-Agent.
Files: `apps/main_api/services/generation.py`.
Findings: W20 open -> resolved.
Metrics: Agent card success, real LLM 0 -> 1; Published card success, real LLM 0.9545 -> 1.
Tests: `test_provider_client::test_client_sends_session_and_user_agent` absent -> passed; `test_provider_client::test_client_without_session_gets_a_fresh_one` absent -> passed.

### F2 (product): Read Responses-API content blocks and fenced JSON
Source: W21, W4. llm_output.reply_text/reply_json read every reply; experts, critic and judge use them.
Files: `apps/main_api/services/llm_output.py`, `apps/main_api/services/orchestrator.py`.
Findings: W21 open -> resolved; W4 open -> resolved.
Metrics: Fenced-JSON job success 0 -> 1; Agent card success, real LLM 0 -> 1.
Tests: `test_llm_output::test_responses_api_content_blocks` absent -> passed; `test_orchestrator_nodes::test_expert_parses_markdown_fenced_json` failed -> passed; `test_orchestrator_iteration::test_expert_reads_responses_api_blocks` absent -> passed.

### F3 (product): Cross-lingual claim verifier replaces lexical overlap
Source: W1, W13, R2, W26. Critic grades each claim with E5 symmetric cosine >= tau (0.805, chosen on the dev split) AND every number and taxon (family names, supported binomials) in the claim appearing in the cited chunk (the LLM judge pass is off, see F18).
Files: `apps/main_api/services/orchestrator.py`, `evals/calibrate_grounding.py`.
Findings: W1 open -> resolved; W13 open -> partial; R2 not_measured -> partial; W26 open -> open.
Metrics: Grounding F1 (test split) 0.7907 -> 1; Grounding false support (all pairs) 0.102 -> 0.0204; True-claim retention (scripted) 0.56 -> 1.
Tests: `test_grounding_and_pipeline::test_indonesian_claim_is_grounded_in_english_evidence` failed -> passed; `test_grounding_and_pipeline::test_shared_token_does_not_ground_a_borrowed_claim` failed -> passed; `test_grounding_and_pipeline::test_grounding_test_split_f1` failed -> passed.
Note: Tau sits in a narrow band (0.795 admits 12% false support on dev): recalibrate when the corpus changes.

### F4 (product): Experts and the critic judge read whole chunks
Source: W14, R3. Removed both 300-character cuts (expert payload and critic LLM pass).
Files: `apps/main_api/services/orchestrator.py`.
Findings: W14 open -> resolved.
Tests: `test_orchestrator_nodes::test_expert_sees_the_whole_chunk` failed -> passed; `test_orchestrator_iteration::test_critic_llm_pass_reads_the_whole_chunk` absent -> passed.

### F5 (product): The whole species slice is the card's evidence
Source: R1, W9, W8. VerifiedRetriever.card_evidence returns every verified chunk of the species when there are at most 20, ordered by category; ranked category-first selection only above that.
Files: `apps/main_api/services/retrieval.py`, `apps/main_api/services/knowledge.py`, `apps/main_api/services/orchestrator.py`.
Findings: R1 open -> resolved; W9 open -> resolved; W8 open -> open.
Metrics: Card evidence completeness (min) n/a -> 1.
Note: W8 (name-biased ranking) no longer decides what a card sees; the ranking itself is unchanged, and a canonical-name passage prefix did not improve it (experiment_passage_prefix).

### F6 (product): Fields are grounded only by their own evidence category; gaps are named
Source: R4, W11. Critic accepts a citation only from the field's categories (substitutes need substitute evidence); experts with no evidence are not called; the card lists fields with no verified evidence.
Files: `apps/main_api/services/orchestrator.py`, `evals/corpus_gaps.py`.
Findings: W11 open -> open.
Tests: `test_orchestrator_iteration::test_substitute_claim_needs_substitute_evidence` absent -> passed; `test_orchestrator_iteration::test_expert_without_evidence_is_not_called` absent -> passed; `test_orchestrator_iteration::test_missing_evidence_is_named_on_the_card` absent -> passed.
Note: The data gap itself (22 of 66 cells, all 11 substitute cells) needs new sourced chunks and human approval: see corpus_gaps.json.

### F7 (product): Two LLM rounds per card
Source: W5. Removed the researcher's LLM sub-query and the writer's post-verification polish.
Files: `apps/main_api/services/orchestrator.py`.
Findings: W5 open -> resolved.
Metrics: LLM calls per card (scripted) 6.82 -> 3.27; Agent card p50 wall time, real LLM (s) 6.04 -> 4.03.
Tests: `test_grounding_and_pipeline::test_card_needs_at_most_two_sequential_llm_rounds` failed -> passed.

### F8 (product): Card cache keyed by the evidence
Source: W6. Species + evidence ids and contents + model + pipeline version; invalidates itself on corpus change.
Files: `apps/main_api/services/orchestrator.py`.
Findings: W6 open -> resolved.
Metrics: Repeat card latency (ms) 832.7 -> 11.3.
Tests: `test_grounding_and_pipeline::test_repeat_card_is_served_from_cache` failed -> passed; `test_orchestrator_iteration::test_cache_key_follows_the_evidence` absent -> passed.

### F9 (product): Embedder warmed at startup
Source: W7. Lifespan starts a background warmup, so the first request does not load the model.
Files: `apps/main_api/main.py`, `apps/main_api/services/embeddings.py`.
Findings: W7 open -> resolved.
Tests: `test_knowledge_api::test_embedder_is_warmed_at_startup` failed -> passed.

### F10 (product): Chunk limit fits the E5 window
Source: W10. chunk_candidate max_tokens 600 -> 480.
Files: `apps/main_api/services/chunking.py`.
Findings: W10 open -> resolved.
Tests: `test_chunking::test_default_chunk_limit_fits_the_e5_window` failed -> passed.

### F11 (product): LLM is an injectable port; scheduling errors are logged
Source: W2, W12. AppDependencies.llm; verify builds the per-card client or uses the port; broad excepts log.
Files: `apps/main_api/ports.py`, `apps/main_api/api/fish.py`, `apps/main_api/services/orchestrator.py`.
Findings: W2 open -> resolved; W12 open -> open.
Tests: `test_knowledge_api::test_verification_schedules_a_job_that_completes` failed -> passed; `test_operator_flow::test_photo_to_grounded_card` failed -> passed.

### F12 (product): A blank key is a provider outage, not a crash
Source: W3. KnowledgeGenerator maps client-construction ValueError to OpenCodeUnavailable (HTTP 502).
Files: `apps/main_api/services/generation.py`.
Findings: W3 open -> resolved.
Tests: `test_provider_client::test_blank_key_is_a_provider_outage_not_a_crash` absent -> passed.

### F13 (product): Publication freezes the graded card
Source: W19. LotService publishes the completed job card for the verified species; the one-call path is only a fallback.
Files: `apps/main_api/services/lots.py`, `apps/main_api/api/lots.py`.
Findings: W19 open -> partial.
Tests: `test_lot_publication::test_published_snapshot_is_the_graded_job_card` absent -> passed; `test_lot_publication::test_graded_card_for_another_species_is_not_published` absent -> passed.

### F14 (product): Nothing groundable yields an honest card, not a failed job
Source: W24. Writer returns an empty card with a limitation when experts ran but no claim was grounded; an all-expert outage still fails.
Files: `apps/main_api/services/orchestrator.py`.
Findings: W24 open -> resolved.
Tests: `test_orchestrator_iteration::test_no_groundable_claim_gives_a_limitation_card` absent -> passed; `test_orchestrator_iteration::test_every_expert_failing_is_an_error_not_an_empty_card` absent -> passed; `test_grounding_and_pipeline::test_llm_outage_fails_the_job_cleanly` passed -> passed.

### F15 (product): Expert prompts require Indonesian
Source: W22. Every expert prompt states the output language.
Files: `apps/main_api/services/orchestrator.py`.
Findings: W22 not_observed -> not_observed.
Tests: `test_orchestrator_iteration::test_expert_prompt_requires_indonesian` absent -> passed.

### F16 (product): Dev taxonomy agrees with the corpus
Source: W18. Synthetic taxonomy fixture uses the corpus identity binomials for 4 species.
Files: `scripts/make_synthetic_taxonomy.py`.
Findings: W18 open -> resolved.

### F17 (product): Quality dashboard served by the API
Source: baseline e2e spec. GET /quality and /api/v1/quality/summary serve generated artifacts.
Files: `apps/main_api/api/quality.py`, `apps/main_api/main.py`.
Tests: `test_operator_flow::test_quality_dashboard_is_served` failed -> passed.
Note: Serves evaluation results; gate behind operator auth before a public deployment.

### F18 (product): Critic's LLM judge pass removed
Source: R2, W5. USE_LLM_JUDGE = False: measured with gpt-5.6-luna, the downgrade-only judge halved held-out recall (1.0 -> 0.5) and still accepted 3 of 14 traps; the verifier alone scores higher on the dev split.
Files: `apps/main_api/services/orchestrator.py`.
Findings: R2 not_measured -> partial; W5 open -> resolved.
Metrics: LLM rounds per card (timestamps) n/a -> 1.
Tests: `test_orchestrator_iteration::test_llm_judge_pass_is_off_in_production` absent -> passed.
Note: R2 stays partial: the shipped verifier accepts 7 of 14 negation/scope traps. Multilingual NLI rejects all 14 but costs ~1.1 s per pair on CPU and loses recall (0.917); confirm on a fresh trap set, on GPU/ONNX, before adopting.

### E1 (evaluation): LLM rounds counted from call timestamps
Source: W5. The wall-time/delay proxy counted verifier compute as rounds; overlapping calls now form one round. The old proxy is still reported.
Files: `evals/fakes.py`, `evals/pipeline_eval.py`.
Findings: W5 open -> resolved.
Metrics: LLM rounds per card (timestamps) n/a -> 1.

### E2 (evaluation): Real-LLM runs use the production client
Source: W20. cost_eval drives make_opencode_go_llm instead of adding the header itself.
Files: `evals/cost_eval.py`.
Findings: W20 open -> resolved.

### E3 (evaluation): Threshold calibration on the dev split
Source: W1, W13. calibrate_grounding sweeps tau for the shipped rule and reports the held-out split.
Files: `evals/calibrate_grounding.py`.
Findings: W1 open -> resolved; W13 open -> partial.
Metrics: Calibrated tau n/a -> 0.805.

### E4 (evaluation): Negation/scope trap set and verifier comparison
Source: R2. 14 traps; E5+exact vs multilingual NLI vs the production LLM judge, pre-registered selection rule.
Files: `evals/datasets/grounding_traps.json`, `evals/experiment_nli_grounding.py`.
Findings: R2 not_measured -> partial.

### E5 (evaluation): Passage-prefix experiment
Source: W8. Tests whether a canonical-name prefix fixes within-species ranking.
Files: `evals/experiment_passage_prefix.py`.
Findings: W8 open -> open.

### E6 (evaluation): Corpus gap task list
Source: R4, W11. Lists every species x category cell without evidence for the data team.
Files: `evals/corpus_gaps.py`.
Findings: W11 open -> open.

### E7 (evaluation): Registry covers the review critique
Source: R1, R2. Computed R1/R2 checks; 'partial' status for mitigated findings; W9/W19/W13 checks follow the new design.
Files: `evals/findings.py`, `evals/probes.py`.
Findings: R1 open -> resolved; R2 not_measured -> partial.

## Target metrics

| Metric | Baseline | Current | Target |
|---|---|---|---|
| Test suite pass rate | 0.7 | 1 | 1 |
| Grounding F1 (test split) | 0.7907 | 1 | 0.85 |
| Grounding false support | 0.102 | 0.0204 | 0.1 |
| True-claim retention (scripted) | 0.56 | 1 | 0.8 |
| Fenced-JSON job success | 0 | 1 | 1 |
| LLM rounds / card | n/a | 1 | 2.5 |
| Repeat card latency (ms) | 832.7 | 11.3 | 50 |
| Embedder cold load (s) | 10.97 | 15.38 | n/a |
| Scoped retrieval MRR | 0.5087 | 0.5087 | 0.7 |
| Scoped recall@6 | 0.9886 | 0.9886 | 1 |
| Agent card success, real LLM | 0 | 1 | 0.9 |
| Published card success, real LLM | 0.9545 | 1 | 1 |
| Agent card p50 wall time (s) | 11.55 | 4.03 | 6 |
| Cost per published card (USD) | 0.0006 | 0.0007 | n/a |
| Cost per agent card (USD) | 0.0014 | 0.0009 | n/a |
| Species ID: field accuracy | 0.8178 | 0.8178 | 0.9 |
| Species ID: wrong and confident (field) | 0.1512 | 0.1512 | 0.05 |
| Species ID: OOD images accepted | 1 | 1 | 0.1 |
| Species ID: background shortcut rate | 0.4245 | 0.4245 | 0.18 |

## Experiments

- **Passage prefix (W8):** hypothesis supported: False. Per-category MRR deltas {'identity': 0.0, 'physical_characteristics': 0.0, 'taste_texture': 0.0, 'processing_methods': 0.0, 'commercial_uses': 0.0}.
- **Calibration (W1, W13):** dev-chosen tau 0.805, shipped 0.805; held-out F1 1.0, false support 0.0.
- **Verifier comparison (R2):** rule: highest dev F1 at dev FSR <= 0.10; within 0.02, fewer traps accepted. Winner A_e5_exact. NLI costs 1095.4 ms per pair on CPU.
  - A_e5_exact: held-out F1 1.0, false support 0.0, traps accepted 7/14
  - B_nli: held-out F1 0.9167, false support 0.0833, traps accepted 0/14
  - C_nli_exact: held-out F1 0.9362, false support 0.0417, traps accepted 0/14
  - D_e5_exact_llm_judge: held-out F1 0.6667, false support 0.0, traps accepted 3/14

## Corpus gaps (R4)

22 of 66 species x category cells have no evidence: {'substitutes': 11, 'taste_texture': 6, 'physical_characteristics': 2, 'processing_methods': 2, 'commercial_uses': 1}. Filling them: research agents -> candidate chunk with a verbatim source quote -> scripts.corpus_pipeline collect -> human review -> approve (signed) -> ingest.

## Still open

| ID | Severity | Status | Worked on | Finding |
|---|---|---|---|---|
| R2 | high | partial | yes | Grounding must catch negated and inverted claims, not only borrowed ones |
| W19 | high | partial | yes | The published card (QR page, buyer matching) bypasses the per-claim critic |
| W13 | medium | partial | yes | Embedding similarity alone separates supported from borrowed claims poorly |
| CV1 | high | open | no | Field photos are misidentified far more often than clean photos |
| CV2 | high | open | no | Confidently wrong on field photos (confidence >= 0.9) |
| CV3 | high | open | no | Abstain threshold 0.0 accepts every unknown fish and non-fish image |
| CV4 | medium | open | no | Background shortcut: species predicted with the fish erased |
| CV5 | medium | open | no | Accuracy drop under image corruption |
| W8 | medium | open | yes | Ranking follows the species name, not the requested category |
| W11 | medium | open | yes | Categories with no evidence at all |
| W12 | medium | open | yes | Broad exception handlers swallow errors silently |
| W26 | medium | open | yes | Critic-guarded agent cards are emptier than published cards |

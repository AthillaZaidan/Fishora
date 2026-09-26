"""Iteration record: every change, its source finding, and its re-test result.

    python -m evals.iteration --name iteration-1 --baseline baseline --current current

The FIXES registry below is the only hand-written part: each entry names the
findings it answers (W* from the baseline registry, R* from the review
critique, CV* from species ID), the files it touched, and the tests and
metrics that verify it. Every number in the report is read from the
Python-generated artifacts of the two runs. Output:

    evals/results/<name>/REPORT.md       findings -> fixes -> re-test, for people
    evals/results/<name>/iteration.json  the same, for tools
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from evals.findings import evaluate, get, improvements, load_artifacts
from evals.run import ROOT, _jsonable


@dataclass(frozen=True)
class Fix:
    id: str
    track: str  # product | evaluation
    title: str
    sources: tuple[str, ...]
    change: str
    files: tuple[str, ...]
    tests: tuple[str, ...] = ()
    metrics: tuple[str, ...] = ()  # "<label>|<artifact.path>"
    note: str = ""


FIXES: tuple[Fix, ...] = (
    Fix("F1", "product", "OpenCode Go session header and user agent", ("W20",),
        "make_opencode_go_llm sends x-opencode-session (one id per card) and a descriptive User-Agent.",
        ("apps/main_api/services/generation.py",),
        ("test_provider_client::test_client_sends_session_and_user_agent",
         "test_provider_client::test_client_without_session_gets_a_fresh_one"),
        ("Agent card success, real LLM|cost_eval.summary.by_path.agent.success_rate",
         "Published card success, real LLM|cost_eval.summary.by_path.published.success_rate")),
    Fix("F2", "product", "Read Responses-API content blocks and fenced JSON", ("W21", "W4"),
        "llm_output.reply_text/reply_json read every reply; experts, critic and judge use them.",
        ("apps/main_api/services/llm_output.py", "apps/main_api/services/orchestrator.py"),
        ("test_llm_output::test_responses_api_content_blocks", "test_orchestrator_nodes::test_expert_parses_markdown_fenced_json",
         "test_orchestrator_iteration::test_expert_reads_responses_api_blocks"),
        ("Fenced-JSON job success|rag_eval.pipeline.fenced.job_success_rate",
         "Agent card success, real LLM|cost_eval.summary.by_path.agent.success_rate")),
    Fix("F3", "product", "Cross-lingual claim verifier replaces lexical overlap", ("W1", "W13", "R2", "W26"),
        "Critic grades each claim with E5 symmetric cosine >= tau (0.805, chosen on the dev split) AND every "
        "number and taxon (family names, supported binomials) in the claim appearing in the cited chunk "
        "(the LLM judge pass is off, see F18).",
        ("apps/main_api/services/orchestrator.py", "evals/calibrate_grounding.py"),
        ("test_grounding_and_pipeline::test_indonesian_claim_is_grounded_in_english_evidence",
         "test_grounding_and_pipeline::test_shared_token_does_not_ground_a_borrowed_claim",
         "test_grounding_and_pipeline::test_grounding_test_split_f1"),
        ("Grounding F1 (test split)|rag_eval.grounding.test.f1",
         "Grounding false support (all pairs)|rag_eval.grounding.all.false_support_rate",
         "True-claim retention (scripted)|rag_eval.pipeline.plain.claim_retention"),
        "Tau sits in a narrow band (0.795 admits 12% false support on dev): recalibrate when the corpus changes."),
    Fix("F4", "product", "Experts and the critic judge read whole chunks", ("W14", "R3"),
        "Removed both 300-character cuts (expert payload and critic LLM pass).",
        ("apps/main_api/services/orchestrator.py",),
        ("test_orchestrator_nodes::test_expert_sees_the_whole_chunk",
         "test_orchestrator_iteration::test_critic_llm_pass_reads_the_whole_chunk")),
    Fix("F5", "product", "The whole species slice is the card's evidence", ("R1", "W9", "W8"),
        "VerifiedRetriever.card_evidence returns every verified chunk of the species when there are at most "
        "20, ordered by category; ranked category-first selection only above that.",
        ("apps/main_api/services/retrieval.py", "apps/main_api/services/knowledge.py",
         "apps/main_api/services/orchestrator.py"),
        (),
        ("Card evidence completeness (min)|rag_eval.retrieval.card_evidence_completeness.min",),
        "W8 (name-biased ranking) no longer decides what a card sees; the ranking itself is unchanged, and a "
        "canonical-name passage prefix did not improve it (experiment_passage_prefix)."),
    Fix("F6", "product", "Fields are grounded only by their own evidence category; gaps are named", ("R4", "W11"),
        "Critic accepts a citation only from the field's categories (substitutes need substitute evidence); "
        "experts with no evidence are not called; the card lists fields with no verified evidence.",
        ("apps/main_api/services/orchestrator.py", "evals/corpus_gaps.py"),
        ("test_orchestrator_iteration::test_substitute_claim_needs_substitute_evidence",
         "test_orchestrator_iteration::test_expert_without_evidence_is_not_called",
         "test_orchestrator_iteration::test_missing_evidence_is_named_on_the_card"),
        (),
        "The data gap itself (22 of 66 cells, all 11 substitute cells) needs new sourced chunks and human "
        "approval: see corpus_gaps.json."),
    Fix("F7", "product", "Two LLM rounds per card", ("W5",),
        "Removed the researcher's LLM sub-query and the writer's post-verification polish.",
        ("apps/main_api/services/orchestrator.py",),
        ("test_grounding_and_pipeline::test_card_needs_at_most_two_sequential_llm_rounds",),
        ("LLM calls per card (scripted)|rag_eval.pipeline.plain.llm_calls_per_card",
         "Agent card p50 wall time, real LLM (s)|cost_eval.summary.by_path.agent.wall_s_per_card.p50")),
    Fix("F8", "product", "Card cache keyed by the evidence", ("W6",),
        "Species + evidence ids and contents + model + pipeline version; invalidates itself on corpus change.",
        ("apps/main_api/services/orchestrator.py",),
        ("test_grounding_and_pipeline::test_repeat_card_is_served_from_cache",
         "test_orchestrator_iteration::test_cache_key_follows_the_evidence"),
        ("Repeat card latency (ms)|rag_eval.pipeline.plain.repeat_latency_ms_mean",)),
    Fix("F9", "product", "Embedder warmed at startup", ("W7",),
        "Lifespan starts a background warmup, so the first request does not load the model.",
        ("apps/main_api/main.py", "apps/main_api/services/embeddings.py"),
        ("test_knowledge_api::test_embedder_is_warmed_at_startup",)),
    Fix("F10", "product", "Chunk limit fits the E5 window", ("W10",),
        "chunk_candidate max_tokens 600 -> 480.", ("apps/main_api/services/chunking.py",),
        ("test_chunking::test_default_chunk_limit_fits_the_e5_window",)),
    Fix("F11", "product", "LLM is an injectable port; scheduling errors are logged", ("W2", "W12"),
        "AppDependencies.llm; verify builds the per-card client or uses the port; broad excepts log.",
        ("apps/main_api/ports.py", "apps/main_api/api/fish.py", "apps/main_api/services/orchestrator.py"),
        ("test_knowledge_api::test_verification_schedules_a_job_that_completes",
         "test_operator_flow::test_photo_to_grounded_card")),
    Fix("F12", "product", "A blank key is a provider outage, not a crash", ("W3",),
        "KnowledgeGenerator maps client-construction ValueError to OpenCodeUnavailable (HTTP 502).",
        ("apps/main_api/services/generation.py",),
        ("test_provider_client::test_blank_key_is_a_provider_outage_not_a_crash",)),
    Fix("F13", "product", "Publication freezes the graded card", ("W19",),
        "LotService publishes the completed job card for the verified species; the one-call path is only a fallback.",
        ("apps/main_api/services/lots.py", "apps/main_api/api/lots.py"),
        ("test_lot_publication::test_published_snapshot_is_the_graded_job_card",
         "test_lot_publication::test_graded_card_for_another_species_is_not_published")),
    Fix("F14", "product", "Nothing groundable yields an honest card, not a failed job", ("W24",),
        "Writer returns an empty card with a limitation when experts ran but no claim was grounded; an all-expert outage still fails.",
        ("apps/main_api/services/orchestrator.py",),
        ("test_orchestrator_iteration::test_no_groundable_claim_gives_a_limitation_card",
         "test_orchestrator_iteration::test_every_expert_failing_is_an_error_not_an_empty_card",
         "test_grounding_and_pipeline::test_llm_outage_fails_the_job_cleanly")),
    Fix("F15", "product", "Expert prompts require Indonesian", ("W22",),
        "Every expert prompt states the output language.", ("apps/main_api/services/orchestrator.py",),
        ("test_orchestrator_iteration::test_expert_prompt_requires_indonesian",)),
    Fix("F16", "product", "Dev taxonomy agrees with the corpus", ("W18",),
        "Synthetic taxonomy fixture uses the corpus identity binomials for 4 species.",
        ("scripts/make_synthetic_taxonomy.py",)),
    Fix("F17", "product", "Quality dashboard served by the API", ("baseline e2e spec",),
        "GET /quality and /api/v1/quality/summary serve generated artifacts.",
        ("apps/main_api/api/quality.py", "apps/main_api/main.py"),
        ("test_operator_flow::test_quality_dashboard_is_served",),
        note="Serves evaluation results; gate behind operator auth before a public deployment."),
    Fix("F18", "product", "Critic's LLM judge pass removed", ("R2", "W5"),
        "USE_LLM_JUDGE = False: measured with gpt-5.6-luna, the downgrade-only judge halved held-out recall "
        "(1.0 -> 0.5) and still accepted 3 of 14 traps; the verifier alone scores higher on the dev split.",
        ("apps/main_api/services/orchestrator.py",),
        ("test_orchestrator_iteration::test_llm_judge_pass_is_off_in_production",),
        ("LLM rounds per card (timestamps)|rag_eval.pipeline.plain.llm_rounds",),
        "R2 stays partial: the shipped verifier accepts 7 of 14 negation/scope traps. Multilingual NLI rejects "
        "all 14 but costs ~1.1 s per pair on CPU and loses recall (0.917); confirm on a fresh trap set, on GPU/ONNX, "
        "before adopting."),
    Fix("E1", "evaluation", "LLM rounds counted from call timestamps", ("W5",),
        "The wall-time/delay proxy counted verifier compute as rounds; overlapping calls now form one round. "
        "The old proxy is still reported.", ("evals/fakes.py", "evals/pipeline_eval.py"),
        metrics=("LLM rounds per card (timestamps)|rag_eval.pipeline.plain.llm_rounds",)),
    Fix("E2", "evaluation", "Real-LLM runs use the production client", ("W20",),
        "cost_eval drives make_opencode_go_llm instead of adding the header itself.", ("evals/cost_eval.py",)),
    Fix("E3", "evaluation", "Threshold calibration on the dev split", ("W1", "W13"),
        "calibrate_grounding sweeps tau for the shipped rule and reports the held-out split.",
        ("evals/calibrate_grounding.py",),
        metrics=("Calibrated tau|grounding_calibration.chosen.tau",)),
    Fix("E4", "evaluation", "Negation/scope trap set and verifier comparison", ("R2",),
        "14 traps; E5+exact vs multilingual NLI vs the production LLM judge, pre-registered selection rule.",
        ("evals/datasets/grounding_traps.json", "evals/experiment_nli_grounding.py")),
    Fix("E5", "evaluation", "Passage-prefix experiment", ("W8",),
        "Tests whether a canonical-name prefix fixes within-species ranking.", ("evals/experiment_passage_prefix.py",)),
    Fix("E6", "evaluation", "Corpus gap task list", ("R4", "W11"),
        "Lists every species x category cell without evidence for the data team.", ("evals/corpus_gaps.py",)),
    Fix("E7", "evaluation", "Registry covers the review critique", ("R1", "R2"),
        "Computed R1/R2 checks; 'partial' status for mitigated findings; W9/W19/W13 checks follow the new design.",
        ("evals/findings.py", "evals/probes.py")),
)


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _metric(artifacts: dict, path: str):
    from evals.findings import metric_value
    return metric_value(artifacts, path)


def build(name: str, baseline: str, current: str) -> dict:
    base, cur = load_artifacts(baseline), load_artifacts(current)
    f_base = {f["id"]: f for f in evaluate(baseline)}
    f_cur = {f["id"]: f for f in evaluate(current)}
    tests_now = {t["name"]: t["status"] for t in get(cur, "tests.tests", []) or []}
    tests_then = {t["name"]: t["status"] for t in get(base, "tests.tests", []) or []}
    fixes = []
    for fx in FIXES:
        fixes.append({
            "id": fx.id, "track": fx.track, "title": fx.title, "sources": list(fx.sources), "change": fx.change,
            "files": list(fx.files), "note": fx.note,
            "findings": [{"id": s, "before": f_base.get(s, {}).get("status"), "after": f_cur.get(s, {}).get("status")}
                         for s in fx.sources if s in f_base or s in f_cur],
            "tests": [{"test": t, "before": tests_then.get(t.split("::")[-1], "absent"),
                       "after": tests_now.get(t.split("::")[-1], "absent")} for t in fx.tests],
            "metrics": [{"metric": m.split("|")[0], "path": m.split("|")[1],
                         "before": _metric(base, m.split("|")[1]), "after": _metric(cur, m.split("|")[1])}
                        for m in fx.metrics],
        })
    cited = {s for fx in FIXES for s in fx.sources}
    carried = [f for f in f_cur.values() if f["status"] in ("open", "partial") ]
    return {
        "name": name, "baseline": baseline, "current": current,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tests": {"baseline": get(base, "tests.totals"), "current": get(cur, "tests.totals"),
                  "coverage_rag": [get(base, "tests.coverage.rag_percent"), get(cur, "tests.coverage.rag_percent")]},
        "findings_status": {"baseline": _count(f_base), "current": _count(f_cur)},
        "fixes": fixes,
        "improvements": improvements(baseline, current),
        "experiments": {
            "passage_prefix": get(cur, "experiment_passage_prefix"),
            "verifier_comparison": get(cur, "experiment_nli_grounding"),
            "calibration": get(cur, "grounding_calibration"),
        },
        "corpus_gaps": get(cur, "corpus_gaps"),
        "still_open": [{"id": f["id"], "severity": f["severity"], "status": f["status"], "title": f["title"],
                        "addressed_this_iteration": f["id"] in cited, "evidence": f["evidence"][:2]} for f in carried],
    }


def _count(findings: dict) -> dict:
    out: dict[str, int] = {}
    for f in findings.values():
        out[f["status"]] = out.get(f["status"], 0) + 1
    return out


def render_markdown(r: dict) -> str:
    L = [f"# {r['name']}: findings, fixes, re-test",
         "",
         f"Generated {r['generated_at']} by `python -m evals.iteration` from `reports/{r['baseline']}` and "
         f"`reports/{r['current']}`. Every number below comes from those run artifacts.",
         "",
         "## Summary",
         "",
         f"- Tests: {r['tests']['baseline'].get('passed')}/{r['tests']['baseline'].get('total')} passing at baseline, "
         f"{r['tests']['current'].get('passed')}/{r['tests']['current'].get('total')} now; RAG-module coverage "
         f"{r['tests']['coverage_rag'][0]}% -> {r['tests']['coverage_rag'][1]}%.",
         f"- Findings at baseline: {r['findings_status']['baseline']}; now: {r['findings_status']['current']}.",
         "",
         "## Fixes",
         ""]
    for fx in r["fixes"]:
        L.append(f"### {fx['id']} ({fx['track']}): {fx['title']}")
        L.append(f"Source: {', '.join(fx['sources'])}. {fx['change']}")
        L.append(f"Files: {', '.join('`' + f + '`' for f in fx['files'])}.")
        if fx["findings"]:
            L.append("Findings: " + "; ".join(f"{f['id']} {f['before']} -> {f['after']}" for f in fx["findings"]) + ".")
        if fx["metrics"]:
            L.append("Metrics: " + "; ".join(f"{m['metric']} {_fmt(m['before'])} -> {_fmt(m['after'])}" for m in fx["metrics"]) + ".")
        if fx["tests"]:
            L.append("Tests: " + "; ".join(f"`{t['test']}` {t['before']} -> {t['after']}" for t in fx["tests"]) + ".")
        if fx["note"]:
            L.append(f"Note: {fx['note']}")
        L.append("")
    L += ["## Target metrics", "", "| Metric | Baseline | Current | Target |", "|---|---|---|---|"]
    for m in r["improvements"]:
        L.append(f"| {m['metric']} | {_fmt(m['baseline'])} | {_fmt(m['current'])} | {_fmt(m['target'])} |")
    exp = r["experiments"]
    L += ["", "## Experiments", ""]
    if exp["passage_prefix"]:
        e = exp["passage_prefix"]
        L.append(f"- **Passage prefix (W8):** hypothesis supported: {e['supported']}. Per-category MRR deltas {e['mrr_delta_by_category']}.")
    if exp["calibration"]:
        c = exp["calibration"]
        L.append(f"- **Calibration (W1, W13):** dev-chosen tau {c['chosen']['tau']}, shipped {c['shipped']['tau']}; "
                 f"held-out F1 {c['chosen']['test']['f1']}, false support {c['chosen']['test']['false_support_rate']}.")
    if exp["verifier_comparison"]:
        v = exp["verifier_comparison"]
        L.append(f"- **Verifier comparison (R2):** rule: {v['rule']}. Winner {v['winner']}. NLI costs {v['nli_ms_per_pair_cpu']} ms per pair on CPU.")
        for k, x in v["variants"].items():
            if x.get("threshold") is None:
                continue
            L.append(f"  - {k}: held-out F1 {x['test']['f1']}, false support {x['test']['false_support_rate']}, "
                     f"traps accepted {x['traps_accepted']}/{x['traps_total']}")
    if r["corpus_gaps"]:
        g = r["corpus_gaps"]
        L += ["", "## Corpus gaps (R4)", "",
              f"{g['missing_cells']} of {g['cells']} species x category cells have no evidence: {g['missing_by_category']}. "
              f"Filling them: {g['how_to_fill']}."]
    L += ["", "## Still open", "", "| ID | Severity | Status | Worked on | Finding |", "|---|---|---|---|---|"]
    for f in sorted(r["still_open"], key=lambda f: (f["status"] != "partial", f["severity"])):
        L.append(f"| {f['id']} | {f['severity']} | {f['status']} | {'yes' if f['addressed_this_iteration'] else 'no'} | {f['title']} |")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(prog="python -m evals.iteration")
    parser.add_argument("--name", default="iteration-1")
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--current", default="current")
    args = parser.parse_args(argv)
    report = build(args.name, args.baseline, args.current)
    out = ROOT / "evals" / "results" / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "iteration.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_jsonable), encoding="utf-8")
    (out / "REPORT.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"[iteration] {len(report['fixes'])} fixes; findings now {report['findings_status']['current']} -> {out}")
    return out


if __name__ == "__main__":
    main()

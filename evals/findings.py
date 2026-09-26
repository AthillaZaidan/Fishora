"""Computed weakness registry.

Each finding is a check over the Python-generated run artifacts in
reports/<label>/ (tests.json, rag_eval.json, cost_eval.json,
model_compare.json, probes.json). The registry holds only the definition:
id, severity, pipeline stage, title and target. Status and evidence are
computed on every run, so a fixed weakness turns "resolved" by itself and a
missing artifact reads "not measured", never a stale claim.

    python -m evals.findings --label baseline   -> reports/baseline/findings.json
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from evals.run import REPORTS_DIR, _jsonable

ARTIFACTS = ("tests", "rag_eval", "cost_eval", "model_compare", "probes", "cv_eval",
             "grounding_calibration", "experiment_nli_grounding", "experiment_passage_prefix", "corpus_gaps")
SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}
# open and partial findings sort before resolved ones
OPEN_STATES = ("open", "partial", "check_error")


def load_artifacts(label: str) -> dict:
    folder = REPORTS_DIR / label
    data = {}
    for name in ARTIFACTS:
        path = folder / f"{name}.json"
        data[name] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    return data


def get(data: dict | None, path: str, default=None):
    node = data
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def test_status(a: dict, module: str, name: str) -> str | None:
    for t in get(a, "tests.tests", []) or []:
        if t["module"] == module and t["name"] == name:
            return t["status"]
    return None


@dataclass(frozen=True)
class Finding:
    id: str
    severity: str  # high | medium | low | info
    stage: str
    title: str
    target: str
    check: Callable[[dict], tuple[str, list[str]]]  # -> (status, evidence lines)


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def _agent(a) -> str:
    """The as-shipped agent path once it completes cards, else the eval-only
    normalised run that priced a working agent path at baseline."""
    shipped = get(a, "cost_eval.summary.by_path.agent.success_rate")
    return "agent" if shipped else "agent_normalized"


def _w1(a):
    g = get(a, "rag_eval.grounding")
    if not g:
        return "not_measured", []
    ev = [f"grader: {g['grader']}",
          f"test-split F1 {g['test']['f1']:.3f}, recall {g['test']['recall']:.3f}",
          f"all 98 pairs: recall {g['all']['recall']:.3f}, false support {g['all']['false_support_rate']:.3f}",
          f"scripted pipeline claim retention {_pct(get(a, 'rag_eval.pipeline.plain.claim_retention'))}"]
    for t in ("test_indonesian_claim_is_grounded_in_english_evidence", "test_shared_token_does_not_ground_a_borrowed_claim"):
        ev.append(f"{t}: {test_status(a, 'test_grounding_and_pipeline', t) or 'n/a'}")
    bad = g["test"]["f1"] < 0.85 or g["test"]["false_support_rate"] > 0.15
    return ("open" if bad else "resolved"), ev


def _w2(a):
    s1 = test_status(a, "test_knowledge_api", "test_verification_schedules_a_job_that_completes")
    s2 = test_status(a, "test_operator_flow", "test_photo_to_grounded_card")
    if s1 is None:
        return "not_measured", []
    return ("resolved" if s1 == s2 == "passed" else "open"), [
        f"verify -> job completes: {s1}", f"photo -> grounded card (e2e): {s2}"]


def _w3(a):
    probe = get(a, "probes.sync_path_blank_key")
    if not probe:
        return "not_measured", []
    # An unhandled crash is HTTP 500; a mapped provider outage (502/503) is the intended answer.
    return ("open" if probe["status_code"] == 500 else "resolved"), [
        f"GET /predictions/{{id}}/knowledge, no job, blank key -> HTTP {probe['status_code']}"]


def _w4(a):
    fenced = get(a, "rag_eval.pipeline.fenced.job_success_rate")
    unit = test_status(a, "test_orchestrator_nodes", "test_expert_parses_markdown_fenced_json")
    if fenced is None:
        return "not_measured", []
    return ("open" if fenced < 1.0 else "resolved"), [
        f"cards completed when the LLM wraps JSON in ``` fences: {_pct(fenced)}", f"unit test: {unit}"]


def _w5(a):
    rounds = get(a, "rag_eval.pipeline.plain.llm_rounds", get(a, "rag_eval.pipeline.plain.sequential_rounds"))
    real = get(a, f"cost_eval.summary.by_path.{_agent(a)}.wall_s_per_card.p50")
    pub = get(a, "cost_eval.summary.by_path.published.wall_s_per_card.p50")
    calls = get(a, f"cost_eval.summary.by_path.{_agent(a)}.llm_calls_per_card.mean")
    if rounds is None:
        return "not_measured", []
    ev = [f"LLM rounds per card (scripted): {rounds}; wall time / call delay {get(a, 'rag_eval.pipeline.plain.sequential_rounds')}"]
    if real:
        ev.append(f"real LLM: agent card p50 {real:.1f}s with {calls} calls vs published card p50 {pub:.1f}s with 1 call")
    return ("open" if rounds > 2.5 else "resolved"), ev


def _w6(a):
    rep = get(a, "rag_eval.pipeline.plain.repeat_latency_ms_mean")
    first = get(a, "rag_eval.pipeline.plain.latency_ms_mean")
    if rep is None:
        return "not_measured", []
    status = "open" if rep >= 50 else "resolved"
    how = "no cache" if status == "open" else "served from the card cache"
    return status, [f"repeat card {rep:.0f} ms vs first card {first:.0f} ms ({how})"]


def _w7(a):
    cold = get(a, "rag_eval.environment.embedder_cold_load_s")
    warm = get(a, "rag_eval.retrieval.latency_ms.p50")
    status = test_status(a, "test_knowledge_api", "test_embedder_is_warmed_at_startup")
    if cold is None:
        return "not_measured", []
    return ("resolved" if status == "passed" else "open"), [
        f"E5 cold load {cold:.1f}s on the first knowledge request; warm query p50 {warm} ms",
        f"startup warmup test: {status}"]


def _w8(a):
    cats = get(a, "rag_eval.retrieval.by_category")
    if not cats:
        return "not_measured", []
    worst = min(cats.items(), key=lambda kv: kv[1]["scoped_mrr"])
    ev = [f"scoped MRR by category: " + ", ".join(f"{k} {v['scoped_mrr']:.2f}" for k, v in cats.items()),
          f"overall scoped MRR {get(a, 'rag_eval.retrieval.scoped.mrr'):.3f}; unfiltered recall@6 {get(a, 'rag_eval.retrieval.global.recall@6'):.3f}"]
    return ("open" if worst[1]["scoped_mrr"] < 0.5 else "resolved"), ev


def _w9(a):
    r6 = get(a, "rag_eval.retrieval.scoped.recall@6")
    if r6 is None:
        return "not_measured", []
    from evals.corpus import load_corpus
    counts = {}
    for c in load_corpus():
        counts[c.species_label] = counts.get(c.species_label, 0) + 1
    comp = get(a, "rag_eval.retrieval.card_evidence_completeness")
    if comp is None:  # baseline harness: the card took the ranked top 6
        over = {s: n for s, n in counts.items() if n > 6}
        return ("open" if over else "resolved"), [
            f"species with more chunks than the 6-chunk cap: {over or 'none'}", f"scoped recall@6 {r6:.3f}"]
    return ("resolved" if comp["min"] >= 1.0 else "open"), [
        f"share of each species' verified chunks that reach its card: mean {comp['mean']:.0%}, min {comp['min']:.0%}",
        f"largest species slice {max(counts.values())} chunks"]


def _w10(a):
    fits = get(a, "rag_eval.corpus.chunker_fits_e5")
    if fits is None:
        return "not_measured", []
    return ("resolved" if fits else "open"), [
        f"chunker max_tokens {get(a, 'rag_eval.corpus.chunker_max_tokens')} vs E5 window 512",
        f"corpus chunks over the limit today: {get(a, 'rag_eval.corpus.chunks_over_e5_limit')}"]


def _w11(a):
    counts = get(a, "rag_eval.corpus.category_counts")
    if not counts:
        return "not_measured", []
    empty = [k for k, v in counts.items() if v == 0]
    fill = get(a, "cost_eval.summary.by_path.published.fields_filled_rate") or {}
    ev = [f"categories with no chunks: {empty or 'none'}",
          "published-card field fill rate: " + ", ".join(f"{k} {v:.0%}" for k, v in fill.items())] if fill else [
          f"categories with no chunks: {empty or 'none'}"]
    return ("open" if empty else "resolved"), ev


def _w12(a):
    probe = get(a, "probes.silent_excepts")
    if not probe:
        return "not_measured", []
    files = {}
    for h in probe["handlers"]:
        files[h["file"].split("/")[-1]] = files.get(h["file"].split("/")[-1], 0) + 1
    return ("open" if probe["count"] else "resolved"), [
        f"{probe['count']} broad except handlers that neither log nor re-raise: {files}"]


def _w13(a):
    sim = get(a, "rag_eval.grounding.similarity")
    if not sim:
        return "not_measured", []
    overlap = sim["negative_max"] >= sim["positive_min"]
    mitigated = get(a, "rag_eval.grounding.grader") == "E5 cosine + numbers" and get(a, "rag_eval.grounding.test.false_support_rate", 1) <= 0.15
    status = "resolved" if not overlap else ("partial" if mitigated else "open")
    return status, [
        f"E5 claim-evidence cosine: supported mean {sim['positive_mean']}, borrowed mean {sim['negative_mean']}",
        f"supported min {sim['positive_min']} vs borrowed max {sim['negative_max']} (overlap: {overlap})"]


def _w14(a):
    s = test_status(a, "test_orchestrator_nodes", "test_expert_sees_the_whole_chunk")
    if s is None:
        return "not_measured", []
    what = "experts receive whole chunks" if s == "passed" else "expert evidence cut at 300 chars"
    return ("resolved" if s == "passed" else "open"), [f"{what}; unit test: {s}"]


def _w18(a):
    tax = get(a, "probes.taxonomy_vs_corpus")
    if not tax or not tax.get("available"):
        return "not_measured", []
    mm = tax["mismatches"]
    return ("open" if mm else "resolved"), [
        f"{m['species']}: seeded {m['seeded']} vs corpus {', '.join(m['corpus'])}" for m in mm] or ["all seeded names agree"]


def _w19(a):
    p = get(a, "probes.publication_path")
    if not p:
        return "not_measured", []
    graded_first = p.get("publish_prefers_graded_card", False)
    fallback_ungraded = p["publish_uses_sync_knowledge_service"] and not p["sync_path_has_claim_critic"]
    status = "resolved" if not fallback_ungraded else ("partial" if graded_first else "open")
    return status, [
        f"publication freezes the graded job card when one exists: {graded_first}",
        f"lot publication uses the sync KnowledgeService: {p['publish_uses_sync_knowledge_service']}",
        f"sync path has a per-claim critic: {p['sync_path_has_claim_critic']}",
        f"buyer matching reads snapshot fields: {', '.join(p['matching_reads_snapshot_fields'])}"]


def _w20(a):
    h = get(a, "probes.opencode_headers")
    if not h:
        return "not_measured", []
    return ("resolved" if h["sends_session_header"] else "open"), [
        f"client sends x-opencode-session: {h['sends_session_header']}; sets a user agent: {h['sets_user_agent']}",
        "without it the gateway answers 400 MissingSessionID"]


def _w21(a):
    shipped = get(a, "cost_eval.summary.by_path.agent")
    fixed = get(a, "cost_eval.summary.by_path.agent_normalized")
    if not shipped:
        return "not_measured", []
    ev = [f"agent path as shipped, real LLM: {_pct(shipped['success_rate'])} cards completed of {shipped['cards']}"]
    if fixed:
        ev.append(f"same run, content read as text (eval-only): {_pct(fixed['success_rate'])} completed")
    status = "open" if shipped["success_rate"] < 0.5 else "resolved"
    cost = shipped["cost_usd_per_card"]["mean"]
    ev.append(f"spend on the failing path: ${cost:.5f} per card, all wasted" if status == "open"
              else f"spend: ${cost:.5f} per completed card")
    return status, ev


def _w22(a):
    paths = get(a, "cost_eval.summary.by_path") or {}
    rates = {p: v.get("english_leak_rate") for p, v in paths.items() if v.get("english_leak_rate") is not None}
    if not rates:
        return "not_measured", []
    leaking = any(r > 0 for r in rates.values())
    return ("open" if leaking else "not_observed"), [
        "cards whose prose is mostly English: " + ", ".join(f"{p} {r:.0%}" for p, r in rates.items()),
        "expert prompts do not state the output language; not seen in final cards in this run"]


def _w24(a):
    path = _agent(a)
    cards = [c for c in (get(a, "cost_eval.cards") or []) if c["path"] == path]
    if not cards:
        return "not_measured", []
    by_species = {}
    for c in cards:
        by_species.setdefault(c["species"], []).append(c["status"])
    failing = {s: f"{v.count('failed')}/{len(v)}" for s, v in by_species.items() if "failed" in v}
    pub = {c["species"]: c["status"] for c in get(a, "cost_eval.cards") if c["path"] == "published"}
    return ("open" if failing else "resolved"), [
        f"{path} failures by species: {failing or 'none'}",
        f"gembolo published card: {pub.get('gembolo', 'n/a')} (only a LIMITATION chunk exists)"]


def _w25(a):
    pub = get(a, "cost_eval.summary.by_path.published")
    if not pub:
        return "not_measured", []
    fails = [f"{c['species']}: {c['error']}" for c in get(a, "cost_eval.cards") if c["path"] == "published" and c["status"] != "completed"]
    return ("open" if pub["success_rate"] < 1 else "resolved"), [
        f"published cards passing the citation gate: {_pct(pub['success_rate'])} of {pub['cards']}", *fails[:3]]


def _w26(a):
    pub = get(a, "cost_eval.summary.by_path.published.fields_filled_rate")
    agent = get(a, f"cost_eval.summary.by_path.{_agent(a)}.fields_filled_rate")
    if not pub or not agent:
        return "not_measured", []
    rows = [f"{f}: published {pub[f]:.0%} vs agent {agent.get(f, 0):.0%}" for f in pub]
    worse = sum(agent.get(f, 0) < pub[f] for f in pub)
    return ("open" if worse >= 3 else "resolved"), [f"fields where the critic-guarded agent card is emptier: {worse}/7", *rows]


def _w27(a):
    paths = get(a, "cost_eval.summary.by_path") or {}
    if not paths:
        return "not_measured", []
    return "info", [
        f"{p}: output tokens {v['output_share_of_cost']:.0%} of cost; reasoning {v['reasoning_share_of_output']:.0%} of output; cached input {v['cache_hit_share_of_input']:.0%}"
        for p, v in paths.items()]


def _w28(a):
    models = get(a, "model_compare.models") or {}
    if not models:
        return "not_measured", []
    rows = [f"{m}: valid {_pct(v['valid_card_rate'])}, ${v['cost_usd_per_card'].get('mean', 0):.5f}/card, p50 {v['latency_s'].get('p50')}s"
            + (f", errors {v['errors']}" if v["errors"] else "") for m, v in models.items()]
    unusable = [m for m, v in models.items() if v["valid_card_rate"] < 0.9]
    return ("open" if unusable else "resolved"), [f"models below 90% valid cards: {unusable or 'none'}", *rows]


# ---------------------------------------------------------------- CV (species identification)

def _cv_prod(a):
    cv = get(a, "cv_eval.models") or {}
    prod = get(a, "cv_eval.production_run")
    return cv.get(prod), cv


def _cv1(a):
    m, _ = _cv_prod(a)
    if not m:
        return "not_measured", []
    lo, hi = m.get("field_accuracy_ci") or (None, None)
    return ("open" if m["field_accuracy"] < 0.9 else "resolved"), [
        f"{m['run']}: clean photos {m['clean_accuracy']:.1%}, field photos {m['field_accuracy']:.1%}"
        + (f" (95% CI {lo:.1%} to {hi:.1%}, n = {m['field_n']})" if lo is not None else ""),
        f"field macro-F1 {m['field_macro_f1']:.3f}"]


def _cv2(a):
    m, _ = _cv_prod(a)
    if not m:
        return "not_measured", []
    return ("open" if m["field_wrong_confident"] > 0.05 else "resolved"), [
        f"{m['run']}: field photos wrong at confidence >= 0.9: {m['field_wrong_confident']:.1%}; ECE {m['field_ece']:.3f}"]


def _cv3(a):
    m, cv = _cv_prod(a)
    if not m:
        return "not_measured", []
    ev = [f"production abstain threshold {m['production_threshold']}: accepts "
          f"{m['ood_accepted_at_threshold']:.0%} of out-of-distribution images as a species"]
    for name, o in m["ood"].items():
        ev.append(f"{name.replace('ood_', '')}: AUROC {o['auroc']:.3f}, FPR at 95% TPR {o['fpr_at_95_tpr']:.0%} (n = {o['n']})")
    return ("open" if (m["ood_accepted_at_threshold"] or 0) > 0.1 else "resolved"), ev


def _cv4(a):
    m, _ = _cv_prod(a)
    if not m or m.get("shortcut_rate") is None:
        return "not_measured", []
    return ("open" if m["shortcut_rate"] > 2 * (m["shortcut_chance"] or 0) else "resolved"), [
        f"{m['run']}: with the fish erased, the true species is still predicted {m['shortcut_rate']:.1%} of the time "
        f"(chance {m['shortcut_chance']:.1%})"]


def _cv5(a):
    m, _ = _cv_prod(a)
    if not m:
        return "not_measured", []
    w = m["worst_corruption"]
    return ("open" if (w["accuracy"] or 1) < 0.97 else "resolved"), [
        f"{m['run']}: worst corruption {w['slice']} {w['accuracy']:.1%}; mean over 7 corruptions {m['corruption_mean_accuracy']:.1%}"]


def _cv6(a):
    m, cv = _cv_prod(a)
    if not m:
        return "not_measured", []
    rows = [f"{r}: field {v['field_accuracy']:.1%}, CPU p50 {v['cpu_p50_ms']:.0f} ms, weights {v['weights_mb']:.0f} MB"
            for r, v in sorted(cv.items(), key=lambda kv: -kv[1]["field_accuracy"])]
    faster_equal = [r for r, v in cv.items() if r != m["run"] and v["field_accuracy"] >= m["field_accuracy"] and v["cpu_p50_ms"] < m["cpu_p50_ms"]]
    return "info", [f"production {m['run']} is on the accuracy/latency frontier: {not faster_equal}", *rows]


# ---------------------------------------------------------------- review critique (R1-R4)

def _r1(a):
    comp = get(a, "rag_eval.retrieval.card_evidence_completeness")
    if comp is None:
        return "open", ["the card takes the ranked top 6 of a 1-7 chunk slice (baseline design)",
                        f"scoped MRR {get(a, 'rag_eval.retrieval.scoped.mrr')} decides which evidence the model sees"]
    return ("resolved" if comp["min"] >= 1.0 else "open"), [
        f"cards receive the whole species slice: completeness mean {comp['mean']:.0%}, min {comp['min']:.0%}",
        "semantic ranking applies only above 20 chunks per species"]


def _r2(a):
    exp = get(a, "experiment_nli_grounding.variants")
    if not exp:
        return "not_measured", []
    from apps.main_api.services.orchestrator import USE_LLM_JUDGE
    key = "D_e5_exact_llm_judge" if USE_LLM_JUDGE and "D_e5_exact_llm_judge" in exp else "A_e5_exact"
    shipped = exp[key]
    name = "shipped verifier (E5 + exact checks" + (" + LLM judge)" if key.startswith("D") else ")")
    ev = [f"{name}: held-out F1 {shipped['test']['f1']}, false support {shipped['test']['false_support_rate']}, "
          f"traps accepted {shipped['traps_accepted']}/{shipped['traps_total']}"]
    for key, label in (("D_e5_exact_llm_judge", "with the LLM judge (removed)"), ("C_nli_exact", "NLI + exact (candidate)")):
        if key in exp and exp[key].get("threshold") is not None:
            ev.append(f"{label}: held-out F1 {exp[key]['test']['f1']}, traps accepted "
                      f"{exp[key]['traps_accepted']}/{exp[key]['traps_total']}")
    return ("resolved" if shipped["traps_accepted"] == 0 else "partial"), ev


REGISTRY: tuple[Finding, ...] = (
    Finding("W1", "high", "guardrail (critic)", "Lexical grounding across languages: Indonesian claims vs English evidence",
            "grounding test F1 ≥ 0.85, false support ≤ 0.15", _w1),
    Finding("W2", "high", "orchestration", "Card jobs cannot complete without a production LLM; the LLM is not an injectable port",
            "verify → job completes; e2e photo → card passes", _w2),
    Finding("W3", "high", "API", "Sync knowledge fallback returns an unhandled 500 when the key is blank",
            "a mapped 502/503, never an unhandled 500", _w3),
    Finding("W4", "high", "generation parsing", "Markdown-fenced LLM JSON fails every expert",
            "fenced-JSON job success = 100%", _w4),
    Finding("W19", "high", "publication", "The published card (QR page, buyer matching) bypasses the per-claim critic",
            "published card passes the same claim grounding", _w19),
    Finding("W20", "high", "LLM client", "OpenCode Go rejects every call: no x-opencode-session header",
            "client sends the session header", _w20),
    Finding("W21", "high", "generation parsing", "Agent path cannot parse Responses-API content blocks",
            "agent card success ≥ 90% with the real LLM", _w21),
    Finding("W5", "medium", "orchestration", "Too many sequential LLM rounds per card",
            "≤ 2.5 rounds per card", _w5),
    Finding("W6", "medium", "orchestration", "No card cache for identical evidence",
            "repeat card < 50 ms", _w6),
    Finding("W7", "medium", "retrieval", "Embedding model loads on the first request",
            "warm at startup", _w7),
    Finding("W8", "medium", "retrieval", "Ranking follows the species name, not the requested category",
            "worst per-category scoped MRR ≥ 0.5", _w8),
    Finding("W9", "medium", "retrieval", "Fixed 6-chunk cap drops evidence",
            "no species above the cap", _w9),
    Finding("W10", "medium", "ingestion", "Chunker limit exceeds the E5 512-token window",
            "max_tokens + 4 ≤ 512", _w10),
    Finding("W11", "medium", "corpus", "Categories with no evidence at all",
            "every category has evidence", _w11),
    Finding("W12", "medium", "observability", "Broad exception handlers swallow errors silently",
            "0 silent handlers", _w12),
    Finding("W13", "medium", "guardrail (critic)", "Embedding similarity alone separates supported from borrowed claims poorly",
            "no score overlap (or add numeric checks)", _w13),
    Finding("W18", "medium", "taxonomy", "Seeded scientific names disagree with the corpus identity chunks",
            "0 mismatches", _w18),
    Finding("W24", "medium", "orchestration", "Agent path fails instead of returning a limitation card",
            "no species fails on the agent path", _w24),
    Finding("W25", "medium", "generation", "Published path intermittently fails its own citation gate",
            "100% published cards valid", _w25),
    Finding("W26", "medium", "guardrail (critic)", "Critic-guarded agent cards are emptier than published cards",
            "agent fill rate ≥ published on ≥ 5/7 fields", _w26),
    Finding("W28", "medium", "LLM provider", "Not every OpenCode Go model is a drop-in replacement",
            "all candidate models ≥ 90% valid", _w28),
    Finding("W14", "low", "context construction", "Expert evidence truncated at 300 characters",
            "experts see whole chunks", _w14),
    Finding("W22", "low", "prompting", "Expert prompts do not require Indonesian output",
            "0% English cards", _w22),
    Finding("W27", "info", "cost", "Where the LLM money goes", "n/a", _w27),
    Finding("R1", "medium", "retrieval", "Retrieval is unnecessary at this corpus size and loses evidence",
            "every species' full evidence reaches its card", _r1),
    Finding("R2", "high", "guardrail (critic)", "Grounding must catch negated and inverted claims, not only borrowed ones",
            "0 of 14 traps accepted", _r2),
    Finding("CV1", "high", "species ID", "Field photos are misidentified far more often than clean photos",
            "field accuracy >= 90%", _cv1),
    Finding("CV2", "high", "species ID", "Confidently wrong on field photos (confidence >= 0.9)",
            "<= 5% wrong-and-confident", _cv2),
    Finding("CV3", "high", "species ID", "Abstain threshold 0.0 accepts every unknown fish and non-fish image",
            "<= 10% of OOD images accepted", _cv3),
    Finding("CV4", "medium", "species ID", "Background shortcut: species predicted with the fish erased",
            "shortcut rate <= 2x chance", _cv4),
    Finding("CV5", "medium", "species ID", "Accuracy drop under image corruption", "worst corruption >= 97%", _cv5),
    Finding("CV6", "info", "species ID", "Backbone accuracy vs CPU cost", "n/a", _cv6),
)


def evaluate(label: str) -> list[dict]:
    artifacts = load_artifacts(label)
    rows = []
    for f in REGISTRY:
        try:
            status, evidence = f.check(artifacts)
        except Exception as exc:  # a malformed artifact must show up, not hide the finding
            status, evidence = "check_error", [f"{type(exc).__name__}: {exc}"]
        rows.append({"id": f.id, "severity": f.severity, "stage": f.stage, "title": f.title,
                     "target": f.target, "status": status, "evidence": evidence})
    rows.sort(key=lambda r: (r["status"] not in OPEN_STATES, SEVERITY_ORDER[r["severity"]],
                             r["id"].rstrip("0123456789"), int(re.sub(r"\D", "", r["id"]) or 0)))
    return rows


# Headline metrics for the improvement view: (label, artifact.path, target, higher_is_better, finding ids).
TARGETS = (
    ("Test suite pass rate", "tests.totals", 1.0, True, ["W1", "W2", "W4"]),
    ("Grounding F1 (test split)", "rag_eval.grounding.test.f1", 0.85, True, ["W1", "W13"]),
    ("Grounding false support", "rag_eval.grounding.all.false_support_rate", 0.10, False, ["W1"]),
    ("True-claim retention (scripted)", "rag_eval.pipeline.plain.claim_retention", 0.80, True, ["W1"]),
    ("Fenced-JSON job success", "rag_eval.pipeline.fenced.job_success_rate", 1.0, True, ["W4"]),
    ("LLM rounds / card", "rag_eval.pipeline.plain.llm_rounds", 2.5, False, ["W5"]),
    ("Repeat card latency (ms)", "rag_eval.pipeline.plain.repeat_latency_ms_mean", 50, False, ["W6"]),
    ("Embedder cold load (s)", "rag_eval.environment.embedder_cold_load_s", None, False, ["W7"]),
    ("Scoped retrieval MRR", "rag_eval.retrieval.scoped.mrr", 0.7, True, ["W8"]),
    ("Scoped recall@6", "rag_eval.retrieval.scoped.recall@6", 1.0, True, ["W9"]),
    ("Agent card success, real LLM", "cost_eval.summary.by_path.agent.success_rate", 0.9, True, ["W20", "W21"]),
    ("Published card success, real LLM", "cost_eval.summary.by_path.published.success_rate", 1.0, True, ["W25"]),
    ("Agent card p50 wall time (s)", "cost_eval.agent.wall_s_per_card.p50", 6.0, False, ["W5"]),
    ("Cost per published card (USD)", "cost_eval.summary.by_path.published.cost_usd_per_card.mean", None, False, ["W27"]),
    ("Cost per agent card (USD)", "cost_eval.agent.cost_usd_per_card.mean", None, False, ["W5", "W27"]),
    ("Species ID: field accuracy", "cv_eval.prod.field_accuracy", 0.90, True, ["CV1"]),
    ("Species ID: wrong and confident (field)", "cv_eval.prod.field_wrong_confident", 0.05, False, ["CV2"]),
    ("Species ID: OOD images accepted", "cv_eval.prod.ood_accepted_at_threshold", 0.10, False, ["CV3"]),
    ("Species ID: background shortcut rate", "cv_eval.prod.shortcut_rate", 0.18, False, ["CV4"]),
)


def metric_value(artifacts: dict, path: str):
    if path.startswith("cv_eval.prod."):
        prod = get(artifacts, "cv_eval.production_run")
        return get(artifacts, f"cv_eval.models.{prod}.{path.removeprefix('cv_eval.prod.')}") if prod else None
    if path.startswith("cost_eval.agent."):
        return get(artifacts, f"cost_eval.summary.by_path.{_agent(artifacts)}.{path.removeprefix('cost_eval.agent.')}")
    if path == "tests.totals":
        t = get(artifacts, "tests.totals")
        return round(t["passed"] / t["total"], 4) if t and t.get("total") else None
    return get(artifacts, path)


def improvements(baseline: str = "baseline", current: str = "current") -> list[dict]:
    base, cur = load_artifacts(baseline), load_artifacts(current)
    rows = []
    for label, path, target, higher, ids in TARGETS:
        b, c = metric_value(base, path), metric_value(cur, path)
        met = None
        if target is not None and b is not None:
            met = b >= target if higher else b <= target
        rows.append({"metric": label, "path": path, "baseline": b, "current": c, "target": target,
                     "higher_is_better": higher, "baseline_meets_target": met, "findings": ids})
    return rows


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(prog="python -m evals.findings")
    parser.add_argument("--label", default="baseline")
    args = parser.parse_args(argv)
    result = {"label": args.label, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "findings": evaluate(args.label), "improvements": improvements(args.label)}
    out = REPORTS_DIR / args.label / "findings.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=_jsonable), encoding="utf-8")
    counts = {}
    for r in result["findings"]:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"[findings] {counts} -> {out}")
    return out


if __name__ == "__main__":
    main()

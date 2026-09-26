"""Iteration 2, step 2: the 2x2 experiment {4-expert graph, writer+critic} x {gpt-5.6-luna, gpt-6-luna}.

    python -m evals.iteration2 --label iteration-2 --judge <model> [--repeat 2] [--species nila,tuna]

Every cell uses the same claim verifier (apps/main_api/services/claim_verifier.py),
the same evidence (whole verified species slice), the same scope rule in the
prompts, and a cleared card cache before each card, so the cells differ only
in architecture and model. Cards are generated with the real LLM.

Each final card is split into claims (list items, prose sentences) and scored
by ``--judge``, a model from another family whose agreement with the locked
gold labels was measured first (evals/iteration2_gold.py). The iteration-1
agent cards are scored by the same judge, so the reference row is comparable.

Per cell: completion, cost, latency p50/p95, LLM calls, claims per card,
judged-supported share of claims (faithfulness), supported claims per card
(useful coverage) and fields filled, with 95% bootstrap CIs over cards.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from evals.corpus import build_store, species_records
from evals.fakes import InMemoryJobRepository, InMemorySpeciesRepository, new_id
from evals.run import REPORTS_DIR

CELLS = [("experts", "gpt-5.6-luna"), ("experts", "gpt-6-luna"),
         ("writer_critic", "gpt-5.6-luna"), ("writer_critic", "gpt-6-luna")]
CARD_FIELDS = ("physical_characteristics", "taste", "texture", "processing_methods",
               "commercial_uses", "similar_or_substitute_species", "potential_buyer_segments")
STAGES = (("You are a fact checker", "verifier"), ("Write claims", "writer"),
          ("The fact checker rejected", "writer_revision"))


def card_claims(card: dict | None) -> list[tuple[str, str]]:
    from apps.main_api.services.claim_verifier import split_field
    if not card:
        return []
    return [(f, t) for f in CARD_FIELDS for t in split_field(f, card.get(f))]


def run_cell(arch: str, model: str, species, store, embedder, settings, repeat: int, effort: str | None = None) -> list[dict]:
    from apps.main_api.services import orchestrator, workflow
    from evals import cost_eval
    from evals.cost_eval import RecordingLLM, make_llm

    cost_eval._STAGE_PREFIXES = STAGES + tuple(p for p in cost_eval._STAGE_PREFIXES if p[1] not in {s for _, s in STAGES})
    species_repo = InMemorySpeciesRepository(species_records())
    jobs = InMemoryJobRepository()
    entry = orchestrator.run_graph if arch == "experts" else workflow.run_workflow
    rows = []
    for rep in range(repeat):
        for record in species:
            job_id = new_id()
            llm = RecordingLLM(make_llm(settings, f"fishora-it2-{job_id}", model, timeout=180, reasoning_effort=effort), model,
                              species=record.normalized_label, path=f"{arch}:{model}")
            jobs.create(job_id, job_id, record.id)
            with _CACHE_LOCK:
                orchestrator.clear_card_cache()
            started = time.perf_counter()
            entry(job_id, record.id, job_id, store, embedder, llm, llm, species_repo, jobs)
            wall = time.perf_counter() - started
            job = jobs.get(job_id)
            calls = llm.records
            starts = sorted((c.started_s, c.started_s + c.latency_s) for c in calls)
            rounds, end = 0, -1.0
            for s, e in starts:
                if s >= end:
                    rounds += 1
                end = max(end, e)
            rows.append({
                "arch": arch, "model": model + (f"@{effort}" if effort else ""), "effort": effort or "default", "repeat": rep, "species": record.normalized_label,
                "status": job.status, "wall_s": round(wall, 2), "llm_calls": len(calls), "llm_rounds": rounds,
                "llm_errors": sum(c.error is not None for c in calls),
                "stages": dict(sorted(defaultdict(int, {s: sum(c.stage == s for c in calls) for s in {c.stage for c in calls}}).items())),
                "input_tokens": sum(c.input_tokens for c in calls), "output_tokens": sum(c.output_tokens for c in calls),
                "reasoning_tokens": sum(c.reasoning_tokens for c in calls),
                "cost_usd": round(sum(c.cost_usd for c in calls), 6),
                "card": job.final_card, "error": job.error,
                "fields_filled": sum(bool((job.final_card or {}).get(f)) for f in CARD_FIELDS),
            })
            print(f"[it2] {arch:13s} {model:12s} {record.normalized_label:12s} {job.status:9s} {wall:5.1f}s "
                  f"{len(calls)} calls ${rows[-1]['cost_usd']:.5f}", flush=True)
    return rows


_CACHE_LOCK = threading.Lock()


def judge_cards(rows: list[dict], judge_model: str, store, embedder, settings) -> None:
    """Score every claim of every final card with the judge (LLM stage only)."""
    from apps.main_api.services import claim_verifier
    from apps.main_api.services.orchestrator import CARD_QUERY
    from apps.main_api.services.retrieval import VerifiedRetriever
    from evals.cost_eval import make_llm

    retriever = VerifiedRetriever(store, embedder)
    records = {s.normalized_label: s for s in species_records()}
    evidence = {lab: retriever.card_evidence(s.id, CARD_QUERY.format(common_name=s.common_name_id)) for lab, s in records.items()}

    def one(row):
        claims = []
        for i, (fld, text) in enumerate(card_claims(row.get("card"))):
            ids = [c.chunk_id for c in evidence[row["species"]] if c.category in claim_verifier.FIELD_EVIDENCE[fld]]
            claims.append(claim_verifier.Claim(id=i, field=fld, text=text, chunk_ids=ids))
        if not claims:
            row["judged"] = []
            return
        for attempt in range(3):
            fresh = [claim_verifier.Claim(id=c.id, field=c.field, text=c.text, chunk_ids=list(c.chunk_ids)) for c in claims]
            llm = make_llm(settings, f"fishora-judge-{new_id()}", judge_model, timeout=180, max_retries=1)
            res = claim_verifier.verify(fresh, evidence[row["species"]], llm, use_llm=True, attempts=1)
            if res.llm_error is None:
                break
        row["judge_error"] = res.llm_error
        print(f"[judge] {row['species']:12s} {'FAILED ' + str(res.llm_error) if res.llm_error else 'ok'}", flush=True)
        # A judge that never answered scores nothing: exclude the card instead of counting 0% supported.
        row["judged"] = None if res.llm_error else [
            {"field": c.field, "text": c.text, "label": c.label, "reason": c.reason, "stage": c.stage} for c in res.claims]

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(one, rows))


def _ci(values: list[float], seed: int = 0) -> list[float]:
    if not values:
        return [None, None]
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choice(values) for _ in values) for _ in range(1000))
    return [round(means[25], 4), round(means[974], 4)]


def summarize(rows: list[dict]) -> dict:
    out = {}
    groups = defaultdict(list)
    for r in rows:
        groups[f"{r['arch']}:{r['model']}"].append(r)
    for cell, rs in groups.items():
        done = [r for r in rs if r["status"] == "completed"]
        judged = [r for r in done if r.get("judged") is not None]
        judge_failed = sum(1 for r in done if r.get("judged") is None and r.get("card"))
        per_card_sup = [sum(c["label"] == "supported" for c in r["judged"]) for r in judged]
        per_card_n = [len(r["judged"]) for r in judged]
        share = [s / n for s, n in zip(per_card_sup, per_card_n) if n]
        unsup = [sum(c["label"] == "unsupported" for c in r["judged"]) / len(r["judged"]) for r in judged if r["judged"]]
        walls = sorted(r["wall_s"] for r in rs)
        out[cell] = {
            "cards": len(rs), "completed": len(done), "completion_rate": round(len(done) / len(rs), 3),
            "cost_usd_per_card": round(statistics.fmean(r["cost_usd"] for r in rs), 6),
            "cost_usd_per_1000_cards": round(1000 * statistics.fmean(r["cost_usd"] for r in rs), 3),
            "wall_s_p50": walls[len(walls) // 2], "wall_s_p95": walls[min(len(walls) - 1, int(0.95 * len(walls)))],
            "llm_calls_mean": round(statistics.fmean(r["llm_calls"] for r in rs), 2),
            "llm_rounds_mean": round(statistics.fmean(r["llm_rounds"] for r in rs), 2),
            "claims_per_card": round(statistics.fmean(per_card_n), 2) if per_card_n else 0,
            "supported_share": round(statistics.fmean(share), 3) if share else None, "supported_share_ci": _ci(share),
            "unsupported_share": round(statistics.fmean(unsup), 3) if unsup else None, "unsupported_share_ci": _ci(unsup),
            "supported_claims_per_card": round(statistics.fmean(per_card_sup), 2) if per_card_sup else 0,
            "supported_claims_per_card_ci": _ci([float(x) for x in per_card_sup]),
            "judged_cards": len(judged), "judge_failed_cards": judge_failed,
            "fields_filled_mean": round(statistics.fmean(r["fields_filled"] for r in done), 2) if done else 0,
        }
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="iteration-2")
    ap.add_argument("--judge", required=True)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--species", default="")
    ap.add_argument("--cells", default="")
    ap.add_argument("--critic-variants", default="e5,no_e5")
    ap.add_argument("--no-reference", action="store_true")
    ap.add_argument("--out", default="matrix.json")
    args = ap.parse_args(argv)

    from apps.main_api.config import MainSettings
    from apps.main_api.services import orchestrator
    from apps.main_api.services.embeddings import LocalE5Embedder

    orchestrator.CRITIC_MODE = "verifier"
    settings = MainSettings()
    embedder = LocalE5Embedder()
    embedder.embed_query("warmup")
    store = build_store(embedder)
    wanted = {s for s in args.species.split(",") if s}
    species = [s for s in species_records() if not wanted or s.normalized_label in wanted]
    if args.cells and "@" in args.cells or args.cells.endswith("@default"):
        cells = []
        for spec in args.cells.split(","):
            arch, _, rest = spec.partition(":")
            model, _, effort = rest.partition("@")
            cells.append((arch, model, None if effort in ("", "default") else effort))
    else:
        cells = [(a, m, None) for a, m in CELLS if not args.cells or f"{a}:{m}" in args.cells.split(",")]

    from apps.main_api.services import claim_verifier
    t0 = time.perf_counter()
    rows = []
    # Variants run one after another: the filter switch is module state.
    for variant in [v for v in args.critic_variants.split(",") if v]:
        claim_verifier.USE_EMBEDDING_FILTER = variant == "e5"
        with ThreadPoolExecutor(max_workers=len(cells)) as pool:
            futures = [pool.submit(run_cell, a, m, species, store, embedder, settings, args.repeat, e) for a, m, e in cells]
            for f in futures:
                for r in f.result():
                    r["arch"] = f"{r['arch']}+{variant}"
                    rows.append(r)

    # iteration-1 reference: the shipped agent cards, scored by the same judge
    ref = json.loads((REPORTS_DIR.parent / "evals" / "results" / "iteration-1" / "cost_eval.json").read_text()) \
        if (REPORTS_DIR.parent / "evals" / "results" / "iteration-1" / "cost_eval.json").exists() else {"cards": []}
    for c in ([] if args.no_reference else ref["cards"]):
        if c["path"] == "agent" and (not wanted or c["species"] in wanted):
            rows.append({"arch": "iteration-1 agent", "model": "gpt-5.6-luna", "repeat": 0, "species": c["species"],
                         "status": c["status"], "wall_s": c["wall_s"], "llm_calls": c["llm_calls"], "llm_rounds": 1,
                         "cost_usd": c["cost_usd"], "card": c.get("card"), "error": c.get("error"),
                         "fields_filled": sum(bool((c.get("card") or {}).get(f)) for f in CARD_FIELDS)})

    judge_cards(rows, args.judge, store, embedder, settings)
    report = {"label": args.label, "judge": args.judge, "repeat": args.repeat, "wall_s": round(time.perf_counter() - t0, 1),
              "summary": summarize(rows), "cards": rows}
    out = REPORTS_DIR / args.label
    out.mkdir(parents=True, exist_ok=True)
    (out / args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    for cell, s in report["summary"].items():
        print(f"{cell:32s} done {s['completion_rate']:.2f} ${s['cost_usd_per_1000_cards']:.2f}/1k p50 {s['wall_s_p50']}s "
              f"p95 {s['wall_s_p95']}s calls {s['llm_calls_mean']} claims {s['claims_per_card']} "
              f"supported {s['supported_share']} ({s['supported_claims_per_card']}/card) fields {s['fields_filled_mean']}")


if __name__ == "__main__":
    main()

"""Iteration 3: more evidence (the atomic-fact writer prompt was tried and not shipped), scored by fact coverage.

    python -m evals.iteration3 --label iteration-3 --judge glm-5.3-flash [--species nila,tuna]

The shipped writer + critic (gpt-6-luna, default effort, E5 critic) on corpus
v2 (66 chunks) with the atomic-fact writer prompt.

Supported claims per card rewards splitting one sentence into many claims, so
the headline metric here is fact coverage: the share of the atomic facts in
the species' evidence (evals/protocol/facts_gold.csv, AI-assisted, reviewed by
the team, SHA-256 locked) that the card states with a claim the judge marks
supported. Three chunks carry no card fact and have no rows (gembolo's
limitation note and two catalogue descriptions for tuna).

Coverage is also reported on the facts of the 49 corpus-v1 chunks only, the
evidence iteration 2 had. The judge first scores faithfulness (as in
evals/iteration2.py), then one call per card maps supported claims to facts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evals.corpus import build_store, corpus_v1_ids, species_records
from evals.fakes import new_id
from evals.iteration2 import _ci, card_claims, judge_cards, run_cell
from evals.run import REPORTS_DIR

PROTOCOL = Path(__file__).resolve().parent / "protocol"
MODEL = "gpt-6-luna"

COVERAGE_PROMPT = """You check which reference facts a fish knowledge card states.
Facts (English, from the verified evidence):
{facts}

Card claims (Indonesian):
{claims}

A fact is covered only if at least one claim states the same information with the same scope
(e.g. a claim about one species of a multi-species label covers only facts about that species;
a list item covers only the item it names). Paraphrase and translation are fine; a vaguer or
partial statement does not cover a fact that is more specific.
Answer only JSON: {{"covered": [{{"fact": "<fact id>", "claims": [<claim number>]}}]}}
"""


def load_facts() -> list[dict]:
    lock = (PROTOCOL / "facts_gold.lock").read_text().split("\n")[0]
    want = lock.split("sha256=")[1].strip()
    got = hashlib.sha256((PROTOCOL / "facts_gold.csv").read_bytes()).hexdigest()
    if got != want:
        raise SystemExit(f"facts_gold.csv changed since it was locked ({got[:12]} != {want[:12]})")
    with open(PROTOCOL / "facts_gold.csv", newline="") as f:
        return list(csv.DictReader(f))


def cover(rows: list[dict], facts: list[dict], chunk_ids: set[str], judge_model: str, settings) -> None:
    """One judge call per card: which facts of its species do the supported claims state?"""
    from apps.main_api.services.llm_output import reply_json
    from evals.cost_eval import make_llm

    by_species = defaultdict(list)
    for f in facts:
        if f["chunk_id"] in chunk_ids:
            by_species[f["species"]].append(f)

    def one(row):
        mine = by_species[row["species"]]
        claims = [c["text"] for c in (row.get("judged") or []) if c["label"] == "supported"]
        row["facts_total"] = len(mine)
        row["covered"] = []
        if not mine or not claims or row.get("judged") is None:
            row["covered"] = None if row.get("judged") is None and row.get("card") else []
            return
        prompt = COVERAGE_PROMPT.format(
            facts="\n".join(f"- {f['fact_id']}: {f['fact']}" for f in mine),
            claims="\n".join(f"{i}. {t}" for i, t in enumerate(claims)))
        valid = {f["fact_id"] for f in mine}
        for _ in range(3):
            try:
                llm = make_llm(settings, f"fishora-cover-{new_id()}", judge_model, timeout=180, max_retries=1)
                data = reply_json(llm.invoke(prompt))
                row["covered"] = sorted({str(c.get("fact")) for c in data.get("covered", [])
                                         if isinstance(c, dict) and str(c.get("fact")) in valid})
                print(f"[cover] {row['species']:12s} {len(row['covered'])}/{len(mine)}", flush=True)
                return
            except Exception as exc:  # judge outage: exclude the card, never score it 0
                row["cover_error"] = type(exc).__name__
        row["covered"] = None

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(one, rows))


def summarize(rows: list[dict], v1_facts: set[str]) -> dict:
    out = {}
    groups = defaultdict(list)
    for r in rows:
        groups[r["cell"]].append(r)
    for cell, rs in groups.items():
        judged = [r for r in rs if r.get("judged") is not None]
        scored = [r for r in rs if r.get("covered") is not None and r.get("facts_total")]
        sup = [sum(c["label"] == "supported" for c in r["judged"]) for r in judged]
        n = [len(r["judged"]) for r in judged]
        cov = [len(r["covered"]) / r["facts_total"] for r in scored]
        cov_v1 = [len(set(r["covered"]) & v1_facts) / r["facts_total_v1"] for r in scored if r["facts_total_v1"]]
        walls = sorted(r["wall_s"] for r in rs)
        out[cell] = {
            "cards": len(rs), "completed": sum(r["status"] == "completed" for r in rs),
            "cost_usd_per_1000_cards": round(1000 * statistics.fmean(r["cost_usd"] for r in rs), 3),
            "wall_s_p50": walls[len(walls) // 2],
            "claims_per_card": round(statistics.fmean(n), 2) if n else 0,
            "supported_share": round(sum(sup) / sum(n), 3) if sum(n) else None,
            "supported_claims_per_card": round(statistics.fmean(sup), 2) if sup else 0,
            "facts_covered_pooled": f"{sum(len(r['covered']) for r in scored)}/{sum(r['facts_total'] for r in scored)}",
            "fact_coverage": round(sum(len(r["covered"]) for r in scored) / sum(r["facts_total"] for r in scored), 3) if scored else None,
            "fact_coverage_card_mean": round(statistics.fmean(cov), 3) if cov else None, "fact_coverage_ci": _ci(cov),
            "fact_coverage_v1_facts": round(statistics.fmean(cov_v1), 3) if cov_v1 else None, "fact_coverage_v1_ci": _ci(cov_v1),
            "claims_per_covered_fact": round(sum(sup) / max(1, sum(len(r["covered"]) for r in scored)), 2),
            "fields_filled_mean": round(statistics.fmean(r["fields_filled"] for r in rs), 2),
            "coverage_scored_cards": len(scored),
        }
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="iteration-3")
    ap.add_argument("--judge", required=True)
    ap.add_argument("--species", default="")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--out", default="matrix.json")
    ap.add_argument("--judge-only", action="store_true", help="re-judge the saved cards.json")
    args = ap.parse_args(argv)

    from apps.main_api.config import MainSettings
    from apps.main_api.services import claim_verifier, orchestrator
    from apps.main_api.services.embeddings import LocalE5Embedder

    facts = load_facts()
    orchestrator.CRITIC_MODE = "verifier"
    claim_verifier.USE_EMBEDDING_FILTER = True
    settings = MainSettings()
    embedder = LocalE5Embedder()
    embedder.embed_query("warmup")
    v1 = corpus_v1_ids()
    store = build_store(embedder)
    wanted = {s for s in args.species.split(",") if s}
    species = [s for s in species_records() if not wanted or s.normalized_label in wanted]

    t0 = time.perf_counter()
    out = REPORTS_DIR / args.label
    out.mkdir(parents=True, exist_ok=True)
    if args.judge_only:
        rows = json.loads((out / "cards.json").read_text())
    else:
        rows = run_cell("writer_critic", MODEL, species, store, embedder, settings, args.repeat)
        for r in rows:
            r["cell"] = "writer_critic:gpt-6-luna+corpus_v2+atomic_writer"
        (out / "cards.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False, default=str))

    v1_facts = {f["fact_id"] for f in facts if f["chunk_id"] in v1}
    per_species_v1 = defaultdict(int)
    for f in facts:
        if f["fact_id"] in v1_facts:
            per_species_v1[f["species"]] += 1
    print(f"[it3] judging faithfulness of {len(rows)} cards", flush=True)
    judge_cards(rows, args.judge, store, embedder, settings)
    print(f"[it3] faithfulness done ({sum(r.get('judged') is None for r in rows)} judge failures); mapping facts", flush=True)
    cover(rows, facts, set(store._chunks), args.judge, settings)
    for r in rows:
        r["facts_total_v1"] = per_species_v1[r["species"]]

    report = {"label": args.label, "judge": args.judge, "model": MODEL, "repeat": args.repeat,
              "facts": {"v1": len(v1_facts), "v2": len(facts)},
              "wall_s": round(time.perf_counter() - t0, 1), "summary": summarize(rows, v1_facts), "cards": rows}
    (out / args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    for cell, s in report["summary"].items():
        print(f"{cell:24s} ${s['cost_usd_per_1000_cards']:.2f}/1k p50 {s['wall_s_p50']}s claims {s['claims_per_card']} "
              f"supported {s['supported_share']} ({s['supported_claims_per_card']}/card) coverage {s['fact_coverage']} "
              f"[{s['facts_covered_pooled']}] v1-facts {s['fact_coverage_v1_facts']} claims/fact {s['claims_per_covered_fact']}")


if __name__ == "__main__":
    main()

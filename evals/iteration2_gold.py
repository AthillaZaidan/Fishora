"""Iteration 2, step 1: grade the locked gold claims with every candidate critic/judge.

    python -m evals.iteration2_gold --label iteration-2 [--judges deepseek-v4.1-flash,glm-5.3-flash]

The gold set is evals/protocol/claims_gold.csv (221 claims from the 44 real
iteration-1 cards; AI-assisted annotation reviewed by the team, SHA-256 locked
in claims_gold.lock). The script refuses to run if the file changed.

Each claim is graded against every verified chunk of its species whose
category may ground its field, the same evidence the card writer saw. Graders:

* ``e5``: the iteration-1 critic (E5 symmetric cosine >= 0.805 + number/taxon checks)
* ``verifier:<model>``: iteration-2 claim_verifier with that model (deterministic + E5 + LLM)
* ``llm_only:<model>``: the LLM stage alone (no E5 filter), to see what E5 adds or removes
* ``judge:<model>``: a judge from another model family, for scoring new cards later

Reported per grader: 3-class accuracy and Cohen's kappa against gold, and the
binary "may reach a card" view (supported vs not): precision, recall, and the
share of gold-unsupported/inferred claims it lets through, with 95% bootstrap CIs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from evals.corpus import build_store, corpus_v1_ids, species_records
from evals.fakes import new_id
from evals.run import REPORTS_DIR

PROTOCOL = Path(__file__).resolve().parent / "protocol"
LABELS = ("supported", "inferred", "unsupported")


def load_gold() -> list[dict]:
    lock = (PROTOCOL / "claims_gold.lock").read_text().split("\n")[0]
    want = lock.split("sha256=")[1].strip()
    got = hashlib.sha256((PROTOCOL / "claims_gold.csv").read_bytes()).hexdigest()
    if got != want:
        raise SystemExit(f"claims_gold.csv changed since it was locked ({got[:12]} != {want[:12]})")
    with open(PROTOCOL / "claims_gold.csv", newline="") as f:
        return list(csv.DictReader(f))


def kappa(gold: list[str], pred: list[str]) -> float:
    n = len(gold)
    po = sum(g == p for g, p in zip(gold, pred)) / n
    pe = sum((gold.count(k) / n) * (pred.count(k) / n) for k in LABELS)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def binary(gold: list[str], pred: list[str], seed: int = 0) -> dict:
    """'May reach a card' = supported. Precision/recall of that decision."""
    def stats(idx):
        tp = sum(gold[i] == "supported" and pred[i] == "supported" for i in idx)
        fp = sum(gold[i] != "supported" and pred[i] == "supported" for i in idx)
        fn = sum(gold[i] == "supported" and pred[i] != "supported" for i in idx)
        neg = sum(gold[i] != "supported" for i in idx)
        return (tp / (tp + fp) if tp + fp else 1.0, tp / (tp + fn) if tp + fn else 1.0, fp / neg if neg else 0.0)
    idx = list(range(len(gold)))
    point = stats(idx)
    rng = random.Random(seed)
    boots = [stats([rng.choice(idx) for _ in idx]) for _ in range(1000)]
    ci = lambda k: [round(sorted(b[k] for b in boots)[25], 3), round(sorted(b[k] for b in boots)[974], 3)]
    return {"precision": round(point[0], 3), "precision_ci": ci(0), "recall": round(point[1], 3), "recall_ci": ci(1),
            "leak_rate": round(point[2], 3), "leak_rate_ci": ci(2)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="iteration-2")
    ap.add_argument("--critic-models", default="gpt-5.6-luna,gpt-6-luna")
    ap.add_argument("--judges", default="deepseek-v4.1-flash,glm-5.3-flash")
    ap.add_argument("--timeout", type=float, default=45)
    ap.add_argument("--out", default="gold_graders.json")
    ap.add_argument("--no-llm-only", action="store_true")
    args = ap.parse_args(argv)

    from apps.main_api.config import MainSettings
    from apps.main_api.services import claim_verifier
    from apps.main_api.services.embeddings import LocalE5Embedder
    from apps.main_api.services.orchestrator import GROUNDING_TAU, _Verifier
    from evals.cost_eval import PRICES, RecordingLLM, make_llm

    gold = load_gold()
    settings = MainSettings()
    embedder = LocalE5Embedder()
    embedder.embed_query("warmup")
    store = build_store(embedder, only=corpus_v1_ids())  # the evidence the gold labels were made against
    records = {s.normalized_label: s for s in species_records()}
    known = tuple(s.scientific_name for s in records.values() if s.scientific_name)
    from apps.main_api.services.orchestrator import CARD_QUERY
    from apps.main_api.services.retrieval import VerifiedRetriever
    retriever = VerifiedRetriever(store, embedder)
    evidence = {lab: retriever.card_evidence(s.id, CARD_QUERY.format(common_name=s.common_name_id)) for lab, s in records.items()}

    def claims_for(rows):
        out = []
        for i, r in enumerate(rows):
            allowed = claim_verifier.FIELD_EVIDENCE[r["field"]]
            ids = [c.chunk_id for c in evidence[r["species"]] if c.category in allowed]
            out.append(claim_verifier.Claim(id=i, field=r["field"], text=r["claim"], chunk_ids=ids))
        return out

    by_species = defaultdict(list)
    for r in gold:
        by_species[r["species"]].append(r)

    graders: dict[str, dict[str, str]] = {}
    costs: dict[str, dict] = {}

    # iteration-1 critic
    e5 = {}
    for sp, rows in by_species.items():
        verifier = _Verifier(embedder, GROUNDING_TAU, known)
        for r, cl in zip(rows, claims_for(rows)):
            chunks = [c for c in evidence[sp] if c.chunk_id in cl.chunk_ids]
            e5[r["claim_id"]] = "supported" if any(verifier.supports(r["claim"], c) for c in chunks) else "unsupported"
    graders["e5"] = e5

    from concurrent.futures import ThreadPoolExecutor

    def run_species(name, model, use_embedder, sp, rows):
        model, _, effort = model.partition("@")
        rec = RecordingLLM(make_llm(settings, f"fishora-gold-{sp}-{new_id()}", model, timeout=args.timeout, max_retries=1,
                                    reasoning_effort=effort or None), model, path=name)
        cl = claims_for(rows)
        res = claim_verifier.verify(cl, evidence[sp], rec, known, embedder=embedder if use_embedder else None, attempts=2)
        failed = res.llm_error is not None
        print(f"[gold] {name:32s} {sp:12s} {'FAILED ' + str(res.llm_error) if failed else 'ok'}", flush=True)
        # A grader that never answered is excluded for that species, not scored as rejecting.
        return {r["claim_id"]: (None if failed else (c.label or "unsupported")) for r, c in zip(rows, res.claims)}, rec.records

    jobs = [(f"verifier:{m}", m, True) for m in args.critic_models.split(",") if m]
    jobs += [] if args.no_llm_only else [(f"llm_only:{m}", m, False) for m in args.critic_models.split(",") if m]
    jobs += [(f"judge:{m}", m, False) for m in args.judges.split(",") if m and m.partition("@")[0] in PRICES]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=16) as pool:
        futs = {(name, sp): pool.submit(run_species, name, model, emb, sp, rows)
                for name, model, emb in jobs for sp, rows in by_species.items()}
        for name, model, _ in jobs:
            preds, recs = {}, []
            for sp in by_species:
                p, r = futs[(name, sp)].result()
                preds.update(p); recs += r
            graders[name] = preds
            costs[name] = {"model": model, "calls": len(recs), "errors": sum(x.error is not None for x in recs),
                           "cost_usd": round(sum(x.cost_usd for x in recs), 5),
                           "excluded_claims": sum(v is None for v in preds.values())}
    print(f"[gold] graded in {time.perf_counter() - t0:.0f}s", flush=True)

    ids = [r["claim_id"] for r in gold]
    g = [r["label"] for r in gold]
    report = {"label": args.label, "n_claims": len(gold), "gold_counts": {k: g.count(k) for k in LABELS},
              "graders": {}, "costs": costs, "predictions": graders}
    for name, preds in graders.items():
        keep = [i for i, cid in enumerate(ids) if preds.get(cid) is not None]
        gg, p = [g[i] for i in keep], [preds[ids[i]] for i in keep]
        report["graders"][name] = {"n_scored": len(keep), "accuracy_3class": round(sum(a == b for a, b in zip(gg, p)) / len(gg), 3),
                                   "kappa_3class": round(kappa(gg, p), 3), **binary(gg, p),
                                   "pred_counts": {k: p.count(k) for k in LABELS}}
    out = REPORTS_DIR / args.label
    out.mkdir(parents=True, exist_ok=True)
    (out / args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    for name, m in report["graders"].items():
        print(f"{name:32s} n {m['n_scored']:3d} acc3 {m['accuracy_3class']:.3f} kappa {m['kappa_3class']:.3f} | precision {m['precision']:.3f} "
              f"recall {m['recall']:.3f} leak {m['leak_rate']:.3f}")


if __name__ == "__main__":
    main()

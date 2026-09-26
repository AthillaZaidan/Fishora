"""Calibrate the cross-lingual grounding threshold tau on the dev split.

    python -m evals.calibrate_grounding --label current

Sweeps tau for the production verifier (orchestrator._Verifier: E5 symmetric
cosine >= tau and number consistency) over the dev half of the grounding
pairs, picks the tau with the highest dev F1 whose false-support rate stays
<= 0.10, and reports that tau on the held-out test half. The shipped
GROUNDING_TAU is compared with the choice; the output is written to
reports/<label>/grounding_calibration.json.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from apps.main_api.services import orchestrator
from evals.corpus import build_store
from evals.grounding_eval import _classification, grounding_pairs
from evals.run import REPORTS_DIR, _jsonable

MAX_FALSE_SUPPORT = 0.10
GRID = [round(0.75 + 0.005 * i, 3) for i in range(31)]  # 0.750 .. 0.900


def main(argv: list[str] | None = None):
    from apps.main_api.services.embeddings import LocalE5Embedder

    parser = argparse.ArgumentParser(prog="python -m evals.calibrate_grounding")
    parser.add_argument("--label", default="current")
    args = parser.parse_args(argv)

    embedder = LocalE5Embedder()
    store = build_store(embedder)
    pairs = grounding_pairs()
    chunks = {p["chunk_id"]: store._to_retrieved(p["chunk_id"], 0.0) for p in pairs}

    # One verifier per tau shares nothing, so precompute the two ingredients.
    cosine, numbers_ok = [], []
    probe = orchestrator._Verifier(embedder, tau=-1.0)  # tau=-1: supports() == numbers check
    for p in pairs:
        chunk = chunks[p["chunk_id"]]
        numbers_ok.append(probe.supports(p["claim"], chunk))
        a = probe._vector(f"claim:{p['claim']}", p["claim"])
        b = probe._vector(f"chunk:{chunk.chunk_id}", chunk.content)
        cosine.append(sum(x * y for x, y in zip(a, b)))

    def predict(tau):
        return [ok and c >= tau for ok, c in zip(numbers_ok, cosine)]

    dev = [i for i, p in enumerate(pairs) if p["split"] == "dev"]
    test = [i for i, p in enumerate(pairs) if p["split"] == "test"]
    sweep = []
    for tau in GRID:
        pred = predict(tau)
        d = _classification([pairs[i] for i in dev], [pred[i] for i in dev])
        sweep.append({"tau": tau, "dev_f1": d["f1"], "dev_false_support": d["false_support_rate"]})
    eligible = [row for row in sweep if row["dev_false_support"] <= MAX_FALSE_SUPPORT]
    chosen = max(eligible, key=lambda r: (r["dev_f1"], -r["tau"]))["tau"] if eligible else None

    def report(tau):
        pred = predict(tau)
        return {"tau": tau,
                "dev": _classification([pairs[i] for i in dev], [pred[i] for i in dev]),
                "test": _classification([pairs[i] for i in test], [pred[i] for i in test])}

    result = {
        "label": args.label,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rule": "E5 symmetric cosine >= tau AND numbers(claim) subset of numbers(chunk)",
        "max_dev_false_support": MAX_FALSE_SUPPORT,
        "chosen": report(chosen) if chosen is not None else None,
        "shipped": report(orchestrator.GROUNDING_TAU),
        "shipped_matches_choice": chosen == orchestrator.GROUNDING_TAU,
        "sweep": sweep,
    }
    out = REPORTS_DIR / args.label / "grounding_calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=_jsonable), encoding="utf-8")
    c = result["chosen"]
    print(f"[calibrate] dev-chosen tau {chosen}; shipped {orchestrator.GROUNDING_TAU}; "
          f"test F1 {c['test']['f1'] if c else 'n/a'}, test false support {c['test']['false_support_rate'] if c else 'n/a'} -> {out}")
    return out


if __name__ == "__main__":
    main()

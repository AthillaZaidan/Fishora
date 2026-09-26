"""Fact coverage for cards generated before the metric existed.

    python -m evals.coverage_rescore --report reports/iteration-2/matrix.json --judge glm-5.3-flash

Cards already carry the judge's faithfulness labels; this adds the fact
coverage of evals/iteration3.py against the corpus-v1 facts (the evidence
those cards were written from) and writes <report>_coverage.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from evals.corpus import corpus_v1_ids
from evals.iteration2 import _ci
from evals.iteration3 import cover, load_facts


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--judge", required=True)
    args = ap.parse_args(argv)

    from apps.main_api.config import MainSettings

    path = Path(args.report)
    rows = json.loads(path.read_text())["cards"]
    cover(rows, load_facts(), set(corpus_v1_ids()), args.judge, MainSettings())
    groups = defaultdict(list)
    for r in rows:
        groups[f"{r['arch']}:{r['model']}"].append(r)
    summary = {}
    for cell, rs in groups.items():
        scored = [r for r in rs if r.get("covered") is not None and r.get("facts_total")]
        cov = [len(r["covered"]) / r["facts_total"] for r in scored]
        summary[cell] = {
            "fact_coverage": round(sum(len(r["covered"]) for r in scored) / sum(r["facts_total"] for r in scored), 3) if scored else None,
            "fact_coverage_card_mean": round(statistics.fmean(cov), 3) if cov else None, "fact_coverage_ci": _ci(cov),
            "facts_covered_pooled": f"{sum(len(r['covered']) for r in scored)}/{sum(r['facts_total'] for r in scored)}",
            "coverage_scored_cards": len(scored),
        }
        print(f"{cell:40s} coverage {summary[cell]['fact_coverage']} [{summary[cell]['facts_covered_pooled']}]", flush=True)
    out = path.with_name(path.stem + "_coverage.json")
    out.write_text(json.dumps({"judge": args.judge, "facts": "corpus_v1", "summary": summary,
                               "cards": [{k: r.get(k) for k in ("arch", "model", "species", "covered", "facts_total")} for r in rows]},
                              indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

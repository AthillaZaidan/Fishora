"""Corpus gaps as a research task list for the data team (R4, W11).

    python -m evals.corpus_gaps --label current

The pipeline can only abstain on a field with no evidence; filling it needs
new candidate chunks from real sources and a human approval, which no code
path may do. This lists every (species, category) cell without a chunk,
with the reason the research stage recorded, and writes
reports/<label>/corpus_gaps.json.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone

from apps.main_api.services.retrieval import CATEGORY_ORDER
from evals.corpus import ROOT, load_corpus, species_records
from evals.run import REPORTS_DIR

COVERAGE = ROOT / "artifacts" / "knowledge_sources" / "coverage-report.json"


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(prog="python -m evals.corpus_gaps")
    parser.add_argument("--label", default="current")
    args = parser.parse_args(argv)

    have = defaultdict(set)
    for chunk in load_corpus():
        have[chunk.species_label].add(chunk.category)
    reasons = {}
    if COVERAGE.exists():
        for row in json.loads(COVERAGE.read_text(encoding="utf-8")).get("unresolved", []):
            reasons[(row["species_label"], row["category"])] = row.get("reason")

    gaps = [
        {"species": s.normalized_label, "category": category,
         "reason": reasons.get((s.normalized_label, category), "no chunk in the candidate corpus")}
        for s in species_records() for category in CATEGORY_ORDER if category not in have[s.normalized_label]
    ]
    by_category = defaultdict(int)
    for g in gaps:
        by_category[g["category"]] += 1
    result = {
        "label": args.label,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cells": len(species_records()) * len(CATEGORY_ORDER),
        "missing_cells": len(gaps),
        "missing_by_category": dict(sorted(by_category.items(), key=lambda kv: -kv[1])),
        "gaps": gaps,
        "how_to_fill": "research agents -> candidate chunk with a verbatim source quote -> "
                       "scripts.corpus_pipeline collect -> human review -> approve (signed) -> ingest",
    }
    out = REPORTS_DIR / args.label / "corpus_gaps.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[gaps] {len(gaps)} of {result['cells']} cells have no evidence; by category {result['missing_by_category']} -> {out}")
    return out


if __name__ == "__main__":
    main()

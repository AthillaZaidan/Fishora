"""Experiment for W8: does prefixing passages with the canonical species name
fix within-species ranking?

    python -m evals.experiment_passage_prefix --label current

Hypothesis: identity chunks win every query because only they contain the
Indonesian common name, so embedding "<common> (<binomial>): <text>" for every
passage should lift the per-category MRR. Measured with the production
retriever on the 88 gold queries, with and without the prefix; the result
goes to reports/<label>/experiment_passage_prefix.json.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from evals import corpus, retrieval_eval
from evals.corpus import load_corpus, species_records
from evals.run import REPORTS_DIR, _jsonable


def main(argv: list[str] | None = None):
    from apps.main_api.services.embeddings import LocalE5Embedder

    parser = argparse.ArgumentParser(prog="python -m evals.experiment_passage_prefix")
    parser.add_argument("--label", default="current")
    args = parser.parse_args(argv)

    embedder = LocalE5Embedder()
    names = {s.normalized_label: (s.common_name_id, s.scientific_name) for s in species_records()}
    original = embedder.embed_passages

    def build(prefix: bool):
        def embed(texts):
            if not prefix:
                return original(texts)
            out = []
            for chunk, text in zip(load_corpus(), texts):
                common, binomial = names[chunk.species_label]
                out.append(f"{common} ({binomial}): {text}" if binomial else f"{common}: {text}")
            return original(out)
        embedder.embed_passages = embed
        try:
            return corpus.build_store(embedder)
        finally:
            embedder.embed_passages = original

    arms = {}
    for name, prefix in (("control", False), ("prefix", True)):
        r = retrieval_eval.evaluate(build(prefix), embedder)
        arms[name] = {"scoped": r["scoped"], "global_species_hit@1": r["global"]["species_hit@1"],
                      "card_query_coverage": r["card_query_coverage"]["mean"],
                      "mrr_by_category": {k: v["scoped_mrr"] for k, v in r["by_category"].items()}}
    delta = {k: round(arms["prefix"]["mrr_by_category"][k] - arms["control"]["mrr_by_category"][k], 4)
             for k in arms["control"]["mrr_by_category"]}
    result = {
        "label": args.label,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hypothesis": "canonical-name passage prefix lifts within-species per-category MRR (W8)",
        "arms": arms,
        "mrr_delta_by_category": delta,
        "supported": any(v > 0.01 for v in delta.values()),
    }
    out = REPORTS_DIR / args.label / "experiment_passage_prefix.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=_jsonable), encoding="utf-8")
    print(f"[experiment] hypothesis supported: {result['supported']}; MRR deltas {delta} -> {out}")
    return out


if __name__ == "__main__":
    main()

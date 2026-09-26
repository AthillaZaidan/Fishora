"""Experiment for R2: embedding similarity vs multilingual NLI entailment as
the claim verifier.

    python -m evals.experiment_nli_grounding --label current

Pre-registered rule (written before the run):
  * variants: A = E5 symmetric cosine >= tau AND exact numbers/taxa (shipped),
    B = NLI P(entailment | chunk => claim) >= theta,
    C = B AND exact numbers/taxa;
  * each threshold is chosen on the dev split: highest dev F1 with dev
    false-support <= 0.10;
  * the variant with the highest dev F1 wins; within 0.02 of it, the one that
    accepts fewer traps wins; test split and trap set are reported only.
Writes reports/<label>/experiment_nli_grounding.json.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

from apps.main_api.services import orchestrator
from evals.corpus import build_store, load_dataset
from evals.grounding_eval import _classification, grounding_pairs
from evals.run import REPORTS_DIR, _jsonable

NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
MAX_FSR = 0.10


def nli_scorer():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(NLI_MODEL, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL, local_files_only=True).eval()
    entail = model.config.label2id.get("entailment", 0)

    def score(premise: str, hypothesis: str) -> float:
        with torch.no_grad():
            enc = tok(premise, hypothesis, truncation=True, max_length=512, return_tensors="pt")
            return float(torch.softmax(model(**enc).logits[0], -1)[entail])
    return score


def choose(scores, labels_dev_idx, pairs, grid):
    best = None
    for t in grid:
        pred = [s >= t for s in scores]
        m = _classification([pairs[i] for i in labels_dev_idx], [pred[i] for i in labels_dev_idx])
        if m["false_support_rate"] <= MAX_FSR and (best is None or m["f1"] > best[1]["f1"]):
            best = (t, m)
    return best


def main(argv: list[str] | None = None):
    from apps.main_api.services.embeddings import LocalE5Embedder

    parser = argparse.ArgumentParser(prog="python -m evals.experiment_nli_grounding")
    parser.add_argument("--label", default="current")
    parser.add_argument("--with-llm", action="store_true",
                        help="also run variant D: the production critic with the real LLM judge (needs OPENCODE_GO_API_KEY)")
    args = parser.parse_args(argv)

    embedder = LocalE5Embedder()
    store = build_store(embedder)
    pairs = grounding_pairs()
    traps = [{**t, "label": False} for t in load_dataset("grounding_traps.json")["traps"]]
    rows = pairs + traps
    chunk = lambda cid: store._to_retrieved(cid, 0.0)

    exact = orchestrator._Verifier(embedder, tau=-1.0)  # tau -1: only numbers and taxa
    cos, ok = [], []
    for r in rows:
        c = chunk(r["chunk_id"])
        ok.append(exact.supports(r["claim"], c))
        a = exact._vector(f"claim:{r['claim']}", r["claim"])
        b = exact._vector(f"chunk:{c.chunk_id}", c.content)
        cos.append(sum(x * y for x, y in zip(a, b)))

    score = nli_scorer()
    t0 = time.perf_counter()
    nli = [score(chunk(r["chunk_id"]).content, r["claim"]) for r in rows]
    nli_ms = (time.perf_counter() - t0) * 1000 / len(rows)

    n = len(pairs)
    dev = [i for i in range(n) if pairs[i]["split"] == "dev"]
    test = [i for i in range(n) if pairs[i]["split"] == "test"]
    trap_idx = list(range(n, len(rows)))
    variants = {
        "A_e5_exact": ([c if k else -1.0 for c, k in zip(cos, ok)], [round(0.75 + 0.005 * i, 3) for i in range(31)]),
        "B_nli": (nli, [round(0.05 * i, 2) for i in range(1, 20)]),
        "C_nli_exact": ([s if k else -1.0 for s, k in zip(nli, ok)], [round(0.05 * i, 2) for i in range(1, 20)]),
    }
    out = {}
    for name, (scores, grid) in variants.items():
        picked = choose(scores, dev, rows, grid)
        if picked is None:
            out[name] = {"threshold": None}
            continue
        t, dev_m = picked
        pred = [s >= t for s in scores]
        out[name] = {
            "threshold": t,
            "dev": dev_m,
            "test": _classification([rows[i] for i in test], [pred[i] for i in test]),
            "traps_accepted": sum(pred[i] for i in trap_idx),
            "traps_total": len(trap_idx),
            "traps_accepted_ids": [rows[i]["chunk_id"] + ":" + rows[i]["kind"] for i in trap_idx if pred[i]],
        }
    if args.with_llm:
        # D: the production critic as it runs with a key: variant A, then the
        # LLM entailment pass (may only downgrade; reads whole chunks, R3).
        from apps.main_api.config import MainSettings
        from apps.main_api.services.generation import make_opencode_go_llm

        llm = make_opencode_go_llm(MainSettings(), session_id="fishora-eval-grounding-judge")
        pred = []
        for r in rows:
            c = chunk(r["chunk_id"])
            state = {"refined_evidence": [c], "expert_outputs": {"physical": {
                "physical_characteristics": r["claim"], "sources": [{"source_id": c.source_id, "chunk_id": c.chunk_id}]}}}
            fixed = {**c.__dict__, "category": "physical_characteristics"}  # grade against the field's own category
            state["refined_evidence"] = [type(c)(**fixed)]
            status = orchestrator.critic_node(state, llm, embedder=embedder)["claim_statuses"][0]
            pred.append(status.status == "supported")
        out["D_e5_exact_llm_judge"] = {
            "threshold": orchestrator.GROUNDING_TAU,
            "dev": _classification([rows[i] for i in dev], [pred[i] for i in dev]),
            "test": _classification([rows[i] for i in test], [pred[i] for i in test]),
            "traps_accepted": sum(pred[i] for i in trap_idx),
            "traps_total": len(trap_idx),
            "traps_accepted_ids": [rows[i]["chunk_id"] + ":" + rows[i]["kind"] for i in trap_idx if pred[i]],
            "judge_model": MainSettings().opencode_go_model,
        }
    ranked = sorted((k for k in out if out[k].get("threshold") is not None and k != "D_e5_exact_llm_judge"),
                    key=lambda k: -out[k]["dev"]["f1"])
    top = out[ranked[0]]["dev"]["f1"]
    close = [k for k in ranked if top - out[k]["dev"]["f1"] <= 0.02]
    winner = min(close, key=lambda k: (out[k]["traps_accepted"], -out[k]["dev"]["f1"]))
    result = {
        "label": args.label,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "nli_model": NLI_MODEL,
        "nli_ms_per_pair_cpu": round(nli_ms, 1),
        "rule": "highest dev F1 at dev FSR <= 0.10; within 0.02, fewer traps accepted",
        "variants": out,
        "winner": winner,
    }
    path = REPORTS_DIR / args.label / "experiment_nli_grounding.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=_jsonable), encoding="utf-8")
    for k, v in out.items():
        if v.get("threshold") is None:
            print(f"[nli] {k}: no threshold meets the false-support cap on dev")
            continue
        print(f"[nli] {k}: thr {v['threshold']} dev F1 {v['dev']['f1']} | test F1 {v['test']['f1']} FSR {v['test']['false_support_rate']} "
              f"recall {v['test']['recall']} | traps accepted {v['traps_accepted']}/{v['traps_total']}")
    print(f"[nli] winner {winner}; NLI {nli_ms:.0f} ms/pair on CPU -> {path}")
    return path


if __name__ == "__main__":
    main()

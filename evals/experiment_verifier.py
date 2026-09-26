"""Experiment for W26 and R2 (iteration 2): which claim verifier, on real claim atoms.

    python -m evals.experiment_verifier --label current

Iteration 1 chose the verifier on synthetic claims (grounding_claims.json), where
E5 + exact checks scored recall 1.0. Real expert claims, split into atoms
(evals/claim_atoms.py), are shorter translations of longer chunks, and E5
drops true ones ("rasa ringan dan berdaging" vs "a mild, meaty flavor": 0.78 <
tau). NLI entailment reads those correctly but accepts some antonyms, and costs
about 0.45 s per pair on CPU. This experiment measures both on the same data.

Pre-registered rule (written before the run):
  * data: dev = synthetic dev + real dev (grounding_real_claims.json, split by
    species); test = synthetic test + real test. Dev traps = the iteration-1
    trap set (grounding_traps.json); test traps = the fresh iteration-2 set in
    grounding_real_claims.json, never used for any choice;
  * variants (premise = chunk, hypothesis = claim; "exact" = every number and
    taxon of the claim is in the chunk):
      A  E5 cosine >= 0.805 AND exact                       (shipped, not re-tuned)
      A2 E5 cosine >= tau AND exact                          (tau re-tuned on dev)
      C  NLI P(entail) >= theta AND exact
      V  A AND NLI P(contradiction) < kappa                  (NLI veto)
      H  (E5 >= 0.805 OR NLI P(entail) >= theta) AND exact AND P(contradiction) < kappa;
  * each variant's thresholds: highest dev F1 with dev false-support <= 0.10
    (grids: tau 0.70-0.90 by 0.005; theta 0.05-0.95 by 0.05; kappa 0.3/0.5/0.7);
  * eligible: the verifier adds at most 5 s per card at p95 on the production
    CPU (pairs per card from the real runs x measured CPU ms per NLI pair);
  * winner: the eligible variant with the highest dev F1; within 0.02 of it,
    fewer dev traps accepted; then fewer NLI pairs per card. Every variant's
    test and fresh-trap numbers are reported, eligible or not.
  * NLI models: mDeBERTa-v3-base (the R2 candidate), and, added before their
    run on the owner's choice, two compact multilingual models (MiniLMv2 L6
    and L12, MNLI + XNLI) that fit the CPU budget. Each model gets its own C,
    V and H variants under the same rule.

Confirmation (added after the first full run, on the owner's decision, so it is
not part of the pre-registered rule): the winner replaces the shipped verifier
only if, on the test split, its F1 is not lower than A's and its false-support
rate is not higher than A's. The first run's winner (H with MiniLM-L12) had test
F1 0.809 vs 0.800 but false support 0.235 vs 0.165, so E5 stayed.

Production verifier (added after the team's iteration 2 was merged, reported
only, never selected on): with --with-llm, the claim_verifier the product now
runs (deterministic checks, E5 filter, one LLM entailment call) grades the same
rows and traps with the production default model, one call per chunk.
Writes reports/<label>/experiment_verifier.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import time
from datetime import datetime, timezone

from apps.main_api.services import orchestrator
from evals.corpus import load_corpus, load_dataset, species_records
from evals.experiment_nli_grounding import NLI_MODEL

NLI_MODELS = {
    "mdeberta": NLI_MODEL,
    "minilm_l6": "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli",
    "minilm_l12": "MoritzLaurer/multilingual-MiniLMv2-L12-mnli-xnli",
}
from evals.grounding_eval import _classification, grounding_pairs
from evals.run import REPORTS_DIR, _jsonable

MAX_FSR = 0.10
CPU_BUDGET_S = 5.0
SHIPPED_TAU = 0.805
TAUS = [round(0.70 + 0.005 * i, 3) for i in range(41)]
THETAS = [round(0.05 * i, 2) for i in range(1, 20)]
KAPPAS = [0.3, 0.5, 0.7]
RUNS = ("iter2-diagnosis", "iter2-claims-2", "iter2-claims-3")


def _nli(device: str, name: str = NLI_MODEL):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(name, local_files_only=True).eval().to(device)
    entail, contra = model.config.label2id["entailment"], model.config.label2id["contradiction"]

    def score(pairs: list[tuple[str, str]], batch: int = 16) -> list[tuple[float, float]]:
        out = []
        with torch.inference_mode():
            for i in range(0, len(pairs), batch):
                chunk = pairs[i:i + batch]
                enc = tok([p for p, _ in chunk], [h for _, h in chunk], truncation=True, max_length=512,
                          padding=True, return_tensors="pt").to(device)
                probs = torch.softmax(model(**enc).logits, -1).cpu()
                out += [(float(r[entail]), float(r[contra])) for r in probs]
        return out
    return score


def _pairs_per_card() -> list[int]:
    """Atom x cited-chunk pairs the critic grades per card, from the real runs."""
    from apps.main_api.services.orchestrator import FIELD_EVIDENCE, _CLAIM_OWNER
    from evals.claim_atoms import claim_atoms

    chunks = {c.id: c for c in load_corpus()}
    counts = []
    for label in RUNS:
        path = REPORTS_DIR / label / "cost_eval.json"
        if not path.exists():
            continue
        for card in json.loads(path.read_text(encoding="utf-8"))["cards"]:
            outputs, total = card.get("expert_outputs") or {}, 0
            for field, owner in _CLAIM_OWNER.items():
                out = outputs.get(owner) or {}
                cited = [s for s in out.get("sources", []) if s.get("chunk_id") in chunks
                         and chunks[s["chunk_id"]].category in FIELD_EVIDENCE[field]]
                total += len(claim_atoms(out.get(field))) * len(cited)
            counts.append(total)
    return counts


def _card_policy(verifier) -> dict:
    """Card-level effect of grading whole claims (shipped) vs atom by atom, with
    the shipped verifier, on the real runs. Truth per atom: some cited chunk of
    the field supports it (the labels). Whole: a field keeps every atom when one
    cited chunk supports the joined claim. Atomic: each atom is kept when one
    cited chunk supports it. Reported, not selected on."""
    from apps.main_api.services.orchestrator import FIELD_EVIDENCE, _CLAIM_OWNER, _claim_text
    from evals.claim_atoms import claim_atoms

    chunks = {c.id: c for c in load_corpus()}
    real = load_dataset("grounding_real_claims.json")
    label = {(p["claim"].casefold(), p["chunk_id"]): p["supported"] for p in real["pairs"]}
    as_chunk = lambda cid: type("C", (), {"content": chunks[cid].content, "chunk_id": cid})()
    totals = {k: 0 for k in ("true", "whole_kept", "whole_kept_true", "atomic_kept", "atomic_kept_true")}
    for run in RUNS:
        path = REPORTS_DIR / run / "cost_eval.json"
        if not path.exists():
            continue
        for card in json.loads(path.read_text(encoding="utf-8"))["cards"]:
            outputs = card.get("expert_outputs") or {}
            for field, owner in _CLAIM_OWNER.items():
                out = outputs.get(owner) or {}
                cited = [s["chunk_id"] for s in out.get("sources", []) if s.get("chunk_id") in chunks
                         and chunks[s["chunk_id"]].category in FIELD_EVIDENCE[field]]
                atoms = claim_atoms(out.get(field))
                if not atoms or not cited:
                    continue
                truth = [any(label.get((a.casefold(), c), False) for c in cited) for a in atoms]
                whole = any(verifier.supports(_claim_text(out.get(field)), as_chunk(c)) for c in cited)
                atomic = [any(verifier.supports(a, as_chunk(c)) for c in cited) for a in atoms]
                totals["true"] += sum(truth)
                if whole:
                    totals["whole_kept"] += len(atoms)
                    totals["whole_kept_true"] += sum(truth)
                totals["atomic_kept"] += sum(atomic)
                totals["atomic_kept_true"] += sum(t and k for t, k in zip(truth, atomic))
    ratio = lambda a, b: round(a / b, 4) if b else None
    return {**totals,
            "whole_precision": ratio(totals["whole_kept_true"], totals["whole_kept"]),
            "whole_recall": ratio(totals["whole_kept_true"], totals["true"]),
            "atomic_precision": ratio(totals["atomic_kept_true"], totals["atomic_kept"]),
            "atomic_recall": ratio(totals["atomic_kept_true"], totals["true"])}


_FIELD_FOR_CATEGORY = {"identity": "physical_characteristics", "physical_characteristics": "physical_characteristics",
                       "taste_texture": "taste", "processing_methods": "processing_methods",
                       "commercial_uses": "commercial_uses", "substitutes": "similar_or_substitute_species"}


def _production_verifier(rows, embedder, known) -> tuple[list[bool], dict]:
    """The claim_verifier the product runs, on every row, grouped by cited chunk."""
    from apps.main_api.config import MainSettings
    from apps.main_api.contracts import RetrievedChunk
    from apps.main_api.services import claim_verifier
    from apps.main_api.services.generation import make_opencode_go_llm

    model = MainSettings.model_fields["opencode_go_model"].default
    settings = MainSettings().model_copy(update={"opencode_go_model": model})
    llm = make_opencode_go_llm(settings, session_id="fishora-eval-claim-verifier")
    corpus = {c.id: c for c in load_corpus()}

    def chunk(cid: str) -> RetrievedChunk:
        c = corpus[cid]
        return RetrievedChunk(chunk_id=cid, species_id=f"species_{c.species_label}", source_id=c.source["id"],
                              source_type=c.source.get("source_type", ""), category=c.category, content=c.content,
                              distance=0.0, chunk_verification_status="verified", source_verification_status="verified",
                              source_title=None, source_publisher=None, source_url=None, source_reviewed_at=None)

    by_chunk: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_chunk.setdefault(r["chunk_id"], []).append(i)
    pred = [False] * len(rows)
    calls, errors = 0, 0
    for cid, ids in by_chunk.items():
        ev = chunk(cid)
        claims = [claim_verifier.Claim(id=n, field=rows[i].get("field") or _FIELD_FOR_CATEGORY[ev.category],
                                       text=rows[i]["claim"], chunk_ids=[cid]) for n, i in enumerate(ids)]
        result = claim_verifier.verify(claims, [ev], llm, known, embedder=embedder)
        calls += result.llm_calls
        errors += result.llm_error is not None
        for n, i in enumerate(ids):
            pred[i] = claims[n].label == "supported"
    return pred, {"model": model, "llm_calls": calls, "chunks_with_llm_error": errors}


def _pick(rows, dev, predict, grid):
    best = None
    for params in grid:
        pred = predict(*params)
        m = _classification([rows[i] for i in dev], [pred[i] for i in dev])
        if m["false_support_rate"] <= MAX_FSR and (best is None or m["f1"] > best[1]["f1"]):
            best = (params, m, pred)
    return best


def main(argv: list[str] | None = None):
    import torch

    from apps.main_api.services.embeddings import LocalE5Embedder

    parser = argparse.ArgumentParser(prog="python -m evals.experiment_verifier")
    parser.add_argument("--label", default="current")
    parser.add_argument("--with-llm", action="store_true",
                        help="also grade every row with the production claim_verifier (needs OPENCODE_GO_API_KEY)")
    args = parser.parse_args(argv)

    content = {c.id: c.content for c in load_corpus()}
    real = load_dataset("grounding_real_claims.json")
    rows = [{**p, "source": "synthetic"} for p in grounding_pairs()]
    rows += [{"claim": p["claim"], "chunk_id": p["chunk_id"], "label": p["supported"], "split": p["split"],
              "source": "real", "field": p["field"]} for p in real["pairs"]]
    n_pairs = len(rows)
    rows += [{**t, "label": False, "split": "trap_dev", "source": "trap_dev"}
             for t in load_dataset("grounding_traps.json")["traps"]]
    rows += [{**t, "label": False, "split": "trap_test", "source": "trap_test"} for t in real["traps"]]
    idx = lambda split, source=None: [i for i, r in enumerate(rows) if r["split"] == split
                                      and (source is None or r["source"] == source)]
    dev, test = idx("dev"), idx("test")
    trap_dev, trap_test = idx("trap_dev"), idx("trap_test")

    embedder = LocalE5Embedder()
    known = [s.scientific_name for s in species_records() if s.scientific_name]
    exact_v = orchestrator._Verifier(embedder, tau=-1.0, known_binomials=known)
    chunk = lambda r: type("C", (), {"content": content[r["chunk_id"]], "chunk_id": r["chunk_id"]})()
    exact, cos = [], []
    for r in rows:
        c = chunk(r)
        exact.append(exact_v.supports(r["claim"], c))  # tau -1: only the number and taxon checks
        a, b = exact_v._vector(f"claim:{r['claim']}", r["claim"]), exact_v._vector(f"chunk:{c.chunk_id}", c.content)
        cos.append(sum(x * y for x, y in zip(a, b)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    per_card = _pairs_per_card()
    p95 = sorted(per_card)[max(0, round(0.95 * (len(per_card) - 1)))] if per_card else None
    e5_ok = lambda tau: [c >= tau and k for c, k in zip(cos, exact)]
    variants = {
        "A_e5_exact": (lambda: e5_ok(SHIPPED_TAU), [()], None),
        "A2_e5_exact_retuned": (lambda tau: e5_ok(tau), [(t,) for t in TAUS], None),
    }
    models = {}
    sample = [(content[r["chunk_id"]], r["claim"]) for r in rows[:16]]
    for key, name in NLI_MODELS.items():
        started = time.perf_counter()
        nli = _nli(device, name)([(content[r["chunk_id"]], r["claim"]) for r in rows])
        nli_s = time.perf_counter() - started
        cpu_score = _nli("cpu", name)
        cpu_score(sample[:8], batch=8)  # warm-up
        t0 = time.perf_counter()
        cpu_score(sample, batch=8)
        cpu_ms = (time.perf_counter() - t0) * 1000 / len(sample)
        models[key] = {"name": name, "cpu_ms_per_pair": round(cpu_ms, 1), f"{device}_s_all_pairs": round(nli_s, 1)}
        ent = [e for e, _ in nli]
        con = [c for _, c in nli]
        variants.update({
            f"C_{key}_exact": (lambda th, ent=ent: [e >= th and k for e, k in zip(ent, exact)],
                               [(t,) for t in THETAS], key),
            f"V_{key}_veto": (lambda ka, con=con: [a and c < ka for a, c in zip(e5_ok(SHIPPED_TAU), con)],
                              [(k,) for k in KAPPAS], key),
            f"H_{key}_e5_or_nli_veto": (lambda th, ka, ent=ent, con=con: [
                ((co >= SHIPPED_TAU) or (e >= th)) and k and c < ka for co, e, k, c in zip(cos, ent, exact, con)],
                list(itertools.product(THETAS, KAPPAS)), key),
        })
    out = {}
    for name, (predict, grid, model_key) in variants.items():
        uses_nli = model_key is not None
        cpu_ms = models[model_key]["cpu_ms_per_pair"] if uses_nli else 0.0
        picked = _pick(rows, dev, predict, grid)
        if picked is None:
            out[name] = {"params": None, "uses_nli": uses_nli}
            continue
        params, dev_m, pred = picked
        added_s = round(p95 * cpu_ms / 1000, 2) if uses_nli and p95 is not None else 0.0
        split = lambda ids: _classification([rows[i] for i in ids], [pred[i] for i in ids])
        out[name] = {
            "params": list(params),
            "uses_nli": uses_nli,
            "nli_model": models[model_key]["name"] if uses_nli else None,
            "dev": dev_m,
            "test": split(test),
            "test_real": split(idx("test", "real")),
            "test_synthetic": split(idx("test", "synthetic")),
            "dev_traps_accepted": sum(pred[i] for i in trap_dev), "dev_traps_total": len(trap_dev),
            "test_traps_accepted": sum(pred[i] for i in trap_test), "test_traps_total": len(trap_test),
            "test_traps_accepted_ids": [f"{rows[i]['id']}:{rows[i]['kind']}" for i in trap_test if pred[i]],
            "cpu_added_s_per_card_p95": added_s,
            "eligible": added_s <= CPU_BUDGET_S,
        }
    ranked = sorted((k for k, v in out.items() if v.get("params") is not None and v["eligible"]),
                    key=lambda k: -out[k]["dev"]["f1"])
    top = out[ranked[0]]["dev"]["f1"]
    close = [k for k in ranked if top - out[k]["dev"]["f1"] <= 0.02]
    winner = min(close, key=lambda k: (out[k]["dev_traps_accepted"], out[k]["uses_nli"], -out[k]["dev"]["f1"]))
    if args.with_llm:
        pred, meta = _production_verifier(rows, embedder, known)
        split = lambda ids: _classification([rows[i] for i in ids], [pred[i] for i in ids])
        out["P_production_claim_verifier"] = {
            "params": [], "uses_nli": False, "nli_model": None, "reported_only": True, **meta,
            "dev": split(dev), "test": split(test), "test_real": split(idx("test", "real")),
            "test_synthetic": split(idx("test", "synthetic")),
            "dev_traps_accepted": sum(pred[i] for i in trap_dev), "dev_traps_total": len(trap_dev),
            "test_traps_accepted": sum(pred[i] for i in trap_test), "test_traps_total": len(trap_test),
            "test_traps_accepted_ids": [f"{rows[i]['id']}:{rows[i]['kind']}" for i in trap_test if pred[i]],
            "cpu_added_s_per_card_p95": 0.0, "eligible": True,
        }
    shipped, pick = out["A_e5_exact"]["test"], out[winner]["test"]
    confirmed = winner == "A_e5_exact" or (
        pick["f1"] >= shipped["f1"] and pick["false_support_rate"] <= shipped["false_support_rate"])
    result = {
        "label": args.label,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "nli_models": models,
        "nli_device": device,
        "pairs_per_card": {"n": len(per_card), "mean": round(statistics.fmean(per_card), 1) if per_card else None,
                           "p95": p95},
        "cpu_budget_s_per_card": CPU_BUDGET_S,
        "counts": {"dev": len(dev), "test": len(test), "dev_traps": len(trap_dev), "test_traps": len(trap_test),
                   "real_pairs": sum(r["source"] == "real" for r in rows[:n_pairs])},
        "rule": "eligible (CPU <= 5 s/card p95); highest dev F1 at dev FSR <= 0.10; within 0.02, "
                "fewer dev traps accepted, then no NLI",
        "variants": out,
        "card_grading_policy": _card_policy(orchestrator._Verifier(embedder, known_binomials=known)),
        "winner": winner,
        "confirmation": "test F1 >= A's and test false support <= A's (added after the first run)",
        "confirmed": confirmed,
        "verifier_in_use": winner if confirmed else "A_e5_exact",
    }
    path = REPORTS_DIR / args.label / "experiment_verifier.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=_jsonable), encoding="utf-8")
    for k, v in out.items():
        if v.get("params") is None:
            print(f"[verifier] {k}: no setting meets the dev false-support cap")
            continue
        print(f"[verifier] {k} {v['params']}: dev F1 {v['dev']['f1']} | test F1 {v['test']['f1']} "
              f"(real {v['test_real']['f1']}, recall {v['test_real']['recall']}, FSR {v['test_real']['false_support_rate']}) "
              f"| traps dev {v['dev_traps_accepted']}/{v['dev_traps_total']} fresh {v['test_traps_accepted']}/{v['test_traps_total']} "
              f"| +{v['cpu_added_s_per_card_p95']}s/card CPU {'eligible' if v['eligible'] else 'NOT eligible'}")
    print(f"[verifier] winner {winner} ({'confirmed' if confirmed else 'not confirmed on test: E5 stays'}); CPU ms/pair {({k: v['cpu_ms_per_pair'] for k, v in models.items()})}, "
          f"pairs/card p95 {p95} -> {path}")
    return path


if __name__ == "__main__":
    main()

"""Grounding (faithfulness proxy): does the critic accept exactly the claims
their cited chunk supports?

Pairs are (Indonesian claim, cited English chunk). Positives: a chunk's own
paraphrase. Hard negatives: another species' paraphrase in the same category
cited against this chunk: the plausible borrowed fact a model produces when
it fills an empty field. Each pair is graded by the orchestrator's real
``critic_node``, so the eval measures whatever grader the code ships.

Chunks alternate into dev/test by sorted id. Thresholds are chosen on dev
only; the headline numbers are test-split numbers.
"""

from __future__ import annotations

import inspect

import numpy as np

from apps.main_api.services import orchestrator
from evals.corpus import load_corpus, load_dataset


def grounding_pairs() -> list[dict]:
    corpus = {chunk.id: chunk for chunk in load_corpus()}
    claims = load_dataset("grounding_claims.json")["claims"]
    missing = set(corpus) - set(claims)
    if missing:
        raise ValueError(f"grounding dataset lacks claims for {sorted(missing)}")
    ordered = sorted(corpus)
    pairs = []
    for index, chunk_id in enumerate(ordered):
        chunk = corpus[chunk_id]
        split = "dev" if index % 2 == 0 else "test"
        pairs.append({"chunk_id": chunk_id, "claim": claims[chunk_id], "label": True,
                      "split": split, "category": chunk.category})
        negative = next(
            (other for other in ordered
             if corpus[other].species_label != chunk.species_label and corpus[other].category == chunk.category),
            None,
        ) or next(other for other in ordered if corpus[other].species_label != chunk.species_label)
        pairs.append({"chunk_id": chunk_id, "claim": claims[negative], "label": False,
                      "split": split, "category": chunk.category, "borrowed_from": negative})
    return pairs


def _critic_accepts(claim: str, chunk, embedder) -> bool:
    # Graded under the physical field, so present the chunk in that field's
    # category: this measures the verifier, not the field-to-category routing
    # (R4), which unit tests cover separately.
    from dataclasses import replace

    chunk = replace(chunk, category="physical_characteristics")
    state = {
        "refined_evidence": [chunk],
        "expert_outputs": {
            "physical": {
                "physical_characteristics": claim,
                "sources": [{"source_id": chunk.source_id, "chunk_id": chunk.chunk_id}],
            }
        },
    }
    kwargs = {}
    if "embedder" in inspect.signature(orchestrator.critic_node).parameters:
        kwargs["embedder"] = embedder
    result = orchestrator.critic_node(state, None, **kwargs)
    status = next(s for s in result["claim_statuses"] if s.field == "physical_characteristics")
    return status.status == "supported"


def _classification(pairs: list[dict], predictions: list[bool]) -> dict:
    tp = sum(p["label"] and y for p, y in zip(pairs, predictions))
    fp = sum((not p["label"]) and y for p, y in zip(pairs, predictions))
    fn = sum(p["label"] and not y for p, y in zip(pairs, predictions))
    tn = sum((not p["label"]) and not y for p, y in zip(pairs, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": len(pairs),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round((tp + tn) / len(pairs), 4),
        "false_support_rate": round(fp / (fp + tn), 4) if fp + tn else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def similarity_analysis(pairs: list[dict], store, embedder) -> dict:
    """Code-independent calibration data: E5 symmetric cosine of each pair,
    and the dev-optimal threshold (max F1 with false-support <= 0.1)."""
    texts = {p["chunk_id"]: store._to_retrieved(p["chunk_id"], 0.0).content for p in pairs}
    claim_vecs = np.array([embedder.embed_query(p["claim"]) for p in pairs])
    chunk_vecs = {cid: np.array(embedder.embed_query(text)) for cid, text in texts.items()}
    scores = [float(claim_vecs[i] @ chunk_vecs[p["chunk_id"]]) for i, p in enumerate(pairs)]
    dev = [(p, s) for p, s in zip(pairs, scores) if p["split"] == "dev"]
    best = None
    for threshold in np.arange(0.70, 0.95, 0.005):
        m = _classification([p for p, _ in dev], [bool(s >= threshold) for _, s in dev])
        if m["false_support_rate"] <= 0.1 and (best is None or m["f1"] > best[1]["f1"]):
            best = (round(float(threshold), 3), m)
    pos = [s for p, s in zip(pairs, scores) if p["label"]]
    neg = [s for p, s in zip(pairs, scores) if not p["label"]]
    return {
        "positive_mean": round(float(np.mean(pos)), 4),
        "negative_mean": round(float(np.mean(neg)), 4),
        "positive_min": round(float(np.min(pos)), 4),
        "negative_max": round(float(np.max(neg)), 4),
        "dev_best_threshold": best[0] if best else None,
        "dev_best": best[1] if best else None,
        "histogram": {
            "bins": [round(0.6 + 0.02 * i, 2) for i in range(21)],
            "positive": np.histogram(pos, bins=20, range=(0.6, 1.0))[0].tolist(),
            "negative": np.histogram(neg, bins=20, range=(0.6, 1.0))[0].tolist(),
        },
    }


def evaluate(store, embedder) -> dict:
    pairs = grounding_pairs()
    predictions = [
        _critic_accepts(p["claim"], store._to_retrieved(p["chunk_id"], 0.0), embedder) for p in pairs
    ]
    test = [(p, y) for p, y in zip(pairs, predictions) if p["split"] == "test"]
    dev = [(p, y) for p, y in zip(pairs, predictions) if p["split"] == "dev"]
    return {
        "pairs": len(pairs),
        "grader": "E5 cosine + numbers" if "embedder" in inspect.signature(orchestrator.critic_node).parameters else "lexical",
        "test": _classification([p for p, _ in test], [y for _, y in test]),
        "dev": _classification([p for p, _ in dev], [y for _, y in dev]),
        "all": _classification(pairs, predictions),
        "errors": [
            {"chunk_id": p["chunk_id"], "label": "supported" if p["label"] else "unsupported",
             "predicted": "supported" if y else "unsupported", "claim": p["claim"][:140]}
            for p, y in zip(pairs, predictions) if p["label"] != y
        ][:12],
        "similarity": similarity_analysis(pairs, store, embedder),
    }

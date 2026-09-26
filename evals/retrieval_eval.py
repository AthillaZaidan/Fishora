"""Retrieval quality: the production VerifiedRetriever over the real corpus.

Context recall/precision in the RAGAS sense, with gold relevance from the
corpus itself: a query about (species, category) is answered by every chunk of
that cell. Two modes:

* scoped: the production path (species filter + category-first selection);
* global: the same embeddings ranked across all species, an eval-only probe
  of how well E5 separates species when the filter is not there to help.
"""

from __future__ import annotations

import inspect
import math
import statistics
import time
from collections import defaultdict

from apps.main_api.services import chunking
from apps.main_api.services.retrieval import CATEGORY_ORDER, VerifiedRetriever
from evals.corpus import load_corpus, load_dataset, species_records

# The fixed card query both knowledge paths use (services/knowledge.py).
CARD_QUERY = (
    "Buat kartu pengetahuan bahasa Indonesia untuk {common_name}: identitas, "
    "ciri fisik, rasa dan tekstur, cara pengolahan, penggunaan komersial, dan "
    "spesies pengganti."
)
E5_MAX_TOKENS = 512


def gold_queries() -> list[dict]:
    corpus = load_corpus()
    cells: dict[tuple[str, str], list[str]] = defaultdict(list)
    for chunk in corpus:
        cells[(chunk.species_label, chunk.category)].append(chunk.id)
    common = {s.normalized_label: s.common_name_id for s in species_records()}
    templates = load_dataset("retrieval_queries.json")["templates"]
    queries = []
    for (label, category), relevant in sorted(cells.items()):
        for template in templates[category]:
            queries.append({
                "species_label": label,
                "category": category,
                "lang": template["lang"],
                "query": template["text"].format(common_name=common[label]),
                "relevant": sorted(relevant),
            })
    return queries


def _dcg(hits: list[bool]) -> float:
    return sum(1.0 / math.log2(rank + 2) for rank, hit in enumerate(hits) if hit)


def _rank_metrics(ranked_ids: list[str], relevant: set[str], k: int) -> dict:
    hits = [cid in relevant for cid in ranked_ids[:k]]
    first = next((rank for rank, cid in enumerate(ranked_ids) if cid in relevant), None)
    ideal = _dcg([True] * min(len(relevant), k))
    return {
        "recall@1": len(set(ranked_ids[:1]) & relevant) / len(relevant),
        "recall@3": len(set(ranked_ids[:3]) & relevant) / len(relevant),
        f"recall@{k}": len(set(ranked_ids[:k]) & relevant) / len(relevant),
        "mrr": 0.0 if first is None else 1.0 / (first + 1),
        f"ndcg@{k}": _dcg(hits) / ideal if ideal else 0.0,
        f"precision@{k}": sum(hits) / max(1, len(ranked_ids[:k])),
    }


def _mean(rows: list[dict], key: str) -> float:
    return round(statistics.fmean(row[key] for row in rows), 4) if rows else 0.0


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(pct / 100 * len(ordered)) - 1))
    return round(ordered[index], 2)


def corpus_health(embedder) -> dict:
    tokenizer = embedder.tokenizer
    lengths = {
        chunk.id: len(tokenizer.encode(chunk.content, add_special_tokens=False))
        for chunk in load_corpus()
    }
    chunk_limit = inspect.signature(chunking.chunk_candidate).parameters["max_tokens"].default
    categories = defaultdict(int)
    for chunk in load_corpus():
        categories[chunk.category] += 1
    return {
        "chunks": len(lengths),
        "species": len({c.species_label for c in load_corpus()}),
        "tokens_max": max(lengths.values()),
        "tokens_mean": round(statistics.fmean(lengths.values()), 1),
        "chunks_over_e5_limit": sum(1 for n in lengths.values() if n > E5_MAX_TOKENS - 2),
        "chunker_max_tokens": chunk_limit,
        "chunker_fits_e5": chunk_limit + 4 <= E5_MAX_TOKENS,  # "passage: " prefix + CLS/SEP
        "category_counts": {c: categories.get(c, 0) for c in CATEGORY_ORDER},
    }


def evaluate(store, embedder, k: int = 6) -> dict:
    retriever = VerifiedRetriever(store, embedder)
    queries = gold_queries()
    scoped_rows, global_rows, latencies = [], [], []
    for q in queries:
        species_id = f"species_{q['species_label']}"
        relevant = set(q["relevant"])
        started = time.perf_counter()
        hits = retriever.retrieve(species_id, q["query"], max_chunks=k)
        latencies.append((time.perf_counter() - started) * 1000)
        row = _rank_metrics([h.chunk_id for h in hits], relevant, k)
        row["species_purity"] = (
            sum(h.species_id == species_id for h in hits) / len(hits) if hits else 1.0
        )
        scoped_rows.append({**q, **row})

        global_hits = store.search_global(embedder.embed_query(q["query"]), limit=k)
        grow = _rank_metrics([h.chunk_id for h in global_hits], relevant, k)
        grow["species_hit@1"] = float(bool(global_hits) and global_hits[0].species_id == species_id)
        grow["species_share@6"] = sum(h.species_id == species_id for h in global_hits) / max(1, len(global_hits))
        global_rows.append({**q, **grow})

    coverage = []
    available = defaultdict(set)
    for chunk in load_corpus():
        available[chunk.species_id].add(chunk.category)
    for species in species_records():
        hits = retriever.retrieve(species.id, CARD_QUERY.format(common_name=species.common_name_id), max_chunks=k)
        covered = {h.category for h in hits}
        coverage.append({
            "species_label": species.normalized_label,
            "available_categories": len(available[species.id]),
            "covered_categories": len(covered & available[species.id]),
            "coverage": len(covered & available[species.id]) / max(1, len(available[species.id])),
            "retrieved": len(hits),
        })

    def summarize(rows, keys):
        return {key: _mean(rows, key) for key in keys}

    scoped_keys = ["recall@1", "recall@3", f"recall@{k}", "mrr", f"ndcg@{k}", f"precision@{k}", "species_purity"]
    global_keys = ["recall@1", "recall@3", f"recall@{k}", "mrr", f"ndcg@{k}", "species_hit@1", "species_share@6"]
    by_lang = {
        lang: summarize([r for r in global_rows if r["lang"] == lang], global_keys)
        for lang in sorted({r["lang"] for r in global_rows})
    }
    by_category = {
        category: {
            "queries": sum(1 for r in scoped_rows if r["category"] == category),
            "scoped_mrr": _mean([r for r in scoped_rows if r["category"] == category], "mrr"),
            "global_species_hit@1": _mean([r for r in global_rows if r["category"] == category], "species_hit@1"),
        }
        for category in CATEGORY_ORDER
        if any(r["category"] == category for r in scoped_rows)
    }
    return {
        "queries": len(queries),
        "k": k,
        "scoped": summarize(scoped_rows, scoped_keys),
        "global": summarize(global_rows, global_keys),
        "global_by_lang": by_lang,
        "by_category": by_category,
        "card_query_coverage": {
            "mean": round(statistics.fmean(c["coverage"] for c in coverage), 4),
            "per_species": coverage,
        },
        "latency_ms": {
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
            "max": round(max(latencies), 2),
        },
        "worst_global_queries": sorted(
            (
                {"query": r["query"], "species": r["species_label"], "category": r["category"], "mrr": round(r["mrr"], 3)}
                for r in global_rows
            ),
            key=lambda r: r["mrr"],
        )[:8],
    }

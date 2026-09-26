"""Retrieval quality gates over the real corpus and the real E5 model."""

import pytest

from apps.main_api.services.retrieval import VerifiedRetriever
from evals import retrieval_eval


@pytest.fixture(scope="module")
def report(store, e5):
    return retrieval_eval.evaluate(store, e5)


def test_gold_set_covers_every_cell_in_both_languages():
    queries = retrieval_eval.gold_queries()
    cells = {(q["species_label"], q["category"]) for q in queries}
    assert len(queries) == 2 * len(cells)
    assert {q["lang"] for q in queries} == {"id", "en"}


def test_scoped_retrieval_never_leaks_other_species(report):
    assert report["scoped"]["species_purity"] == 1.0


def test_scoped_recall_at_6(report):
    assert report["scoped"]["recall@6"] >= 0.95


def test_card_query_covers_every_available_category(report):
    assert report["card_query_coverage"]["mean"] == 1.0


def test_embedding_separates_species_without_the_filter(report):
    assert report["global"]["species_hit@1"] >= 0.9


def test_warm_query_latency(report):
    assert report["latency_ms"]["p95"] < 150


def test_unverified_rows_are_never_returned(store, e5):
    from dataclasses import replace

    chunk_id = "chunk_nila_taste_001"
    original = store._chunks[chunk_id]
    store._chunks[chunk_id] = replace(original, verification_status="candidate")
    try:
        hits = VerifiedRetriever(store, e5).retrieve("species_nila", "rasa ikan nila")
        assert chunk_id not in {h.chunk_id for h in hits}
    finally:
        store._chunks[chunk_id] = original

from dataclasses import replace

import pytest

from apps.main_api.contracts import RetrievedChunk
from apps.main_api.services.retrieval import (
    CATEGORY_ORDER,
    VerifiedRetriever,
    _category_first_selection,
    _validate_query_vector,
)

BASE = RetrievedChunk(
    chunk_id="c", species_id="species_nila", source_id="s", source_type="t", category="identity",
    content="x", distance=0.0, chunk_verification_status="verified", source_verification_status="verified",
    source_title="T", source_publisher=None, source_url=None, source_reviewed_at=None,
)


def _hit(chunk_id: str, category: str, distance: float) -> RetrievedChunk:
    return replace(BASE, chunk_id=chunk_id, category=category, distance=distance)


def test_one_per_category_first_then_nearest_fill():
    candidates = sorted([
        _hit("a1", "identity", 0.10), _hit("a2", "identity", 0.11), _hit("a3", "identity", 0.12),
        _hit("t1", "taste_texture", 0.30), _hit("p1", "processing_methods", 0.40),
    ], key=lambda h: h.distance)
    picked = [h.chunk_id for h in _category_first_selection(candidates, 4)]
    assert picked == ["a1", "t1", "p1", "a2"]


def test_selection_respects_max_chunks():
    candidates = [_hit(f"c{i}", CATEGORY_ORDER[i % 6], i / 10) for i in range(12)]
    assert len(_category_first_selection(candidates, 3)) == 3


class _Embedder:
    model_name = "intfloat/multilingual-e5-base"

    def embed_query(self, text):
        return [1.0] + [0.0] * 767


class _Repo:
    def __init__(self):
        self.calls = []

    def search_verified(self, species_id, vector, model, limit):
        self.calls.append(limit)
        return []


def test_zero_and_negative_max_chunks():
    retriever = VerifiedRetriever(_Repo(), _Embedder())
    assert retriever.retrieve("species_nila", "q", max_chunks=0) == []
    with pytest.raises(ValueError):
        retriever.retrieve("species_nila", "q", max_chunks=-1)


def test_rejects_foreign_embedding_model():
    embedder = _Embedder()
    embedder.model_name = "other-model"
    with pytest.raises(ValueError):
        VerifiedRetriever(_Repo(), embedder).retrieve("species_nila", "q")


@pytest.mark.parametrize("vector", [[1.0] * 3, [float("nan")] + [0.0] * 767, [2.0] + [0.0] * 767])
def test_malformed_query_vectors_rejected(vector):
    with pytest.raises(ValueError):
        _validate_query_vector(vector)

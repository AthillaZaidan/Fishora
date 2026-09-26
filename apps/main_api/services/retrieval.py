"""Verified, species-scoped, category-diverse retrieval.

Internal service for generation (Task 7): given a stored verified species id
and a query, return up to ``max_chunks`` verified chunks of that species --
one nearest row per category first, in a fixed category order, then the
nearest unused rows. No public route exposes retrieval and no caller-provided
species id is ever accepted: callers hold the species id only after
verification (Task 7).
"""

from __future__ import annotations

import math

from apps.main_api.contracts import RetrievedChunk
from apps.main_api.services.embeddings import E5_DIMENSION, E5_MODEL_NAME

# Knowledge-card evidence (R1, W9, W8). A species slice is 1 to 7 chunks and
# about 1,000 tokens at most, so the whole verified slice fits one prompt and
# semantic ranking can only lose evidence: the 6-chunk cap dropped tuna's
# seventh chunk and name-biased ranking decided what the model saw. Up to
# FULL_CONTEXT_MAX chunks the card gets the whole slice; above it, the
# category-first ranked selection with CARD_MAX_CHUNKS applies.
FULL_CONTEXT_MAX = 20
CARD_MAX_CHUNKS = 10

CATEGORY_ORDER = [
    "identity",
    "physical_characteristics",
    "taste_texture",
    "processing_methods",
    "commercial_uses",
    "substitutes",
]


class VerifiedRetriever:
    """Category-diverse nearest-neighbor retrieval over verified chunks."""

    def __init__(self, knowledge_repo, embedder):
        self._repo = knowledge_repo
        self._embedder = embedder

    def retrieve(self, species_id: str, query: str, max_chunks: int = 6) -> list[RetrievedChunk]:
        if max_chunks < 0:
            raise ValueError(f"max_chunks must be >= 0, got {max_chunks}")
        if max_chunks == 0:
            return []
        # The query vector must come from exactly the E5 model the store was
        # indexed with; anything else would rank against incomparable vectors.
        if self._embedder.model_name != E5_MODEL_NAME:
            raise ValueError(
                f"embedder model must be {E5_MODEL_NAME!r}, got {self._embedder.model_name!r}"
            )
        # The embedder applies the E5 ``query: `` prefix; retrieval only ever
        # embeds through embed_query, never as passages.
        query_vector = self._embedder.embed_query(query)
        _validate_query_vector(query_vector)
        # Bounded fetch window: every category's nearest chunk must lie inside
        # it for full diversification (36 candidates at max_chunks=6).
        # ponytail: fixed window; a category whose nearest row ranks beyond
        # max_chunks * len(CATEGORY_ORDER) is skipped, raise the multiplier
        # only if recall at that depth ever matters.
        candidates = self._repo.search_verified(
            species_id, query_vector, E5_MODEL_NAME, limit=max_chunks * len(CATEGORY_ORDER)
        )
        return _category_first_selection(candidates, max_chunks)

    def card_evidence(self, species_id: str, query: str) -> list[RetrievedChunk]:
        """Evidence for one knowledge card: the whole verified species slice when
        it has at most FULL_CONTEXT_MAX chunks, ordered by category then
        distance; otherwise the ranked, category-first selection."""
        if self._embedder.model_name != E5_MODEL_NAME:
            raise ValueError(
                f"embedder model must be {E5_MODEL_NAME!r}, got {self._embedder.model_name!r}"
            )
        query_vector = self._embedder.embed_query(query)
        _validate_query_vector(query_vector)
        candidates = self._repo.search_verified(
            species_id, query_vector, E5_MODEL_NAME, limit=FULL_CONTEXT_MAX + 1
        )
        if len(candidates) <= FULL_CONTEXT_MAX:
            rank = {category: i for i, category in enumerate(CATEGORY_ORDER)}
            return sorted(candidates, key=lambda c: (rank.get(c.category, len(rank)), c.distance, c.chunk_id))
        return self.retrieve(species_id, query, max_chunks=CARD_MAX_CHUNKS)


def _validate_query_vector(vector: list[float]) -> None:
    """Reject a malformed query embedding before it reaches the store."""
    if len(vector) != E5_DIMENSION:
        raise ValueError(
            f"query embedding has {len(vector)} dimensions; expected {E5_DIMENSION}"
        )
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("query embedding contains non-finite values")
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isclose(norm, 1.0, rel_tol=1e-3):
        raise ValueError(f"query embedding is not L2-normalized (norm {norm:.4f})")


def _category_first_selection(
    candidates: list[RetrievedChunk], max_chunks: int
) -> list[RetrievedChunk]:
    """One nearest chunk per category in CATEGORY_ORDER, then nearest unused.

    ``candidates`` arrive sorted by (cosine distance, chunk id). Category
    rounds and the distance fill both stop at ``max_chunks``; categories with
    no candidates are skipped, so the result can be shorter than max_chunks.
    """
    by_category: dict[str, list[RetrievedChunk]] = {}
    for candidate in candidates:
        by_category.setdefault(candidate.category, []).append(candidate)

    selected: list[RetrievedChunk] = []
    used: set[str] = set()
    for category in CATEGORY_ORDER:
        if len(selected) == max_chunks:
            break
        for candidate in by_category.get(category, []):
            if candidate.chunk_id not in used:
                selected.append(candidate)
                used.add(candidate.chunk_id)
                break
    for candidate in candidates:
        if len(selected) == max_chunks:
            break
        if candidate.chunk_id not in used:
            selected.append(candidate)
            used.add(candidate.chunk_id)
    return selected
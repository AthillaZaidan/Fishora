"""Eval corpus: the real candidate chunks behind an in-memory knowledge store.

``InMemoryKnowledgeRepository.search_verified`` applies the same four filters
as ``SqlKnowledgeRepository.search_verified`` (species, chunk status, source
status, embedding model) and the same ordering (cosine distance, then chunk
id), so the retriever under test is exercised exactly as in production. Exact
cosine over the species slice is what pgvector computes here too: there is no
ANN index, so the results are identical, not approximate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

from apps.main_api.contracts import (
    KnowledgeChunkWrite,
    KnowledgeSourceWrite,
    RetrievedChunk,
    SpeciesRecord,
)

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_DIR = ROOT / "artifacts" / "knowledge_sources" / "candidates"
DATASET_DIR = Path(__file__).resolve().parent / "datasets"

# Relational identity per label. Scientific names follow the corpus identity
# chunks and generation.TAXONOMY_GUARDRAILS; common names are the UI labels.
SPECIES_FIXTURE: dict[str, tuple[str, str | None, str, str]] = {
    "bandeng": ("Bandeng", "Chanos chanos", "species", "VERIFIED_TAXONOMY"),
    "gelama_bunga": ("Gelama Bunga", "Nibea albiflora", "species", "VERIFIED_TAXONOMY"),
    "gembolo": ("Gembolo", None, "unresolved", "TAXONOMY_REVIEW_REQUIRED"),
    "gulamah": ("Gulamah", "Johnius trachycephalus", "species", "VERIFIED_TAXONOMY"),
    "kembung": ("Kembung", "Rastrelliger faughni", "species", "VERIFIED_TAXONOMY"),
    "kuniran": ("Kuniran", "Upeneus moluccensis", "species", "VERIFIED_TAXONOMY"),
    "mujair": ("Mujair", "Oreochromis mossambicus", "species", "VERIFIED_TAXONOMY"),
    "nila": ("Nila", "Oreochromis niloticus", "species", "VERIFIED_TAXONOMY"),
    "senangin": ("Senangin", "Eleutheronema tetradactylum", "species", "VERIFIED_TAXONOMY"),
    "tenggiri": ("Tenggiri", "Scomberomorus commerson", "species", "MEDIUM_CONFIDENCE_LABEL_AMBIGUITY"),
    "tuna": ("Tuna", "Thunnus spp.", "genus", "MIXED_TAXONOMY"),
}


def species_records() -> list[SpeciesRecord]:
    return [
        SpeciesRecord(
            id=f"species_{label}",
            normalized_label=label,
            common_name_id=common,
            scientific_name=scientific,
            taxonomic_rank=rank,
            taxonomy_status=status,
            notes=None,
        )
        for label, (common, scientific, rank, status) in SPECIES_FIXTURE.items()
    ]


@dataclass(frozen=True)
class CorpusChunk:
    id: str
    species_label: str
    category: str
    content: str
    source: dict

    @property
    def species_id(self) -> str:
        return f"species_{self.species_label}"


@lru_cache(maxsize=1)
def load_corpus() -> tuple[CorpusChunk, ...]:
    chunks = []
    for path in sorted(CANDIDATE_DIR.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        chunk = record["chunk"]
        chunks.append(
            CorpusChunk(
                id=chunk["id"],
                species_label=chunk["species_label"],
                category=chunk["category"],
                content=chunk["content"],
                source=record["source"],
            )
        )
    if not chunks:
        raise FileNotFoundError(f"no candidate chunks under {CANDIDATE_DIR}")
    return tuple(chunks)


def load_dataset(name: str) -> dict:
    return json.loads((DATASET_DIR / name).read_text(encoding="utf-8"))


class InMemoryKnowledgeRepository:
    """KnowledgeRepository port over numpy, same contract as the SQL store."""

    def __init__(self):
        self._sources: dict[str, KnowledgeSourceWrite] = {}
        self._chunks: dict[str, KnowledgeChunkWrite] = {}
        self._matrix: np.ndarray | None = None
        self._order: list[str] = []
        self.search_calls = 0

    def embedding_models_in_store(self) -> set[str]:
        return {chunk.embedding_model for chunk in self._chunks.values()}

    def insert_verified(
        self,
        sources: Sequence[KnowledgeSourceWrite],
        chunks: Sequence[KnowledgeChunkWrite],
    ) -> int:
        for source in sources:
            self._sources[source.id] = source
        for chunk in chunks:
            self._chunks[chunk.id] = chunk
        self._order = sorted(self._chunks)
        self._matrix = np.array([self._chunks[cid].embedding for cid in self._order], dtype=np.float32)
        return len(chunks)

    def search_verified(
        self,
        species_id: str,
        query_vector: list[float],
        embedding_model: str,
        limit: int,
    ) -> list[RetrievedChunk]:
        self.search_calls += 1
        if self._matrix is None:
            return []
        distances = 1.0 - self._matrix @ np.asarray(query_vector, dtype=np.float32)
        rows = []
        for index, chunk_id in enumerate(self._order):
            chunk = self._chunks[chunk_id]
            source = self._sources.get(chunk.source_id)
            if (
                chunk.species_id == species_id
                and chunk.verification_status == "verified"
                and source is not None
                and source.verification_status == "verified"
                and chunk.embedding_model == embedding_model
            ):
                rows.append((float(distances[index]), chunk_id))
        rows.sort()
        return [self._to_retrieved(chunk_id, distance) for distance, chunk_id in rows[:limit]]

    def search_global(self, query_vector: list[float], limit: int) -> list[RetrievedChunk]:
        """Unfiltered ranking across every species: an eval-only probe of how
        well the embedding separates species, not a production code path."""
        distances = 1.0 - self._matrix @ np.asarray(query_vector, dtype=np.float32)
        ranked = sorted(zip(distances.tolist(), self._order))[:limit]
        return [self._to_retrieved(chunk_id, distance) for distance, chunk_id in ranked]

    def _to_retrieved(self, chunk_id: str, distance: float) -> RetrievedChunk:
        chunk = self._chunks[chunk_id]
        source = self._sources[chunk.source_id]
        return RetrievedChunk(
            chunk_id=chunk.id,
            species_id=chunk.species_id,
            source_id=chunk.source_id,
            source_type=source.source_type,
            category=chunk.category,
            content=chunk.content,
            distance=distance,
            chunk_verification_status=chunk.verification_status,
            source_verification_status=source.verification_status,
            source_title=source.title,
            source_publisher=source.publisher,
            source_url=source.url,
            source_reviewed_at=source.reviewed_at,
        )


def corpus_v1_ids() -> frozenset[str]:
    """The 49 chunks the iteration-1/2 experiments and gold labels were built on."""
    path = Path(__file__).resolve().parent / "protocol" / "corpus_v1_chunks.txt"
    return frozenset(line.strip() for line in path.read_text().splitlines() if line.strip())


def build_store(embedder, only: frozenset[str] | None = None) -> InMemoryKnowledgeRepository:
    """Embed the candidate corpus as passages and load it as verified rows.

    Candidates are treated as verified for evaluation only; production
    ingestion still requires the signed human approval manifest. ``only``
    pins the store to a chunk subset (e.g. ``corpus_v1_ids()``).
    """
    corpus = tuple(c for c in load_corpus() if only is None or c.id in only)
    vectors = embedder.embed_passages([chunk.content for chunk in corpus])
    sources = {
        chunk.source["id"]: KnowledgeSourceWrite(
            id=chunk.source["id"],
            title=chunk.source["title"],
            source_type=chunk.source["source_type"],
            url=chunk.source.get("url"),
            publisher=chunk.source.get("publisher"),
            reviewed_at=None,
            verification_status="verified",
        )
        for chunk in corpus
    }
    store = InMemoryKnowledgeRepository()
    store.insert_verified(
        list(sources.values()),
        [
            KnowledgeChunkWrite(
                id=chunk.id,
                species_id=chunk.species_id,
                source_id=chunk.source["id"],
                category=chunk.category,
                content=chunk.content,
                embedding=vector,
                embedding_model=embedder.model_name,
                verification_status="verified",
            )
            for chunk, vector in zip(corpus, vectors)
        ],
    )
    return store

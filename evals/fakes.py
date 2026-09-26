"""In-memory ports and a scripted LLM for offline evaluation and tests.

``ScriptedLLM`` stands in for the OpenCode Go model. It answers each
orchestrator prompt deterministically and sleeps ``delay`` seconds per call so
latency and the number of sequential LLM rounds are measurable. Its experts
emit two kinds of claims, both tied to a chunk in their evidence:

* a *true* claim: the Indonesian paraphrase of a chunk of the field's own
  category, citing that chunk;
* a *hallucination*, only when the field has no evidence of its category: a
  plausible claim borrowed from another species, citing an unrelated chunk.

That is the realistic failure the critic must stop (a model filling an empty
field with a borrowed fact and a real citation). Every emitted claim is logged
in ``emitted`` so retention and leakage can be scored exactly.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from dataclasses import dataclass, field, replace

from apps.contracts import CVCandidate, CVPredictionEnvelope
from apps.main_api.contracts import KnowledgeJobRecord, PredictionRecord, SpeciesRecord

# The card field each expert fills and the evidence category that supports it.
FIELD_CATEGORY = {
    "physical_characteristics": "physical_characteristics",
    "taste": "taste_texture",
    "texture": "taste_texture",
    "processing_methods": "processing_methods",
    "commercial_uses": "commercial_uses",
    "similar_or_substitute_species": "substitutes",
    "potential_buyer_segments": "commercial_uses",
}
EXPERT_FIELDS = {
    "physical": ("physical_characteristics",),
    "taste": ("taste", "texture"),
    "commercial": ("processing_methods", "commercial_uses"),
    "substitute": ("similar_or_substitute_species", "potential_buyer_segments"),
}
LIST_FIELDS = {
    "processing_methods", "commercial_uses",
    "similar_or_substitute_species", "potential_buyer_segments",
}
# Expert prompts open with these phrases (orchestrator._EXPERT_PROMPTS).
_EXPERT_MARKERS = {
    "Tulis physical_characteristics": "physical",
    "Tulis taste dan texture": "taste",
    "Tulis processing_methods": "commercial",
    "Tulis similar_or_substitute_species": "substitute",
}
_CHUNK_LINE = re.compile(r"\[chunk_id: ([a-z0-9_]+)\] \[source_id: ([a-z0-9_]+)\] \[([a-z_]+)\] ?(.*)")


@dataclass
class EmittedClaim:
    species_label: str
    field: str
    text: str
    kind: str  # "true" | "hallucination"
    chunk_id: str


@dataclass
class ScriptedLLM:
    claims: dict[str, str]
    chunk_species: dict[str, str]
    chunk_category: dict[str, str]
    delay: float = 0.0
    fenced: bool = False
    calls: int = 0
    prompts: list[str] = field(default_factory=list)
    spans: list[tuple[float, float]] = field(default_factory=list)  # (start, end) per call
    emitted: list[EmittedClaim] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def invoke(self, prompt):
        text = prompt if isinstance(prompt, str) else str(prompt)
        with self._lock:
            self.calls += 1
            self.prompts.append(text)
        started = time.perf_counter()
        if self.delay:
            time.sleep(self.delay)
        with self._lock:
            self.spans.append((started, time.perf_counter()))
        if text.startswith("Untuk spesies"):
            return "query tambahan untuk kategori yang kosong"
        if text.startswith("Untuk setiap field"):
            return "{}"  # a neutral critic: measures the deterministic gate alone
        if text.startswith("Perbaiki bahasa"):
            return text.split("\n", 1)[1] if "\n" in text else "{}"
        for marker, expert in _EXPERT_MARKERS.items():
            if text.startswith(marker):
                return self._wrap(json.dumps(self._expert(expert, text), ensure_ascii=False))
        return "{}"

    def _wrap(self, payload: str) -> str:
        return f"```json\n{payload}\n```" if self.fenced else payload

    def _expert(self, expert: str, prompt: str) -> dict:
        evidence = [m.groups() for m in map(_CHUNK_LINE.match, prompt.splitlines()) if m]
        data: dict = {name: ([] if name in LIST_FIELDS else None) for name in EXPERT_FIELDS[expert]}
        sources: list[dict] = []
        if not evidence:
            data["sources"] = sources
            return data
        species = self.chunk_species[evidence[0][0]]
        for name in EXPERT_FIELDS[expert]:
            wanted = FIELD_CATEGORY[name]
            own = [row for row in evidence if row[2] == wanted or (wanted == "physical_characteristics" and row[2] == "identity")]
            if own:
                chunk_id, source_id = own[0][0], own[0][1]
                text, kind = self.claims[chunk_id], "true"
            else:
                borrowed = self._borrowed_claim(species, wanted)
                if borrowed is None:
                    continue
                chunk_id, source_id = evidence[0][0], evidence[0][1]
                text, kind = borrowed, "hallucination"
            data[name] = [text] if name in LIST_FIELDS else text
            sources.append({"source_id": source_id, "chunk_id": chunk_id})
            with self._lock:
                self.emitted.append(EmittedClaim(species, name, text, kind, chunk_id))
        data["sources"] = sources
        return data

    def _borrowed_claim(self, species: str, category: str) -> str | None:
        for chunk_id in sorted(self.claims):
            if self.chunk_species[chunk_id] != species and self.chunk_category[chunk_id] == category:
                return self.claims[chunk_id]
        for chunk_id in sorted(self.claims):
            if self.chunk_species[chunk_id] != species:
                return self.claims[chunk_id]
        return None


class InMemorySpeciesRepository:
    def __init__(self, species: list[SpeciesRecord]):
        self._by_id = {s.id: s for s in species}

    def get_by_normalized_label(self, label: str) -> SpeciesRecord | None:
        return next((s for s in self._by_id.values() if s.normalized_label == label), None)

    def get_by_id(self, species_id: str) -> SpeciesRecord | None:
        return self._by_id.get(species_id)

    def list_all(self) -> list[SpeciesRecord]:
        return list(self._by_id.values())


class InMemoryPredictionRepository:
    def __init__(self):
        self.rows: dict[str, PredictionRecord] = {}

    def create(self, prediction_id, image_reference, predicted_species_id, confidence, top_candidates, model_version):
        record = PredictionRecord(
            id=prediction_id,
            image_reference=image_reference,
            predicted_species_id=predicted_species_id,
            confidence=confidence,
            top_candidates=top_candidates,
            model_version=model_version,
            verification_status="pending",
        )
        self.rows[prediction_id] = record
        return record

    def get(self, prediction_id):
        return self.rows.get(prediction_id)

    def verify(self, prediction_id, verified_species_id, verification_status):
        record = replace(self.rows[prediction_id], verified_species_id=verified_species_id,
                         verification_status=verification_status)
        self.rows[prediction_id] = record
        return record


class InMemoryJobRepository:
    def __init__(self):
        self.rows: dict[str, KnowledgeJobRecord] = {}

    def create(self, job_id, prediction_id, species_id):
        self.rows[job_id] = KnowledgeJobRecord(id=job_id, prediction_id=prediction_id,
                                               species_id=species_id, status="processing")
        return self.rows[job_id]

    def get(self, job_id):
        return self.rows.get(job_id)

    def update(self, job_id, status=None, **fields):
        row = self.rows.get(job_id)
        if row is None:
            return None
        if status is not None:
            fields["status"] = status
        self.rows[job_id] = replace(row, **fields)
        return self.rows[job_id]

    def list_by_prediction(self, prediction_id):
        return [row for row in self.rows.values() if row.prediction_id == prediction_id]


class InMemoryImageStore:
    def __init__(self):
        self.saved: dict[str, bytes] = {}

    def save(self, prediction_id, image_bytes, content_type):
        reference = f"memory://{prediction_id}"
        self.saved[reference] = image_bytes
        return reference

    def delete(self, image_reference):
        self.saved.pop(image_reference, None)


class FixedCVClient:
    """CV stub answering one fixed label, so the API flow is deterministic."""

    def __init__(self, label: str, confidence: float = 0.91):
        self.label = label
        self.confidence = confidence

    def predict(self, image_bytes, *, filename, content_type):
        others = [lbl for lbl in ("nila", "mujair", "bandeng") if lbl != self.label][:2]
        return CVPredictionEnvelope(
            model_version="eval-fixed-cv",
            status="confident_prediction",
            prediction=CVCandidate(label=self.label, confidence=self.confidence),
            top_candidates=[CVCandidate(label=self.label, confidence=self.confidence)]
            + [CVCandidate(label=lbl, confidence=0.03) for lbl in others],
            threshold=0.5,
        )


def llm_rounds(spans: list[tuple[float, float]]) -> int:
    """Sequential LLM rounds: calls that overlap in time form one round."""
    rounds, end = 0, float("-inf")
    for start, stop in sorted(spans):
        if start >= end:
            rounds += 1
            end = stop
        else:
            end = max(end, stop)
    return rounds


def new_id() -> str:
    return uuid.uuid4().hex

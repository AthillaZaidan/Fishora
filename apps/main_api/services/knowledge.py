"""Verification-gated knowledge cards, made synchronously.

The only retrieval key is the prediction's stored ``verified_species_id``:
pending predictions are rejected before any retrieval, and no caller-supplied
species id or raw label ever reaches the retriever.

This is the path for a prediction with no background job (a manually declared
catch) and the fallback at lot publication. It runs the same graded graph as
the job, ``orchestrator.grade_card``, so a card served or frozen here has had
every claim checked against its cited chunk (W19). The former one-call
generator, which had no per-claim critic, is no longer used by the product.
"""

from apps.main_api.errors import (
    InvalidGeneratedKnowledge,
    OpenCodeUnavailable,
    PredictionNotFound,
    PredictionNotVerified,
    UnsupportedSpecies,
)
from apps.main_api.services.generation import KnowledgeCard, KnowledgeResponse


class KnowledgeService:
    def __init__(self, prediction_repo, species_repo, knowledge_repo, embedder, llm):
        self._prediction_repo = prediction_repo
        self._species_repo = species_repo
        self._knowledge_repo = knowledge_repo
        self._embedder = embedder
        self._llm = llm

    def get_for_prediction(self, prediction_id: str) -> KnowledgeResponse:
        from apps.main_api.services.orchestrator import grade_card

        record = self._prediction_repo.get(prediction_id)
        if record is None:
            raise PredictionNotFound(prediction_id)
        if record.verification_status not in ("confirmed", "corrected") or record.verified_species_id is None:
            raise PredictionNotVerified(prediction_id)
        species = self._species_repo.get_by_id(record.verified_species_id)
        if species is None:
            raise UnsupportedSpecies(record.verified_species_id)
        if self._llm is None:
            # No key: a provider outage (W3). Serving the graph's empty card here
            # would read as "nothing verifiable", and publication would freeze it.
            raise OpenCodeUnavailable("opencode go generation unavailable", [])

        result = grade_card(species.id, prediction_id=record.id, knowledge_repo=self._knowledge_repo,
                            embedder=self._embedder, llm=self._llm, species_repo=self._species_repo)
        final, error = result.get("final_card"), result.get("error")
        chunk_ids = [chunk.chunk_id for chunk in result.get("refined_evidence") or []]
        if final is None or error:
            if error and "every expert failed" in error:
                raise OpenCodeUnavailable("opencode go generation unavailable", chunk_ids)
            raise InvalidGeneratedKnowledge(error or "knowledge generation failed", chunk_ids)
        card = final if isinstance(final, KnowledgeCard) else KnowledgeCard.model_validate(final)
        return KnowledgeResponse(prediction_id=record.id, species_id=species.id, card=card)

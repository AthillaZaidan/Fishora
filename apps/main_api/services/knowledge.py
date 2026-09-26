"""Verification-gated knowledge cards.

The only retrieval key is the prediction's stored ``verified_species_id``:
pending predictions are rejected before any retrieval, and no caller-supplied
species id or raw label ever reaches the retriever.
"""

from apps.main_api.errors import PredictionNotVerified, PredictionNotFound, UnsupportedSpecies
from apps.main_api.services.generation import KnowledgeGenerator, KnowledgeResponse

# ponytail: fixed query, one per card request; tune per category only if
# retrieval quality ever measurably suffers.
CARD_QUERY = (
    "Buat kartu pengetahuan bahasa Indonesia untuk {common_name}: identitas, "
    "ciri fisik, rasa dan tekstur, cara pengolahan, penggunaan komersial, dan "
    "spesies pengganti."
)


class KnowledgeService:
    def __init__(self, prediction_repo, species_repo, retriever, generator: KnowledgeGenerator):
        self._prediction_repo = prediction_repo
        self._species_repo = species_repo
        self._retriever = retriever
        self._generator = generator

    def get_for_prediction(self, prediction_id: str) -> KnowledgeResponse:
        record = self._prediction_repo.get(prediction_id)
        if record is None:
            raise PredictionNotFound(prediction_id)
        if record.verification_status not in ("confirmed", "corrected") or record.verified_species_id is None:
            raise PredictionNotVerified(prediction_id)
        species = self._species_repo.get_by_id(record.verified_species_id)
        if species is None:
            raise UnsupportedSpecies(record.verified_species_id)

        query = CARD_QUERY.format(common_name=species.common_name_id)
        # The whole species slice when it is small (R1); see VerifiedRetriever.card_evidence.
        card_evidence = getattr(self._retriever, "card_evidence", None)
        evidence = (card_evidence(record.verified_species_id, query) if card_evidence
                    else self._retriever.retrieve(record.verified_species_id, query))
        card = self._generator.generate(species, evidence)
        return KnowledgeResponse(prediction_id=record.id, species_id=species.id, card=card)
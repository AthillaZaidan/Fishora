from fastapi import APIRouter, Request
from pydantic import BaseModel

from apps.main_api.services.generation import KnowledgeCard
from apps.main_api.services.lots import LotService

router = APIRouter(prefix="/api/v1/discover")


class DiscoverResponse(BaseModel):
    public_slug: str
    # The page a diner opens from a printed card reads the lot's reviews by id.
    lot_id: str
    species_id: str
    card: KnowledgeCard


@router.get("/{public_slug}", response_model=DiscoverResponse)
def discover(public_slug: str, request: Request):
    deps = request.app.state.deps
    lot = LotService(prediction_repo=deps.prediction_repo, lot_repo=deps.lot_repo).get_by_slug(public_slug)
    if not lot.knowledge_snapshot:
        # A lot can reach the marketplace with no card when generation was down
        # at publication. The operator may already have printed a QR for it, so
        # 404 here turns a physical card into a dead end. Fall back to the
        # species identity, which is stored and true, and say outright that the
        # rest is missing rather than implying it with empty fields.
        return DiscoverResponse(
            public_slug=lot.public_slug,
            lot_id=lot.id,
            species_id=lot.species_id,
            card=_identity_only_card(request, lot.species_id),
        )
    return DiscoverResponse(
        public_slug=lot.public_slug,
        lot_id=lot.id,
        species_id=lot.species_id,
        card=KnowledgeCard.model_validate(lot.knowledge_snapshot),
    )


CARD_PENDING = (
    "The knowledge card for this lot is not available yet. The name and taxonomy "
    "below come from the species database, not from a source search."
)


def _identity_only_card(request: Request, species_id: str) -> KnowledgeCard:
    species = request.app.state.deps.species_repo.get_by_id(species_id)
    if species is None:
        from apps.main_api.errors import UnsupportedSpecies

        raise UnsupportedSpecies(species_id)
    return KnowledgeCard(
        common_name=species.common_name_id,
        scientific_name=species.scientific_name,
        taxonomy_status=species.taxonomy_status,
        physical_characteristics=None,
        taste=None,
        texture=None,
        processing_methods=[],
        commercial_uses=[],
        similar_or_substitute_species=[],
        potential_buyer_segments=[],
        limitations=[CARD_PENDING],
        sources=[],
    )

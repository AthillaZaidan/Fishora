import pytest

from apps.main_api.contracts import RetrievedChunk, SpeciesRecord
from apps.main_api.errors import InvalidGeneratedKnowledge
from apps.main_api.services.generation import GeneratedKnowledgeCard, KnowledgeGenerator

NILA = SpeciesRecord("species_nila", "nila", "Nila", "Oreochromis niloticus", "species", "VERIFIED_TAXONOMY", None)
TUNA = SpeciesRecord("species_tuna", "tuna", "Tuna", "Thunnus albacares", "genus", "MIXED_TAXONOMY", None)
CHUNK = RetrievedChunk(
    chunk_id="chunk_nila_taste_001", species_id="species_nila", source_id="fao_en_niletilapia",
    source_type="species_fact_sheet", category="taste_texture", content="mild, delicious flavour",
    distance=0.1, chunk_verification_status="verified", source_verification_status="verified",
    source_title="FAO Nile tilapia", source_publisher="FAO", source_url="https://fao.org", source_reviewed_at=None,
)


def _generated(**overrides) -> GeneratedKnowledgeCard:
    fields = dict(
        common_name="x", scientific_name=None, taxonomy_status="x", physical_characteristics=None,
        taste="Rasa ringan", texture=None, processing_methods=[], commercial_uses=[],
        similar_or_substitute_species=[], potential_buyer_segments=[], limitations=[],
        sources=[{"source_id": "fao_en_niletilapia"}],
    )
    fields.update(overrides)
    return GeneratedKnowledgeCard(**fields)


def test_empty_evidence_never_builds_the_llm_client():
    def factory():
        raise AssertionError("LLM client constructed for empty evidence")

    card = KnowledgeGenerator(factory).generate(NILA, [])
    assert card.sources == [] and card.limitations[0] == "Informasi belum tersedia"


def test_relational_identity_overrides_generated_text():
    card = KnowledgeGenerator().build_card(NILA, [CHUNK], _generated(common_name="Salah"))
    assert (card.common_name, card.scientific_name) == ("Nila", "Oreochromis niloticus")
    assert card.sources[0].title == "FAO Nile tilapia"


def test_citation_to_unretrieved_source_is_rejected():
    with pytest.raises(InvalidGeneratedKnowledge):
        KnowledgeGenerator().build_card(NILA, [CHUNK], _generated(sources=[{"source_id": "made_up"}]))


def test_uncited_card_with_evidence_is_rejected():
    with pytest.raises(InvalidGeneratedKnowledge):
        KnowledgeGenerator().build_card(NILA, [CHUNK], _generated(sources=[]))


def test_tuna_taxonomy_guardrail_pins_genus():
    card = KnowledgeGenerator().empty_card(TUNA)
    assert card.scientific_name == "Thunnus spp."
    assert any("Thunnus spp." in limitation for limitation in card.limitations)

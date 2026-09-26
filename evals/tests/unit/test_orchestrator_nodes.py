"""Expert, critic and writer node behavior with fake LLMs (no model load)."""

import json

from apps.main_api.contracts import RetrievedChunk, SpeciesRecord
from apps.main_api.services import orchestrator

NILA = SpeciesRecord("species_nila", "nila", "Nila", "Oreochromis niloticus", "species", "VERIFIED_TAXONOMY", None)
TAIL = "fillets without small bones"
CHUNK = RetrievedChunk(
    chunk_id="chunk_nila_taste_001", species_id="species_nila", source_id="fao_en_niletilapia",
    source_type="species_fact_sheet", category="taste_texture",
    content=("Nile tilapia fillets have a mild flavour. " * 9) + TAIL,
    distance=0.1, chunk_verification_status="verified", source_verification_status="verified",
    source_title="FAO", source_publisher="FAO", source_url="https://fao.org", source_reviewed_at=None,
)
ANSWER = {"taste": "Rasa ringan", "texture": None,
          "sources": [{"source_id": "fao_en_niletilapia", "chunk_id": "chunk_nila_taste_001"}]}


class _LLM:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def test_expert_parses_markdown_fenced_json():
    llm = _LLM("```json\n" + json.dumps(ANSWER) + "\n```")
    out = orchestrator._expert_node("taste", {"refined_evidence": [CHUNK]}, llm)
    assert out.get("taste") == "Rasa ringan" and "error" not in out


def test_expert_sees_the_whole_chunk():
    llm = _LLM(json.dumps(ANSWER))
    orchestrator._expert_node("taste", {"refined_evidence": [CHUNK]}, llm)
    assert TAIL in llm.prompts[0]


def test_expert_without_llm_claims_nothing():
    out = orchestrator._expert_node("taste", {"refined_evidence": [CHUNK]}, None)
    assert out["taste"] is None and out["sources"] == []


def test_expert_citation_outside_its_evidence_is_dropped():
    answer = {**ANSWER, "sources": [{"source_id": "other", "chunk_id": "chunk_tuna_taste_001"}]}
    out = orchestrator._expert_node("taste", {"refined_evidence": [CHUNK]}, _LLM(json.dumps(answer)))
    assert out["sources"] == []


def test_critic_llm_outage_never_upgrades():
    state = {
        "refined_evidence": [CHUNK],
        "expert_outputs": {"taste": {"taste": "Pesan tidak terkait", "sources": []}},
    }
    result = orchestrator.critic_node(state, _LLM(RuntimeError("down")))
    taste = next(s for s in result["claim_statuses"] if s.field == "taste")
    assert taste.status == "unsupported"


def test_writer_with_no_evidence_returns_empty_card():
    out = orchestrator.writer_node({"refined_evidence": []}, None, NILA)
    assert out["final_card"].limitations[0] == "Informasi belum tersedia"


def test_writer_keeps_only_supported_claims():
    state = {
        "refined_evidence": [CHUNK],
        "expert_outputs": {"taste": {"taste": "Rasa ringan", "texture": "Keras",
                                     "sources": ANSWER["sources"]}},
        "claim_statuses": [
            orchestrator.ClaimStatus(field="taste", status="supported",
                                     chunk_ids=["chunk_nila_taste_001"], reason="ok"),
            orchestrator.ClaimStatus(field="texture", status="unsupported", chunk_ids=[], reason="no"),
        ],
    }
    card = orchestrator.writer_node(state, None, NILA)["final_card"]
    assert card.taste == "Rasa ringan" and card.texture is None
    assert [s.source_id for s in card.sources] == ["fao_en_niletilapia"]

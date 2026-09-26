"""Iteration-1 orchestrator behaviour: block content (W21), skipped experts
(W11), no-claim cards (W24), outage handling, and the cache key (W6)."""

import json
from dataclasses import replace

from langchain_core.messages import AIMessage

from apps.main_api.contracts import RetrievedChunk, SpeciesRecord
from apps.main_api.services import orchestrator

NILA = SpeciesRecord("species_nila", "nila", "Nila", "Oreochromis niloticus", "species", "VERIFIED_TAXONOMY", None)
TASTE = RetrievedChunk(
    chunk_id="chunk_nila_taste_001", species_id="species_nila", source_id="fao_en_niletilapia",
    source_type="species_fact_sheet", category="taste_texture", content="Nile tilapia fillets have a mild flavour.",
    distance=0.1, chunk_verification_status="verified", source_verification_status="verified",
    source_title="FAO", source_publisher="FAO", source_url="https://fao.org", source_reviewed_at=None,
)
ANSWER = {"taste": "Rasa ringan", "texture": None,
          "sources": [{"source_id": "fao_en_niletilapia", "chunk_id": "chunk_nila_taste_001"}]}


class _LLM:
    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def invoke(self, prompt):
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def test_expert_reads_responses_api_blocks():
    llm = _LLM(AIMessage(content=[{"type": "text", "text": json.dumps(ANSWER)}]))
    out = orchestrator._expert_node("taste", {"refined_evidence": [TASTE]}, llm)
    assert out["taste"] == "Rasa ringan" and "error" not in out


def test_expert_without_evidence_is_not_called():
    llm = _LLM(json.dumps(ANSWER))
    out = orchestrator._expert_node("substitute", {"refined_evidence": [TASTE]}, llm)
    assert llm.calls == 0 and out.get("skipped")


def test_expert_prompt_requires_indonesian():
    assert all("bahasa Indonesia" in prompt for prompt in orchestrator._EXPERT_PROMPTS.values())


def test_no_groundable_claim_gives_a_limitation_card():
    state = {
        "refined_evidence": [TASTE],
        "expert_outputs": {"taste": {"taste": None, "texture": None, "sources": []},
                           "physical": {"skipped": True}, "commercial": {"skipped": True}, "substitute": {"skipped": True}},
        "claim_statuses": [],
    }
    out = orchestrator.writer_node(state, None, NILA)
    assert out["final_card"].limitations[0] == orchestrator.NO_GROUNDED_CLAIM
    assert out["final_card"].sources == [] and "error" not in out


def test_every_expert_failing_is_an_error_not_an_empty_card():
    state = {
        "refined_evidence": [TASTE],
        "expert_outputs": {"taste": {"error": "expert generation failed", "sources": []},
                           "physical": {"skipped": True}, "commercial": {"skipped": True}, "substitute": {"skipped": True}},
        "claim_statuses": [],
    }
    out = orchestrator.writer_node(state, None, NILA)
    assert out["final_card"] is None and out["error"]


def test_critic_llm_pass_reads_the_whole_chunk():
    """R3: the LLM judge used to see only the first 300 characters."""
    tail = "TAIL-FACT-AFTER-300-CHARACTERS"
    long = replace(TASTE, content=("Nile tilapia fillets have a mild flavour. " * 9) + tail)
    llm = _LLM("{}")
    state = {"refined_evidence": [long],
             # shares "tilapia"/"fillets" with the chunk, so the lexical fallback accepts it
             "expert_outputs": {"taste": {"taste": "Fillets tilapia terasa ringan", "sources": ANSWER["sources"]}}}
    llm.prompts = []
    original = llm.invoke
    llm.invoke = lambda prompt: (llm.prompts.append(prompt), original(prompt))[1]
    statuses = orchestrator.critic_node(state, None)["claim_statuses"]  # lexical fallback: supported
    by_chunk = {long.chunk_id: long}
    orchestrator._llm_downgrade(statuses, by_chunk, llm)  # the judge pass itself (off in production)
    assert llm.prompts and tail in llm.prompts[0]


def test_llm_judge_pass_is_off_in_production():
    """F18: measured harmful (R2 experiment), so the critic does not call the LLM."""
    llm = _LLM("{}")
    state = {"refined_evidence": [TASTE],
             "expert_outputs": {"taste": {"taste": "Fillets tilapia terasa ringan", "sources": ANSWER["sources"]}}}
    orchestrator.critic_node(state, llm)
    assert orchestrator.USE_LLM_JUDGE is False and llm.calls == 0


def test_substitute_claim_needs_substitute_evidence():
    """R4: a commercial chunk cannot ground a substitute claim."""
    commercial = replace(TASTE, chunk_id="c_comm", category="commercial_uses",
                         content="Tilapia is sold fresh and frozen to restaurants.")
    status = orchestrator._grade_claim(
        "similar_or_substitute_species", ["Mujair"], [{"chunk_id": "c_comm"}], {"c_comm": commercial})
    assert status.status == "unsupported"


def test_missing_evidence_is_named_on_the_card():
    text = orchestrator.missing_evidence_limitation([TASTE])
    assert "spesies pengganti" in text and "rasa" not in text.split(":")[1].split(",")[0]


def test_cache_key_follows_the_evidence():
    llm = _LLM("{}")
    a = orchestrator.card_cache_key("species_nila", [TASTE], llm)
    assert a == orchestrator.card_cache_key("species_nila", [TASTE], llm)
    changed = replace(TASTE, content=TASTE.content + " Firm flesh.")
    assert a != orchestrator.card_cache_key("species_nila", [changed], llm)
    assert a != orchestrator.card_cache_key("species_mujair", [TASTE], llm)

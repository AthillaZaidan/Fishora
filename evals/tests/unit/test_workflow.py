"""Iteration-2 writer-critic workflow and claim verifier (no model, no network)."""

import json
from dataclasses import replace

from apps.main_api.contracts import RetrievedChunk, SpeciesRecord
from apps.main_api.services import claim_verifier, workflow

NILA = SpeciesRecord("species_nila", "nila", "Nila", "Oreochromis niloticus", "species", "VERIFIED_TAXONOMY", None)
TASTE = RetrievedChunk(
    chunk_id="chunk_nila_taste_001", species_id="species_nila", source_id="fao_en_niletilapia",
    source_type="species_fact_sheet", category="taste_texture", content="Nile tilapia fillets have a mild flavour.",
    distance=0.1, chunk_verification_status="verified", source_verification_status="verified",
    source_title="FAO", source_publisher="FAO", source_url="https://fao.org", source_reviewed_at=None,
)
PHYSICAL = replace(TASTE, chunk_id="chunk_nila_physical_001", category="physical_characteristics",
                   content="Nile tilapia reaches 60 cm SL and 4.3 kg.")
EVIDENCE = [TASTE, PHYSICAL]


class _Script:
    """Writer and verifier replies in order; records every prompt."""

    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return json.dumps(reply) if isinstance(reply, dict) else reply


def _writer(*claims):
    return {"claims": [{"field": f, "chunk_ids": ids, "teks": t} for f, ids, t in claims]}


def _verdicts(*labels):
    return {"verdicts": [{"id": i, "alasan": "x", "label": lab} for i, lab in enumerate(labels)]}


def test_only_supported_claims_reach_the_card():
    llm = _Script(
        _writer(("taste", ["chunk_nila_taste_001"], "Filet nila berasa ringan."),
                ("physical_characteristics", ["chunk_nila_physical_001"], "Panjang baku mencapai 60 cm.")),
        _verdicts("supported", "inferred"),
    )
    out = workflow.generate(NILA, EVIDENCE, llm)
    card = workflow.assemble(NILA, EVIDENCE, out["result"])
    assert card.taste == "Filet nila berasa ringan."
    assert card.physical_characteristics is None  # "inferred" never reaches a card
    assert out["llm_calls"] == 2 and not out["revised"]


def test_numbers_missing_from_the_cited_chunk_are_rejected_without_the_llm():
    claims = [claim_verifier.Claim(0, "physical_characteristics", "Panjang baku mencapai 80 cm.", ["chunk_nila_physical_001"])]
    llm = _Script()  # would fail if called
    res = claim_verifier.verify(claims, EVIDENCE, llm)
    assert res.claims[0].label == "unsupported" and res.claims[0].stage == "deterministic" and not llm.prompts


def test_a_citation_from_another_category_is_rejected():
    claims = [claim_verifier.Claim(0, "taste", "Rasa ringan.", ["chunk_nila_physical_001"])]
    res = claim_verifier.verify(claims, EVIDENCE, _Script())
    assert res.claims[0].label == "unsupported"


def test_verifier_outage_is_fail_closed_and_reported():
    claims = [claim_verifier.Claim(0, "taste", "Filet nila berasa ringan.", ["chunk_nila_taste_001"])]
    res = claim_verifier.verify(claims, EVIDENCE, _Script(TimeoutError(), TimeoutError()))
    assert res.claims[0].label == "unsupported" and res.llm_error == "TimeoutError" and res.llm_calls == 2


def test_revision_happens_at_most_once():
    llm = _Script(
        _writer(("taste", ["chunk_nila_taste_001"], "Tebakan pertama."),
                ("taste", ["chunk_nila_taste_001"], "Tebakan kedua.")),
        _verdicts("unsupported", "unsupported"),
        _writer(("taste", ["chunk_nila_taste_001"], "Filet nila berasa ringan.")),
        _verdicts("supported"),
    )
    out = workflow.generate(NILA, EVIDENCE, llm)
    assert out["revised"] and out["llm_calls"] == 4 and not llm.replies
    assert [c.text for c in out["result"].supported()] == ["Filet nila berasa ringan."]


def test_writer_prompt_requires_indonesian_and_keeping_scope():
    assert "bahasa Indonesia" in workflow.WRITER_PROMPT and "cakupan" in workflow.WRITER_PROMPT


def test_prose_fields_are_split_into_sentences():
    assert claim_verifier.split_field("physical_characteristics", "Tubuh pipih. Panjang 60 cm.") == ["Tubuh pipih.", "Panjang 60 cm."]
    assert claim_verifier.split_field("commercial_uses", ["Segar", " ", "Beku"]) == ["Segar", "Beku"]

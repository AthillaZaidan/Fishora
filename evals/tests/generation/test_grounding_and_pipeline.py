"""Generation-side gates: the critic's grounding and the full card pipeline.

Pipeline numbers come from the production ``run_graph`` driven by the
scripted LLM (see evals/fakes.py); they measure guardrails, not prose.
"""

import pytest

from apps.main_api.services import orchestrator
from evals import grounding_eval, pipeline_eval

DELAY = 0.05


@pytest.fixture(scope="module")
def grounding(store, e5):
    return grounding_eval.evaluate(store, e5)


@pytest.fixture(scope="module")
def plain(store, e5):
    return pipeline_eval.run_scenario(store, e5, delay=DELAY, fenced=False)


@pytest.fixture(scope="module")
def fenced(store, e5):
    return pipeline_eval.run_scenario(store, e5, delay=0.0, fenced=True, repeat=False)


def _grade(claim: str, chunk, e5) -> str:
    return "supported" if grounding_eval._critic_accepts(claim, chunk, e5) else "unsupported"


def test_indonesian_claim_is_grounded_in_english_evidence(store, e5):
    chunk = store._to_retrieved("chunk_kembung_physical_001", 0.0)
    claim = ("Mencapai 24 cm dan 750 g, kepala lebih panjang dari tinggi badan, 2 sampai 6 "
             "bintik besar di pangkal sirip punggung pertama.")
    assert _grade(claim, chunk, e5) == "supported"


def test_shared_token_does_not_ground_a_borrowed_claim(store, e5):
    """'Indo-Pasifik' shares the token 'indo' with the croaker chunk; the
    claim is still about milkfish and must not be accepted."""
    chunk = store._to_retrieved("chunk_gulamah_identity_001", 0.0)
    claim = "Chanos chanos, satu-satunya anggota famili Chanidae, hidup di laut, payau dan tawar di Indo-Pasifik Barat."
    assert _grade(claim, chunk, e5) == "unsupported"


def test_wrong_numbers_are_not_grounded(store, e5):
    chunk = store._to_retrieved("chunk_kembung_physical_001", 0.0)
    claim = "Mencapai 60 cm dan 4,3 kg, kepala lebih panjang dari tinggi badan."
    assert _grade(claim, chunk, e5) == "unsupported"


def test_grounding_test_split_f1(grounding):
    assert grounding["test"]["f1"] >= 0.85


def test_grounding_false_support_rate(grounding):
    assert grounding["test"]["false_support_rate"] <= 0.15


def test_every_card_job_completes(plain):
    assert plain["job_success_rate"] == 1.0


def test_true_claims_survive_into_the_card(plain):
    assert plain["claim_retention"] >= 0.8


def test_borrowed_claims_do_not_leak(plain):
    assert plain["hallucination_leakage"] <= 0.1


def test_every_citation_belongs_to_the_species(plain):
    assert plain["citation_validity"] == 1.0


def test_card_needs_at_most_two_sequential_llm_rounds(plain):
    assert plain["sequential_rounds"] <= 2.5


def test_repeat_card_is_served_from_cache(plain):
    assert plain["repeat_latency_ms_mean"] < 50


def test_markdown_fenced_llm_output_still_completes(fenced):
    assert fenced["job_success_rate"] == 1.0


def test_llm_outage_fails_the_job_cleanly(store, e5, species):
    from evals.fakes import InMemoryJobRepository, InMemorySpeciesRepository

    class Down:
        def invoke(self, prompt):
            raise ConnectionError("opencode down")

    jobs = InMemoryJobRepository()
    jobs.create("j1", "j1", "species_nila")
    clear = getattr(orchestrator, "clear_card_cache", None)
    if clear:
        clear()
    orchestrator.run_graph("j1", "species_nila", "j1", store, e5, Down(), Down(),
                           InMemorySpeciesRepository(species), jobs)
    job = jobs.get("j1")
    assert job.status == "failed" and job.final_card is None

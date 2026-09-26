"""FastAPI app wired with in-memory ports: verification gate, job lifecycle,
startup warmup."""

import threading

from fastapi.testclient import TestClient

from evals.tests.conftest import png_bytes, sign_in


def _identify(client) -> str:
    response = client.post("/api/v1/fish/identify", files={"file": ("f.png", png_bytes(), "image/png")})
    assert response.status_code == 200, response.text
    return response.json()["prediction_id"]


def test_health_reports_seeded_taxonomy(app_factory):
    app, _ = app_factory()
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok", "taxonomy_seeded": True}


def test_pending_prediction_gets_no_knowledge(app_factory):
    app, _ = app_factory()
    with TestClient(app) as client:
        sign_in(client)
        prediction_id = _identify(client)
        assert client.get(f"/api/v1/predictions/{prediction_id}/knowledge").status_code == 409


def test_verification_schedules_a_job_that_completes(app_factory):
    app, deps = app_factory("nila")
    with TestClient(app) as client:
        sign_in(client)
        prediction_id = _identify(client)
        verify = client.post("/api/v1/fish/verify", json={
            "prediction_id": prediction_id, "verified_species_id": "species_nila"})
        assert verify.json()["verification_status"] == "confirmed"
        job = client.get(f"/api/v1/jobs/{prediction_id}").json()
        assert job["status"] == "completed", job


def test_correction_retrieves_for_the_corrected_species(app_factory):
    app, _ = app_factory("nila")
    with TestClient(app) as client:
        sign_in(client)
        prediction_id = _identify(client)
        client.post("/api/v1/fish/verify", json={
            "prediction_id": prediction_id, "verified_species_id": "species_mujair"})
        body = client.get(f"/api/v1/predictions/{prediction_id}/knowledge").json()
        assert body["species_id"] == "species_mujair"
        assert body["card"]["common_name"] == "Mujair"


def test_embedder_is_warmed_at_startup(app_factory):
    app, deps = app_factory()
    warmed = threading.Event()
    deps.embedder = type("Spy", (), {
        "model_name": deps.embedder.model_name,
        "warmup": lambda self: warmed.set(),
    })()
    with TestClient(app):
        assert warmed.wait(2.0)


def _declare(client, species_id="species_nila") -> str:
    response = client.post("/api/v1/fish/manual", data={"species_id": species_id},
                           files={"file": ("f.png", png_bytes(), "image/png")})
    assert response.status_code == 200, response.text
    return response.json()["prediction_id"]


def test_manual_catch_card_is_graded_by_the_critic(app_factory):
    """W19: a catch with no background job gets its card from the same pipeline
    as the job, so every claim the scripted writer borrowed from another species is dropped."""
    from evals.corpus import load_corpus
    from evals.pipeline_eval import _claim_in_card

    from apps.main_api.services.orchestrator import clear_card_cache

    clear_card_cache()  # a cached card would skip the writer this test watches
    app, deps = app_factory("kembung")
    own_sources = {c.source["id"] for c in load_corpus() if c.species_label == "kembung"}
    with TestClient(app) as client:
        sign_in(client)
        # Kembung has no taste/texture evidence, so the scripted writer borrows a claim for it.
        prediction_id = _declare(client, "species_kembung")
        assert deps.job_repo.list_by_prediction(prediction_id) == []
        emitted_before = len(deps.llm.emitted)
        response = client.get(f"/api/v1/predictions/{prediction_id}/knowledge")
    assert response.status_code == 200, response.text
    card = response.json()["card"]
    emitted = deps.llm.emitted[emitted_before:]
    assert any(c.kind == "hallucination" for c in emitted), "the scripted model must try to over-claim"
    assert not any(c.kind == "hallucination" and _claim_in_card(card, c.field, c.text) for c in emitted)
    assert any(c.kind == "true" and _claim_in_card(card, c.field, c.text) for c in emitted)
    assert card["sources"] and {s["source_id"] for s in card["sources"]} <= own_sources


def test_manual_catch_without_a_key_is_a_provider_outage(app_factory):
    """W3 on the graded sync path: no key is a mapped 502, never a 500 or an empty card."""
    app, deps = app_factory("nila", llm=None)
    with TestClient(app) as client:
        sign_in(client)
        prediction_id = _declare(client)
        response = client.get(f"/api/v1/predictions/{prediction_id}/knowledge")
    assert response.status_code == 502, response.text


def test_card_job_records_a_stage_trace(app_factory):
    """W12: the job keeps what the run saw and decided, per the judge methodology §6.1."""
    from apps.main_api.services.orchestrator import clear_card_cache

    clear_card_cache()  # a cache hit runs no writer, and its trace says only that
    app, deps = app_factory("nila")
    with TestClient(app) as client:
        sign_in(client)
        prediction_id = _identify(client)
        client.post("/api/v1/fish/verify", json={
            "prediction_id": prediction_id, "verified_species_id": "species_nila"})
    [job] = deps.job_repo.list_by_prediction(prediction_id)
    trace = job.trace
    assert job.status == "completed" and trace, job
    assert {"pipeline", "model", "evidence", "claims", "llm_calls", "timings_ms"} <= set(trace)
    assert trace["evidence"] and all({"chunk_id", "category", "distance"} <= set(e) for e in trace["evidence"])
    assert trace["claims"] and all({"field", "text", "label", "chunk_ids"} <= set(c) for c in trace["claims"])
    assert trace["llm_calls"] >= 2 and not trace["cache_hit"]
    assert {"research", "write_and_verify", "assemble", "total"} <= set(trace["timings_ms"])

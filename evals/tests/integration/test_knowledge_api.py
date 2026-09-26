"""FastAPI app wired with in-memory ports: verification gate, job lifecycle,
startup warmup."""

import threading

from fastapi.testclient import TestClient

from evals.tests.conftest import png_bytes


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
        prediction_id = _identify(client)
        assert client.get(f"/api/v1/predictions/{prediction_id}/knowledge").status_code == 409


def test_verification_schedules_a_job_that_completes(app_factory):
    app, deps = app_factory("nila")
    with TestClient(app) as client:
        prediction_id = _identify(client)
        verify = client.post("/api/v1/fish/verify", json={
            "prediction_id": prediction_id, "verified_species_id": "species_nila"})
        assert verify.json()["verification_status"] == "confirmed"
        job = client.get(f"/api/v1/jobs/{prediction_id}").json()
        assert job["status"] == "completed", job


def test_correction_retrieves_for_the_corrected_species(app_factory):
    app, _ = app_factory("nila")
    with TestClient(app) as client:
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

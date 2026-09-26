"""End to end through the HTTP API: photo -> identification -> operator
verification -> background agent graph -> grounded knowledge card; plus the
quality dashboard that reports on all of it."""

from fastapi.testclient import TestClient

from evals.corpus import load_corpus
from evals.tests.conftest import png_bytes


def test_photo_to_grounded_card(app_factory):
    app, _ = app_factory("tuna")
    own_sources = {c.source["id"] for c in load_corpus() if c.species_label == "tuna"}
    with TestClient(app) as client:
        identified = client.post("/api/v1/fish/identify",
                                 files={"file": ("tuna.png", png_bytes(), "image/png")}).json()
        assert identified["prediction"]["normalized_label"] == "tuna"
        prediction_id = identified["prediction_id"]

        client.post("/api/v1/fish/verify", json={
            "prediction_id": prediction_id, "verified_species_id": "species_tuna"})
        response = client.get(f"/api/v1/predictions/{prediction_id}/knowledge")
        assert response.status_code == 200, response.text
        card = response.json()["card"]

    assert card["scientific_name"] == "Thunnus spp."  # relational guardrail wins
    assert card["sources"], "a card built from evidence must cite it"
    assert {s["source_id"] for s in card["sources"]} <= own_sources
    assert card["taste"] or card["physical_characteristics"]


def test_quality_dashboard_is_served(app_factory):
    app, _ = app_factory()
    with TestClient(app) as client:
        page = client.get("/quality")
        assert page.status_code == 200 and "Fishora RAG quality" in page.text
        summary = client.get("/api/v1/quality/summary")
        assert summary.status_code == 200
        assert {"runs", "comparison"} <= set(summary.json())

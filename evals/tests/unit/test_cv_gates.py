"""The gated CV export may refuse a photo; nothing is saved and the API says why."""

import pytest
from fastapi.testclient import TestClient

from apps.contracts import CVCandidate, CVPredictionEnvelope
from apps.main_api.errors import PhotoRejected
from apps.main_api.main import create_main_app
from apps.main_api.ports import AppDependencies
from apps.main_api.services.identification import IdentificationService
from evals.corpus import species_records
from evals.fakes import InMemoryImageStore, InMemoryPredictionRepository, InMemorySpeciesRepository
from evals.tests.conftest import png_bytes, sign_in

pytestmark = pytest.mark.unit

GATE = {"fish_prob": 0.02, "fish_threshold": 0.5, "knn_distance": 0.9, "knn_threshold": 0.4}


class GatedCV:
    def __init__(self, status: str):
        self.status = status

    def predict(self, image_bytes, *, filename, content_type):
        top = [CVCandidate(label=label, confidence=c) for label, c in (("nila", 0.6), ("mujair", 0.3), ("bandeng", 0.1))]
        return CVPredictionEnvelope(model_version="fishora-vit-b-gated", status=self.status, prediction=top[0],
                                    top_candidates=top, threshold=0.0, gate=GATE)


def _service(status):
    images, predictions = InMemoryImageStore(), InMemoryPredictionRepository()
    service = IdentificationService(GatedCV(status), InMemorySpeciesRepository(species_records()), predictions,
                                    images, max_image_bytes=10 * 1024 * 1024)
    return service, images, predictions


@pytest.mark.parametrize("status,reason", [("rejected_not_fish", "not_fish"),
                                            ("rejected_unknown_species", "unknown_species")])
def test_a_rejected_photo_saves_nothing(status, reason):
    service, images, predictions = _service(status)
    with pytest.raises(PhotoRejected) as caught:
        service.identify(png_bytes(), filename="f.png", content_type="image/png")
    assert caught.value.reason == reason
    assert not images.saved and not predictions.rows


def test_an_accepted_gated_photo_is_still_a_prediction():
    service, _, predictions = _service("low_confidence_human_verification_required")
    result = service.identify(png_bytes(), filename="f.png", content_type="image/png")
    assert result.prediction.normalized_label == "nila" and result.prediction_id in predictions.rows


@pytest.mark.parametrize("status,detail", [("rejected_not_fish", "photo rejected: not a fish"),
                                            ("rejected_unknown_species", "photo rejected: unknown species")])
def test_the_api_answers_422_with_the_gate_that_refused(status, detail):
    deps = AppDependencies(cv_client=GatedCV(status), species_repo=InMemorySpeciesRepository(species_records()),
                           prediction_repo=InMemoryPredictionRepository(), image_store=InMemoryImageStore(),
                           embedder=type("E5Stub", (), {"model_name": "intfloat/multilingual-e5-base"})())
    with TestClient(create_main_app(deps=deps)) as client:
        sign_in(client)
        response = client.post("/api/v1/fish/identify", files={"file": ("f.png", png_bytes(), "image/png")})
    assert response.status_code == 422 and response.json()["detail"] == detail

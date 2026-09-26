import uuid
from dataclasses import dataclass
from typing import Literal

from apps.common.image import validate_image_bytes
from apps.main_api.errors import PhotoRejected, UnsupportedCvLabel


@dataclass(frozen=True)
class SpeciesCandidate:
    species_id: str
    normalized_label: str
    confidence: float


@dataclass(frozen=True)
class IdentificationResult:
    prediction_id: str
    model_version: str
    status: Literal["confident_prediction", "low_confidence_human_verification_required"]
    prediction: SpeciesCandidate
    top_candidates: list[SpeciesCandidate]
    threshold: float
    verification_status: Literal["pending"] = "pending"


class IdentificationService:
    """Trust boundary -> CV -> label mapping -> image persistence -> prediction persistence.

    Order matters: CV failures or unsupported labels raise before any image file
    or prediction row is written.
    """

    def __init__(self, cv_client, species_repo, prediction_repo, image_store, max_image_bytes: int):
        self._cv_client = cv_client
        self._species_repo = species_repo
        self._prediction_repo = prediction_repo
        self._image_store = image_store
        self._max_image_bytes = max_image_bytes

    def identify(self, image_bytes: bytes, *, filename: str, content_type: str) -> IdentificationResult:
        validate_image_bytes(image_bytes, content_type, self._max_image_bytes)  # 400/413/415 before CV

        envelope = self._cv_client.predict(image_bytes, filename=filename, content_type=content_type)
        if envelope.status.startswith("rejected_"):
            # Refused by the CV gates: raise before any image or prediction is
            # written, as for a CV failure. A rejected photo identifies nothing.
            raise PhotoRejected(envelope.status.removeprefix("rejected_"))

        prediction = self._map(envelope.prediction.label, envelope.prediction.confidence)
        top_candidates = [self._map(candidate.label, candidate.confidence) for candidate in envelope.top_candidates]

        prediction_id = uuid.uuid4().hex
        image_reference = self._image_store.save(prediction_id, image_bytes, content_type)
        try:
            record = self._prediction_repo.create(
                prediction_id=prediction_id,
                image_reference=image_reference,
                predicted_species_id=prediction.species_id,
                confidence=prediction.confidence,
                top_candidates=[
                    {"species_id": c.species_id, "normalized_label": c.normalized_label, "confidence": c.confidence}
                    for c in top_candidates
                ],
                model_version=envelope.model_version,
            )
        except Exception:
            # Compensation: remove only the image saved for this attempt; the
            # prediction row was never committed. Successful persistence never deletes.
            self._image_store.delete(image_reference)
            raise
        return IdentificationResult(
            prediction_id=record.id,
            model_version=envelope.model_version,
            status=_reportable_status(envelope.status, envelope.threshold),
            prediction=prediction,
            top_candidates=top_candidates,
            threshold=envelope.threshold,
            verification_status=record.verification_status,
        )

    def _map(self, label: str, confidence: float) -> SpeciesCandidate:
        species = self._species_repo.get_by_normalized_label(label)
        if species is None:
            raise UnsupportedCvLabel(label)
        return SpeciesCandidate(species_id=species.id, normalized_label=species.normalized_label, confidence=confidence)


def _reportable_status(status: str, threshold: float) -> str:
    """Downgrade confidence the model was never in a position to claim.

    A threshold of zero accepts every candidate, so `confident_prediction` under
    it carries no information: a plain grey image scored about 0.99 against the
    shipped export. Repeating that verdict would tell an operator the model is
    sure when it has not been asked a question it could fail. The threshold and
    the confidences are still reported exactly as measured.
    """
    if threshold <= 0.0:
        return "low_confidence_human_verification_required"
    return status

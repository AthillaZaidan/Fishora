from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CVCandidate(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)


class CVGate(BaseModel):
    """The gated export's two rejection checks, as measured on this photo."""
    model_config = ConfigDict(extra="forbid")
    fish_prob: float
    fish_threshold: float
    knn_distance: float
    knn_threshold: float


class CVPredictionEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_version: str
    # The gated export (ai/results/fishora_vit_b_gated) may refuse a photo: it
    # does not show a fish, or it shows a fish outside the 11 known species.
    status: Literal[
        "confident_prediction",
        "low_confidence_human_verification_required",
        "rejected_not_fish",
        "rejected_unknown_species",
    ]
    prediction: CVCandidate
    top_candidates: list[CVCandidate] = Field(min_length=3, max_length=3)
    threshold: float = Field(ge=0.0, le=1.0)
    gate: CVGate | None = None


class ImageValidationError(Exception):
    """Shared trust-boundary error raised before either service invokes inference."""

    def __init__(self, status_code: Literal[400, 413, 415], message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
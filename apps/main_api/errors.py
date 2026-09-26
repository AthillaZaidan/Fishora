from decimal import Decimal


class CvUnavailable(Exception):
    """Upstream CV failure: connection error, timeout, non-2xx, or unparsable body.

    Maps to HTTP 503. The message must never contain the internal CV URL,
    credentials, or request headers; the API layer emits a fixed generic detail.
    """


class PredictionNotFound(Exception):
    """No stored prediction for the given id. Maps to HTTP 404."""

    def __init__(self, prediction_id: str):
        super().__init__(f"prediction {prediction_id!r} not found")
        self.prediction_id = prediction_id


class UnsupportedSpecies(Exception):
    """The verified species id is not one of the seeded supported rows. Maps to HTTP 422."""

    def __init__(self, species_id: str):
        super().__init__(f"species {species_id!r} is not supported")
        self.species_id = species_id


class PhotoRejected(Exception):
    """The CV gates refused the photo: not a fish, or not one of the known species.
    Nothing is saved; the operator retakes the photo or names the species by hand."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason  # "not_fish" or "unknown_species"


class UnsupportedCvLabel(Exception):
    """The CV service returned a label outside the seeded taxonomy (upstream contract error).

    Maps to HTTP 502 and must never persist a prediction or image file.
    """

    def __init__(self, label: str):
        super().__init__(f"cv returned unsupported label {label!r}")
        self.label = label


class PredictionNotVerified(Exception):
    """The prediction exists but has no verified identity yet. Maps to HTTP 409.

    Only a stored verified_species_id (confirmed or corrected) may drive
    retrieval; pending predictions must never reach the retriever.
    """

    def __init__(self, prediction_id: str):
        super().__init__(f"prediction {prediction_id!r} is not verified")
        self.prediction_id = prediction_id


class OpenCodeUnavailable(Exception):
    """OpenCode Go timeout/connection failure during generation.

    Maps to HTTP 502. Carries the retrieved chunk ids so the 502 body can
    surface evidence ids for diagnosis, but never credentials, internal
    URLs, headers, or the raw upstream error.
    """

    def __init__(self, message: str, retrieved_chunk_ids: list[str]):
        super().__init__(message)
        self.retrieved_chunk_ids = list(retrieved_chunk_ids)


class RetrievalUnavailable(Exception):
    """Evidence retrieval cannot run at all, e.g. the embedding stack is absent.

    Distinct from OpenCodeUnavailable: nothing was retrieved, so there are no
    chunk ids to report. Maps to HTTP 502.
    """


class InvalidGeneratedKnowledge(Exception):
    """Generated JSON failed decoding, strict schema validation, or citation
    checks against the retrieved evidence. Maps to HTTP 502 and is safe to
    retry: nothing was persisted."""

    def __init__(self, message: str, retrieved_chunk_ids: list[str]):
        super().__init__(message)
        self.retrieved_chunk_ids = list(retrieved_chunk_ids)


class InvalidLot(Exception):
    """Lot fields fail domain checks (non-positive quantity/price, bad size). Maps to HTTP 422."""

    def __init__(self, message: str):
        super().__init__(message)


class LotNotFound(Exception):
    """No stored lot for the given id. Maps to HTTP 404."""

    def __init__(self, lot_id: str):
        super().__init__(f"lot {lot_id!r} not found")
        self.lot_id = lot_id


class LotAlreadyPublished(Exception):
    """A lot already exists for this prediction. Maps to HTTP 409.

    HANDOFF 11 models Prediction 1 -> 0..1 AuctionLot: one landed catch is one
    auction, so republishing must be refused rather than double-listing it.
    """

    def __init__(self, prediction_id: str):
        super().__init__(f"prediction {prediction_id!r} already has a published lot")
        self.prediction_id = prediction_id


class LotClosed(Exception):
    """The auction window is over or the lot is no longer active. Maps to HTTP 409."""

    def __init__(self, lot_id: str):
        super().__init__(f"lot {lot_id!r} is closed")
        self.lot_id = lot_id


class BidOutbid(Exception):
    """The bid is at or below the current highest. Maps to HTTP 409 with that floor."""

    def __init__(self, current_highest_per_kg: Decimal):
        super().__init__("bid must exceed current highest")
        self.current_highest_per_kg = current_highest_per_kg


class LotNotAllocatable(Exception):
    """Allocation requires a closed lot with at least one bid. Maps to HTTP 409."""

    def __init__(self, lot_id: str, message: str = "lot is not allocatable"):
        super().__init__(message)
        self.lot_id = lot_id


class Unauthenticated(Exception):
    """No valid session cookie. Maps to HTTP 401."""


class Forbidden(Exception):
    """Authenticated but the role or identity is not allowed. Maps to HTTP 403."""

    def __init__(self, message: str = "forbidden"):
        super().__init__(message)
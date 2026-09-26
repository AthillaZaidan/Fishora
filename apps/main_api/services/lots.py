"""Verification-gated lot publication, listing, bidding, and allocation.

Species identity is copied from the prediction's stored verified_species_id.
Callers never supply a species id, mirroring KnowledgeService.
Bid races are serialised inside the lot repository (row lock).
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4

from apps.main_api.contracts import BidRecord, LotRecord
from apps.main_api.errors import (
    InvalidLot,
    LotAlreadyPublished,
    LotNotFound,
    PredictionNotFound,
    PredictionNotVerified,
)
from apps.main_api.services.matching import fold_words, lot_characteristics, lot_uses

DEFAULT_AUCTION_HOURS = 4
# A lot that never closes cannot be allocated, and one that closes instantly
# cannot be bid on, so the window is bounded rather than free-form.
MIN_AUCTION_HOURS = 1
MAX_AUCTION_HOURS = 72
_POSITIVE_SIZES = {"S", "M", "L"}


class LotService:
    def __init__(
        self,
        prediction_repo,
        lot_repo,
        *,
        auction_hours: int = DEFAULT_AUCTION_HOURS,
        landing_point_repo=None,
        knowledge_service=None,
        job_repo=None,
    ):
        self._prediction_repo = prediction_repo
        self._lot_repo = lot_repo
        self._auction_hours = auction_hours
        self._landing_point_repo = landing_point_repo
        self._knowledge_service = knowledge_service
        self._job_repo = job_repo

    def _graded_card(self, prediction_id: str, species_id: str) -> dict | None:
        """The critic-graded card the operator was shown, if its job completed
        for the species the prediction is verified as now. Publishing it keeps
        the lot, the QR page and buyer matching on the same checked card (W19)."""
        if self._job_repo is None:
            return None
        try:
            jobs = self._job_repo.list_by_prediction(prediction_id)
        except Exception:
            return None
        for job in reversed(jobs):
            if job.status == "completed" and job.final_card and job.species_id == species_id:
                return job.final_card
        return None

    def publish(
        self,
        *,
        prediction_id: str,
        operator_id: str,
        quantity_kg: Decimal,
        starting_price_per_kg: Decimal,
        size_category: str,
        landing_point_id: str,
        auction_hours: int | None = None,
        seller_fisher_group: str | None = None,
        now: datetime | None = None,
    ) -> LotRecord:
        hours = self._auction_hours if auction_hours is None else int(auction_hours)
        if not MIN_AUCTION_HOURS <= hours <= MAX_AUCTION_HOURS:
            raise ValueError(
                f"auction_hours must be between {MIN_AUCTION_HOURS} and {MAX_AUCTION_HOURS}"
            )
        record = self._prediction_repo.get(prediction_id)
        if record is None:
            raise PredictionNotFound(prediction_id)
        if record.verification_status not in ("confirmed", "corrected") or record.verified_species_id is None:
            raise PredictionNotVerified(prediction_id)
        if quantity_kg <= 0 or starting_price_per_kg <= 0:
            raise InvalidLot("quantity and starting price must be greater than zero")
        if size_category not in _POSITIVE_SIZES:
            raise InvalidLot("size_category must be S, M, or L")
        if self._lot_repo.get_by_prediction(prediction_id) is not None:
            raise LotAlreadyPublished(prediction_id)

        starts = now or datetime.now(timezone.utc)
        lot_id = uuid4().hex
        label = record.verified_species_id.removeprefix("species_")
        snapshot = self._graded_card(prediction_id, record.verified_species_id)
        if snapshot is None and self._knowledge_service is not None:
            # Best effort. The catch is landed and the auction has to open; a
            # knowledge card that cannot be generated is a degraded listing, not
            # a reason to refuse publication. The lot page and the discover page
            # both already render without a snapshot.
            try:
                snapshot = self._knowledge_service.get_for_prediction(
                    prediction_id
                ).card.model_dump(mode="json")
            except Exception:
                snapshot = None
        lot = LotRecord(
            id=lot_id,
            prediction_id=record.id,
            operator_id=operator_id,
            species_id=record.verified_species_id,
            landing_point_id=landing_point_id,
            quantity_kg=quantity_kg,
            size_category=size_category,
            starting_price_per_kg=starting_price_per_kg,
            status="active",
            auction_starts_at=starts,
            auction_ends_at=starts + timedelta(hours=hours),
            public_slug=f"{label}-{lot_id[:8]}",
            knowledge_snapshot=snapshot,
            seller_fisher_group=seller_fisher_group,
        )
        return self._lot_repo.create(lot)

    def get(self, lot_id: str) -> LotRecord:
        lot = self._lot_repo.get(lot_id)
        if lot is None:
            raise LotNotFound(lot_id)
        return lot

    def get_by_slug(self, public_slug: str) -> LotRecord:
        lot = self._lot_repo.get_by_slug(public_slug)
        if lot is None:
            raise LotNotFound(public_slug)
        return lot

    def list_lots(
        self,
        *,
        species_ids: list[str] | None = None,
        intended_uses: list[str] | None = None,
        characteristics: list[str] | None = None,
        min_price: Decimal | None = None,
        max_price: Decimal | None = None,
        min_quantity: Decimal | None = None,
        max_quantity: Decimal | None = None,
        status: str | None = None,
        operator_id: str | None = None,
        buyer_lat: float | None = None,
        buyer_lon: float | None = None,
        serviceability_radius_km: float | None = None,
    ) -> list[LotRecord]:
        lots = self._lot_repo.all()
        wanted_species = set(species_ids or ())
        wanted_uses = fold_words(list(intended_uses or ()))
        wanted_characteristics = fold_words(list(characteristics or ()))
        filtered = []
        for lot in lots:
            if wanted_species and lot.species_id not in wanted_species:
                continue
            # OR within a term list, AND across the two lists: the matching
            # engine scores each criterion on set intersection, and the buyer
            # preview reads this endpoint to predict that score.
            if wanted_uses and not wanted_uses & fold_words(lot_uses(lot)):
                continue
            if wanted_characteristics and not wanted_characteristics & fold_words(
                lot_characteristics(lot)
            ):
                continue
            if status and lot.status != status:
                continue
            if operator_id and lot.operator_id != operator_id:
                continue
            if min_price is not None and lot.starting_price_per_kg < min_price:
                continue
            if max_price is not None and lot.starting_price_per_kg > max_price:
                continue
            if min_quantity is not None and lot.quantity_kg < min_quantity:
                continue
            if max_quantity is not None and lot.quantity_kg > max_quantity:
                continue
            if buyer_lat is not None and buyer_lon is not None and self._landing_point_repo is not None:
                from apps.main_api.services.geo import DEFAULT_SERVICEABILITY_RADIUS_KM, within_serviceability

                radius = (
                    serviceability_radius_km
                    if serviceability_radius_km is not None
                    else DEFAULT_SERVICEABILITY_RADIUS_KM
                )
                point = self._landing_point_repo.get(lot.landing_point_id)
                if point is None or not within_serviceability(
                    buyer_lat, buyer_lon, point.latitude, point.longitude, radius
                ):
                    continue
            filtered.append(lot)
        return filtered

    def place_bid(
        self,
        lot_id: str,
        buyer_id: str,
        amount_per_kg: Decimal,
        now: datetime | None = None,
    ) -> BidRecord:
        if amount_per_kg <= 0:
            raise InvalidLot("bid amount must be greater than zero")
        return self._lot_repo.place_bid(lot_id, buyer_id, amount_per_kg, now=now)

    def list_bids(self, lot_id: str) -> list[BidRecord]:
        self.get(lot_id)
        return self._lot_repo.list_bids(lot_id)

    def current_highest(self, lot_id: str) -> Decimal | None:
        return self._lot_repo.highest(lot_id)

    def close(self, lot_id: str) -> LotRecord:
        return self._lot_repo.close(lot_id)

    def allocate(self, lot_id: str, now: datetime | None = None) -> LotRecord:
        return self._lot_repo.allocate(lot_id, now=now)

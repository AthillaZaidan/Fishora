"""Verification-gated lot publication, listing, bidding, and allocation.

Species identity is copied from the prediction's stored verified_species_id.
Callers never supply a species id, mirroring KnowledgeService.
Bid races are serialised inside the lot repository (row lock).

An auction ends at auction_ends_at whether or not anyone closes it: every read
closes expired lots first, so no sweeper process is needed and a lot past its
end is never served as live.
"""

import logging
from dataclasses import replace
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
from apps.main_api.services.reference_cards import fill_from_reference

# The durations the operator can pick. Market testing asked for short windows:
# a landed catch is sold the same morning, not over days.
AUCTION_MINUTE_OPTIONS = (30, 60, 120, 180)
DEFAULT_AUCTION_MINUTES = 60
# How many lots one catch can be split into.
MAX_LOT_COUNT = 50
_POSITIVE_SIZES = {"S", "M", "L"}

logger = logging.getLogger(__name__)


class LotService:
    def __init__(
        self,
        prediction_repo,
        lot_repo,
        *,
        auction_minutes: int = DEFAULT_AUCTION_MINUTES,
        landing_point_repo=None,
        knowledge_service=None,
        job_repo=None,
        image_store=None,
    ):
        self._prediction_repo = prediction_repo
        self._lot_repo = lot_repo
        self._auction_minutes = auction_minutes
        self._landing_point_repo = landing_point_repo
        self._knowledge_service = knowledge_service
        self._job_repo = job_repo
        self._image_store = image_store

    def _graded_card(self, prediction_id: str, species_id: str) -> dict | None:
        """The critic-graded card the operator was shown, if its job completed
        for the species the prediction is verified as now. Publishing it keeps
        the lot, the QR page and buyer matching on the same checked card (W19)."""
        if self._job_repo is None:
            return None
        try:
            jobs = self._job_repo.list_by_prediction(prediction_id)
        except Exception:
            logger.exception("could not read knowledge jobs for prediction %s", prediction_id)
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
        lot_count: int = 1,
        auction_minutes: int | None = None,
        seller_fisher_group: str | None = None,
        now: datetime | None = None,
    ) -> list[LotRecord]:
        """Publish a verified catch as `lot_count` lots of `quantity_kg` each.

        Every lot is its own auction with its own winner, so several buyers can
        each take part of one catch. All lots share the catch's species, price,
        window and knowledge card.
        """
        minutes = self._auction_minutes if auction_minutes is None else int(auction_minutes)
        if minutes not in AUCTION_MINUTE_OPTIONS:
            raise InvalidLot(f"auction_minutes must be one of {AUCTION_MINUTE_OPTIONS}")
        if not 1 <= lot_count <= MAX_LOT_COUNT:
            raise InvalidLot(f"lot_count must be between 1 and {MAX_LOT_COUNT}")
        record = self._prediction_repo.get(prediction_id)
        if record is None:
            raise PredictionNotFound(prediction_id)
        if record.verification_status not in ("confirmed", "corrected") or record.verified_species_id is None:
            raise PredictionNotVerified(prediction_id)
        if quantity_kg <= 0 or starting_price_per_kg <= 0:
            raise InvalidLot("quantity and starting price must be greater than zero")
        if size_category not in _POSITIVE_SIZES:
            raise InvalidLot("size_category must be S, M, or L")
        # The landing point is where buyers collect and what the geo filter
        # measures from; an unknown id would publish a lot no radius can find.
        if self._landing_point_repo is not None and self._landing_point_repo.get(landing_point_id) is None:
            raise InvalidLot(f"unknown landing_point_id {landing_point_id!r}")
        if self._lot_repo.get_by_prediction(prediction_id) is not None:
            raise LotAlreadyPublished(prediction_id)

        starts = now or datetime.now(timezone.utc)
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
                logger.warning("lot for prediction %s published without a card", prediction_id, exc_info=True)
                snapshot = None
        # Empty fields fall back to the species' reference notes, labelled as such.
        snapshot = fill_from_reference(snapshot, record.verified_species_id)
        lots = []
        for index in range(1, lot_count + 1):
            lot_id = uuid4().hex
            lots.append(
                LotRecord(
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
                    auction_ends_at=starts + timedelta(minutes=minutes),
                    public_slug=f"{label}-{lot_id[:8]}",
                    knowledge_snapshot=snapshot,
                    seller_fisher_group=seller_fisher_group,
                    batch_index=index,
                    batch_size=lot_count,
                )
            )
        return self._lot_repo.create_many(lots)

    def _close_expired(self, now: datetime | None) -> datetime:
        """Persist the end of every auction that has run out, and return the
        clock the caller should judge the lots it reads next against."""
        clock = now or datetime.now(timezone.utc)
        close_expired = getattr(self._lot_repo, "close_expired", None)
        if callable(close_expired):
            try:
                close_expired(clock)
            except Exception:
                # A read must not fail because the write behind it did; the
                # in-memory settle below still reports the lot as ended.
                logger.exception("could not close expired auctions")
        return clock

    @staticmethod
    def _settled(lot: LotRecord, clock: datetime) -> LotRecord:
        # Covers a lot that expired between the UPDATE and the SELECT, and a
        # repository without close_expired.
        if lot.status == "active" and clock >= lot.auction_ends_at:
            return replace(lot, status="closed")
        return lot

    def get(self, lot_id: str, now: datetime | None = None) -> LotRecord:
        clock = self._close_expired(now)
        lot = self._lot_repo.get(lot_id)
        if lot is None:
            raise LotNotFound(lot_id)
        return self._settled(lot, clock)

    def get_by_slug(self, public_slug: str, now: datetime | None = None) -> LotRecord:
        clock = self._close_expired(now)
        lot = self._lot_repo.get_by_slug(public_slug)
        if lot is None:
            raise LotNotFound(public_slug)
        return self._settled(lot, clock)

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
        allocated_buyer_id: str | None = None,
        buyer_lat: float | None = None,
        buyer_lon: float | None = None,
        serviceability_radius_km: float | None = None,
        now: datetime | None = None,
    ) -> list[LotRecord]:
        clock = self._close_expired(now)
        lots = [self._settled(lot, clock) for lot in self._lot_repo.all()]
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
            if allocated_buyer_id and lot.allocated_buyer_id != allocated_buyer_id:
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

    def photo(self, lot: LotRecord):
        """(path, content_type) of the catch photo the lot was identified from,
        or None when there is none (no store, or a store that cannot serve)."""
        find = getattr(self._image_store, "find", None)
        return find(lot.prediction_id) if callable(find) else None

    def close(self, lot_id: str) -> LotRecord:
        return self._lot_repo.close(lot_id)

    def allocate(self, lot_id: str, now: datetime | None = None) -> LotRecord:
        return self._lot_repo.allocate(lot_id, now=now)


def bidder_labels(bids: list[BidRecord]) -> dict[str, str]:
    """Public names for the buyers on one lot: "Bidder 1" is whoever bid first.

    Buyer ids are account identifiers, not something other buyers should read.
    Numbering by first bid keeps a buyer's label stable for the whole auction,
    so the history still shows one buyer raising against another.
    """
    labels: dict[str, str] = {}
    for bid in sorted(bids, key=lambda bid: (bid.created_at, bid.id)):
        if bid.buyer_id not in labels:
            labels[bid.buyer_id] = f"Bidder {len(labels) + 1}"
    return labels


def winning_bid(bids: list[BidRecord]) -> BidRecord | None:
    """The bid allocation picks: highest amount, earliest on a tie (as the repository does)."""
    if not bids:
        return None
    return min(bids, key=lambda bid: (-bid.amount_per_kg, bid.created_at, bid.id))

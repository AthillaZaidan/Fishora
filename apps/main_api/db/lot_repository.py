"""SQL lot/bid store. Bid placement locks the lot row so concurrent equal bids cannot both win."""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Callable
from uuid import uuid4

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from apps.main_api.contracts import BidRecord, LandingPointRecord, LotRecord
from apps.main_api.db.models import Bid, LandingPoint, Lot
from apps.main_api.errors import (
    BidOutbid,
    LotAlreadyPublished,
    LotClosed,
    LotNotAllocatable,
    LotNotFound,
)


class SqlLotRepository:
    def __init__(self, session_factory: Callable[[], Session]):
        self._session_factory = session_factory

    def create(self, lot: LotRecord) -> LotRecord:
        return self.create_many([lot])[0]

    def create_many(self, lots: list[LotRecord]) -> list[LotRecord]:
        """One transaction: a batch is published whole or not at all."""
        with self._session_factory() as session:
            session.add_all([self._to_row(lot) for lot in lots])
            try:
                session.commit()
            except IntegrityError as exc:
                # The service pre-checks, so reaching here means two publishes
                # raced. Same domain error either way, never a 500.
                session.rollback()
                if "uq_lots_prediction_batch" not in str(exc.orig):
                    raise
                raise LotAlreadyPublished(lots[0].prediction_id) from exc
        return lots

    @staticmethod
    def _to_row(lot: LotRecord) -> Lot:
        return Lot(
            id=lot.id,
            prediction_id=lot.prediction_id,
            operator_id=lot.operator_id,
            species_id=lot.species_id,
            landing_point_id=lot.landing_point_id,
            quantity_kg=lot.quantity_kg,
            size_category=lot.size_category,
            starting_price_per_kg=lot.starting_price_per_kg,
            status=lot.status,
            auction_starts_at=lot.auction_starts_at,
            auction_ends_at=lot.auction_ends_at,
            knowledge_snapshot=lot.knowledge_snapshot,
            public_slug=lot.public_slug,
            allocated_buyer_id=lot.allocated_buyer_id,
            seller_fisher_group=lot.seller_fisher_group,
            batch_index=lot.batch_index,
            batch_size=lot.batch_size,
        )

    def get(self, lot_id: str) -> LotRecord | None:
        with self._session_factory() as session:
            return self._to_lot(session.get(Lot, lot_id))

    def get_by_slug(self, public_slug: str) -> LotRecord | None:
        with self._session_factory() as session:
            row = session.scalar(select(Lot).where(Lot.public_slug == public_slug))
            return self._to_lot(row)

    def get_by_prediction(self, prediction_id: str) -> LotRecord | None:
        with self._session_factory() as session:
            # The first lot of the batch: a catch publishes as lots 1..N.
            row = session.scalar(
                select(Lot).where(Lot.prediction_id == prediction_id).order_by(Lot.batch_index).limit(1)
            )
            return self._to_lot(row)

    def all(self) -> list[LotRecord]:
        with self._session_factory() as session:
            rows = session.scalars(select(Lot).order_by(Lot.created_at.desc())).all()
            return [self._to_lot(row) for row in rows]

    def close_expired(self, now: datetime | None = None) -> int:
        """Close every active lot whose auction has ended; returns how many.

        Nothing else would: there is no sweeper process, and a lot's status
        otherwise only changes inside place_bid and allocate. Readers call this
        first, so a lot past its end is never listed, searched or recommended
        as live. One indexed UPDATE, and a no-op when nothing has expired.
        """
        clock = now or datetime.now(timezone.utc)
        with self._session_factory() as session:
            result = session.execute(
                update(Lot)
                .where(Lot.status == "active", Lot.auction_ends_at <= clock)
                .values(status="closed")
            )
            session.commit()
            return result.rowcount or 0

    def highest(self, lot_id: str) -> Decimal | None:
        with self._session_factory() as session:
            return session.scalar(select(func.max(Bid.amount_per_kg)).where(Bid.lot_id == lot_id))

    def list_bids(self, lot_id: str) -> list[BidRecord]:
        with self._session_factory() as session:
            rows = session.scalars(
                select(Bid).where(Bid.lot_id == lot_id).order_by(Bid.created_at.desc())
            ).all()
            return [self._to_bid(row) for row in rows]

    def place_bid(
        self,
        lot_id: str,
        buyer_id: str,
        amount_per_kg: Decimal,
        now: datetime | None = None,
    ) -> BidRecord:
        clock = now or datetime.now(timezone.utc)
        with self._session_factory() as session:
            lot = session.execute(select(Lot).where(Lot.id == lot_id).with_for_update()).scalar_one_or_none()
            if lot is None:
                raise LotNotFound(lot_id)
            if lot.status != "active" or clock >= lot.auction_ends_at:
                if lot.status == "active":
                    lot.status = "closed"
                    session.commit()
                raise LotClosed(lot_id)
            highest = session.scalar(select(func.max(Bid.amount_per_kg)).where(Bid.lot_id == lot_id))
            if highest is not None:
                if amount_per_kg <= highest:
                    raise BidOutbid(Decimal(highest))
            elif amount_per_kg < lot.starting_price_per_kg:
                raise BidOutbid(Decimal(lot.starting_price_per_kg))
            bid = Bid(
                id=uuid4().hex,
                lot_id=lot_id,
                buyer_id=buyer_id,
                amount_per_kg=amount_per_kg,
            )
            session.add(bid)
            session.commit()
            session.refresh(bid)
            return self._to_bid(bid)

    def close(self, lot_id: str) -> LotRecord:
        with self._session_factory() as session:
            lot = session.execute(select(Lot).where(Lot.id == lot_id).with_for_update()).scalar_one_or_none()
            if lot is None:
                raise LotNotFound(lot_id)
            if lot.status == "allocated":
                raise LotNotAllocatable(lot_id, "allocated lot cannot be closed")
            if lot.status != "closed":
                lot.status = "closed"
                session.commit()
                session.refresh(lot)
            return self._to_lot(lot)

    def allocate(self, lot_id: str, now: datetime | None = None) -> LotRecord:
        clock = now or datetime.now(timezone.utc)
        with self._session_factory() as session:
            lot = session.execute(select(Lot).where(Lot.id == lot_id).with_for_update()).scalar_one_or_none()
            if lot is None:
                raise LotNotFound(lot_id)
            if lot.status == "allocated":
                return self._to_lot(lot)
            if lot.status == "active" and clock >= lot.auction_ends_at:
                lot.status = "closed"
            if lot.status != "closed":
                raise LotNotAllocatable(lot_id, "allocation requires a closed lot")
            winner = session.scalar(
                select(Bid).where(Bid.lot_id == lot_id).order_by(Bid.amount_per_kg.desc(), Bid.created_at.asc())
            )
            if winner is None:
                raise LotNotAllocatable(lot_id, "closed lot has no bids")
            lot.allocated_buyer_id = winner.buyer_id
            lot.status = "allocated"
            session.commit()
            session.refresh(lot)
            return self._to_lot(lot)

    @staticmethod
    def _to_lot(row: Lot | None) -> LotRecord | None:
        if row is None:
            return None
        return LotRecord(
            id=row.id,
            prediction_id=row.prediction_id,
            operator_id=row.operator_id,
            species_id=row.species_id,
            landing_point_id=row.landing_point_id,
            quantity_kg=Decimal(row.quantity_kg),
            size_category=row.size_category,
            starting_price_per_kg=Decimal(row.starting_price_per_kg),
            status=row.status,
            auction_starts_at=row.auction_starts_at,
            auction_ends_at=row.auction_ends_at,
            public_slug=row.public_slug,
            knowledge_snapshot=row.knowledge_snapshot,
            allocated_buyer_id=row.allocated_buyer_id,
            seller_fisher_group=row.seller_fisher_group,
            batch_index=row.batch_index,
            batch_size=row.batch_size,
        )

    @staticmethod
    def _to_bid(row: Bid) -> BidRecord:
        return BidRecord(
            id=row.id,
            lot_id=row.lot_id,
            buyer_id=row.buyer_id,
            amount_per_kg=Decimal(row.amount_per_kg),
            created_at=row.created_at,
        )


class SqlLandingPointRepository:
    def __init__(self, session_factory: Callable[[], Session]):
        self._session_factory = session_factory

    def get(self, landing_point_id: str) -> LandingPointRecord | None:
        with self._session_factory() as session:
            row = session.get(LandingPoint, landing_point_id)
            if row is None:
                return None
            return LandingPointRecord(
                id=row.id, name=row.name, latitude=row.latitude, longitude=row.longitude
            )

    def all(self) -> list[LandingPointRecord]:
        with self._session_factory() as session:
            rows = session.scalars(select(LandingPoint)).all()
            return [
                LandingPointRecord(id=row.id, name=row.name, latitude=row.latitude, longitude=row.longitude)
                for row in rows
            ]

    def upsert(self, point: LandingPointRecord) -> LandingPointRecord:
        with self._session_factory() as session:
            row = session.get(LandingPoint, point.id)
            if row is None:
                session.add(
                    LandingPoint(
                        id=point.id,
                        name=point.name,
                        latitude=point.latitude,
                        longitude=point.longitude,
                    )
                )
            else:
                row.name = point.name
                row.latitude = point.latitude
                row.longitude = point.longitude
            session.commit()
        return point

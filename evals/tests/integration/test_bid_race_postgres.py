"""Bid races against a real PostgreSQL: the lot row lock is what serialises
them, and an in-memory twin cannot show that. Skipped unless
FISHORA_DATABASE_URL is set in the environment; every row it writes is removed.
"""

import os
import threading
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, delete, select
from sqlalchemy.orm import sessionmaker

from apps.main_api.contracts import LotRecord
from apps.main_api.errors import BidOutbid

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.environ.get("FISHORA_DATABASE_URL"), reason="needs FISHORA_DATABASE_URL"),
]


@pytest.fixture
def lot():
    from apps.main_api.db.lot_repository import SqlLotRepository
    from apps.main_api.db.models import Bid, Lot, Prediction

    engine = create_engine(os.environ["FISHORA_DATABASE_URL"], pool_size=12)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    key = f"test_race_{uuid4().hex[:8]}"
    with factory() as session:
        session.add(Prediction(id=key, image_reference="test", predicted_species_id="species_nila",
                               confidence=1.0, top_candidates=[], model_version="test",
                               verification_status="confirmed", verified_species_id="species_nila"))
        session.commit()
    repo = SqlLotRepository(factory)
    now = datetime.now(timezone.utc)
    repo.create(LotRecord(
        id=key, prediction_id=key, operator_id="op_test", species_id="species_nila",
        landing_point_id="lp_muara_angke", quantity_kg=Decimal("10"), size_category="M",
        starting_price_per_kg=Decimal("20000"), status="active", auction_starts_at=now,
        auction_ends_at=now + timedelta(minutes=30), public_slug=key,
    ))
    try:
        yield repo, factory, key
    finally:
        with factory() as session:
            session.execute(delete(Bid).where(Bid.lot_id == key))
            session.execute(delete(Lot).where(Lot.id == key))
            session.execute(delete(Prediction).where(Prediction.id == key))
            session.commit()
        engine.dispose()


def _race(repo, key, amounts):
    barrier = threading.Barrier(len(amounts))
    accepted, refused = [], []

    def bid(index, amount):
        barrier.wait()
        try:
            accepted.append(repo.place_bid(key, f"buyer_{index}", Decimal(amount)))
        except BidOutbid:
            refused.append(amount)

    threads = [threading.Thread(target=bid, args=(i, amount)) for i, amount in enumerate(amounts)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return accepted, refused


def test_equal_simultaneous_bids_accept_exactly_one(lot):
    repo, _, key = lot
    accepted, refused = _race(repo, key, ["25000"] * 8)
    assert len(accepted) == 1 and len(refused) == 7
    assert [bid.amount_per_kg for bid in repo.list_bids(key)] == [Decimal("25000")]


def test_simultaneous_bids_leave_a_strictly_rising_history(lot):
    repo, _, key = lot
    _race(repo, key, [str(21000 + 500 * i) for i in range(10)])
    history = sorted(repo.list_bids(key), key=lambda bid: bid.created_at)
    amounts = [bid.amount_per_kg for bid in history]
    # Every accepted bid beat the one before it, so in time order they rise,
    # and the newest bid is the highest: the winner the lot allocates to.
    assert amounts == sorted(set(amounts))
    assert repo.highest(key) == amounts[-1]


def test_a_bid_is_dated_when_it_takes_the_lock_not_when_it_started_waiting(lot):
    from apps.main_api.db.models import Lot

    repo, factory, key = lot
    holder = factory()
    holder.execute(select(Lot).where(Lot.id == key).with_for_update())
    placed = []
    waiter = threading.Thread(target=lambda: placed.append(repo.place_bid(key, "buyer_a", Decimal("20000"))))
    waiter.start()
    time.sleep(1.0)
    released_at = datetime.now(timezone.utc)
    holder.rollback()
    holder.close()
    waiter.join()
    # Dated at its transaction's start, it would sit before a bid that beat it
    # to the lock (and bid lower) while it waited.
    assert placed[0].created_at >= released_at - timedelta(milliseconds=50)

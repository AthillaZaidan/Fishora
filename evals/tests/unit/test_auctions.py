"""Auction lifecycle as the API serves it: an ended auction reads as closed
everywhere without a sweeper, bidders are labelled rather than identified,
allocation reports the real winner, a buyer reads only their own wins, and a
lot serves its catch photo."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from apps.main_api.contracts import BidRecord, LandingPointRecord, LotRecord
from apps.main_api.errors import (
    BidOutbid,
    InvalidLot,
    LotClosed,
    LotNotAllocatable,
    LotNotFound,
)
from apps.main_api.services.landing_points import DEMO_LANDING_POINTS
from apps.main_api.services.lots import LotService, bidder_labels, winning_bid
from evals.fakes import InMemoryPredictionRepository

pytestmark = pytest.mark.unit

T0 = datetime(2026, 9, 26, 6, 0, tzinfo=timezone.utc)


class Lots:
    """In-memory twin of SqlLotRepository: the same status rules, no database."""

    def __init__(self):
        self.rows: dict[str, LotRecord] = {}
        self.bids: list[BidRecord] = []
        self.tick = 0

    def create_many(self, lots):
        self.rows.update({lot.id: lot for lot in lots})
        return lots

    def get(self, lot_id):
        return self.rows.get(lot_id)

    def get_by_slug(self, slug):
        return next((lot for lot in self.rows.values() if lot.public_slug == slug), None)

    def get_by_prediction(self, prediction_id):
        return next((lot for lot in self.rows.values() if lot.prediction_id == prediction_id), None)

    def all(self):
        return list(self.rows.values())

    def close_expired(self, now=None):
        clock = now or datetime.now(timezone.utc)
        expired = [lot for lot in self.rows.values() if lot.status == "active" and lot.auction_ends_at <= clock]
        for lot in expired:
            self.rows[lot.id] = replace(lot, status="closed")
        return len(expired)

    def highest(self, lot_id):
        amounts = [bid.amount_per_kg for bid in self.bids if bid.lot_id == lot_id]
        return max(amounts) if amounts else None

    def list_bids(self, lot_id):
        return sorted((b for b in self.bids if b.lot_id == lot_id), key=lambda b: b.created_at, reverse=True)

    def place_bid(self, lot_id, buyer_id, amount_per_kg, now=None):
        clock = now or datetime.now(timezone.utc)
        lot = self.rows.get(lot_id)
        if lot is None:
            raise LotNotFound(lot_id)
        if lot.status != "active" or clock >= lot.auction_ends_at:
            if lot.status == "active":
                self.rows[lot_id] = replace(lot, status="closed")
            raise LotClosed(lot_id)
        highest = self.highest(lot_id)
        if highest is not None and amount_per_kg <= highest:
            raise BidOutbid(highest)
        if highest is None and amount_per_kg < lot.starting_price_per_kg:
            raise BidOutbid(lot.starting_price_per_kg, "bid must be at least the starting price")
        self.tick += 1
        bid = BidRecord(id=uuid4().hex, lot_id=lot_id, buyer_id=buyer_id, amount_per_kg=amount_per_kg,
                        created_at=T0 + timedelta(seconds=self.tick))
        self.bids.append(bid)
        return bid

    def close(self, lot_id):
        lot = self.rows[lot_id]
        if lot.status == "allocated":
            raise LotNotAllocatable(lot_id, "allocated lot cannot be closed")
        self.rows[lot_id] = replace(lot, status="closed")
        return self.rows[lot_id]

    def allocate(self, lot_id, now=None):
        clock = now or datetime.now(timezone.utc)
        lot = self.rows[lot_id]
        if lot.status == "allocated":
            return lot
        if lot.status == "active" and clock >= lot.auction_ends_at:
            lot = replace(lot, status="closed")
        if lot.status != "closed":
            raise LotNotAllocatable(lot_id, "allocation requires a closed lot")
        winner = winning_bid([bid for bid in self.bids if bid.lot_id == lot_id])
        if winner is None:
            raise LotNotAllocatable(lot_id, "closed lot has no bids")
        self.rows[lot_id] = replace(lot, status="allocated", allocated_buyer_id=winner.buyer_id)
        return self.rows[lot_id]


class LandingPoints:
    def __init__(self):
        self.rows = {point.id: point for point in DEMO_LANDING_POINTS}

    def get(self, landing_point_id):
        return self.rows.get(landing_point_id)

    def all(self):
        return list(self.rows.values())


def _lot(lot_id: str, *, ends_in: timedelta, status: str = "active", operator: str = "op_rian", **fields):
    now = datetime.now(timezone.utc)
    return LotRecord(
        id=lot_id, prediction_id=f"p_{lot_id}", operator_id=operator, species_id="species_nila",
        landing_point_id="lp_muara_angke", quantity_kg=Decimal("10"), size_category="M",
        starting_price_per_kg=Decimal("20000"), status=status, auction_starts_at=now - timedelta(hours=1),
        auction_ends_at=now + ends_in, public_slug=f"nila-{lot_id}", **fields,
    )


def _service(lots=None, **kwargs):
    return LotService(InMemoryPredictionRepository(), lots or Lots(), landing_point_repo=LandingPoints(), **kwargs)


# --- An ended auction is closed wherever it is read -------------------------------


def test_an_ended_auction_is_not_listed_as_active_and_its_close_is_persisted():
    lots = Lots()
    lots.create_many([_lot("live", ends_in=timedelta(minutes=5)), _lot("ended", ends_in=-timedelta(seconds=1))])
    service = _service(lots)
    assert [lot.id for lot in service.list_lots(status="active")] == ["live"]
    assert [lot.id for lot in service.list_lots(status="closed")] == ["ended"]
    assert lots.rows["ended"].status == "closed"


def test_get_and_slug_reads_report_an_ended_auction_as_closed():
    lots = Lots()
    lots.create_many([_lot("ended", ends_in=-timedelta(minutes=1))])
    service = _service(lots)
    assert service.get("ended").status == "closed"
    assert service.get_by_slug("nila-ended").status == "closed"


def test_a_repository_without_close_expired_still_reads_closed():
    class Legacy(Lots):
        close_expired = None

    lots = Legacy()
    lots.create_many([_lot("ended", ends_in=-timedelta(minutes=1))])
    service = _service(lots)
    assert service.get("ended").status == "closed"
    assert service.list_lots(status="active") == []


def test_a_bid_after_the_end_is_refused():
    lots = Lots()
    lots.create_many([_lot("ended", ends_in=-timedelta(seconds=1))])
    with pytest.raises(LotClosed):
        _service(lots).place_bid("ended", "buyer_dewi", Decimal("25000"))


def test_an_ended_auction_can_be_allocated_to_its_highest_bidder():
    lots = Lots()
    lots.create_many([_lot("a", ends_in=timedelta(minutes=5))])
    service = _service(lots)
    service.place_bid("a", "buyer_dewi", Decimal("21000"))
    service.place_bid("a", "buyer_budi", Decimal("22000"))
    lots.rows["a"] = replace(lots.rows["a"], auction_ends_at=datetime.now(timezone.utc) - timedelta(seconds=1))
    assert service.allocate("a").allocated_buyer_id == "buyer_budi"


# --- Publication checks the landing point ------------------------------------------


def _verified_predictions():
    predictions = InMemoryPredictionRepository()
    predictions.create("p1", "images/p1.jpg", "species_nila", 0.9, [], "v")
    predictions.verify("p1", "species_nila", "confirmed")
    return predictions


def _publish(landing_point_id):
    service = LotService(_verified_predictions(), Lots(), landing_point_repo=LandingPoints())
    return service.publish(prediction_id="p1", operator_id="op_rian", quantity_kg=Decimal("10"),
                           starting_price_per_kg=Decimal("20000"), size_category="M",
                           landing_point_id=landing_point_id)


def test_publish_refuses_an_unknown_landing_point():
    with pytest.raises(InvalidLot):
        _publish("lp_nowhere")


def test_publish_accepts_a_seeded_landing_point():
    [lot] = _publish("lp_cilacap")
    assert lot.landing_point_id == "lp_cilacap"


# --- Labels and winners ---------------------------------------------------------


def _bid(buyer, amount, second):
    return BidRecord(id=f"b{second}", lot_id="a", buyer_id=buyer, amount_per_kg=Decimal(amount),
                     created_at=T0 + timedelta(seconds=second))


def test_bidders_are_numbered_by_their_first_bid():
    bids = [_bid("buyer_b", "22000", 2), _bid("buyer_a", "21000", 1), _bid("buyer_a", "23000", 3)]
    assert bidder_labels(bids) == {"buyer_a": "Bidder 1", "buyer_b": "Bidder 2"}


def test_the_winning_bid_is_the_highest_and_the_earliest_on_a_tie():
    bids = [_bid("buyer_b", "22000", 2), _bid("buyer_a", "22000", 1), _bid("buyer_c", "21000", 3)]
    assert winning_bid(bids).buyer_id == "buyer_a"
    assert winning_bid([]) is None


# --- Through the API -------------------------------------------------------------


@pytest.fixture
def api(tmp_path):
    from apps.main_api.api.lots import landing_router
    from apps.main_api.main import create_main_app
    from apps.main_api.ports import AppDependencies
    from apps.main_api.services.image_store import FilesystemImageStore
    from apps.main_api.services.session import COOKIE_NAME, SessionUser
    from evals.fakes import FixedCVClient, InMemorySpeciesRepository

    lots = Lots()
    deps = AppDependencies(
        cv_client=FixedCVClient("nila"),
        species_repo=InMemorySpeciesRepository([]),
        prediction_repo=_verified_predictions(),
        image_store=FilesystemImageStore(tmp_path),
        embedder=object(),  # never used: no knowledge repo, so no card is made
        lot_repo=lots,
        landing_point_repo=LandingPoints(),
    )
    app = create_main_app(deps=deps)
    # main.py registers this router; the test does not depend on that line.
    app.include_router(landing_router)

    def client(user_id: str | None = None, role: str = "buyer") -> TestClient:
        test_client = TestClient(app)
        if user_id is not None:
            token = deps.session_service.dump(SessionUser(id=user_id, role=role, name=user_id, username=user_id))
            test_client.cookies.set(COOKIE_NAME, token)
        return test_client

    return lots, deps, client


def test_an_ended_auction_reads_closed_through_list_get_and_search(api):
    lots, _, client = api
    lots.create_many([_lot("ended", ends_in=-timedelta(seconds=1)), _lot("live", ends_in=timedelta(minutes=5))])
    guest = client()
    assert [lot["id"] for lot in guest.get("/api/v1/lots", params={"status": "active"}).json()] == ["live"]
    assert guest.get("/api/v1/lots/ended").json()["status"] == "closed"
    assert [lot["id"] for lot in guest.get("/api/v1/lots/search").json()["matches"]] == ["live"]


def test_a_bid_after_the_end_is_a_409(api):
    lots, _, client = api
    lots.create_many([_lot("ended", ends_in=-timedelta(seconds=1))])
    response = client("buyer_dewi").post("/api/v1/lots/ended/bids", json={"amount_per_kg": "30000"})
    assert response.status_code == 409


def test_a_bid_under_the_floor_is_a_409_naming_the_floor(api):
    lots, _, client = api
    lots.create_many([_lot("a", ends_in=timedelta(minutes=5))])
    dewi = client("buyer_dewi")
    under = dewi.post("/api/v1/lots/a/bids", json={"amount_per_kg": "19999"})
    assert under.status_code == 409
    assert under.json() == {"detail": "bid must be at least the starting price", "current_highest_per_kg": "20000"}
    assert dewi.post("/api/v1/lots/a/bids", json={"amount_per_kg": "20000"}).status_code == 200
    tie = client("buyer_budi").post("/api/v1/lots/a/bids", json={"amount_per_kg": "20000"})
    assert tie.status_code == 409 and tie.json()["detail"] == "bid must exceed current highest"


@pytest.mark.parametrize("amount", ["20000.001", "99999999999999"])
def test_a_bid_the_price_column_cannot_hold_is_a_422(api, amount):
    # NUMERIC(12, 2): a third decimal would round onto a tie with the bid it
    # beat, and an oversized amount overflowed into a 500.
    lots, _, client = api
    lots.create_many([_lot("a", ends_in=timedelta(minutes=5))])
    assert client("buyer_dewi").post("/api/v1/lots/a/bids", json={"amount_per_kg": amount}).status_code == 422
    assert lots.bids == []


@pytest.mark.parametrize(
    "field,value",
    [("quantity_kg", "1.0005"), ("quantity_kg", "9999999999999"), ("starting_price_per_kg", "99999999999999")],
)
def test_a_lot_the_numeric_columns_cannot_hold_is_a_422(api, field, value):
    lots, _, client = api
    body = {"prediction_id": "p1", "quantity_kg": "10", "starting_price_per_kg": "20000",
            "size_category": "M", "landing_point_id": "lp_karangsong", field: value}
    assert client("op_rian", "operator").post("/api/v1/lots", json=body).status_code == 422
    assert lots.rows == {}


def test_bid_history_shows_labels_not_buyer_ids(api):
    lots, _, client = api
    lots.create_many([_lot("a", ends_in=timedelta(minutes=5))])
    dewi, budi = client("buyer_dewi"), client("buyer_budi")
    placed = dewi.post("/api/v1/lots/a/bids", json={"amount_per_kg": "21000"}).json()
    assert placed["bidder"] == "You" and placed["is_you"] is True
    budi.post("/api/v1/lots/a/bids", json={"amount_per_kg": "22000"})

    public = client().get("/api/v1/lots/a/bids").json()
    assert [(bid["bidder"], bid["buyer_id"]) for bid in public] == [("Bidder 2", None), ("Bidder 1", None)]
    assert "buyer_dewi" not in str(public) and "buyer_budi" not in str(public)

    as_dewi = dewi.get("/api/v1/lots/a/bids").json()
    assert [(bid["bidder"], bid["is_you"]) for bid in as_dewi] == [("Bidder 2", False), ("You", True)]
    assert [bid["buyer_id"] for bid in as_dewi] == [None, "buyer_dewi"]

    as_operator = client("op_rian", "operator").get("/api/v1/lots/a/bids").json()
    assert [bid["buyer_id"] for bid in as_operator] == ["buyer_budi", "buyer_dewi"]


def test_allocation_reports_the_real_winner_and_hides_it_from_others(api):
    lots, _, client = api
    lots.create_many([_lot("a", ends_in=timedelta(minutes=5))])
    client("buyer_dewi").post("/api/v1/lots/a/bids", json={"amount_per_kg": "21000"})
    client("buyer_budi").post("/api/v1/lots/a/bids", json={"amount_per_kg": "22500"})
    operator = client("op_rian", "operator")
    assert operator.post("/api/v1/lots/a/close").json()["status"] == "closed"

    allocated = operator.post("/api/v1/lots/a/allocate").json()
    assert allocated["allocated_buyer_id"] == "buyer_budi"
    assert allocated["winner_label"] == "Bidder 2"
    assert Decimal(allocated["winning_amount_per_kg"]) == Decimal("22500")

    assert client().get("/api/v1/lots/a").json()["allocated_buyer_id"] is None
    loser = client("buyer_dewi").get("/api/v1/lots/a").json()
    assert loser["allocated_buyer_id"] is None and loser["won_by_you"] is False
    winner = client("buyer_budi").get("/api/v1/lots/a").json()
    assert winner["allocated_buyer_id"] == "buyer_budi" and winner["won_by_you"] is True
    assert operator.get("/api/v1/lots/a").json()["allocated_buyer_id"] == "buyer_budi"


def test_won_lots_are_scoped_to_the_signed_in_buyer(api):
    lots, _, client = api
    lots.create_many([
        _lot("mine", ends_in=-timedelta(minutes=1), status="allocated", allocated_buyer_id="buyer_dewi"),
        _lot("theirs", ends_in=-timedelta(minutes=1), status="allocated", allocated_buyer_id="buyer_budi"),
        _lot("open", ends_in=timedelta(minutes=5)),
    ])
    won = client("buyer_dewi").get("/api/v1/lots", params={"won": 1}).json()
    assert [lot["id"] for lot in won] == ["mine"] and won[0]["won_by_you"] is True
    assert client().get("/api/v1/lots", params={"won": 1}).status_code == 401
    assert client("op_rian", "operator").get("/api/v1/lots", params={"won": 1}).status_code == 403


def test_a_lot_serves_its_catch_photo(api, tmp_path):
    lots, deps, client = api
    lots.create_many([_lot("pic", ends_in=timedelta(minutes=5)), _lot("bare", ends_in=timedelta(minutes=5))])
    deps.image_store.save("p_pic", b"\x89PNG fake", "image/png")
    guest = client()
    assert guest.get("/api/v1/lots/pic").json()["photo_url"] == "/api/v1/lots/pic/photo"
    photo = guest.get("/api/v1/lots/pic/photo")
    assert photo.status_code == 200 and photo.content == b"\x89PNG fake"
    assert photo.headers["content-type"] == "image/png"
    assert guest.get("/api/v1/lots/bare").json()["photo_url"] is None
    assert guest.get("/api/v1/lots/bare/photo").status_code == 404
    assert guest.get("/api/v1/lots/nope/photo").status_code == 404


def test_landing_points_are_listed_and_checked_on_publish(api):
    _, _, client = api
    points = client().get("/api/v1/landing-points").json()
    assert [point["id"] for point in points] == [point.id for point in DEMO_LANDING_POINTS]
    operator = client("op_rian", "operator")
    body = {"prediction_id": "p1", "quantity_kg": "10", "starting_price_per_kg": "20000",
            "size_category": "M", "landing_point_id": "lp_nowhere"}
    assert operator.post("/api/v1/lots", json=body).status_code == 422
    published = operator.post("/api/v1/lots", json={**body, "landing_point_id": "lp_karangsong"})
    assert published.status_code == 200 and published.json()[0]["landing_point_id"] == "lp_karangsong"


def test_the_image_store_only_finds_names_it_writes(tmp_path):
    from apps.main_api.services.image_store import FilesystemImageStore

    store = FilesystemImageStore(tmp_path / "images")
    (tmp_path / "secret.png").write_bytes(b"x")
    assert store.find("../secret") is None
    store.save("p1", b"jpeg", "image/jpeg")
    path, content_type = store.find("p1")
    assert path.read_bytes() == b"jpeg" and content_type == "image/jpeg"

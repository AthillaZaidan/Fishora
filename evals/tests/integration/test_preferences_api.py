"""Buyer preferences API: read back, role checks, and recommendations that follow a save.

Builds the app directly rather than through ``app_factory``: nothing here needs
the embedding model, and the lot, landing point and preference ports are ones
that fixture does not provide.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from apps.main_api.contracts import LotRecord
from apps.main_api.services.landing_points import DEMO_LANDING_POINTS
from apps.main_api.services.session import COOKIE_NAME, SessionService, SessionUser

NOW = datetime.now(timezone.utc)

TENGGIRI_CARD = {
    "taste": "Savory and not too oily.",
    "texture": "Firm with fine fibers and white flesh.",
    "processing_methods": ["Fillet", "Smoked", "Fish balls"],
    "commercial_uses": ["Restaurants", "Fish ball processing", "Catering"],
    "potential_buyer_segments": ["Fish ball processors", "Seafood restaurants"],
}
MUJAIR_CARD = {
    "taste": "Mild with a slightly earthy taste.",
    "texture": "Firm and fibrous.",
    "processing_methods": ["Fried", "Grilled"],
    "commercial_uses": ["Wet markets", "Catering"],
    "potential_buyer_segments": ["Market traders", "Caterers"],
}


class InMemoryPreferenceRepository:
    def __init__(self):
        self._rows = {}

    def get(self, buyer_id):
        return self._rows.get(buyer_id)

    def upsert(self, record):
        self._rows[record.buyer_id] = record
        return record


class InMemoryLotRepository:
    def __init__(self, lots):
        self._lots = {lot.id: lot for lot in lots}

    def all(self):
        return list(self._lots.values())

    def get(self, lot_id):
        return self._lots.get(lot_id)

    def highest(self, lot_id):
        return None


class InMemoryLandingPointRepository:
    def __init__(self, points):
        self._points = {point.id: point for point in points}

    def get(self, point_id):
        return self._points.get(point_id)

    def all(self):
        return list(self._points.values())


def _lot(lot_id, card, landing, *, price="68000", status="active"):
    return LotRecord(
        id=lot_id, prediction_id=f"p_{lot_id}", operator_id="op_rian", species_id="species_tenggiri",
        landing_point_id=landing, quantity_kg=Decimal("120"), size_category="L",
        starting_price_per_kg=Decimal(price), status=status, auction_starts_at=NOW,
        auction_ends_at=NOW + timedelta(hours=2), public_slug=lot_id, knowledge_snapshot=card,
    )


LOTS = [
    _lot("jakarta_tenggiri", TENGGIRI_CARD, "lp_muara_angke"),
    _lot("cilacap_mujair", MUJAIR_CARD, "lp_cilacap", price="26000"),
    _lot("cilacap_closed", MUJAIR_CARD, "lp_cilacap", status="allocated"),
]

DEWI = SessionUser(id="buyer_dewi", role="buyer", name="Dewi", username="dewi")
OTHER_BUYER = SessionUser(id="buyer_other", role="buyer", name="Other", username="other")
RIAN = SessionUser(id="op_rian", role="operator", name="Rian", username="rian")

JAKARTA = {"latitude": -6.2, "longitude": 106.85}
CILACAP = {"latitude": -7.72, "longitude": 109.0}


def _profile(**overrides):
    body = {
        "business_type": "restaurant",
        "intended_uses": ["fillet", "smoked"],
        "characteristics": ["savory", "firm"],
        "max_price_per_kg": "70000",
        "min_quantity_kg": "50",
        **JAKARTA,
    }
    body.update(overrides)
    return body


@pytest.fixture
def app():
    from apps.main_api.main import create_main_app
    from apps.main_api.ports import AppDependencies

    stub = object()
    deps = AppDependencies(
        # The five ports that mark the bundle complete, so the lifespan never
        # reaches for Postgres. None of them is exercised by these routes.
        cv_client=stub, species_repo=stub, prediction_repo=stub, image_store=stub, embedder=stub,
        lot_repo=InMemoryLotRepository(LOTS),
        landing_point_repo=InMemoryLandingPointRepository(DEMO_LANDING_POINTS),
        preference_repo=InMemoryPreferenceRepository(),
        session_service=SessionService(),
    )
    return create_main_app(deps=deps)


def _client(app, user: SessionUser | None):
    client = TestClient(app)
    if user is not None:
        client.cookies.set(COOKIE_NAME, app.state.deps.session_service.dump(user))
    return client


def test_get_before_saving_is_404(app):
    with _client(app, DEWI) as client:
        assert client.get("/api/v1/buyers/buyer_dewi/preferences").status_code == 404


def test_put_then_get_round_trips_the_profile(app):
    with _client(app, DEWI) as client:
        saved = client.put("/api/v1/buyers/buyer_dewi/preferences", json=_profile())
        assert saved.status_code == 200, saved.text
        body = client.get("/api/v1/buyers/buyer_dewi/preferences").json()
    assert body["business_type"] == "restaurant"
    assert body["intended_uses"] == ["fillet", "smoked"]
    assert body["characteristics"] == ["savory", "firm"]
    assert Decimal(body["max_price_per_kg"]) == Decimal("70000")
    assert Decimal(body["min_quantity_kg"]) == Decimal("50")
    assert (body["latitude"], body["longitude"]) == (JAKARTA["latitude"], JAKARTA["longitude"])


def test_put_normalises_legacy_business_type_and_duplicate_choices(app):
    with _client(app, DEWI) as client:
        body = client.put("/api/v1/buyers/buyer_dewi/preferences", json=_profile(
            business_type="rumah_makan", intended_uses=["fillet", " Fillet ", ""],
        )).json()
    assert body["business_type"] == "restaurant"
    assert body["intended_uses"] == ["fillet"]


@pytest.mark.parametrize("field,value", [
    ("max_price_per_kg", "0"), ("latitude", 120), ("business_type", ""),
])
def test_put_rejects_invalid_values(app, field, value):
    with _client(app, DEWI) as client:
        response = client.put("/api/v1/buyers/buyer_dewi/preferences", json=_profile(**{field: value}))
    assert response.status_code == 422


@pytest.mark.parametrize("method,path", [
    ("get", "/api/v1/buyers/buyer_dewi/preferences"),
    ("put", "/api/v1/buyers/buyer_dewi/preferences"),
    ("post", "/api/v1/buyers/buyer_dewi/preferences/preview"),
    ("get", "/api/v1/buyers/buyer_dewi/recommendations"),
])
def test_signed_out_is_401(app, method, path):
    with _client(app, None) as client:
        kwargs = {"json": _profile()} if method in {"put", "post"} else {}
        assert getattr(client, method)(path, **kwargs).status_code == 401


@pytest.mark.parametrize("user", [RIAN, OTHER_BUYER], ids=["operator", "other-buyer"])
@pytest.mark.parametrize("method,path", [
    ("get", "/api/v1/buyers/buyer_dewi/preferences"),
    ("put", "/api/v1/buyers/buyer_dewi/preferences"),
    ("post", "/api/v1/buyers/buyer_dewi/preferences/preview"),
    ("get", "/api/v1/buyers/buyer_dewi/recommendations"),
])
def test_only_the_buyer_themself_may_touch_the_profile(app, user, method, path):
    with _client(app, user) as client:
        kwargs = {"json": _profile()} if method in {"put", "post"} else {}
        assert getattr(client, method)(path, **kwargs).status_code == 403


def test_operator_cannot_write_preferences_under_their_own_id(app):
    with _client(app, RIAN) as client:
        response = client.put("/api/v1/buyers/op_rian/preferences", json=_profile())
    assert response.status_code == 403


def test_recommendations_follow_the_saved_location(app):
    with _client(app, DEWI) as client:
        assert client.get("/api/v1/buyers/buyer_dewi/recommendations").json()["profile_missing"] is True

        client.put("/api/v1/buyers/buyer_dewi/preferences", json=_profile())
        in_jakarta = client.get("/api/v1/buyers/buyer_dewi/recommendations").json()
        assert [item["lot"]["id"] for item in in_jakarta["items"]] == ["jakarta_tenggiri"]
        top = in_jakarta["items"][0]
        assert top["matched"] is True and top["score"] >= in_jakarta["match_threshold"]
        assert {reason["criterion"] for reason in top["reasons"]} >= {"business_type", "distance"}

        # Moving to Cilacap brings its active lot in range; the allocated one stays out.
        client.put("/api/v1/buyers/buyer_dewi/preferences", json=_profile(
            intended_uses=["fried"], characteristics=["mild"], business_type="catering", **CILACAP,
        ))
        in_cilacap = client.get("/api/v1/buyers/buyer_dewi/recommendations").json()
    assert [item["lot"]["id"] for item in in_cilacap["items"]] == ["cilacap_mujair"]
    assert in_cilacap["items"][0]["matched"] is True


def test_preview_counts_what_saving_would_recommend(app):
    draft = _profile(characteristics=["oily"], intended_uses=["grilled"])
    with _client(app, DEWI) as client:
        preview = client.post("/api/v1/buyers/buyer_dewi/preferences/preview", json=draft).json()
        # Nothing was stored by previewing.
        assert client.get("/api/v1/buyers/buyer_dewi/preferences").status_code == 404
        client.put("/api/v1/buyers/buyer_dewi/preferences", json=draft)
        saved = client.get("/api/v1/buyers/buyer_dewi/recommendations").json()
    assert preview["in_range"] == len(saved["items"]) == 1
    assert preview["matched"] == sum(item["matched"] for item in saved["items"])
    # "not too oily" on the card is not a match for a buyer who wants oily fish.
    assert preview["matched"] == 0
    assert preview["nearest_landing_point"]["id"] == "lp_muara_angke"


def test_preview_reports_nearest_landing_point_when_none_is_in_range(app):
    bandung = _profile(latitude=-6.9175, longitude=107.6191)
    with _client(app, DEWI) as client:
        body = client.post("/api/v1/buyers/buyer_dewi/preferences/preview", json=bandung).json()
    assert body["in_range"] == 0 and body["matched"] == 0
    assert body["nearest_landing_point"]["distance_km"] > body["radius_km"]


def test_reason_details_are_english(app):
    with _client(app, DEWI) as client:
        client.put("/api/v1/buyers/buyer_dewi/preferences", json=_profile())
        items = client.get("/api/v1/buyers/buyer_dewi/recommendations").json()["items"]
    text = " ".join(reason["detail"] for reason in items[0]["reasons"])
    assert "Suits how you use fish" in text and "cocok" not in text


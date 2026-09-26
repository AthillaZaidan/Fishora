"""Accounts and sessions: password hashing, signed expiring tokens, the
register/login/logout API, and the operator guard on the fish, jobs and
quality routes."""

import hashlib
import hmac
import json
from base64 import urlsafe_b64encode

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from apps.main_api.config import MainSettings
from apps.main_api.db.models import User
from apps.main_api.db.user_repository import InMemoryUserRepository, SqlUserRepository
from apps.main_api.main import create_main_app
from apps.main_api.ports import AppDependencies
from apps.main_api.services.session import (
    COOKIE_NAME,
    SESSION_TTL_SECONDS,
    SessionService,
    SessionUser,
    UserRecord,
    UsernameTaken,
    hash_password,
    seed_demo_users,
    verify_password,
)
from evals.corpus import species_records
from evals.fakes import FixedCVClient, InMemoryImageStore, InMemoryPredictionRepository, InMemorySpeciesRepository
from evals.tests.conftest import png_bytes, sign_in

pytestmark = pytest.mark.unit

RIAN = SessionUser(id="op_rian", role="operator", name="Rian Setiawan", username="rian")


def _app(settings=None):
    """The in-memory app with no database and no model: accounts live in memory."""
    deps = AppDependencies(cv_client=FixedCVClient("nila"), species_repo=InMemorySpeciesRepository(species_records()),
                           prediction_repo=InMemoryPredictionRepository(), image_store=InMemoryImageStore(),
                           embedder=type("E5Stub", (), {"model_name": "intfloat/multilingual-e5-base"})())
    return create_main_app(settings=settings, deps=deps)


def _register(client, **overrides):
    body = {"username": "sari_01", "password": "correct horse", "display_name": "Sari Wulandari", "role": "buyer"}
    return client.post("/api/v1/auth/register", json={**body, **overrides})


# Hashing


def test_a_password_hash_verifies_only_its_own_password():
    stored = hash_password("correct horse")
    assert stored.startswith("scrypt$") and "correct horse" not in stored
    assert verify_password("correct horse", stored)
    assert not verify_password("correct hors", stored)


def test_each_hash_has_its_own_salt():
    assert hash_password("same") != hash_password("same")


@pytest.mark.parametrize("stored", ["", "plain", "md5$x$y", "scrypt$1$2$3$!!$??", "scrypt$a$b$c$d$e"])
def test_a_malformed_hash_never_verifies(stored):
    assert not verify_password("anything", stored)


# Tokens


def test_a_token_expires_after_its_ttl():
    now = [1_000_000.0]
    sessions = SessionService("secret", InMemoryUserRepository(), clock=lambda: now[0])
    token = sessions.dump(RIAN)
    now[0] += SESSION_TTL_SECONDS - 1
    assert sessions.load(token) == RIAN
    now[0] += 1
    assert sessions.load(token) is None


def test_a_token_from_another_secret_or_tampered_is_refused():
    token = SessionService("one", InMemoryUserRepository()).dump(RIAN)
    assert SessionService("two", InMemoryUserRepository()).load(token) is None
    payload, signature = token.rsplit(".", 1)
    assert SessionService("one", InMemoryUserRepository()).load(f"{payload}x.{signature}") is None


def test_the_old_public_secret_signs_nothing():
    """A token forged with the constant the repository used to ship is refused
    when no secret is configured, and so is one with no expiry."""
    payload = urlsafe_b64encode(json.dumps({"id": "op_rian", "role": "operator", "name": "x",
                                            "username": "rian"}).encode())
    forged = f"{payload.decode()}.{hmac.new(b'fishora-dev-session', payload, hashlib.sha256).hexdigest()}"
    assert SessionService(users=InMemoryUserRepository()).load(forged) is None


def test_services_without_a_secret_share_the_process_secret():
    token = SessionService(users=InMemoryUserRepository()).dump(RIAN)
    assert SessionService(users=InMemoryUserRepository()).load(token) == RIAN


# Repositories


def test_the_sql_repository_refuses_a_duplicate_username_and_seeds_once():
    engine = create_engine("sqlite://")
    User.__table__.create(engine)
    users = SqlUserRepository(sessionmaker(bind=engine, expire_on_commit=False))
    seed_demo_users(users)
    seed_demo_users(users)  # idempotent
    rian = users.get_by_username("rian")
    assert rian.id == "op_rian" and rian.role == "operator" and rian.display_name == "Rian Setiawan"
    assert users.get_by_username("dewi").id == "buyer_dewi"
    assert verify_password("demo", rian.password_hash)
    with pytest.raises(UsernameTaken):
        users.create(UserRecord(id="op_other", username="rian", display_name="Other", role="operator",
                                password_hash=hash_password("whatever1")))


# API


def test_register_signs_the_new_account_in():
    with TestClient(_app()) as client:
        response = _register(client)
        assert response.status_code == 201, response.text
        body = response.json()
        assert body["username"] == "sari_01" and body["role"] == "buyer" and body["name"] == "Sari Wulandari"
        assert body["id"].startswith("buyer_")
        cookie = response.headers["set-cookie"]
        assert f"Max-Age={SESSION_TTL_SECONDS}" in cookie and "HttpOnly" in cookie and "Secure" not in cookie
        assert client.get("/api/v1/auth/me").json() == body


def test_a_registered_account_can_sign_in_again():
    app = _app()
    with TestClient(app) as client:
        created = _register(client, username="Budi_Fisher", role="operator").json()
    with TestClient(app) as client:
        assert created["username"] == "budi_fisher"  # stored lower-case
        response = client.post("/api/v1/auth/login", json={"username": "BUDI_FISHER", "password": "correct horse"})
        assert response.status_code == 200 and response.json() == created


def test_a_taken_username_is_409():
    with TestClient(_app()) as client:
        assert _register(client).status_code == 201
        assert _register(client, display_name="Someone Else").status_code == 409
        assert _register(client, username="rian").status_code == 409  # the seeded demo account


@pytest.mark.parametrize("overrides", [
    {"username": "ab"},
    {"username": "x" * 33},
    {"username": "has space"},
    {"username": "dash-ed"},
    {"password": "short"},
    {"role": "admin"},
    {"display_name": "   "},
])
def test_register_validates_its_fields(overrides):
    with TestClient(_app()) as client:
        assert _register(client, **overrides).status_code == 422


def test_the_demo_accounts_sign_in():
    with TestClient(_app()) as client:
        assert sign_in(client, "rian").get("/api/v1/auth/me").json()["id"] == "op_rian"
        assert sign_in(client, "dewi").get("/api/v1/auth/me").json()["id"] == "buyer_dewi"


def test_wrong_credentials_get_one_generic_401():
    with TestClient(_app()) as client:
        wrong = client.post("/api/v1/auth/login", json={"username": "rian", "password": "nope"})
        unknown = client.post("/api/v1/auth/login", json={"username": "nobody", "password": "demo"})
        assert wrong.status_code == unknown.status_code == 401
        assert wrong.json() == unknown.json() == {"detail": "invalid username or password"}
        assert "set-cookie" not in wrong.headers


def test_logout_ends_the_session():
    with TestClient(_app()) as client:
        sign_in(client)
        response = client.post("/api/v1/auth/logout")
        assert response.status_code == 200 and f'{COOKIE_NAME}=""' in response.headers["set-cookie"]
        assert client.get("/api/v1/auth/me").status_code == 401


def test_the_secure_cookie_setting_reaches_the_cookie():
    settings = MainSettings(_env_file=None, database_url="postgresql+psycopg://t@localhost/t",
                            session_secret="s" * 32, session_cookie_secure=True)
    with TestClient(_app(settings)) as client:
        response = client.post("/api/v1/auth/login", json={"username": "rian", "password": "demo"})
        assert response.status_code == 200 and "Secure" in response.headers["set-cookie"]


GUARDED = [
    ("post", "/api/v1/fish/identify", {"files": {"file": ("f.png", b"", "image/png")}}),
    ("post", "/api/v1/fish/verify", {"json": {"prediction_id": "p", "verified_species_id": "species_nila"}}),
    ("post", "/api/v1/fish/manual", {"data": {"species_id": "species_nila"},
                                     "files": {"file": ("f.png", b"", "image/png")}}),
    ("get", "/api/v1/predictions/p/knowledge", {}),
    ("get", "/api/v1/jobs/p", {}),
    ("get", "/quality", {}),
    ("get", "/api/v1/quality/summary", {}),
]


@pytest.mark.parametrize("method,path,kwargs", GUARDED)
def test_operator_routes_refuse_guests_and_buyers(method, path, kwargs):
    with TestClient(_app()) as client:
        assert getattr(client, method)(path, **kwargs).status_code == 401
        sign_in(client, "dewi")
        assert getattr(client, method)(path, **kwargs).status_code == 403


def test_an_operator_gets_through_the_guard():
    with TestClient(_app()) as client:
        sign_in(client)
        identified = client.post("/api/v1/fish/identify", files={"file": ("f.png", png_bytes(), "image/png")})
        assert identified.status_code == 200, identified.text
        assert client.get("/api/v1/quality/summary").status_code == 200

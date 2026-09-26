"""Accounts, password hashing and signed session cookies."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
import secrets
import time
from base64 import urlsafe_b64decode, urlsafe_b64encode
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Callable, Protocol

from fastapi import Request

from apps.main_api.errors import Forbidden, Unauthenticated

logger = logging.getLogger(__name__)

COOKIE_NAME = "fishora_session"
SESSION_TTL_SECONDS = 7 * 24 * 60 * 60
ROLES = ("operator", "buyer")
USERNAME_PATTERN = re.compile(r"^[a-z0-9_]{3,32}$")
MIN_PASSWORD_LENGTH = 8

# The two public demo accounts. Their ids are what scripts/seed_demo_lots.py
# and the demo data point at, so they are fixed rather than generated.
DEMO_USERS = (
    {"id": "op_rian", "username": "rian", "display_name": "Rian Setiawan", "role": "operator", "password": "demo"},
    {"id": "buyer_dewi", "username": "dewi", "display_name": "Dewi Anggraini", "role": "buyer", "password": "demo"},
)

# scrypt at n=2**14 costs about 16 MB and a tenth of a second: slow enough to
# make guessing a leaked hash expensive, fast enough for a sign-in.
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P, _SCRYPT_LEN = 2**14, 8, 1, 32


def hash_password(password: str) -> str:
    """`scrypt$n$r$p$salt$hash`, so the cost can be raised later without
    invalidating hashes made at the old one."""
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=_SCRYPT_LEN)
    return "$".join(["scrypt", str(_SCRYPT_N), str(_SCRYPT_R), str(_SCRYPT_P), _b64(salt), _b64(digest)])


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, expected = stored.split("$")
        if scheme != "scrypt":
            return False
        expected_bytes = _unb64(expected)
        digest = hashlib.scrypt(password.encode("utf-8"), salt=_unb64(salt), n=int(n), r=int(r), p=int(p),
                                dklen=len(expected_bytes))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest, expected_bytes)


def _b64(raw: bytes) -> str:
    return urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    return urlsafe_b64decode(text + "=" * (-len(text) % 4))


# Checked against when the username does not exist, so a miss costs the same
# scrypt run as a wrong password and response time does not reveal which
# usernames are registered.
@lru_cache(maxsize=1)
def _decoy_hash() -> str:
    return hash_password(secrets.token_urlsafe(16))


@dataclass(frozen=True)
class UserRecord:
    id: str
    username: str
    display_name: str
    role: str
    password_hash: str
    created_at: datetime | None = None


class UsernameTaken(Exception):
    """Registration asked for a username that already exists. Maps to HTTP 409."""

    def __init__(self, username: str):
        super().__init__(f"username {username!r} is taken")
        self.username = username


class UserRepository(Protocol):
    def get_by_username(self, username: str) -> UserRecord | None: ...
    def create(self, record: UserRecord) -> UserRecord: ...


@lru_cache(maxsize=None)
def _demo_hash(password: str) -> str:
    # Every in-memory app (each test builds one) seeds the demo accounts; one
    # scrypt per process instead of one per app.
    return hash_password(password)


def seed_demo_users(users: UserRepository) -> None:
    """Create the demo accounts that are missing. Existing rows, including a
    changed password, are left alone."""
    for demo in DEMO_USERS:
        if users.get_by_username(demo["username"]) is not None:
            continue
        try:
            users.create(UserRecord(
                id=demo["id"], username=demo["username"], display_name=demo["display_name"],
                role=demo["role"], password_hash=_demo_hash(demo["password"]),
            ))
        except UsernameTaken:
            pass  # another worker seeded it first


@lru_cache(maxsize=1)
def process_secret() -> str:
    """A random secret for a process with none configured. Cached, so every
    SessionService in the process signs with the same key."""
    logger.warning(
        "FISHORA_SESSION_SECRET is not set; using a random secret, so sessions "
        "end when the API restarts and are not shared between workers"
    )
    return secrets.token_urlsafe(32)


@dataclass(frozen=True)
class SessionUser:
    id: str
    role: str
    name: str
    username: str


class SessionService:
    def __init__(
        self,
        secret: str | None = None,
        users: UserRepository | None = None,
        *,
        cookie_secure: bool = False,
        ttl_seconds: int = SESSION_TTL_SECONDS,
        clock: Callable[[], float] = time.time,
    ):
        # Resolved on first use: the production app builds a placeholder
        # service before its settings exist and replaces it at startup, and
        # that placeholder must not warn about a secret that is in fact set.
        self._configured_secret = secret
        if users is None:
            from apps.main_api.db.user_repository import InMemoryUserRepository

            users = InMemoryUserRepository()
            seed_demo_users(users)
        self.users = users
        self.cookie_secure = cookie_secure
        self.ttl_seconds = ttl_seconds
        self._clock = clock

    @property
    def _secret(self) -> bytes:
        return (self._configured_secret or process_secret()).encode("utf-8")

    def login(self, username: str, password: str) -> SessionUser:
        record = self.users.get_by_username(username.strip().lower())
        if record is None:
            verify_password(password, _decoy_hash())
            raise Unauthenticated()
        if not verify_password(password, record.password_hash):
            raise Unauthenticated()
        return _session_user(record)

    def register(self, username: str, password: str, display_name: str, role: str) -> SessionUser:
        """Validation of the fields is the API model's job; this assigns the id
        and hashes. Raises UsernameTaken."""
        prefix = "op" if role == "operator" else "buyer"
        record = self.users.create(UserRecord(
            id=f"{prefix}_{secrets.token_hex(6)}",
            username=username,
            display_name=display_name,
            role=role,
            password_hash=hash_password(password),
        ))
        return _session_user(record)

    def dump(self, user: SessionUser) -> str:
        issued = int(self._clock())
        payload = urlsafe_b64encode(json.dumps({
            "id": user.id, "role": user.role, "name": user.name, "username": user.username,
            "iat": issued, "exp": issued + self.ttl_seconds,
        }).encode("utf-8"))
        signature = hmac.new(self._secret, payload, hashlib.sha256).hexdigest()
        return f"{payload.decode('ascii')}.{signature}"

    def load(self, token: str | None) -> SessionUser | None:
        if not token or "." not in token:
            return None
        payload, signature = token.rsplit(".", 1)
        try:
            expected = hmac.new(self._secret, payload.encode("ascii"), hashlib.sha256).hexdigest()
        except UnicodeEncodeError:
            return None
        if not hmac.compare_digest(expected, signature):
            return None
        try:
            data = json.loads(urlsafe_b64decode(payload.encode("ascii")))
            # The cookie's max_age is a request to the browser; the signed
            # expiry is what actually ends a stolen or replayed token.
            if not isinstance(data.get("exp"), int) or data["exp"] <= self._clock():
                return None
            return SessionUser(
                id=data["id"], role=data["role"], name=data["name"], username=data["username"]
            )
        except (KeyError, ValueError, AttributeError, json.JSONDecodeError):
            return None


def _session_user(record: UserRecord) -> SessionUser:
    return SessionUser(id=record.id, role=record.role, name=record.display_name, username=record.username)


def sessions_of(request: Request) -> SessionService:
    service = request.app.state.deps.session_service
    if service is None:
        # create_main_app always installs one; a bare fallback here would sign
        # with a different key from the one that issued the cookie.
        raise RuntimeError("no session service configured")
    return service


def current_user(request: Request) -> SessionUser:
    user = sessions_of(request).load(request.cookies.get(COOKIE_NAME))
    if user is None:
        raise Unauthenticated()
    return user


def require_role(request: Request, role: str) -> SessionUser:
    user = current_user(request)
    if user.role != role:
        raise Forbidden(f"{role} role required")
    return user

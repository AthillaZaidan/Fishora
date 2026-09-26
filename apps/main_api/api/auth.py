from typing import Literal

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field, field_validator

from apps.main_api.errors import Unauthenticated
from apps.main_api.services.session import (
    COOKIE_NAME,
    MIN_PASSWORD_LENGTH,
    SessionService,
    SessionUser,
    UsernameTaken,
    current_user,
    sessions_of,
)

router = APIRouter(prefix="/api/v1/auth")

# Upper bound only so a megabyte "password" cannot be sent through scrypt.
MAX_PASSWORD_LENGTH = 256


class LoginRequest(BaseModel):
    username: str = Field(max_length=64)
    password: str = Field(max_length=MAX_PASSWORD_LENGTH)


class RegisterRequest(BaseModel):
    username: str = Field(pattern=r"^[a-z0-9_]{3,32}$")
    password: str = Field(min_length=MIN_PASSWORD_LENGTH, max_length=MAX_PASSWORD_LENGTH)
    display_name: str = Field(min_length=1, max_length=80)
    role: Literal["operator", "buyer"]

    # Usernames are compared lower-case at sign-in, so they are stored that way.
    @field_validator("username", mode="before")
    @classmethod
    def _fold_username(cls, value):
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("display_name", mode="before")
    @classmethod
    def _trim_display_name(cls, value):
        return value.strip() if isinstance(value, str) else value


class SessionResponse(BaseModel):
    id: str
    role: str
    name: str
    username: str


def _body(user: SessionUser) -> SessionResponse:
    return SessionResponse(id=user.id, role=user.role, name=user.name, username=user.username)


def _sign_in(response: Response, sessions: SessionService, user: SessionUser) -> SessionResponse:
    response.set_cookie(
        COOKIE_NAME,
        sessions.dump(user),
        max_age=sessions.ttl_seconds,
        httponly=True,
        samesite="lax",
        secure=sessions.cookie_secure,
        path="/",
    )
    return _body(user)


@router.post("/login", response_model=SessionResponse)
def login(payload: LoginRequest, request: Request, response: Response):
    sessions = sessions_of(request)
    try:
        user = sessions.login(payload.username, payload.password)
    except Unauthenticated:
        # One message for an unknown user and a wrong password, so the answer
        # does not reveal which usernames exist.
        raise HTTPException(status_code=401, detail="invalid username or password") from None
    return _sign_in(response, sessions, user)


@router.post("/register", response_model=SessionResponse, status_code=201)
def register(payload: RegisterRequest, request: Request, response: Response):
    sessions = sessions_of(request)
    try:
        user = sessions.register(payload.username, payload.password, payload.display_name, payload.role)
    except UsernameTaken:
        raise HTTPException(status_code=409, detail="username is taken") from None
    return _sign_in(response, sessions, user)


@router.post("/logout")
def logout(request: Request, response: Response):
    # Same attributes as when it was set, or a browser may keep the old cookie.
    response.delete_cookie(
        COOKIE_NAME, path="/", httponly=True, samesite="lax", secure=sessions_of(request).cookie_secure,
    )
    return {"ok": True}


@router.get("/me", response_model=SessionResponse)
def me(request: Request):
    return _body(current_user(request))

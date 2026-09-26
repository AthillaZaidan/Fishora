from threading import Lock
from typing import Callable

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from apps.main_api.db.models import User
from apps.main_api.services.session import UserRecord, UsernameTaken


class SqlUserRepository:
    def __init__(self, session_factory: Callable[[], Session]):
        self._session_factory = session_factory

    def get_by_username(self, username: str) -> UserRecord | None:
        with self._session_factory() as session:
            row = session.scalar(select(User).where(User.username == username))
            return None if row is None else self._to_record(row)

    def create(self, record: UserRecord) -> UserRecord:
        with self._session_factory() as session:
            row = User(
                id=record.id,
                username=record.username,
                display_name=record.display_name,
                role=record.role,
                password_hash=record.password_hash,
            )
            session.add(row)
            try:
                session.commit()
            except IntegrityError as exc:
                # The unique index decides, so two simultaneous sign-ups for
                # one name cannot both pass a read-then-insert check.
                session.rollback()
                raise UsernameTaken(record.username) from exc
            session.refresh(row)
            return self._to_record(row)

    @staticmethod
    def _to_record(row: User) -> UserRecord:
        return UserRecord(
            id=row.id,
            username=row.username,
            display_name=row.display_name,
            role=row.role,
            password_hash=row.password_hash,
            created_at=row.created_at,
        )


class InMemoryUserRepository:
    """Accounts for an app built without a database (the in-memory test app).
    They last as long as the process."""

    def __init__(self):
        self._rows: dict[str, UserRecord] = {}
        self._lock = Lock()

    def get_by_username(self, username: str) -> UserRecord | None:
        return self._rows.get(username)

    def create(self, record: UserRecord) -> UserRecord:
        with self._lock:
            if record.username in self._rows:
                raise UsernameTaken(record.username)
            self._rows[record.username] = record
        return record

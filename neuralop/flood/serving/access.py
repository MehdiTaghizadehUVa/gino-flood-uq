"""Access-control policy for gated FGN serving."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Iterable, Protocol

from neuralop.flood.serving.repository import RunRecord

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _normalize_email(value: str) -> str:
    cleaned = (value or "").strip().lower()
    if not _EMAIL_RE.match(cleaned):
        raise ValueError(f"Invalid email address: {value!r}")
    return cleaned


@dataclass(frozen=True)
class User:
    user_id: str
    email: str
    is_admin: bool = False
    disclaimer_acknowledged: bool = False


@dataclass(frozen=True)
class AccessRecord:
    email: str
    is_admin: bool = False
    disclaimer_acknowledged: bool = False


class AccessRepository(Protocol):
    def list_users(self) -> list[AccessRecord]: ...
    def get_user(self, email: str) -> AccessRecord | None: ...
    def upsert_user(
        self,
        email: str,
        *,
        is_admin: bool | None = None,
        disclaimer_acknowledged: bool | None = None,
    ) -> AccessRecord: ...
    def remove_user(self, email: str) -> bool: ...


class InMemoryAccessRepository:
    """Thread-safe access repository for tests and single-process development."""

    def __init__(self, *, allowed_emails: Iterable[str] = (), admin_emails: Iterable[str] = ()) -> None:
        self._records: dict[str, AccessRecord] = {}
        self._lock = threading.RLock()
        for email in allowed_emails:
            cleaned = (email or "").strip().lower()
            if cleaned:
                self._records[cleaned] = AccessRecord(email=cleaned)
        for email in admin_emails:
            cleaned = (email or "").strip().lower()
            if cleaned:
                self._records[cleaned] = AccessRecord(email=cleaned, is_admin=True)

    def list_users(self) -> list[AccessRecord]:
        with self._lock:
            return [self._records[email] for email in sorted(self._records)]

    def get_user(self, email: str) -> AccessRecord | None:
        cleaned = (email or "").strip().lower()
        with self._lock:
            return self._records.get(cleaned)

    def upsert_user(
        self,
        email: str,
        *,
        is_admin: bool | None = None,
        disclaimer_acknowledged: bool | None = None,
    ) -> AccessRecord:
        cleaned = _normalize_email(email)
        with self._lock:
            current = self._records.get(cleaned, AccessRecord(email=cleaned))
            updated = AccessRecord(
                email=cleaned,
                is_admin=current.is_admin if is_admin is None else bool(is_admin),
                disclaimer_acknowledged=(
                    current.disclaimer_acknowledged
                    if disclaimer_acknowledged is None
                    else bool(disclaimer_acknowledged)
                ),
            )
            self._records[cleaned] = updated
            return updated

    def remove_user(self, email: str) -> bool:
        cleaned = _normalize_email(email)
        with self._lock:
            return self._records.pop(cleaned, None) is not None


class AccessDenied(PermissionError):
    pass


class AccessPolicy:
    """Email allowlist + owner/admin policy used by routes and workers.

    The policy is backed by an AccessRepository so production deployments can
    persist allowlist, admin, and disclaimer state in Postgres while tests keep a
    simple in-memory adapter.
    """

    def __init__(
        self,
        *,
        allowed_emails: Iterable[str],
        admin_emails: Iterable[str] = (),
        repository: AccessRepository | None = None,
        open_authenticated_access: bool = False,
    ) -> None:
        self.repository = repository or InMemoryAccessRepository()
        self.open_authenticated_access = bool(open_authenticated_access)
        for email in allowed_emails:
            if email and email.strip():
                self.repository.upsert_user(email, is_admin=False)
        for email in admin_emails:
            if email and email.strip():
                self.repository.upsert_user(email, is_admin=True)

    @property
    def allowed_emails(self) -> set[str]:
        return {record.email for record in self.repository.list_users()}

    @property
    def admin_emails(self) -> set[str]:
        return {record.email for record in self.repository.list_users() if record.is_admin}

    def list_users(self) -> list[dict[str, object]]:
        """Return a sorted snapshot of the current allowlist for admin display."""
        return [
            {
                "email": record.email,
                "is_admin": record.is_admin,
                "disclaimer_acknowledged": record.disclaimer_acknowledged,
            }
            for record in self.repository.list_users()
        ]

    def add_allowed(self, email: str, *, admin: bool = False) -> dict[str, object]:
        """Idempotently grant access to ``email``. Returns the updated record."""
        record = self.repository.upsert_user(email, is_admin=True if admin else None)
        return {
            "email": record.email,
            "is_admin": record.is_admin,
            "disclaimer_acknowledged": record.disclaimer_acknowledged,
        }

    def remove_allowed(self, email: str) -> bool:
        """Revoke access for ``email``. Returns True iff the email was present."""
        return self.repository.remove_user(email)

    def set_admin(self, email: str, *, is_admin: bool) -> dict[str, object]:
        """Promote or demote ``email``. Promotion implies allowlisting."""
        record = self.repository.upsert_user(email, is_admin=bool(is_admin))
        return {
            "email": record.email,
            "is_admin": record.is_admin,
            "disclaimer_acknowledged": record.disclaimer_acknowledged,
        }

    def acknowledge_disclaimer(self, user: User) -> User:
        user = self.require_allowed(user)
        record = self.repository.upsert_user(user.email, disclaimer_acknowledged=True)
        return User(
            user_id=user.user_id,
            email=record.email,
            is_admin=record.is_admin or user.is_admin,
            disclaimer_acknowledged=True,
        )

    def normalize_user(self, user: User) -> User:
        email = user.email.strip().lower()
        record = self.repository.get_user(email)
        is_admin = user.is_admin or bool(record and record.is_admin)
        disclaimer_acknowledged = user.disclaimer_acknowledged or bool(
            record and record.disclaimer_acknowledged
        )
        if user.disclaimer_acknowledged and record and not record.disclaimer_acknowledged:
            self.repository.upsert_user(email, disclaimer_acknowledged=True)
            disclaimer_acknowledged = True
        return User(
            user_id=user.user_id,
            email=email,
            is_admin=is_admin,
            disclaimer_acknowledged=disclaimer_acknowledged,
        )

    def require_allowed(self, user: User) -> User:
        user = self.normalize_user(user)
        allowed = self.repository.get_user(user.email) is not None
        if not allowed and self.open_authenticated_access:
            record = self.repository.upsert_user(
                user.email,
                is_admin=user.is_admin,
                disclaimer_acknowledged=user.disclaimer_acknowledged,
            )
            return User(
                user_id=record.email,
                email=record.email,
                is_admin=record.is_admin,
                disclaimer_acknowledged=record.disclaimer_acknowledged,
            )
        if not allowed and not user.is_admin:
            raise AccessDenied(f"Email is not allowed to use this service: {user.email}")
        return user

    def require_disclaimer(self, user: User) -> User:
        user = self.require_allowed(user)
        if not user.disclaimer_acknowledged:
            raise AccessDenied("Research-only disclaimer must be acknowledged before submitting runs.")
        return user

    def can_read_run(self, user: User, record: RunRecord) -> bool:
        try:
            user = self.require_allowed(user)
        except AccessDenied:
            return False
        return user.is_admin or record.spec.user_id == user.user_id

    def require_run_read(self, user: User, record: RunRecord) -> None:
        if not self.can_read_run(user, record):
            raise AccessDenied("User cannot access this run.")

    def can_administer(self, user: User) -> bool:
        return self.normalize_user(user).is_admin

    def require_admin(self, user: User) -> None:
        if not self.can_administer(user):
            raise AccessDenied("Admin privileges are required.")

"""Authentication adapters for the serving API.

The preferred production deployment is Google OAuth at the reverse-proxy layer
(e.g. oauth2-proxy/Caddy forward-auth), with FastAPI receiving a trusted email
header. Keeping OAuth outside the inference process prevents auth concerns from
leaking into model-serving modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from neuralop.flood.serving.access import User


class AuthenticationError(PermissionError):
    """Raised when a request cannot be mapped to an authenticated user."""


@dataclass(frozen=True)
class TrustedHeaderAuth:
    """Build User objects from reverse-proxy-authenticated headers."""

    allowed_emails: frozenset[str]
    admin_emails: frozenset[str] = frozenset()
    email_headers: tuple[str, ...] = (
        "x-auth-request-email",
        "x-forwarded-email",
        "x-goog-authenticated-user-email",
    )
    disclaimer_header: str = "x-fgn-disclaimer-accepted"

    @staticmethod
    def from_lists(
        *,
        allowed_emails: Iterable[str],
        admin_emails: Iterable[str] = (),
    ) -> "TrustedHeaderAuth":
        return TrustedHeaderAuth(
            allowed_emails=frozenset(_clean_email(x) for x in allowed_emails if _clean_email(x)),
            admin_emails=frozenset(_clean_email(x) for x in admin_emails if _clean_email(x)),
        )

    def user_from_headers(self, headers) -> User:
        email = ""
        for header in self.email_headers:
            value = headers.get(header)
            if value:
                email = _clean_email(value)
                break
        if not email:
            raise AuthenticationError("Missing authenticated email header.")
        if self.allowed_emails and email not in self.allowed_emails and email not in self.admin_emails:
            raise AuthenticationError(f"Email is not allowlisted: {email}")
        accepted = str(headers.get(self.disclaimer_header, "")).strip().lower() in {"1", "true", "yes", "accepted"}
        return User(
            user_id=email,
            email=email,
            is_admin=email in self.admin_emails,
            disclaimer_acknowledged=accepted,
        )


def _clean_email(value: str) -> str:
    text = str(value or "").strip()
    if ":" in text and text.lower().startswith("accounts.google.com"):
        text = text.rsplit(":", 1)[-1]
    return text.lower()

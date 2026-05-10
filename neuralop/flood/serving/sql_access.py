"""SQL-backed access repository for FGN serving users."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from neuralop.flood.serving.access import AccessRecord, _normalize_email


class SqlAccessRepository:
    """Persist allowlist/admin/disclaimer state in the serving database."""

    def __init__(self, database_url: str) -> None:
        try:
            import sqlalchemy as sa
        except Exception as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError("SqlAccessRepository requires SQLAlchemy; install neuraloperator[serve].") from exc
        self.sa = sa
        self.engine = sa.create_engine(database_url, future=True)
        self.table = sa.Table(
            "fgn_serving_users",
            sa.MetaData(),
            sa.Column("email", sa.String, primary_key=True),
            sa.Column("is_admin", sa.Boolean, nullable=False, default=False),
            sa.Column("disclaimer_acknowledged", sa.Boolean, nullable=False, default=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )
        self.table.metadata.create_all(self.engine)

    def _row_to_record(self, row) -> AccessRecord:
        row = getattr(row, "_mapping", row)
        return AccessRecord(
            email=str(row["email"]),
            is_admin=bool(row["is_admin"]),
            disclaimer_acknowledged=bool(row["disclaimer_acknowledged"]),
        )

    def list_users(self) -> list[AccessRecord]:
        with self.engine.begin() as conn:
            rows = conn.execute(self.sa.select(self.table).order_by(self.table.c.email.asc())).all()
        return [self._row_to_record(row) for row in rows]

    def get_user(self, email: str) -> AccessRecord | None:
        cleaned = (email or "").strip().lower()
        if not cleaned:
            return None
        with self.engine.begin() as conn:
            row = conn.execute(self.sa.select(self.table).where(self.table.c.email == cleaned)).first()
        return None if row is None else self._row_to_record(row)

    def upsert_user(
        self,
        email: str,
        *,
        is_admin: bool | None = None,
        disclaimer_acknowledged: bool | None = None,
    ) -> AccessRecord:
        cleaned = _normalize_email(email)
        now = datetime.now(timezone.utc)
        current = self.get_user(cleaned)
        values = {
            "email": cleaned,
            "is_admin": current.is_admin if current is not None and is_admin is None else bool(is_admin),
            "disclaimer_acknowledged": (
                current.disclaimer_acknowledged
                if current is not None and disclaimer_acknowledged is None
                else bool(disclaimer_acknowledged)
            ),
            "updated_at": now,
        }
        if current is None:
            values["created_at"] = now
            with self.engine.begin() as conn:
                conn.execute(self.table.insert().values(**values))
        else:
            with self.engine.begin() as conn:
                conn.execute(self.table.update().where(self.table.c.email == cleaned).values(**values))
        return self.get_user(cleaned)  # type: ignore[return-value]

    def remove_user(self, email: str) -> bool:
        cleaned = _normalize_email(email)
        with self.engine.begin() as conn:
            result = conn.execute(self.table.delete().where(self.table.c.email == cleaned))
        return int(result.rowcount or 0) > 0

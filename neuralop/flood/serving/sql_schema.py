"""Helpers for safe SQL schema initialization."""

from __future__ import annotations


SCHEMA_INIT_ADVISORY_LOCK_KEY = 4207101501


def create_all_safely(sa, engine, metadata, *, tables=None) -> None:
    """Create SQLAlchemy tables under a Postgres advisory lock.

    The API, worker, and cleanup containers can start at the same time during a
    lab deploy. PostgreSQL DDL is transactional, but concurrent
    ``metadata.create_all`` calls can still race while creating table composite
    types. A single advisory lock keeps startup idempotent without introducing a
    migration dependency for the current additive schema shims.
    """

    if engine.dialect.name != "postgresql":
        metadata.create_all(engine, tables=tables)
        return

    with engine.begin() as conn:
        conn.execute(sa.text("SELECT pg_advisory_xact_lock(:lock_key)"), {"lock_key": SCHEMA_INIT_ADVISORY_LOCK_KEY})
        metadata.create_all(conn, tables=tables)

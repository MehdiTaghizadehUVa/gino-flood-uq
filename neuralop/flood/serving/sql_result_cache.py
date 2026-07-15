"""SQL-backed result-cache repository adapter."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Sequence

from neuralop.flood.serving.result_cache import (
    RESULT_CACHE_SCHEMA_VERSION,
    ResultCacheEntry,
    ResultCacheEntryStatus,
    ResultCacheFailureResolution,
    ResultCacheLookupStatus,
    ResultCacheRepository,
    ResultCacheReservation,
    ResultCacheRunLink,
    ResultCacheRunRole,
    ResultCacheRunStatus,
)
from neuralop.flood.serving.sql_schema import create_all_safely


class SqlResultCacheRepository(ResultCacheRepository):
    def __init__(self, database_url: str) -> None:
        try:
            import sqlalchemy as sa
        except Exception as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError("SqlResultCacheRepository requires SQLAlchemy.") from exc
        self.sa = sa
        self.engine = sa.create_engine(database_url, future=True)
        metadata = sa.MetaData()
        self.entries = sa.Table(
            "fgn_result_cache_entries",
            metadata,
            sa.Column("cache_key", sa.String, primary_key=True),
            sa.Column("status", sa.String, nullable=False, index=True),
            sa.Column("producer_run_id", sa.String, nullable=True, index=True),
            sa.Column("artifact_manifest_json", sa.Text, nullable=False),
            sa.Column("schema_version", sa.Integer, nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )
        self.links = sa.Table(
            "fgn_result_cache_run_links",
            metadata,
            sa.Column("run_id", sa.String, primary_key=True),
            sa.Column("cache_key", sa.String, nullable=False, index=True),
            sa.Column("role", sa.String, nullable=False, index=True),
            sa.Column("status", sa.String, nullable=False, index=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )
        create_all_safely(self.sa, self.engine, metadata)

    def reserve_or_find(self, cache_key: str, producer_run_id: str) -> ResultCacheReservation:
        now = datetime.now(timezone.utc)
        with self.engine.begin() as conn:
            row = conn.execute(self.sa.select(self.entries).where(self.entries.c.cache_key == cache_key)).first()
            entry = self._row_to_entry(row) if row is not None else None
            if entry and entry.status == ResultCacheEntryStatus.READY:
                self._upsert_link(
                    conn,
                    ResultCacheRunLink(
                        run_id=producer_run_id,
                        cache_key=cache_key,
                        role=ResultCacheRunRole.HIT,
                        status=ResultCacheRunStatus.WAITING,
                        created_at=now,
                        updated_at=now,
                    ),
                )
                return ResultCacheReservation(ResultCacheLookupStatus.HIT, cache_key, entry)
            if entry and entry.status == ResultCacheEntryStatus.ACTIVE and entry.producer_run_id != producer_run_id:
                self._upsert_link(
                    conn,
                    ResultCacheRunLink(
                        run_id=producer_run_id,
                        cache_key=cache_key,
                        role=ResultCacheRunRole.WAITER,
                        status=ResultCacheRunStatus.WAITING,
                        created_at=now,
                        updated_at=now,
                    ),
                )
                return ResultCacheReservation(ResultCacheLookupStatus.WAITING, cache_key, entry)
            next_entry = ResultCacheEntry(
                cache_key=cache_key,
                status=ResultCacheEntryStatus.ACTIVE,
                producer_run_id=producer_run_id,
                created_at=entry.created_at if entry else now,
                updated_at=now,
            )
            self._upsert_entry(conn, next_entry)
            self._upsert_link(
                conn,
                ResultCacheRunLink(
                    run_id=producer_run_id,
                    cache_key=cache_key,
                    role=ResultCacheRunRole.PRODUCER,
                    status=ResultCacheRunStatus.ACTIVE,
                    created_at=now,
                    updated_at=now,
                ),
            )
            return ResultCacheReservation(ResultCacheLookupStatus.MISS, cache_key, next_entry)

    def publish_ready(self, producer_run_id: str, artifact_manifest: Sequence[str]) -> ResultCacheEntry | None:
        now = datetime.now(timezone.utc)
        with self.engine.begin() as conn:
            link = self._link_for_run(conn, producer_run_id)
            if link is None:
                return None
            entry = self._entry_for_key(conn, link.cache_key)
            if entry is None or entry.producer_run_id != producer_run_id:
                return None
            ready = ResultCacheEntry(
                cache_key=entry.cache_key,
                status=ResultCacheEntryStatus.READY,
                producer_run_id=producer_run_id,
                artifact_manifest=tuple(dict.fromkeys(str(x) for x in artifact_manifest)),
                schema_version=entry.schema_version,
                created_at=entry.created_at,
                updated_at=now,
            )
            self._upsert_entry(conn, ready)
            self._upsert_link(
                conn,
                ResultCacheRunLink(
                    run_id=producer_run_id,
                    cache_key=link.cache_key,
                    role=ResultCacheRunRole.PRODUCER,
                    status=ResultCacheRunStatus.MATERIALIZED,
                    created_at=link.created_at,
                    updated_at=now,
                ),
            )
            return ready

    def handle_producer_failed(self, producer_run_id: str) -> ResultCacheFailureResolution:
        now = datetime.now(timezone.utc)
        with self.engine.begin() as conn:
            link = self._link_for_run(conn, producer_run_id)
            if link is None:
                return ResultCacheFailureResolution(cache_key=None)
            entry = self._entry_for_key(conn, link.cache_key)
            if entry is None or entry.producer_run_id != producer_run_id:
                return ResultCacheFailureResolution(cache_key=link.cache_key)
            waiters = self._waiting_links(conn, link.cache_key)
            self._upsert_link(
                conn,
                ResultCacheRunLink(
                    run_id=producer_run_id,
                    cache_key=link.cache_key,
                    role=link.role,
                    status=ResultCacheRunStatus.FAILED,
                    created_at=link.created_at,
                    updated_at=now,
                ),
            )
            if not waiters:
                self._upsert_entry(
                    conn,
                    ResultCacheEntry(
                        cache_key=entry.cache_key,
                        status=ResultCacheEntryStatus.FAILED,
                        producer_run_id=None,
                        artifact_manifest=(),
                        schema_version=entry.schema_version,
                        created_at=entry.created_at,
                        updated_at=now,
                    ),
                )
                return ResultCacheFailureResolution(cache_key=link.cache_key)
            promoted = waiters[0]
            self._upsert_entry(
                conn,
                ResultCacheEntry(
                    cache_key=entry.cache_key,
                    status=ResultCacheEntryStatus.ACTIVE,
                    producer_run_id=promoted.run_id,
                    artifact_manifest=(),
                    schema_version=entry.schema_version,
                    created_at=entry.created_at,
                    updated_at=now,
                ),
            )
            self._upsert_link(
                conn,
                ResultCacheRunLink(
                    run_id=promoted.run_id,
                    cache_key=promoted.cache_key,
                    role=ResultCacheRunRole.PRODUCER,
                    status=ResultCacheRunStatus.ACTIVE,
                    created_at=promoted.created_at,
                    updated_at=now,
                ),
            )
            return ResultCacheFailureResolution(
                cache_key=link.cache_key,
                promoted_run_id=promoted.run_id,
                waiting_run_ids=tuple(waiter.run_id for waiter in waiters[1:]),
            )

    def list_waiting_runs(self, cache_key: str) -> list[ResultCacheRunLink]:
        with self.engine.begin() as conn:
            return self._waiting_links(conn, cache_key)

    def mark_materialized(self, run_id: str, cache_key: str, *, role: ResultCacheRunRole) -> ResultCacheRunLink:
        now = datetime.now(timezone.utc)
        with self.engine.begin() as conn:
            current = self._link_for_run(conn, run_id)
            link = ResultCacheRunLink(
                run_id=run_id,
                cache_key=cache_key,
                role=role,
                status=ResultCacheRunStatus.MATERIALIZED,
                created_at=current.created_at if current else now,
                updated_at=now,
            )
            self._upsert_link(conn, link)
            return link

    def link_for_run(self, run_id: str) -> ResultCacheRunLink | None:
        with self.engine.begin() as conn:
            return self._link_for_run(conn, run_id)

    def entry_for_key(self, cache_key: str) -> ResultCacheEntry | None:
        with self.engine.begin() as conn:
            return self._entry_for_key(conn, cache_key)

    def _upsert_entry(self, conn, entry: ResultCacheEntry) -> None:
        payload = {
            "cache_key": entry.cache_key,
            "status": entry.status.value,
            "producer_run_id": entry.producer_run_id,
            "artifact_manifest_json": json.dumps(list(entry.artifact_manifest)),
            "schema_version": int(entry.schema_version),
            "created_at": entry.created_at,
            "updated_at": entry.updated_at,
        }
        existing = conn.execute(self.sa.select(self.entries.c.cache_key).where(self.entries.c.cache_key == entry.cache_key)).first()
        if existing:
            conn.execute(self.entries.update().where(self.entries.c.cache_key == entry.cache_key).values(**payload))
        else:
            conn.execute(self.entries.insert().values(**payload))

    def _upsert_link(self, conn, link: ResultCacheRunLink) -> None:
        payload = {
            "run_id": link.run_id,
            "cache_key": link.cache_key,
            "role": link.role.value,
            "status": link.status.value,
            "created_at": link.created_at,
            "updated_at": link.updated_at,
        }
        existing = conn.execute(self.sa.select(self.links.c.run_id).where(self.links.c.run_id == link.run_id)).first()
        if existing:
            conn.execute(self.links.update().where(self.links.c.run_id == link.run_id).values(**payload))
        else:
            conn.execute(self.links.insert().values(**payload))

    def _entry_for_key(self, conn, cache_key: str) -> ResultCacheEntry | None:
        row = conn.execute(self.sa.select(self.entries).where(self.entries.c.cache_key == cache_key)).first()
        return self._row_to_entry(row) if row is not None else None

    def _link_for_run(self, conn, run_id: str) -> ResultCacheRunLink | None:
        row = conn.execute(self.sa.select(self.links).where(self.links.c.run_id == run_id)).first()
        return self._row_to_link(row) if row is not None else None

    def _waiting_links(self, conn, cache_key: str) -> list[ResultCacheRunLink]:
        rows = conn.execute(
            self.sa.select(self.links)
            .where(self.links.c.cache_key == cache_key)
            .where(self.links.c.role == ResultCacheRunRole.WAITER.value)
            .where(self.links.c.status == ResultCacheRunStatus.WAITING.value)
            .order_by(self.links.c.created_at.asc())
        ).all()
        return [self._row_to_link(row) for row in rows]

    @staticmethod
    def _row_to_entry(row) -> ResultCacheEntry:
        row = getattr(row, "_mapping", row)
        try:
            artifacts = tuple(str(x) for x in json.loads(row["artifact_manifest_json"]))
        except Exception:
            artifacts = ()
        return ResultCacheEntry(
            cache_key=row["cache_key"],
            status=ResultCacheEntryStatus(row["status"]),
            producer_run_id=row["producer_run_id"],
            artifact_manifest=artifacts,
            schema_version=int(row["schema_version"] or RESULT_CACHE_SCHEMA_VERSION),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _row_to_link(row) -> ResultCacheRunLink:
        row = getattr(row, "_mapping", row)
        return ResultCacheRunLink(
            run_id=row["run_id"],
            cache_key=row["cache_key"],
            role=ResultCacheRunRole(row["role"]),
            status=ResultCacheRunStatus(row["status"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

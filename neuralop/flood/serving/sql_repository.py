"""SQL-backed run repository adapter.

This adapter intentionally stores the immutable RunSpec as JSON and keeps large
artifacts out of the database. It supports PostgreSQL in production and SQLite
for local smoke tests through SQLAlchemy URLs.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Iterable

from neuralop.flood.serving.repository import (
    RunRecord,
    default_progress_for_status,
    default_progress_label_for_status,
    _runtime_seconds,
)
from neuralop.flood.serving.run_spec import RunSpec, RunStatus, ensure_transition


class SqlRunRepository:
    def __init__(self, database_url: str) -> None:
        try:
            import sqlalchemy as sa
        except Exception as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError("SqlRunRepository requires SQLAlchemy; install neuraloperator[serve].") from exc
        self.sa = sa
        self.engine = sa.create_engine(database_url, future=True)
        self.table = sa.Table(
            "fgn_serving_runs",
            sa.MetaData(),
            sa.Column("run_id", sa.String, primary_key=True),
            sa.Column("user_id", sa.String, nullable=False, index=True),
            sa.Column("status", sa.String, nullable=False, index=True),
            sa.Column("spec_json", sa.Text, nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("pinned", sa.Boolean, nullable=False, default=False),
            sa.Column("failure_reason", sa.Text, nullable=True),
            sa.Column("internal_log_ref", sa.Text, nullable=True),
            sa.Column("progress", sa.Float, nullable=True),
            sa.Column("progress_label", sa.Text, nullable=True),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("runtime_seconds", sa.Float, nullable=True),
        )
        self.table.metadata.create_all(self.engine)
        self._ensure_runtime_columns()

    def _ensure_runtime_columns(self) -> None:
        """Add runtime/progress columns to existing deployments.

        ``metadata.create_all`` deliberately avoids mutating existing tables.
        The lab PC already has a Postgres database, so startup needs this
        additive, non-destructive schema shim instead of asking operators to
        rebuild the volume.
        """
        existing = {column["name"] for column in self.sa.inspect(self.engine).get_columns(self.table.name)}
        runtime_columns = (
            self.table.c.progress,
            self.table.c.progress_label,
            self.table.c.started_at,
            self.table.c.completed_at,
            self.table.c.runtime_seconds,
        )
        with self.engine.begin() as conn:
            for column in runtime_columns:
                if column.name in existing:
                    continue
                column_type = column.type.compile(dialect=self.engine.dialect)
                conn.execute(self.sa.text(f"ALTER TABLE {self.table.name} ADD COLUMN {column.name} {column_type}"))

    def _row_to_record(self, row) -> RunRecord:
        row = getattr(row, "_mapping", row)
        spec_payload = json.loads(row["spec_json"])
        spec = RunSpec(
            run_id=spec_payload["run_id"],
            user_id=spec_payload["user_id"],
            bundle_id=spec_payload["bundle_id"],
            input_hash=spec_payload["input_hash"],
            forecast_steps=int(spec_payload["forecast_steps"]),
            output_detail=spec_payload.get("output_detail", "standard"),
            request_full_hdf5=bool(spec_payload.get("request_full_hdf5", True)),
            request_animation=bool(spec_payload.get("request_animation", False)),
            ensemble_count=int(spec_payload.get("ensemble_count", 3)),
            members_per_ensemble=int(spec_payload.get("members_per_ensemble", 20)),
            calibration_mode=spec_payload.get("calibration_mode", "calibrated_default"),
            exceedance_thresholds_m=tuple(float(x) for x in spec_payload.get("exceedance_thresholds_m", (0.01, 0.05, 0.1, 0.3, 0.5))),
            seed=int(spec_payload.get("seed", 123)),
            label=spec_payload.get("label"),
            created_at=_parse_datetime(spec_payload.get("created_at")) or row["created_at"],
        )
        return RunRecord(
            spec=spec,
            status=RunStatus(row["status"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            pinned=bool(row["pinned"]),
            failure_reason=row["failure_reason"],
            internal_log_ref=row["internal_log_ref"],
            progress=row.get("progress"),
            progress_label=row.get("progress_label"),
            started_at=_parse_datetime(row.get("started_at")),
            completed_at=_parse_datetime(row.get("completed_at")),
            runtime_seconds=row.get("runtime_seconds"),
        )

    def create(self, spec: RunSpec) -> RunRecord:
        now = datetime.now(timezone.utc)
        payload = {
            "run_id": spec.run_id,
            "user_id": spec.user_id,
            "status": RunStatus.SUBMITTED.value,
            "spec_json": json.dumps(spec.manifest(), sort_keys=True),
            "created_at": now,
            "updated_at": now,
            "pinned": False,
            "progress": default_progress_for_status(RunStatus.SUBMITTED),
            "progress_label": default_progress_label_for_status(RunStatus.SUBMITTED),
        }
        with self.engine.begin() as conn:
            conn.execute(self.table.insert().values(**payload))
        return self.get(spec.run_id)

    def get(self, run_id: str) -> RunRecord:
        with self.engine.begin() as conn:
            row = conn.execute(self.sa.select(self.table).where(self.table.c.run_id == run_id)).first()
        if row is None:
            raise KeyError(f"Unknown run_id: {run_id}")
        return self._row_to_record(row)

    def list_for_user(self, user_id: str) -> Iterable[RunRecord]:
        with self.engine.begin() as conn:
            rows = conn.execute(
                self.sa.select(self.table)
                .where(self.table.c.user_id == user_id)
                .order_by(self.table.c.created_at.desc())
            ).all()
        return [self._row_to_record(row) for row in rows]

    def list_all(self) -> Iterable[RunRecord]:
        with self.engine.begin() as conn:
            rows = conn.execute(
                self.sa.select(self.table).order_by(self.table.c.created_at.desc())
            ).all()
        return [self._row_to_record(row) for row in rows]

    def transition(self, run_id: str, status: RunStatus, *, failure_reason: str | None = None) -> RunRecord:
        current = self.get(run_id)
        ensure_transition(current.status, status)
        now = datetime.now(timezone.utc)
        started_at = current.started_at or (now if status == RunStatus.RUNNING else None)
        completed_at = current.completed_at
        if status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELED} and started_at is not None:
            completed_at = completed_at or now
        values = {
            "status": status.value,
            "updated_at": now,
            "progress": default_progress_for_status(status),
            "progress_label": default_progress_label_for_status(status),
            "started_at": started_at,
            "completed_at": completed_at,
            "runtime_seconds": current.runtime_seconds or _runtime_seconds(started_at, completed_at),
        }
        if status == RunStatus.FAILED:
            values["failure_reason"] = failure_reason
        with self.engine.begin() as conn:
            conn.execute(self.table.update().where(self.table.c.run_id == run_id).values(**values))
        return self.get(run_id)

    def update_progress(self, run_id: str, progress: float, *, label: str | None = None) -> RunRecord:
        values = {
            "progress": max(0.0, min(1.0, float(progress))),
            "updated_at": datetime.now(timezone.utc),
        }
        if label is not None:
            values["progress_label"] = str(label)
        with self.engine.begin() as conn:
            conn.execute(self.table.update().where(self.table.c.run_id == run_id).values(**values))
        return self.get(run_id)

    def set_pinned(self, run_id: str, pinned: bool) -> RunRecord:
        with self.engine.begin() as conn:
            conn.execute(
                self.table.update()
                .where(self.table.c.run_id == run_id)
                .values(pinned=bool(pinned), updated_at=datetime.now(timezone.utc))
            )
        return self.get(run_id)


def _parse_datetime(value):
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None

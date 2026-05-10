"""SQL-backed run repository adapter.

This adapter intentionally stores the immutable RunSpec as JSON and keeps large
artifacts out of the database. It supports PostgreSQL in production and SQLite
for local smoke tests through SQLAlchemy URLs.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from typing import Iterable

from neuralop.flood.serving.repository import RunRecord
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
        )
        self.table.metadata.create_all(self.engine)

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
            request_full_hdf5=bool(spec_payload.get("request_full_hdf5", False)),
            request_animation=bool(spec_payload.get("request_animation", False)),
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
        values = {"status": status.value, "updated_at": datetime.now(timezone.utc)}
        if status == RunStatus.FAILED:
            values["failure_reason"] = failure_reason
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
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None

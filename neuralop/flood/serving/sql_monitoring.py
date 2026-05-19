"""SQL-backed monitoring repository for drift reports and candidates."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from neuralop.flood.serving.drift import DriftTestResult
from neuralop.flood.serving.monitoring import (
    CandidateRecord,
    CandidateStatus,
    DescriptorLogEntry,
    HecrasErrorRecord,
    MonitoringFlag,
    MonitoringPhase,
    MonitoringReport,
)


class SqlMonitoringRepository:
    """Persist monitoring reports and retraining candidates in the serving DB."""

    def __init__(self, database_url: str) -> None:
        try:
            import sqlalchemy as sa
        except Exception as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError("SqlMonitoringRepository requires SQLAlchemy; install neuraloperator[serve].") from exc
        self.sa = sa
        self.engine = sa.create_engine(database_url, future=True)
        metadata = sa.MetaData()
        self.reports = sa.Table(
            "fgn_monitoring_reports",
            metadata,
            sa.Column("report_id", sa.String, primary_key=True),
            sa.Column("run_id", sa.String, nullable=False, index=True),
            sa.Column("user_id", sa.String, nullable=False, index=True),
            sa.Column("phase", sa.String, nullable=False, index=True),
            sa.Column("monitoring_bundle_id", sa.String, nullable=False),
            sa.Column("input_hash", sa.String, nullable=False, index=True),
            sa.Column("descriptors_json", sa.Text, nullable=False),
            sa.Column("scores_json", sa.Text, nullable=False),
            sa.Column("flags_json", sa.Text, nullable=False),
            sa.Column("candidate_recommended", sa.Boolean, nullable=False, default=False),
            sa.Column("candidate_reason", sa.Text, nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )
        self.candidates = sa.Table(
            "fgn_retraining_candidates",
            metadata,
            sa.Column("candidate_id", sa.String, primary_key=True),
            sa.Column("run_id", sa.String, nullable=False, unique=True, index=True),
            sa.Column("owner_email", sa.String, nullable=False, index=True),
            sa.Column("input_hash", sa.String, nullable=False, index=True),
            sa.Column("monitoring_bundle_id", sa.String, nullable=False),
            sa.Column("model_bundle_id", sa.String, nullable=False, index=True),
            sa.Column("candidate_score", sa.Float, nullable=False),
            sa.Column("reason", sa.Text, nullable=False),
            sa.Column("status", sa.String, nullable=False, index=True),
            sa.Column("report_ids_json", sa.Text, nullable=False),
            sa.Column("package_artifacts_json", sa.Text, nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("reviewed_by", sa.String, nullable=True),
            sa.Column("admin_notes", sa.Text, nullable=True),
        )
        self.descriptor_logs = sa.Table(
            "fgn_monitoring_descriptor_log",
            metadata,
            sa.Column("descriptor_log_id", sa.String, primary_key=True),
            sa.Column("run_id", sa.String, nullable=False, index=True),
            sa.Column("phase", sa.String, nullable=False, index=True),
            sa.Column("monitoring_bundle_id", sa.String, nullable=False, index=True),
            sa.Column("descriptor_schema_version", sa.Integer, nullable=False),
            sa.Column("descriptors_json", sa.Text, nullable=False),
            sa.Column("scores_json", sa.Text, nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, index=True),
        )
        self.hecras_errors = sa.Table(
            "fgn_hecras_error_records",
            metadata,
            sa.Column("error_record_id", sa.String, primary_key=True),
            sa.Column("candidate_id", sa.String, nullable=False, unique=True, index=True),
            sa.Column("run_id", sa.String, nullable=False, index=True),
            sa.Column("monitoring_bundle_id", sa.String, nullable=False, index=True),
            sa.Column("fgn_descriptors_json", sa.Text, nullable=False),
            sa.Column("hecras_descriptors_json", sa.Text, nullable=False),
            sa.Column("error_descriptors_json", sa.Text, nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, index=True),
        )
        self.drift_results = sa.Table(
            "fgn_drift_test_results",
            metadata,
            sa.Column("test_id", sa.String, primary_key=True),
            sa.Column("test_type", sa.String, nullable=False, index=True),
            sa.Column("descriptor_name", sa.String, nullable=True),
            sa.Column("drift_detected", sa.Boolean, nullable=False),
            sa.Column("test_statistic", sa.Float, nullable=False),
            sa.Column("threshold", sa.Float, nullable=False),
            sa.Column("details_json", sa.Text, nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, index=True),
        )
        metadata.create_all(self.engine)

    def create_report(self, report: MonitoringReport) -> MonitoringReport:
        payload = {
            "report_id": report.report_id,
            "run_id": report.run_id,
            "user_id": report.user_id,
            "phase": report.phase.value,
            "monitoring_bundle_id": report.monitoring_bundle_id,
            "input_hash": report.input_hash,
            "descriptors_json": json.dumps(report.descriptors, sort_keys=True),
            "scores_json": json.dumps(report.scores, sort_keys=True),
            "flags_json": json.dumps([flag.as_dict() for flag in report.flags], sort_keys=True),
            "candidate_recommended": bool(report.candidate_recommended),
            "candidate_reason": report.candidate_reason,
            "created_at": report.created_at,
        }
        with self.engine.begin() as conn:
            conn.execute(self.reports.insert().values(**payload))
        return report

    def list_reports_for_run(self, run_id: str) -> list[MonitoringReport]:
        with self.engine.begin() as conn:
            rows = conn.execute(
                self.sa.select(self.reports)
                .where(self.reports.c.run_id == run_id)
                .order_by(self.reports.c.created_at.asc())
            ).all()
        return [self._row_to_report(row) for row in rows]

    def latest_report_for_run(self, run_id: str, phase: MonitoringPhase) -> MonitoringReport | None:
        with self.engine.begin() as conn:
            row = conn.execute(
                self.sa.select(self.reports)
                .where(self.reports.c.run_id == run_id)
                .where(self.reports.c.phase == phase.value)
                .order_by(self.reports.c.created_at.desc())
            ).first()
        return None if row is None else self._row_to_report(row)

    def upsert_candidate(self, candidate: CandidateRecord) -> CandidateRecord:
        current = self._candidate_for_id(candidate.candidate_id)
        if current is None:
            with self.engine.begin() as conn:
                conn.execute(self.candidates.insert().values(**self._candidate_values(candidate)))
            return self.get_candidate(candidate.candidate_id)
        merged = CandidateRecord(
            candidate_id=current.candidate_id,
            run_id=current.run_id,
            owner_email=current.owner_email,
            input_hash=current.input_hash,
            monitoring_bundle_id=current.monitoring_bundle_id,
            model_bundle_id=current.model_bundle_id,
            candidate_score=max(float(current.candidate_score), float(candidate.candidate_score)),
            reason=_merge_reason(current.reason, candidate.reason),
            status=current.status,
            report_ids=tuple(dict.fromkeys([*current.report_ids, *candidate.report_ids])),
            package_artifacts=tuple(dict.fromkeys([*current.package_artifacts, *candidate.package_artifacts])),
            created_at=current.created_at,
            updated_at=datetime.now(timezone.utc),
            reviewed_by=current.reviewed_by,
            admin_notes=current.admin_notes,
        )
        with self.engine.begin() as conn:
            conn.execute(
                self.candidates.update()
                .where(self.candidates.c.candidate_id == candidate.candidate_id)
                .values(**self._candidate_values(merged, include_created=False))
            )
        return self.get_candidate(candidate.candidate_id)

    def get_candidate(self, candidate_id: str) -> CandidateRecord:
        found = self._candidate_for_id(candidate_id)
        if found is None:
            raise KeyError(f"Unknown candidate_id: {candidate_id}")
        return found

    def candidate_for_run(self, run_id: str) -> CandidateRecord | None:
        with self.engine.begin() as conn:
            row = conn.execute(
                self.sa.select(self.candidates).where(self.candidates.c.run_id == run_id)
            ).first()
        return None if row is None else self._row_to_candidate(row)

    def list_candidates(self) -> list[CandidateRecord]:
        with self.engine.begin() as conn:
            rows = conn.execute(
                self.sa.select(self.candidates).order_by(self.candidates.c.created_at.desc())
            ).all()
        return [self._row_to_candidate(row) for row in rows]

    def update_candidate_status(
        self,
        candidate_id: str,
        status: CandidateStatus,
        *,
        reviewed_by: str | None = None,
        admin_notes: str | None = None,
    ) -> CandidateRecord:
        values = {
            "status": status.value,
            "updated_at": datetime.now(timezone.utc),
        }
        if reviewed_by is not None:
            values["reviewed_by"] = reviewed_by
        if admin_notes is not None:
            values["admin_notes"] = admin_notes
        with self.engine.begin() as conn:
            result = conn.execute(
                self.candidates.update()
                .where(self.candidates.c.candidate_id == candidate_id)
                .values(**values)
            )
        if int(result.rowcount or 0) == 0:
            raise KeyError(f"Unknown candidate_id: {candidate_id}")
        return self.get_candidate(candidate_id)

    def log_descriptors(self, entry: DescriptorLogEntry) -> DescriptorLogEntry:
        payload = {
            "descriptor_log_id": entry.descriptor_log_id,
            "run_id": entry.run_id,
            "phase": entry.phase.value,
            "monitoring_bundle_id": entry.monitoring_bundle_id,
            "descriptor_schema_version": entry.descriptor_schema_version,
            "descriptors_json": json.dumps(entry.descriptors, sort_keys=True),
            "scores_json": json.dumps(entry.scores, sort_keys=True),
            "created_at": entry.created_at,
        }
        with self.engine.begin() as conn:
            existing = conn.execute(
                self.sa.select(self.descriptor_logs)
                .where(self.descriptor_logs.c.descriptor_log_id == entry.descriptor_log_id)
            ).first()
            if existing is None:
                conn.execute(self.descriptor_logs.insert().values(**payload))
            else:
                conn.execute(
                    self.descriptor_logs.update()
                    .where(self.descriptor_logs.c.descriptor_log_id == entry.descriptor_log_id)
                    .values(**payload)
                )
        return entry

    def list_descriptor_logs(
        self,
        *,
        phase: MonitoringPhase | None = None,
        after: datetime | None = None,
        before: datetime | None = None,
        limit: int | None = None,
    ) -> list[DescriptorLogEntry]:
        query = self.sa.select(self.descriptor_logs)
        if phase is not None:
            query = query.where(self.descriptor_logs.c.phase == phase.value)
        if after is not None:
            query = query.where(self.descriptor_logs.c.created_at > after)
        if before is not None:
            query = query.where(self.descriptor_logs.c.created_at < before)
        query = query.order_by(self.descriptor_logs.c.created_at.asc())
        if limit is not None:
            query = query.limit(limit)
        with self.engine.begin() as conn:
            rows = conn.execute(query).all()
        return [self._row_to_descriptor_log(row) for row in rows]

    def _row_to_descriptor_log(self, row) -> DescriptorLogEntry:
        row = getattr(row, "_mapping", row)
        return DescriptorLogEntry(
            descriptor_log_id=row["descriptor_log_id"],
            run_id=row["run_id"],
            phase=MonitoringPhase(row["phase"]),
            monitoring_bundle_id=row["monitoring_bundle_id"],
            descriptor_schema_version=int(row["descriptor_schema_version"]),
            descriptors=json.loads(row["descriptors_json"]),
            scores=json.loads(row["scores_json"]),
            created_at=_parse_datetime(row["created_at"]),
        )

    def create_hecras_error_record(self, record: HecrasErrorRecord) -> HecrasErrorRecord:
        payload = {
            "error_record_id": record.error_record_id,
            "candidate_id": record.candidate_id,
            "run_id": record.run_id,
            "monitoring_bundle_id": record.monitoring_bundle_id,
            "fgn_descriptors_json": json.dumps(record.fgn_descriptors, sort_keys=True),
            "hecras_descriptors_json": json.dumps(record.hecras_descriptors, sort_keys=True),
            "error_descriptors_json": json.dumps(record.error_descriptors, sort_keys=True),
            "created_at": record.created_at,
        }
        with self.engine.begin() as conn:
            conn.execute(self.hecras_errors.insert().values(**payload))
        return record

    def get_hecras_error_record(self, candidate_id: str) -> HecrasErrorRecord | None:
        with self.engine.begin() as conn:
            row = conn.execute(
                self.sa.select(self.hecras_errors)
                .where(self.hecras_errors.c.candidate_id == candidate_id)
            ).first()
        return None if row is None else self._row_to_hecras_error(row)

    def list_hecras_error_records(self, *, limit: int | None = None) -> list[HecrasErrorRecord]:
        query = self.sa.select(self.hecras_errors).order_by(self.hecras_errors.c.created_at.desc())
        if limit is not None:
            query = query.limit(limit)
        with self.engine.begin() as conn:
            rows = conn.execute(query).all()
        return [self._row_to_hecras_error(row) for row in rows]

    def create_drift_test_result(self, result: DriftTestResult) -> DriftTestResult:
        payload = {
            "test_id": result.test_id,
            "test_type": result.test_type,
            "descriptor_name": result.descriptor_name,
            "drift_detected": result.drift_detected,
            "test_statistic": result.test_statistic,
            "threshold": result.threshold,
            "details_json": json.dumps(result.details) if result.details else None,
            "created_at": result.created_at,
        }
        with self.engine.begin() as conn:
            conn.execute(self.drift_results.insert().values(**payload))
        return result

    def list_drift_test_results(
        self, *, after: datetime | None = None, limit: int | None = None,
    ) -> list[DriftTestResult]:
        query = self.sa.select(self.drift_results)
        if after is not None:
            query = query.where(self.drift_results.c.created_at > after)
        query = query.order_by(self.drift_results.c.created_at.desc())
        if limit is not None:
            query = query.limit(limit)
        with self.engine.begin() as conn:
            rows = conn.execute(query).all()
        return [self._row_to_drift_result(row) for row in rows]

    def latest_drift_tests(self) -> list[DriftTestResult]:
        with self.engine.begin() as conn:
            latest_row = conn.execute(
                self.sa.select(self.drift_results.c.created_at)
                .order_by(self.drift_results.c.created_at.desc())
                .limit(1)
            ).first()
            if latest_row is None:
                return []
            latest_time = _parse_datetime(getattr(latest_row, "_mapping", latest_row)["created_at"])
            rows = conn.execute(
                self.sa.select(self.drift_results)
                .where(self.drift_results.c.created_at == latest_time)
            ).all()
        return [self._row_to_drift_result(row) for row in rows]

    def _row_to_drift_result(self, row) -> DriftTestResult:
        row = getattr(row, "_mapping", row)
        details = json.loads(row["details_json"]) if row["details_json"] else None
        return DriftTestResult(
            test_id=row["test_id"],
            test_type=row["test_type"],
            descriptor_name=row["descriptor_name"],
            drift_detected=bool(row["drift_detected"]),
            test_statistic=float(row["test_statistic"]),
            threshold=float(row["threshold"]),
            details=details,
            created_at=_parse_datetime(row["created_at"]),
        )

    def _row_to_hecras_error(self, row) -> HecrasErrorRecord:
        row = getattr(row, "_mapping", row)
        return HecrasErrorRecord(
            error_record_id=row["error_record_id"],
            candidate_id=row["candidate_id"],
            run_id=row["run_id"],
            monitoring_bundle_id=row["monitoring_bundle_id"],
            fgn_descriptors=json.loads(row["fgn_descriptors_json"]),
            hecras_descriptors=json.loads(row["hecras_descriptors_json"]),
            error_descriptors=json.loads(row["error_descriptors_json"]),
            created_at=_parse_datetime(row["created_at"]),
        )

    def _candidate_for_id(self, candidate_id: str) -> CandidateRecord | None:
        with self.engine.begin() as conn:
            row = conn.execute(
                self.sa.select(self.candidates).where(self.candidates.c.candidate_id == candidate_id)
            ).first()
        return None if row is None else self._row_to_candidate(row)

    def _row_to_report(self, row) -> MonitoringReport:
        row = getattr(row, "_mapping", row)
        flags = []
        for raw in json.loads(row["flags_json"]):
            flags.append(
                MonitoringFlag(
                    code=raw["code"],
                    message=raw["message"],
                    severity=raw.get("severity", "warning"),
                    descriptor=raw.get("descriptor"),
                    value=raw.get("value"),
                    reference_low=raw.get("reference_low"),
                    reference_high=raw.get("reference_high"),
                )
            )
        return MonitoringReport(
            report_id=row["report_id"],
            run_id=row["run_id"],
            user_id=row["user_id"],
            phase=MonitoringPhase(row["phase"]),
            monitoring_bundle_id=row["monitoring_bundle_id"],
            input_hash=row["input_hash"],
            descriptors=json.loads(row["descriptors_json"]),
            scores=json.loads(row["scores_json"]),
            flags=tuple(flags),
            candidate_recommended=bool(row["candidate_recommended"]),
            candidate_reason=row["candidate_reason"],
            created_at=_parse_datetime(row["created_at"]),
        )

    def _row_to_candidate(self, row) -> CandidateRecord:
        row = getattr(row, "_mapping", row)
        return CandidateRecord(
            candidate_id=row["candidate_id"],
            run_id=row["run_id"],
            owner_email=row["owner_email"],
            input_hash=row["input_hash"],
            monitoring_bundle_id=row["monitoring_bundle_id"],
            model_bundle_id=row["model_bundle_id"],
            candidate_score=float(row["candidate_score"]),
            reason=row["reason"],
            status=CandidateStatus(row["status"]),
            report_ids=tuple(json.loads(row["report_ids_json"])),
            package_artifacts=tuple(json.loads(row["package_artifacts_json"])),
            created_at=_parse_datetime(row["created_at"]),
            updated_at=_parse_datetime(row["updated_at"]),
            reviewed_by=row["reviewed_by"],
            admin_notes=row["admin_notes"],
        )

    def _candidate_values(self, candidate: CandidateRecord, *, include_created: bool = True) -> dict:
        values = {
            "candidate_id": candidate.candidate_id,
            "run_id": candidate.run_id,
            "owner_email": candidate.owner_email,
            "input_hash": candidate.input_hash,
            "monitoring_bundle_id": candidate.monitoring_bundle_id,
            "model_bundle_id": candidate.model_bundle_id,
            "candidate_score": float(candidate.candidate_score),
            "reason": candidate.reason,
            "status": candidate.status.value,
            "report_ids_json": json.dumps(list(candidate.report_ids), sort_keys=True),
            "package_artifacts_json": json.dumps(list(candidate.package_artifacts), sort_keys=True),
            "updated_at": candidate.updated_at,
            "reviewed_by": candidate.reviewed_by,
            "admin_notes": candidate.admin_notes,
        }
        if include_created:
            values["created_at"] = candidate.created_at
        return values


def _parse_datetime(value) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _merge_reason(a: str, b: str) -> str:
    parts = []
    for item in [*a.split(","), *b.split(",")]:
        item = item.strip()
        if item and item not in parts:
            parts.append(item)
    return ", ".join(parts)

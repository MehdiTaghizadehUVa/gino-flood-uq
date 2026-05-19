"""Drift screening and retraining-candidate capture for FGN serving.

This module is intentionally independent from FastAPI, Celery, SQLAlchemy, and
the FGN inference implementation. It owns the Phase 1 monitoring contract:
screen forcing inputs, summarize completed forecast/UQ outputs, and preserve
candidate events for later HEC-RAS labeling.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Iterable, Mapping, Protocol, Sequence

import numpy as np

from neuralop.flood.serving.drift import DriftTestResult
from neuralop.flood.serving.forcing import ForcingInput
from neuralop.flood.serving.run_spec import RunSpec
from neuralop.flood.serving.storage import ArtifactStore


class MonitoringError(ValueError):
    """Raised when a monitoring bundle or report is malformed."""


class MonitoringPhase(str, Enum):
    PRE_RUN = "PRE_RUN"
    POST_RUN = "POST_RUN"


class CandidateStatus(str, Enum):
    NEW = "NEW"
    REVIEWED = "REVIEWED"
    SELECTED_FOR_HECRAS = "SELECTED_FOR_HECRAS"
    REJECTED = "REJECTED"
    SIMULATED = "SIMULATED"


@dataclass(frozen=True)
class MonitoringFlag:
    code: str
    message: str
    severity: str = "warning"
    descriptor: str | None = None
    value: float | None = None
    reference_low: float | None = None
    reference_high: float | None = None
    details: dict[str, object] | None = None

    def as_dict(self) -> dict[str, object]:
        d: dict[str, object] = {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "descriptor": self.descriptor,
            "value": self.value,
            "reference_low": self.reference_low,
            "reference_high": self.reference_high,
        }
        if self.details is not None:
            d["details"] = self.details
        return d


@dataclass(frozen=True)
class MonitoringReport:
    report_id: str
    run_id: str
    user_id: str
    phase: MonitoringPhase
    monitoring_bundle_id: str
    input_hash: str
    descriptors: dict[str, float | int | None]
    scores: dict[str, float | None]
    flags: tuple[MonitoringFlag, ...] = ()
    candidate_recommended: bool = False
    candidate_reason: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def public_payload(self) -> dict[str, object]:
        return {
            "report_id": self.report_id,
            "run_id": self.run_id,
            "user_id": self.user_id,
            "phase": self.phase.value,
            "monitoring_bundle_id": self.monitoring_bundle_id,
            "input_hash": self.input_hash,
            "descriptors": self.descriptors,
            "scores": self.scores,
            "flags": [flag.as_dict() for flag in self.flags],
            "candidate_recommended": self.candidate_recommended,
            "candidate_reason": self.candidate_reason,
            "created_at": self.created_at.isoformat(),
            "language": (
                "Research performance-risk screening only; this is not proof "
                "of model error or operational flood guidance."
            ),
        }

    def screening_payload(self) -> dict[str, object]:
        return {
            "available": True,
            "monitoring_bundle_id": self.monitoring_bundle_id,
            "input_novelty_score": self.scores.get("input_novelty_score", 0.0),
            "candidate_recommended": self.candidate_recommended,
            "flags": [flag.as_dict() for flag in self.flags],
        }


@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: str
    run_id: str
    owner_email: str
    input_hash: str
    monitoring_bundle_id: str
    model_bundle_id: str
    candidate_score: float
    reason: str
    status: CandidateStatus = CandidateStatus.NEW
    report_ids: tuple[str, ...] = ()
    package_artifacts: tuple[str, ...] = ()
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reviewed_by: str | None = None
    admin_notes: str | None = None

    def public_payload(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "run_id": self.run_id,
            "owner_email": self.owner_email,
            "input_hash": self.input_hash,
            "monitoring_bundle_id": self.monitoring_bundle_id,
            "model_bundle_id": self.model_bundle_id,
            "candidate_score": self.candidate_score,
            "reason": self.reason,
            "status": self.status.value,
            "report_ids": list(self.report_ids),
            "package_artifacts": list(self.package_artifacts),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "reviewed_by": self.reviewed_by,
            "admin_notes": self.admin_notes,
        }


@dataclass(frozen=True)
class DescriptorLogEntry:
    descriptor_log_id: str
    run_id: str
    phase: MonitoringPhase
    monitoring_bundle_id: str
    descriptor_schema_version: int
    descriptors: dict[str, float | int | None]
    scores: dict[str, float | None]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class HecrasErrorRecord:
    error_record_id: str
    candidate_id: str
    run_id: str
    monitoring_bundle_id: str
    fgn_descriptors: dict[str, float | int | None]
    hecras_descriptors: dict[str, float | int | None]
    error_descriptors: dict[str, float | None]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def validate_hecras_path(path: str) -> Path:
    import os as _os
    root = _os.environ.get("FGN_HECRAS_RESULT_ROOT")
    if not root:
        raise ValueError("FGN_HECRAS_RESULT_ROOT not configured.")
    resolved = Path(path).resolve()
    allowed = Path(root).resolve()
    if not str(resolved).startswith(str(allowed) + _os.sep) and resolved != allowed:
        raise PermissionError(f"HEC-RAS path must be under {allowed}.")
    if not resolved.exists():
        raise FileNotFoundError(f"HEC-RAS HDF not found: {resolved}")
    return resolved


class MonitoringRepository(Protocol):
    def create_report(self, report: MonitoringReport) -> MonitoringReport: ...
    def list_reports_for_run(self, run_id: str) -> list[MonitoringReport]: ...
    def latest_report_for_run(self, run_id: str, phase: MonitoringPhase) -> MonitoringReport | None: ...
    def upsert_candidate(self, candidate: CandidateRecord) -> CandidateRecord: ...
    def get_candidate(self, candidate_id: str) -> CandidateRecord: ...
    def candidate_for_run(self, run_id: str) -> CandidateRecord | None: ...
    def list_candidates(self) -> list[CandidateRecord]: ...
    def update_candidate_status(
        self,
        candidate_id: str,
        status: CandidateStatus,
        *,
        reviewed_by: str | None = None,
        admin_notes: str | None = None,
    ) -> CandidateRecord: ...
    def log_descriptors(self, entry: DescriptorLogEntry) -> DescriptorLogEntry: ...
    def list_descriptor_logs(
        self,
        *,
        phase: MonitoringPhase | None = None,
        after: datetime | None = None,
        before: datetime | None = None,
        limit: int | None = None,
    ) -> list[DescriptorLogEntry]: ...
    def create_hecras_error_record(self, record: HecrasErrorRecord) -> HecrasErrorRecord: ...
    def get_hecras_error_record(self, candidate_id: str) -> HecrasErrorRecord | None: ...
    def list_hecras_error_records(self, *, limit: int | None = None) -> list[HecrasErrorRecord]: ...
    def create_drift_test_result(self, result: DriftTestResult) -> DriftTestResult: ...
    def list_drift_test_results(self, *, after: datetime | None = None, limit: int | None = None) -> list[DriftTestResult]: ...
    def latest_drift_tests(self) -> list[DriftTestResult]: ...


class InMemoryMonitoringRepository:
    """Thread-safe monitoring repository for tests and local development."""

    def __init__(self) -> None:
        self._reports: dict[str, MonitoringReport] = {}
        self._candidates: dict[str, CandidateRecord] = {}
        self._descriptor_logs: dict[str, DescriptorLogEntry] = {}
        self._hecras_errors: dict[str, HecrasErrorRecord] = {}
        self._drift_results: list[DriftTestResult] = []
        self._lock = RLock()

    def create_report(self, report: MonitoringReport) -> MonitoringReport:
        with self._lock:
            self._reports[report.report_id] = report
            return report

    def list_reports_for_run(self, run_id: str) -> list[MonitoringReport]:
        with self._lock:
            return sorted(
                [report for report in self._reports.values() if report.run_id == run_id],
                key=lambda report: report.created_at,
            )

    def latest_report_for_run(self, run_id: str, phase: MonitoringPhase) -> MonitoringReport | None:
        reports = [r for r in self.list_reports_for_run(run_id) if r.phase == phase]
        return reports[-1] if reports else None

    def upsert_candidate(self, candidate: CandidateRecord) -> CandidateRecord:
        with self._lock:
            current = self._candidates.get(candidate.candidate_id)
            if current is None:
                self._candidates[candidate.candidate_id] = candidate
                return candidate
            merged = replace(
                current,
                candidate_score=max(float(current.candidate_score), float(candidate.candidate_score)),
                reason=_merge_reason(current.reason, candidate.reason),
                report_ids=tuple(dict.fromkeys([*current.report_ids, *candidate.report_ids])),
                package_artifacts=tuple(dict.fromkeys([*current.package_artifacts, *candidate.package_artifacts])),
                updated_at=datetime.now(timezone.utc),
            )
            self._candidates[candidate.candidate_id] = merged
            return merged

    def get_candidate(self, candidate_id: str) -> CandidateRecord:
        with self._lock:
            if candidate_id not in self._candidates:
                raise KeyError(f"Unknown candidate_id: {candidate_id}")
            return self._candidates[candidate_id]

    def candidate_for_run(self, run_id: str) -> CandidateRecord | None:
        with self._lock:
            for candidate in self._candidates.values():
                if candidate.run_id == run_id:
                    return candidate
            return None

    def list_candidates(self) -> list[CandidateRecord]:
        with self._lock:
            return sorted(self._candidates.values(), key=lambda item: item.created_at, reverse=True)

    def update_candidate_status(
        self,
        candidate_id: str,
        status: CandidateStatus,
        *,
        reviewed_by: str | None = None,
        admin_notes: str | None = None,
    ) -> CandidateRecord:
        with self._lock:
            current = self.get_candidate(candidate_id)
            updated = replace(
                current,
                status=status,
                reviewed_by=reviewed_by if reviewed_by is not None else current.reviewed_by,
                admin_notes=admin_notes if admin_notes is not None else current.admin_notes,
                updated_at=datetime.now(timezone.utc),
            )
            self._candidates[candidate_id] = updated
            return updated

    def log_descriptors(self, entry: DescriptorLogEntry) -> DescriptorLogEntry:
        with self._lock:
            self._descriptor_logs[entry.descriptor_log_id] = entry
            return entry

    def list_descriptor_logs(
        self,
        *,
        phase: MonitoringPhase | None = None,
        after: datetime | None = None,
        before: datetime | None = None,
        limit: int | None = None,
    ) -> list[DescriptorLogEntry]:
        with self._lock:
            entries = list(self._descriptor_logs.values())
            if phase is not None:
                entries = [e for e in entries if e.phase == phase]
            if after is not None:
                entries = [e for e in entries if e.created_at > after]
            if before is not None:
                entries = [e for e in entries if e.created_at < before]
            entries.sort(key=lambda e: e.created_at)
            if limit is not None:
                entries = entries[-limit:]
            return entries

    def create_hecras_error_record(self, record: HecrasErrorRecord) -> HecrasErrorRecord:
        with self._lock:
            self._hecras_errors[record.candidate_id] = record
            return record

    def get_hecras_error_record(self, candidate_id: str) -> HecrasErrorRecord | None:
        with self._lock:
            return self._hecras_errors.get(candidate_id)

    def list_hecras_error_records(self, *, limit: int | None = None) -> list[HecrasErrorRecord]:
        with self._lock:
            records = sorted(self._hecras_errors.values(), key=lambda r: r.created_at, reverse=True)
            if limit is not None:
                records = records[:limit]
            return records

    def create_drift_test_result(self, result: DriftTestResult) -> DriftTestResult:
        with self._lock:
            self._drift_results.append(result)
            return result

    def list_drift_test_results(
        self, *, after: datetime | None = None, limit: int | None = None,
    ) -> list[DriftTestResult]:
        with self._lock:
            results = list(self._drift_results)
            if after is not None:
                results = [r for r in results if r.created_at > after]
            results.sort(key=lambda r: r.created_at, reverse=True)
            if limit is not None:
                results = results[:limit]
            return results

    def latest_drift_tests(self) -> list[DriftTestResult]:
        with self._lock:
            if not self._drift_results:
                return []
            latest_time = max(r.created_at for r in self._drift_results)
            return [r for r in self._drift_results if r.created_at == latest_time]


@dataclass(frozen=True)
class MonitoringBundle:
    """Versioned reference distribution for drift screening."""

    bundle_id: str
    descriptor_percentiles: dict[str, dict[str, float]]
    forecast_descriptor_percentiles: dict[str, dict[str, float]] = field(default_factory=dict)
    candidate_score_threshold: float = 0.8
    warning_percentile_low: str = "p05"
    warning_percentile_high: str = "p95"
    candidate_percentile_low: str = "p01"
    candidate_percentile_high: str = "p99"
    control_sample_modulus: int = 0
    provenance: dict[str, object] = field(default_factory=dict)
    # Phase 2 fields — all default to safe values for backwards compatibility
    monitoring_bundle_schema_version: int = 1
    descriptor_schema_version: int = 1
    transform_spec: dict[str, str] = field(default_factory=dict)
    reference_population: dict[str, int] = field(default_factory=dict)
    created_from_commit: str | None = None
    forcing_covariance: dict[str, object] | None = None
    forecast_covariance: dict[str, object] | None = None
    heuristic_reference_percentiles: dict[str, dict[str, float]] | None = None
    covariance_exclude_descriptors: list[str] = field(default_factory=list)

    @staticmethod
    def load(path: str | Path) -> "MonitoringBundle":
        raw_path = Path(path).expanduser()
        with raw_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise MonitoringError(f"Monitoring bundle {raw_path} must be a JSON object.")
        thresholds = payload.get("thresholds", {})
        if not isinstance(thresholds, Mapping):
            thresholds = {}
        descriptor_percentiles = _coerce_percentiles(
            payload.get("descriptor_percentiles")
            or payload.get("reference", {}).get("descriptors")
            or {}
        )
        forecast_descriptor_percentiles = _coerce_percentiles(
            payload.get("forecast_descriptor_percentiles")
            or payload.get("reference", {}).get("forecast_descriptors")
            or {}
        )
        heuristic_raw = payload.get("heuristic_reference_percentiles")
        heuristic_percentiles = _coerce_percentiles(heuristic_raw) if isinstance(heuristic_raw, Mapping) else None
        heuristic_percentiles = heuristic_percentiles or None
        covariance_exclude = payload.get("covariance_exclude_descriptors", [])
        if not isinstance(covariance_exclude, list):
            covariance_exclude = []
        bundle = MonitoringBundle(
            bundle_id=str(payload.get("bundle_id") or payload.get("monitoring_bundle_id") or raw_path.stem),
            descriptor_percentiles=descriptor_percentiles,
            forecast_descriptor_percentiles=forecast_descriptor_percentiles,
            candidate_score_threshold=float(thresholds.get("candidate_score_threshold", 0.8)),
            warning_percentile_low=str(thresholds.get("warning_percentile_low", "p05")),
            warning_percentile_high=str(thresholds.get("warning_percentile_high", "p95")),
            candidate_percentile_low=str(thresholds.get("candidate_percentile_low", "p01")),
            candidate_percentile_high=str(thresholds.get("candidate_percentile_high", "p99")),
            control_sample_modulus=int(thresholds.get("control_sample_modulus", 0) or 0),
            provenance=dict(payload.get("provenance", {})) if isinstance(payload.get("provenance"), Mapping) else {},
            monitoring_bundle_schema_version=int(payload.get("monitoring_bundle_schema_version", 1)),
            descriptor_schema_version=int(payload.get("descriptor_schema_version", 1)),
            transform_spec=dict(payload.get("transform_spec", {})) if isinstance(payload.get("transform_spec"), Mapping) else {},
            reference_population=dict(payload.get("reference_population", {})) if isinstance(payload.get("reference_population"), Mapping) else {},
            created_from_commit=str(payload["created_from_commit"]) if payload.get("created_from_commit") else None,
            forcing_covariance=dict(payload["forcing_covariance"]) if isinstance(payload.get("forcing_covariance"), Mapping) else None,
            forecast_covariance=dict(payload["forecast_covariance"]) if isinstance(payload.get("forecast_covariance"), Mapping) else None,
            heuristic_reference_percentiles=heuristic_percentiles,
            covariance_exclude_descriptors=[str(x) for x in covariance_exclude],
        )
        return bundle.validate()

    def validate(self) -> "MonitoringBundle":
        if not self.bundle_id:
            raise MonitoringError("Monitoring bundle requires a non-empty bundle_id.")
        if not self.descriptor_percentiles:
            raise MonitoringError("Monitoring bundle requires descriptor_percentiles.")
        for name, percentiles in self.descriptor_percentiles.items():
            _validate_percentile_entry(name, percentiles)
        for name, percentiles in self.forecast_descriptor_percentiles.items():
            _validate_percentile_entry(name, percentiles)
        if not 0.0 <= float(self.candidate_score_threshold) <= 1.0:
            raise MonitoringError("candidate_score_threshold must be in [0, 1].")
        if int(self.control_sample_modulus) < 0:
            raise MonitoringError("control_sample_modulus must be >= 0.")
        return self


class MonitoringOrchestrator:
    """Coordinates monitoring reports, candidate decisions, and packages."""

    def __init__(
        self,
        *,
        bundle: MonitoringBundle,
        repository: MonitoringRepository,
        run_artifact_store: ArtifactStore,
        candidate_artifact_store: ArtifactStore,
    ) -> None:
        self.bundle = bundle
        self.repository = repository
        self.run_artifact_store = run_artifact_store
        self.candidate_artifact_store = candidate_artifact_store

    @property
    def available(self) -> bool:
        return True

    def preview_forcing(self, *, user_id: str, forcing: ForcingInput, bundle_id: str) -> MonitoringReport:
        """Score a forcing input without persisting reports or artifacts."""
        preview_spec = RunSpec.new(
            user_id=user_id,
            bundle_id=bundle_id,
            input_hash=forcing.input_hash,
            forecast_steps=forcing.forecast_steps,
            label="validation-preview",
        )
        return _build_report(
            run_spec=preview_spec,
            phase=MonitoringPhase.PRE_RUN,
            monitoring_bundle=self.bundle,
            descriptors=build_forcing_descriptors(forcing),
            score_prefix="input",
        )

    def screen_forcing(
        self,
        *,
        run_spec: RunSpec,
        owner_email: str,
        forcing: ForcingInput,
        forcing_csv: bytes,
        model_bundle_metadata: Mapping[str, object],
    ) -> MonitoringReport:
        descriptors = build_forcing_descriptors(forcing)
        report = _build_report(
            run_spec=run_spec,
            phase=MonitoringPhase.PRE_RUN,
            monitoring_bundle=self.bundle,
            descriptors=descriptors,
            score_prefix="input",
        )
        self.repository.create_report(report)
        self.repository.log_descriptors(DescriptorLogEntry(
            descriptor_log_id=f"{run_spec.run_id}:{MonitoringPhase.PRE_RUN.value}:{self.bundle.bundle_id}",
            run_id=run_spec.run_id,
            phase=MonitoringPhase.PRE_RUN,
            monitoring_bundle_id=self.bundle.bundle_id,
            descriptor_schema_version=self.bundle.descriptor_schema_version,
            descriptors=descriptors,
            scores=dict(report.scores),
        ))
        self.run_artifact_store.put_json(run_spec.run_id, "forcing_descriptors", descriptors)
        self.run_artifact_store.put_json(run_spec.run_id, "monitoring_report_pre_run", report.public_payload())
        if report.candidate_recommended:
            self._preserve_candidate_package(
                run_spec=run_spec,
                owner_email=owner_email,
                report=report,
                model_bundle_metadata=model_bundle_metadata,
                source_artifacts=("forcing.csv", "run_manifest.json", "forcing_descriptors.json", "monitoring_report_pre_run.json"),
                extra_payloads={"uploaded_forcing.csv": forcing_csv},
            )
        return report

    def evaluate_completed_run(
        self,
        *,
        run_spec: RunSpec,
        owner_email: str,
        raw_summary: Mapping[str, object],
        calibrated_summary: Mapping[str, object],
        comparison_summary: Mapping[str, object],
        model_bundle_metadata: Mapping[str, object],
    ) -> MonitoringReport:
        descriptors = build_forecast_descriptors(
            raw_summary=raw_summary,
            calibrated_summary=calibrated_summary,
            comparison_summary=comparison_summary,
        )
        report = _build_report(
            run_spec=run_spec,
            phase=MonitoringPhase.POST_RUN,
            monitoring_bundle=self.bundle,
            descriptors=descriptors,
            score_prefix="forecast",
        )
        self.repository.create_report(report)
        self.repository.log_descriptors(DescriptorLogEntry(
            descriptor_log_id=f"{run_spec.run_id}:{MonitoringPhase.POST_RUN.value}:{self.bundle.bundle_id}",
            run_id=run_spec.run_id,
            phase=MonitoringPhase.POST_RUN,
            monitoring_bundle_id=self.bundle.bundle_id,
            descriptor_schema_version=self.bundle.descriptor_schema_version,
            descriptors=descriptors,
            scores=dict(report.scores),
        ))
        self.run_artifact_store.put_json(run_spec.run_id, "forecast_descriptors", descriptors)
        self.run_artifact_store.put_json(run_spec.run_id, "monitoring_report_post_run", report.public_payload())
        if report.candidate_recommended:
            self._preserve_candidate_package(
                run_spec=run_spec,
                owner_email=owner_email,
                report=report,
                model_bundle_metadata=model_bundle_metadata,
                source_artifacts=(
                    "forcing.csv",
                    "run_manifest.json",
                    "forcing_descriptors.json",
                    "monitoring_report_pre_run.json",
                    "forecast_descriptors.json",
                    "monitoring_report_post_run.json",
                    "raw_summary.json",
                    "calibrated_summary.json",
                    "comparison_summary.json",
                ),
            )
        return report

    def monitoring_payload_for_run(self, run_id: str) -> dict[str, object]:
        reports = self.repository.list_reports_for_run(run_id)
        candidate = self.repository.candidate_for_run(run_id)
        return {
            "available": True,
            "monitoring_bundle_id": self.bundle.bundle_id,
            "reports": [report.public_payload() for report in reports],
            "pre_run": _latest_payload(reports, MonitoringPhase.PRE_RUN),
            "post_run": _latest_payload(reports, MonitoringPhase.POST_RUN),
            "candidate": candidate.public_payload() if candidate else None,
        }

    def list_candidates(self) -> list[dict[str, object]]:
        return [candidate.public_payload() for candidate in self.repository.list_candidates()]

    def update_candidate_status(
        self,
        candidate_id: str,
        status: CandidateStatus | str,
        *,
        reviewed_by: str | None = None,
        admin_notes: str | None = None,
    ) -> dict[str, object]:
        updated = self.repository.update_candidate_status(
            candidate_id,
            CandidateStatus(str(status)),
            reviewed_by=reviewed_by,
            admin_notes=admin_notes,
        )
        return updated.public_payload()

    def _preserve_candidate_package(
        self,
        *,
        run_spec: RunSpec,
        owner_email: str,
        report: MonitoringReport,
        model_bundle_metadata: Mapping[str, object],
        source_artifacts: Sequence[str],
        extra_payloads: Mapping[str, bytes] | None = None,
    ) -> CandidateRecord:
        candidate_id = candidate_id_for_run(run_spec.run_id)
        copied: list[str] = []
        for artifact_id in source_artifacts:
            try:
                data = self.run_artifact_store.read_bytes(run_spec.run_id, artifact_id)
            except (FileNotFoundError, KeyError):
                continue
            self.candidate_artifact_store.put_bytes(
                candidate_id,
                artifact_id,
                data,
                content_type=_content_type_for_artifact(artifact_id),
            )
            copied.append(artifact_id)
        for artifact_id, data in (extra_payloads or {}).items():
            self.candidate_artifact_store.put_bytes(
                candidate_id,
                artifact_id,
                data,
                content_type=_content_type_for_artifact(artifact_id),
            )
            copied.append(artifact_id)
        manifest = {
            "candidate_id": candidate_id,
            "run_id": run_spec.run_id,
            "owner_email": owner_email,
            "input_hash": run_spec.input_hash,
            "model_bundle_id": run_spec.bundle_id,
            "monitoring_bundle_id": self.bundle.bundle_id,
            "candidate_reason": report.candidate_reason,
            "candidate_score": _report_candidate_score(report),
            "report_ids": [report.report_id],
            "model_bundle": dict(model_bundle_metadata),
            "monitoring_report": report.public_payload(),
            "preserved_artifacts": copied,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.candidate_artifact_store.put_json(candidate_id, "candidate_manifest", manifest)
        copied.append("candidate_manifest.json")
        candidate = CandidateRecord(
            candidate_id=candidate_id,
            run_id=run_spec.run_id,
            owner_email=owner_email,
            input_hash=run_spec.input_hash,
            monitoring_bundle_id=self.bundle.bundle_id,
            model_bundle_id=run_spec.bundle_id,
            candidate_score=_report_candidate_score(report),
            reason=report.candidate_reason or "monitoring_recommended",
            report_ids=(report.report_id,),
            package_artifacts=tuple(dict.fromkeys(copied)),
        )
        return self.repository.upsert_candidate(candidate)


def candidate_id_for_run(run_id: str) -> str:
    return f"candidate_{run_id}"


def build_forcing_descriptors(forcing: ForcingInput) -> dict[str, float | int]:
    stage = np.asarray(forcing.stage, dtype=np.float64)
    precip = np.asarray(forcing.precipitation, dtype=np.float64)
    dt_h = float(forcing.dt_seconds) / 3600.0
    time_h = np.arange(stage.shape[0], dtype=np.float64) * dt_h
    stage_diff = np.diff(stage) / dt_h if stage.shape[0] > 1 and dt_h > 0 else np.array([], dtype=np.float64)
    precip_positive = precip > 1.0e-9
    wet_rows = int(np.count_nonzero(precip_positive))
    peak_stage_idx = int(np.argmax(stage)) if stage.size else 0
    peak_precip_idx = int(np.argmax(precip)) if precip.size else 0
    return {
        "n_rows": int(stage.shape[0]),
        "forecast_steps": int(forcing.forecast_steps),
        "duration_hours": float(max(0, stage.shape[0] - 1) * dt_h),
        "stage_min": _finite_float(np.min(stage)),
        "stage_max": _finite_float(np.max(stage)),
        "stage_range": _finite_float(np.max(stage) - np.min(stage)),
        "stage_mean": _finite_float(np.mean(stage)),
        "stage_peak_lead_hours": _finite_float(time_h[peak_stage_idx]) if time_h.size else 0.0,
        "stage_max_rise_rate_per_hour": _finite_float(np.max(stage_diff)) if stage_diff.size else 0.0,
        "stage_max_fall_rate_per_hour": _finite_float(abs(np.min(stage_diff))) if stage_diff.size else 0.0,
        "precipitation_total": _finite_float(np.sum(precip)),
        "precipitation_max": _finite_float(np.max(precip)),
        "precipitation_mean": _finite_float(np.mean(precip)),
        "precipitation_active_hours": float(wet_rows * dt_h),
        "precipitation_peak_lead_hours": _finite_float(time_h[peak_precip_idx]) if time_h.size else 0.0,
        "peak_precip_to_peak_stage_lag_hours": _finite_float((peak_stage_idx - peak_precip_idx) * dt_h),
    }


def build_forecast_descriptors(
    *,
    raw_summary: Mapping[str, object],
    calibrated_summary: Mapping[str, object],
    comparison_summary: Mapping[str, object],
) -> dict[str, float | int | None]:
    threshold_deltas = comparison_summary.get(
        "delta_peak_expected_area_fraction_wettable_percentage_points_by_threshold_m", {}
    )
    max_abs_delta = 0.0
    if isinstance(threshold_deltas, Mapping):
        numeric = [abs(float(v)) for v in threshold_deltas.values() if _is_number(v)]
        max_abs_delta = max(numeric) if numeric else 0.0
    raw_peak_fraction = _maybe_float(
        raw_summary.get("peak_expected_flooded_area_fraction_wettable_gt_0.05m")
    )
    cal_peak_fraction = _maybe_float(
        calibrated_summary.get("peak_expected_flooded_area_fraction_wettable_gt_0.05m")
    )
    descriptors: dict[str, float | int | None] = {
        "calibrated_max_mean_wd_m": _maybe_float(calibrated_summary.get("max_mean_wd_m")),
        "raw_max_mean_wd_m": _maybe_float(raw_summary.get("max_mean_wd_m")),
        "peak_expected_flooded_area_km2_gt_0.05m": _maybe_float(
            calibrated_summary.get("peak_expected_flooded_area_km2_gt_0.05m")
        ),
        "peak_expected_flooded_area_fraction_wettable_gt_0.05m": cal_peak_fraction,
        "raw_peak_expected_flooded_area_fraction_wettable_gt_0.05m": raw_peak_fraction,
        "peak_area_weighted_iqr_wd_m": _maybe_float(calibrated_summary.get("peak_area_weighted_iqr_wd_m")),
        "peak_area_weighted_central_90_wd_m": _maybe_float(
            calibrated_summary.get("peak_area_weighted_central_90_wd_m")
        ),
        "uncertainty_to_signal_ratio": _maybe_float(calibrated_summary.get("uncertainty_to_signal_ratio")),
        "calibration_peak_area_shift_percentage_points_gt_0.05m": _maybe_float(
            comparison_summary.get(
                "delta_peak_expected_flooded_area_fraction_wettable_percentage_points_gt_0.05m"
            )
        ),
        "max_abs_calibration_shift_percentage_points_by_threshold": max_abs_delta,
    }
    descriptors.update(_threshold_forecast_descriptors(calibrated_summary))
    return descriptors


def _threshold_forecast_descriptors(summary: Mapping[str, object]) -> dict[str, float | None]:
    nested = summary.get("exceedance_by_threshold_m")
    if not isinstance(nested, Mapping):
        return {}
    descriptors: dict[str, float | None] = {}
    for raw_thr, raw_metrics in nested.items():
        if not isinstance(raw_metrics, Mapping):
            continue
        threshold = _maybe_float(raw_metrics.get("threshold_m"))
        if threshold is None:
            threshold = _maybe_float(raw_thr)
        if threshold is None:
            continue
        suffix = _threshold_suffix(threshold)
        descriptors[f"peak_expected_area_fraction_wettable_gt_{suffix}m"] = _maybe_float(
            raw_metrics.get("peak_expected_area_fraction_wettable")
        )
        descriptors[f"peak_expected_area_km2_gt_{suffix}m"] = _maybe_float(
            raw_metrics.get("peak_expected_area_km2")
        )
        descriptors[f"peak_high_confidence_area_fraction_wettable_gt_{suffix}m"] = _maybe_float(
            raw_metrics.get("peak_high_confidence_area_fraction_wettable")
        )
        descriptors[f"peak_expected_area_lead_hours_gt_{suffix}m"] = _maybe_float(
            raw_metrics.get("peak_expected_area_lead_hours")
        )
    return descriptors


def _threshold_suffix(threshold: float) -> str:
    text = f"{float(threshold):g}"
    return text if "." in text else f"{text}.0"


def build_monitoring_bundle_from_forcings(
    *,
    bundle_id: str,
    forcings: Iterable[ForcingInput],
    provenance: Mapping[str, object] | None = None,
    transform_spec: Mapping[str, str] | None = None,
    covariance_exclude_descriptors: Sequence[str] | None = None,
    heuristic_reference_percentiles: Mapping[str, dict[str, float]] | None = None,
) -> dict[str, object]:
    rows = [build_forcing_descriptors(forcing) for forcing in forcings]
    if not rows:
        raise MonitoringError("At least one reference forcing is required.")
    keys = sorted({key for row in rows for key, value in row.items() if _is_number(value)})
    percentiles = {}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows if _is_number(row.get(key))], dtype=np.float64)
        percentiles[key] = _percentile_payload(values)

    ts = dict(transform_spec or {})
    exclude = list(covariance_exclude_descriptors or [])

    forcing_covariance = _compute_covariance_block(rows, keys, ts, exclude)

    payload: dict[str, object] = {
        "bundle_id": bundle_id,
        "monitoring_bundle_schema_version": 2 if forcing_covariance else 1,
        "descriptor_schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "descriptor_percentiles": percentiles,
        "forecast_descriptor_percentiles": {},
        "transform_spec": ts,
        "reference_population": {"n_reference_forcing": len(rows)},
        "covariance_exclude_descriptors": exclude,
        "thresholds": {
            "warning_percentile_low": "p05",
            "warning_percentile_high": "p95",
            "candidate_percentile_low": "p01",
            "candidate_percentile_high": "p99",
            "candidate_score_threshold": 0.8,
            "control_sample_modulus": 0,
        },
        "provenance": dict(provenance or {}),
    }
    if forcing_covariance:
        payload["forcing_covariance"] = forcing_covariance
    if heuristic_reference_percentiles:
        payload["heuristic_reference_percentiles"] = dict(heuristic_reference_percentiles)
    return payload


def _compute_covariance_block(
    rows: list[dict[str, float | int | None]],
    keys: list[str],
    transform_spec: dict[str, str],
    exclude: list[str],
) -> dict[str, object] | None:
    """Compute regularized covariance + empirical Mahalanobis distance percentiles."""
    cov_keys = [k for k in keys if k not in set(exclude)]
    if len(cov_keys) < 2 or len(rows) <= len(cov_keys):
        return None

    transformed_rows: list[list[float]] = []
    for row in rows:
        numeric = {k: float(row.get(k, 0) or 0) for k in cov_keys}
        t = _apply_transforms(numeric, transform_spec)
        transformed_rows.append([t[k] for k in cov_keys])

    X = np.array(transformed_rows, dtype=np.float64)
    mean_vec = np.mean(X, axis=0)
    cov_mat = np.cov(X, rowvar=False)
    p = len(cov_keys)
    alpha = 0.01 * float(np.trace(cov_mat)) / p
    cov_reg = cov_mat + alpha * np.eye(p)
    inv_cov = np.linalg.inv(cov_reg)

    dists: list[float] = []
    for row_vec in X:
        diff = row_vec - mean_vec
        d2 = float(diff @ inv_cov @ diff)
        dists.append(math.sqrt(max(0.0, d2)))
    dist_pcts = np.percentile(dists, [50, 75, 90, 95, 99])

    return {
        "descriptor_names": cov_keys,
        "mean_vector": mean_vec.tolist(),
        "inv_covariance_matrix": inv_cov.tolist(),
        "n_reference": len(rows),
        "regularization_alpha": float(alpha),
        "empirical_distance_percentiles": {
            "p50": float(dist_pcts[0]),
            "p75": float(dist_pcts[1]),
            "p90": float(dist_pcts[2]),
            "p95": float(dist_pcts[3]),
            "p99": float(dist_pcts[4]),
        },
    }


def _build_report(
    *,
    run_spec: RunSpec,
    phase: MonitoringPhase,
    monitoring_bundle: MonitoringBundle,
    descriptors: dict[str, float | int | None],
    score_prefix: str,
) -> MonitoringReport:
    reference = (
        monitoring_bundle.descriptor_percentiles
        if phase == MonitoringPhase.PRE_RUN
        else monitoring_bundle.forecast_descriptor_percentiles
    )
    score, flags = _score_descriptors(descriptors, reference, monitoring_bundle)
    mv_prefix = "forcing" if phase == MonitoringPhase.PRE_RUN else "forecast"
    mv_score, mv_flags = _score_multivariate(descriptors, monitoring_bundle, mv_prefix)
    heuristic_score, heuristic_flags = (
        _post_run_reference_score(descriptors, monitoring_bundle) if phase == MonitoringPhase.POST_RUN else (0.0, [])
    )
    flags = [*flags, *mv_flags, *heuristic_flags]
    score = max(score, mv_score, heuristic_score)
    control = _is_control_sample(run_spec.input_hash, monitoring_bundle.control_sample_modulus)
    candidate = score >= monitoring_bundle.candidate_score_threshold or any(
        flag.severity == "candidate" for flag in flags
    )
    reason = None
    if candidate:
        reason = flags[0].code if flags else "candidate_score_threshold"
    elif control:
        candidate = True
        reason = "deterministic_control_sample"
        flags.append(
            MonitoringFlag(
                code="deterministic_control_sample",
                message="Deterministic control sample selected for retraining-set balance.",
                severity="info",
            )
        )
    return MonitoringReport(
        report_id=_report_id(run_spec.run_id, phase),
        run_id=run_spec.run_id,
        user_id=run_spec.user_id,
        phase=phase,
        monitoring_bundle_id=monitoring_bundle.bundle_id,
        input_hash=run_spec.input_hash,
        descriptors=descriptors,
        scores={
            f"{score_prefix}_novelty_score" if phase == MonitoringPhase.PRE_RUN else "forecast_monitoring_score": score,
            "candidate_score": score if candidate else (0.05 if control else score),
        },
        flags=tuple(flags),
        candidate_recommended=candidate,
        candidate_reason=reason,
    )


def _apply_transforms(
    descriptors: dict[str, float],
    transform_spec: dict[str, str],
) -> dict[str, float]:
    eps = 0.001
    result: dict[str, float] = {}
    for key, value in descriptors.items():
        transform = transform_spec.get(key, "identity")
        if transform == "log1p":
            result[key] = math.log1p(value)
        elif transform == "logit_bounded":
            clamped = max(eps, min(1 - eps, value))
            result[key] = math.log(clamped / (1 - clamped))
        elif transform == "identity":
            result[key] = value
        else:
            result[key] = value
    return result


def _score_multivariate(
    descriptors: Mapping[str, float | int | None],
    bundle: MonitoringBundle,
    phase_prefix: str,
) -> tuple[float, list[MonitoringFlag]]:
    covariance = bundle.forcing_covariance if phase_prefix == "forcing" else bundle.forecast_covariance
    if covariance is None:
        return 0.0, []

    desc_names: list[str] = covariance["descriptor_names"]
    mean_vec: list[float] = covariance["mean_vector"]
    inv_cov: list[list[float]] = covariance["inv_covariance_matrix"]
    empirical: dict[str, float] = covariance["empirical_distance_percentiles"]

    aligned: list[float] = []
    for name in desc_names:
        val = descriptors.get(name)
        if val is None:
            return 0.0, []
        aligned.append(float(val))

    transformed = _apply_transforms(dict(zip(desc_names, aligned)), bundle.transform_spec)
    x = [transformed[n] for n in desc_names]

    p = len(x)
    diff = [x[i] - mean_vec[i] for i in range(p)]

    d_sq = 0.0
    for i in range(p):
        for j in range(p):
            d_sq += diff[i] * inv_cov[i][j] * diff[j]
    d = math.sqrt(max(0.0, d_sq))

    if d < empirical.get("p90", float("inf")):
        score = 0.0
    elif d < empirical.get("p95", float("inf")):
        score = 0.4
    elif d < empirical.get("p99", float("inf")):
        score = 0.6
    else:
        score = 1.0

    if score == 0.0:
        return 0.0, []

    diag = [inv_cov[i][i] for i in range(p)]
    std_devs = []
    for i in range(p):
        sd = 1.0 / math.sqrt(diag[i]) if diag[i] > 0 else 1.0
        std_devs.append((desc_names[i], abs(diff[i]) / sd))
    std_devs.sort(key=lambda t: t[1], reverse=True)
    top_3 = std_devs[:3]

    flag = MonitoringFlag(
        code="multivariate_outlier",
        message=(
            f"Mahalanobis distance {d:.2f} exceeds empirical "
            f"{'p99' if score >= 1.0 else 'p95' if score >= 0.6 else 'p90'}. "
            f"Largest standardized deviations: "
            + ", ".join(f"{n}={z:.2f}" for n, z in top_3)
        ),
        severity="candidate" if score >= 1.0 else "warning",
        details={
            "d_mahal": d,
            "empirical_percentile_exceeded": "p99" if score >= 1.0 else "p95" if score >= 0.6 else "p90",
            "top_3_standardized_deviations": {n: z for n, z in top_3},
        },
    )
    return score, [flag]


def _score_descriptors(
    descriptors: Mapping[str, float | int | None],
    reference: Mapping[str, Mapping[str, float]],
    bundle: MonitoringBundle,
) -> tuple[float, list[MonitoringFlag]]:
    score = 0.0
    flags: list[MonitoringFlag] = []
    for key, raw in descriptors.items():
        if not _is_number(raw):
            continue
        percentiles = reference.get(key)
        if not percentiles:
            continue
        value = float(raw)
        warn_low = percentiles.get(bundle.warning_percentile_low)
        warn_high = percentiles.get(bundle.warning_percentile_high)
        cand_low = percentiles.get(bundle.candidate_percentile_low)
        cand_high = percentiles.get(bundle.candidate_percentile_high)
        if cand_low is not None and value < cand_low:
            score = max(score, 1.0)
            flags.append(_reference_flag(key, value, cand_low, cand_high, "below_candidate_reference", "candidate"))
        elif cand_high is not None and value > cand_high:
            score = max(score, 1.0)
            flags.append(_reference_flag(key, value, cand_low, cand_high, "above_candidate_reference", "candidate"))
        elif warn_low is not None and value < warn_low:
            score = max(score, 0.6)
            flags.append(_reference_flag(key, value, warn_low, warn_high, "below_reference_warning", "warning"))
        elif warn_high is not None and value > warn_high:
            score = max(score, 0.6)
            flags.append(_reference_flag(key, value, warn_low, warn_high, "above_reference_warning", "warning"))
    return score, flags


def _post_run_reference_score(
    descriptors: Mapping[str, float | int | None],
    bundle: MonitoringBundle,
) -> tuple[float, list[MonitoringFlag]]:
    ref = bundle.heuristic_reference_percentiles
    if ref is None:
        return _post_run_heuristic_score_legacy(descriptors)

    flags: list[MonitoringFlag] = []
    score = 0.0
    ratio = _maybe_float(descriptors.get("uncertainty_to_signal_ratio"))
    iqr = _maybe_float(descriptors.get("peak_area_weighted_iqr_wd_m"))
    area_fraction = _maybe_float(descriptors.get("peak_expected_flooded_area_fraction_wettable_gt_0.05m")) or 0.0
    shift = abs(_maybe_float(descriptors.get("max_abs_calibration_shift_percentage_points_by_threshold")) or 0.0)

    ratio_ref = ref.get("uncertainty_to_signal_ratio", {})
    ratio_p95 = ratio_ref.get("p95", 0.5)
    ratio_p99 = ratio_ref.get("p99", 0.5)
    if ratio is not None and ratio >= ratio_p95:
        s = 1.0 if ratio >= ratio_p99 else 0.6
        score = max(score, s)
        flags.append(MonitoringFlag(
            code="high_uncertainty_to_signal",
            message="Forecast has high ensemble uncertainty relative to predicted signal.",
            severity="candidate" if s >= 1.0 else "warning",
            descriptor="uncertainty_to_signal_ratio",
            value=ratio,
            reference_high=ratio_p95,
        ))

    iqr_area_ref = ref.get("iqr_area_product", {})
    iqr_area_p95 = iqr_area_ref.get("p95", 0.02)
    iqr_area_p99 = iqr_area_ref.get("p99", 0.02)
    if iqr is not None:
        iqr_area_product = iqr * area_fraction
        if iqr_area_product >= iqr_area_p95:
            s = 1.0 if iqr_area_product >= iqr_area_p99 else 0.6
            score = max(score, s)
            flags.append(MonitoringFlag(
                code="high_impact_high_uncertainty",
                message="Forecast combines substantial affected area with nontrivial uncertainty width.",
                severity="candidate" if s >= 1.0 else "warning",
                descriptor="peak_area_weighted_iqr_wd_m",
                value=iqr,
                reference_high=iqr_area_p95,
            ))

    shift_ref = ref.get("max_abs_calibration_shift_pp", {})
    shift_p95 = shift_ref.get("p95", 5.0)
    shift_p99 = shift_ref.get("p99", 5.0)
    if shift >= shift_p95:
        s = 1.0 if shift >= shift_p99 else 0.6
        score = max(score, s)
        flags.append(MonitoringFlag(
            code="large_calibration_shift",
            message="Calibration materially shifted the exceedance footprint.",
            severity="candidate" if s >= 1.0 else "warning",
            descriptor="max_abs_calibration_shift_percentage_points_by_threshold",
            value=shift,
            reference_high=shift_p95,
        ))

    return score, flags


def _post_run_heuristic_score_legacy(
    descriptors: Mapping[str, float | int | None],
) -> tuple[float, list[MonitoringFlag]]:
    flags: list[MonitoringFlag] = []
    score = 0.0
    ratio = _maybe_float(descriptors.get("uncertainty_to_signal_ratio"))
    iqr = _maybe_float(descriptors.get("peak_area_weighted_iqr_wd_m"))
    shift = abs(_maybe_float(descriptors.get("max_abs_calibration_shift_percentage_points_by_threshold")) or 0.0)
    area_fraction = _maybe_float(descriptors.get("peak_expected_flooded_area_fraction_wettable_gt_0.05m")) or 0.0
    if ratio is not None and ratio >= 0.5:
        score = max(score, 0.9)
        flags.append(
            MonitoringFlag(
                code="high_uncertainty_to_signal",
                message="Forecast has high ensemble uncertainty relative to predicted signal.",
                severity="candidate",
                descriptor="uncertainty_to_signal_ratio",
                value=ratio,
                reference_high=0.5,
            )
        )
    if iqr is not None and area_fraction >= 0.20 and iqr >= 0.10:
        score = max(score, 0.85)
        flags.append(
            MonitoringFlag(
                code="high_impact_high_uncertainty",
                message="Forecast combines substantial affected area with nontrivial uncertainty width.",
                severity="candidate",
                descriptor="peak_area_weighted_iqr_wd_m",
                value=iqr,
                reference_high=0.10,
            )
        )
    if shift >= 5.0:
        score = max(score, 0.85)
        flags.append(
            MonitoringFlag(
                code="large_calibration_shift",
                message="Calibration materially shifted the exceedance footprint.",
                severity="candidate",
                descriptor="max_abs_calibration_shift_percentage_points_by_threshold",
                value=shift,
                reference_high=5.0,
            )
        )
    return score, flags


def _reference_flag(
    descriptor: str,
    value: float,
    low: float | None,
    high: float | None,
    code: str,
    severity: str,
) -> MonitoringFlag:
    if code.startswith("above"):
        message = f"{descriptor} is above the reference envelope for train/test events."
    else:
        message = f"{descriptor} is below the reference envelope for train/test events."
    return MonitoringFlag(
        code=code,
        message=message,
        severity=severity,
        descriptor=descriptor,
        value=value,
        reference_low=low,
        reference_high=high,
    )


def _latest_payload(reports: Sequence[MonitoringReport], phase: MonitoringPhase) -> dict[str, object] | None:
    matching = [report for report in reports if report.phase == phase]
    return matching[-1].public_payload() if matching else None


def _merge_reason(a: str, b: str) -> str:
    parts = []
    for item in [*a.split(","), *b.split(",")]:
        item = item.strip()
        if item and item not in parts:
            parts.append(item)
    return ", ".join(parts)


def _report_candidate_score(report: MonitoringReport) -> float:
    value = report.scores.get("candidate_score")
    return float(value) if _is_number(value) else 0.0


def _report_id(run_id: str, phase: MonitoringPhase) -> str:
    digest = hashlib.sha256(f"{run_id}:{phase.value}".encode("utf-8")).hexdigest()[:16]
    return f"monitoring_{phase.value.lower()}_{digest}"


def _is_control_sample(input_hash: str, modulus: int) -> bool:
    if int(modulus) <= 0 or not input_hash:
        return False
    try:
        value = int(str(input_hash)[:12], 16)
    except ValueError:
        value = int(hashlib.sha256(str(input_hash).encode("utf-8")).hexdigest()[:12], 16)
    return value % int(modulus) == 0


def _coerce_percentiles(raw: object) -> dict[str, dict[str, float]]:
    if not isinstance(raw, Mapping):
        return {}
    coerced: dict[str, dict[str, float]] = {}
    for key, value in raw.items():
        if not isinstance(value, Mapping):
            continue
        coerced[str(key)] = {str(k): float(v) for k, v in value.items() if _is_number(v)}
    return coerced


def _validate_percentile_entry(name: str, percentiles: Mapping[str, float]) -> None:
    for required in ("p01", "p05", "p50", "p95", "p99"):
        if required not in percentiles:
            raise MonitoringError(f"Descriptor {name!r} is missing percentile {required}.")
    ordered = [float(percentiles[key]) for key in ("p01", "p05", "p50", "p95", "p99")]
    if ordered != sorted(ordered):
        raise MonitoringError(f"Descriptor {name!r} percentiles must be monotone.")


def _percentile_payload(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def _finite_float(value: object) -> float:
    out = float(value)
    if not math.isfinite(out):
        return 0.0
    return out


def _maybe_float(value: object) -> float | None:
    if not _is_number(value):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value))


def _content_type_for_artifact(artifact_id: str) -> str:
    if artifact_id.endswith(".json"):
        return "application/json"
    if artifact_id.endswith(".csv"):
        return "text/csv"
    if artifact_id.endswith(".svg"):
        return "image/svg+xml"
    if artifact_id.endswith(".png"):
        return "image/png"
    if artifact_id.endswith(".gif"):
        return "image/gif"
    if artifact_id.endswith((".h5", ".hdf5")):
        return "application/x-hdf5"
    return "application/octet-stream"


def copy_candidate_tree(src: Path, dst: Path) -> None:
    """Utility for operators/tests: copy candidate packages without run cleanup."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

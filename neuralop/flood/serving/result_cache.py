"""Exact-match result cache for repeated FGN serving runs.

The cache is deliberately scientific rather than byte-oriented: two uploads
match only when their parsed forcing arrays and all output-affecting run
options match. User-facing run records stay private; cached scientific
artifacts are copied into a new owned run directory.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Iterable, Mapping, Protocol, Sequence

from neuralop.flood.serving.forcing import ForcingInput
from neuralop.flood.serving.model_bundle import FGNModelBundle
from neuralop.flood.serving.run_spec import RunSpec
from neuralop.flood.serving.storage import ArtifactRef, ArtifactStore


RESULT_CACHE_SCHEMA_VERSION = 1
RESULT_CACHE_PRODUCT_SCHEMA_VERSION = "fgn-serving-products-v1"

_USER_SCOPED_ARTIFACTS = {
    "forcing.csv",
    "run_manifest.json",
    "cache_manifest.json",
    "forcing_descriptors.json",
    "forecast_descriptors.json",
    "monitoring_report_pre_run.json",
    "monitoring_report_post_run.json",
    "performance_timing.json",
}


class ResultCacheEntryStatus(str, Enum):
    ACTIVE = "ACTIVE"
    READY = "READY"
    FAILED = "FAILED"


class ResultCacheRunRole(str, Enum):
    PRODUCER = "PRODUCER"
    WAITER = "WAITER"
    HIT = "HIT"


class ResultCacheRunStatus(str, Enum):
    ACTIVE = "ACTIVE"
    WAITING = "WAITING"
    MATERIALIZED = "MATERIALIZED"
    FAILED = "FAILED"


class ResultCacheLookupStatus(str, Enum):
    MISS = "MISS"
    HIT = "HIT"
    WAITING = "WAITING"


@dataclass(frozen=True)
class ResultCacheEntry:
    cache_key: str
    status: ResultCacheEntryStatus
    producer_run_id: str | None
    artifact_manifest: tuple[str, ...] = ()
    schema_version: int = RESULT_CACHE_SCHEMA_VERSION
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class ResultCacheRunLink:
    run_id: str
    cache_key: str
    role: ResultCacheRunRole
    status: ResultCacheRunStatus
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class ResultCacheReservation:
    status: ResultCacheLookupStatus
    cache_key: str
    entry: ResultCacheEntry | None = None


@dataclass(frozen=True)
class ResultCacheFailureResolution:
    cache_key: str | None
    promoted_run_id: str | None = None
    waiting_run_ids: tuple[str, ...] = ()


class ResultCacheRepository(Protocol):
    def reserve_or_find(self, cache_key: str, producer_run_id: str) -> ResultCacheReservation: ...
    def publish_ready(self, producer_run_id: str, artifact_manifest: Sequence[str]) -> ResultCacheEntry | None: ...
    def handle_producer_failed(self, producer_run_id: str) -> ResultCacheFailureResolution: ...
    def list_waiting_runs(self, cache_key: str) -> list[ResultCacheRunLink]: ...
    def mark_materialized(self, run_id: str, cache_key: str, *, role: ResultCacheRunRole) -> ResultCacheRunLink: ...
    def link_for_run(self, run_id: str) -> ResultCacheRunLink | None: ...
    def entry_for_key(self, cache_key: str) -> ResultCacheEntry | None: ...


class InMemoryResultCacheRepository:
    """Thread-safe test/development cache repository adapter."""

    def __init__(self) -> None:
        self._entries: dict[str, ResultCacheEntry] = {}
        self._links: dict[str, ResultCacheRunLink] = {}
        self._lock = RLock()

    def reserve_or_find(self, cache_key: str, producer_run_id: str) -> ResultCacheReservation:
        now = datetime.now(timezone.utc)
        with self._lock:
            entry = self._entries.get(cache_key)
            if entry and entry.status == ResultCacheEntryStatus.READY:
                self._links[producer_run_id] = ResultCacheRunLink(
                    run_id=producer_run_id,
                    cache_key=cache_key,
                    role=ResultCacheRunRole.HIT,
                    status=ResultCacheRunStatus.WAITING,
                    created_at=now,
                    updated_at=now,
                )
                return ResultCacheReservation(ResultCacheLookupStatus.HIT, cache_key, entry)
            if entry and entry.status == ResultCacheEntryStatus.ACTIVE and entry.producer_run_id != producer_run_id:
                self._links[producer_run_id] = ResultCacheRunLink(
                    run_id=producer_run_id,
                    cache_key=cache_key,
                    role=ResultCacheRunRole.WAITER,
                    status=ResultCacheRunStatus.WAITING,
                    created_at=now,
                    updated_at=now,
                )
                return ResultCacheReservation(ResultCacheLookupStatus.WAITING, cache_key, entry)
            next_entry = ResultCacheEntry(
                cache_key=cache_key,
                status=ResultCacheEntryStatus.ACTIVE,
                producer_run_id=producer_run_id,
                created_at=entry.created_at if entry else now,
                updated_at=now,
            )
            self._entries[cache_key] = next_entry
            self._links[producer_run_id] = ResultCacheRunLink(
                run_id=producer_run_id,
                cache_key=cache_key,
                role=ResultCacheRunRole.PRODUCER,
                status=ResultCacheRunStatus.ACTIVE,
                created_at=self._links.get(producer_run_id, ResultCacheRunLink(
                    run_id=producer_run_id,
                    cache_key=cache_key,
                    role=ResultCacheRunRole.PRODUCER,
                    status=ResultCacheRunStatus.ACTIVE,
                )).created_at,
                updated_at=now,
            )
            return ResultCacheReservation(ResultCacheLookupStatus.MISS, cache_key, next_entry)

    def publish_ready(self, producer_run_id: str, artifact_manifest: Sequence[str]) -> ResultCacheEntry | None:
        now = datetime.now(timezone.utc)
        with self._lock:
            link = self._links.get(producer_run_id)
            if link is None:
                return None
            entry = self._entries.get(link.cache_key)
            if entry is None or entry.producer_run_id != producer_run_id:
                return None
            ready = replace(
                entry,
                status=ResultCacheEntryStatus.READY,
                artifact_manifest=tuple(dict.fromkeys(str(x) for x in artifact_manifest)),
                updated_at=now,
            )
            self._entries[link.cache_key] = ready
            self._links[producer_run_id] = replace(
                link,
                role=ResultCacheRunRole.PRODUCER,
                status=ResultCacheRunStatus.MATERIALIZED,
                updated_at=now,
            )
            return ready

    def handle_producer_failed(self, producer_run_id: str) -> ResultCacheFailureResolution:
        now = datetime.now(timezone.utc)
        with self._lock:
            link = self._links.get(producer_run_id)
            if link is None:
                return ResultCacheFailureResolution(cache_key=None)
            entry = self._entries.get(link.cache_key)
            if entry is None or entry.producer_run_id != producer_run_id:
                return ResultCacheFailureResolution(cache_key=link.cache_key)
            waiters = self.list_waiting_runs(link.cache_key)
            self._links[producer_run_id] = replace(link, status=ResultCacheRunStatus.FAILED, updated_at=now)
            if not waiters:
                self._entries[link.cache_key] = replace(
                    entry,
                    status=ResultCacheEntryStatus.FAILED,
                    producer_run_id=None,
                    updated_at=now,
                )
                return ResultCacheFailureResolution(cache_key=link.cache_key)
            promoted = waiters[0]
            self._entries[link.cache_key] = replace(
                entry,
                status=ResultCacheEntryStatus.ACTIVE,
                producer_run_id=promoted.run_id,
                artifact_manifest=(),
                updated_at=now,
            )
            self._links[promoted.run_id] = replace(
                promoted,
                role=ResultCacheRunRole.PRODUCER,
                status=ResultCacheRunStatus.ACTIVE,
                updated_at=now,
            )
            return ResultCacheFailureResolution(
                cache_key=link.cache_key,
                promoted_run_id=promoted.run_id,
                waiting_run_ids=tuple(waiter.run_id for waiter in waiters[1:]),
            )

    def list_waiting_runs(self, cache_key: str) -> list[ResultCacheRunLink]:
        with self._lock:
            return [
                link
                for link in self._links.values()
                if link.cache_key == cache_key and link.role == ResultCacheRunRole.WAITER and link.status == ResultCacheRunStatus.WAITING
            ]

    def mark_materialized(self, run_id: str, cache_key: str, *, role: ResultCacheRunRole) -> ResultCacheRunLink:
        now = datetime.now(timezone.utc)
        with self._lock:
            current = self._links.get(run_id)
            link = ResultCacheRunLink(
                run_id=run_id,
                cache_key=cache_key,
                role=role,
                status=ResultCacheRunStatus.MATERIALIZED,
                created_at=current.created_at if current else now,
                updated_at=now,
            )
            self._links[run_id] = link
            return link

    def link_for_run(self, run_id: str) -> ResultCacheRunLink | None:
        with self._lock:
            return self._links.get(run_id)

    def entry_for_key(self, cache_key: str) -> ResultCacheEntry | None:
        with self._lock:
            return self._entries.get(cache_key)


class ResultCache:
    """Coordinates cache repository records with artifact storage."""

    def __init__(
        self,
        *,
        repository: ResultCacheRepository,
        run_artifact_store: ArtifactStore,
        cache_artifact_store: ArtifactStore,
    ) -> None:
        self.repository = repository
        self.run_artifact_store = run_artifact_store
        self.cache_artifact_store = cache_artifact_store

    def reserve_or_find(self, cache_key: str, producer_run_id: str) -> ResultCacheReservation:
        return self.repository.reserve_or_find(cache_key, producer_run_id)

    def publish_completed(self, producer_run_id: str) -> ResultCacheEntry | None:
        link = self.repository.link_for_run(producer_run_id)
        if link is None or link.role != ResultCacheRunRole.PRODUCER:
            return None
        copied: list[str] = []
        for ref in self.run_artifact_store.list(producer_run_id):
            if not _is_cacheable_artifact(ref.artifact_id):
                continue
            _copy_artifact_between_stores(
                source_store=self.run_artifact_store,
                target_store=self.cache_artifact_store,
                source_run_id=producer_run_id,
                target_run_id=link.cache_key,
                artifact_id=ref.artifact_id,
                content_type=ref.content_type,
            )
            copied.append(ref.artifact_id)
        manifest = self.repository.publish_ready(producer_run_id, copied)
        self.cache_artifact_store.put_json(
            link.cache_key,
            "result_cache_manifest",
            {
                "cache_key_prefix": link.cache_key[:12],
                "schema_version": RESULT_CACHE_SCHEMA_VERSION,
                "product_schema_version": RESULT_CACHE_PRODUCT_SCHEMA_VERSION,
                "artifact_ids": copied,
                "published_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        return manifest

    def materialize_hit(self, target_run_id: str, cache_key: str, *, role: ResultCacheRunRole) -> tuple[str, ...]:
        started = time.perf_counter()
        entry = self.repository.entry_for_key(cache_key)
        if entry is None or entry.status != ResultCacheEntryStatus.READY:
            raise RuntimeError(f"Result cache entry is not ready for key {cache_key[:12]}.")
        cache_refs = {ref.artifact_id: ref for ref in self.cache_artifact_store.list(cache_key)}
        copied: list[str] = []
        for artifact_id in entry.artifact_manifest:
            content_type = (
                cache_refs[artifact_id].content_type if artifact_id in cache_refs else "application/octet-stream"
            )
            self._copy_cache_artifact(cache_key, target_run_id, artifact_id, content_type=content_type)
            copied.append(artifact_id)
        elapsed = time.perf_counter() - started
        self.run_artifact_store.put_json(
            target_run_id,
            "cache_manifest",
            {
                "materialized_from_cache": True,
                "cache_key_prefix": cache_key[:12],
                "schema_version": RESULT_CACHE_SCHEMA_VERSION,
                "product_schema_version": RESULT_CACHE_PRODUCT_SCHEMA_VERSION,
                "artifact_ids": copied,
                "materialized_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        self.run_artifact_store.put_json(
            target_run_id,
            "performance_timing",
            {
                "cache": {
                    "materialized_from_cache": True,
                    "cache_key_prefix": cache_key[:12],
                    "materialized_artifact_count": len(copied),
                },
                "phases": {"cache_materialization": {"seconds": elapsed}},
            },
        )
        self.repository.mark_materialized(target_run_id, cache_key, role=role)
        return tuple(copied)

    def handle_producer_failed(self, producer_run_id: str) -> ResultCacheFailureResolution:
        return self.repository.handle_producer_failed(producer_run_id)

    def waiting_runs_for_key(self, cache_key: str) -> list[ResultCacheRunLink]:
        return self.repository.list_waiting_runs(cache_key)

    def metadata_for_run(self, run_id: str, *, include_admin: bool = False) -> dict[str, object]:
        link = self.repository.link_for_run(run_id)
        if link is None:
            return {
                "enabled": True,
                "mode": "none",
                "materialized_from_cache": False,
                "waiting_for_cached_result": False,
            }
        entry = self.repository.entry_for_key(link.cache_key)
        role = link.role.value.lower()
        payload: dict[str, object] = {
            "enabled": True,
            "mode": role,
            "status": link.status.value,
            "entry_status": entry.status.value if entry else None,
            "materialized_from_cache": link.role in {ResultCacheRunRole.HIT, ResultCacheRunRole.WAITER}
            and link.status == ResultCacheRunStatus.MATERIALIZED,
            "waiting_for_cached_result": link.status == ResultCacheRunStatus.WAITING,
            "cache_key_prefix": link.cache_key[:12],
        }
        if include_admin:
            payload["cache_key"] = link.cache_key
            payload["producer_run_id"] = entry.producer_run_id if entry else None
        return payload

    def _copy_cache_artifact(
        self,
        cache_key: str,
        target_run_id: str,
        artifact_id: str,
        *,
        content_type: str,
    ) -> ArtifactRef:
        return _copy_artifact_between_stores(
            source_store=self.cache_artifact_store,
            target_store=self.run_artifact_store,
            source_run_id=cache_key,
            target_run_id=target_run_id,
            artifact_id=artifact_id,
            content_type=content_type,
        )


def build_result_cache_key(
    *,
    run_spec: RunSpec,
    forcing_input: ForcingInput,
    bundle: FGNModelBundle,
    product_schema_version: str = RESULT_CACHE_PRODUCT_SCHEMA_VERSION,
) -> str:
    payload = {
        "schema_version": RESULT_CACHE_SCHEMA_VERSION,
        "product_schema_version": product_schema_version,
        "forcing": {
            "stage": _float_list(forcing_input.stage),
            "precipitation": _float_list(forcing_input.precipitation),
            "dt_seconds": int(forcing_input.dt_seconds),
            "forecast_steps": int(forcing_input.forecast_steps),
        },
        "run": {
            "forecast_steps": int(run_spec.forecast_steps),
            "output_detail": run_spec.output_detail,
            "request_full_hdf5": bool(run_spec.request_full_hdf5),
            "request_animation": bool(run_spec.request_animation),
            "ensemble_count": int(run_spec.ensemble_count),
            "members_per_ensemble": int(run_spec.members_per_ensemble),
            "calibration_mode": run_spec.calibration_mode,
            "exceedance_thresholds_m": [float(x) for x in run_spec.exceedance_thresholds_m],
            "seed": int(run_spec.seed),
        },
        "bundle": {
            "bundle_id": bundle.bundle_id,
            "git_commit": bundle.git_commit,
            "mesh_hash": bundle.mesh_hash,
            "calibration_version": _calibration_identity(bundle),
            "initial_condition": _initial_condition_identity(bundle),
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _float_list(values) -> list[float]:
    return [float(f"{float(value):.8g}") for value in values]


def _calibration_identity(bundle: FGNModelBundle) -> dict[str, str]:
    return {
        "coefficients": _path_identity(bundle.calibration_coefficients_path),
        "isotonic": _path_identity(bundle.isotonic_curves_path),
    }


def _initial_condition_identity(bundle: FGNModelBundle) -> dict[str, object]:
    config = getattr(bundle, "initial_condition", None)
    if config is None:
        payload: dict[str, object] = {}
    elif hasattr(config, "public_metadata"):
        payload = dict(config.public_metadata())
        if getattr(config, "library_path", None) is not None:
            payload["library_path"] = str(config.library_path)
    elif isinstance(config, Mapping):
        payload = dict(config)
    else:
        payload = {"repr": repr(config)}
    library_path = payload.get("library_path")
    if library_path:
        payload["library_identity"] = _path_identity(Path(library_path))
    return payload


def _path_identity(path: str | Path) -> str:
    p = Path(path)
    try:
        stat = p.stat()
    except OSError:
        return str(path)
    return f"{p.name}:{stat.st_size}:{int(stat.st_mtime_ns)}"


def _is_cacheable_artifact(artifact_id: str) -> bool:
    return artifact_id not in _USER_SCOPED_ARTIFACTS and not artifact_id.startswith("monitoring_")


def _copy_artifact_between_stores(
    *,
    source_store: ArtifactStore,
    target_store: ArtifactStore,
    source_run_id: str,
    target_run_id: str,
    artifact_id: str,
    content_type: str,
) -> ArtifactRef:
    """Copy a cached artifact without routing large files through Python bytes.

    Local production stores keep run artifacts and cache artifacts on the same
    data root. In that case a hardlink is effectively instant and still safe:
    deleting a user's run path only unlinks that user's directory entry, while
    the shared cache package remains available. If hardlinks are unavailable
    (different filesystems, Windows mount limits, permissions), fall back to
    shutil.copy2. Non-local test stores keep the byte-oriented protocol path.
    """
    source_path_fn = getattr(source_store, "_artifact_path", None)
    target_path_fn = getattr(target_store, "_artifact_path", None)
    if callable(source_path_fn) and callable(target_path_fn):
        source_path = Path(source_path_fn(source_run_id, artifact_id))
        target_path = Path(target_path_fn(target_run_id, artifact_id))
        if not source_path.exists() or not source_path.is_file():
            raise FileNotFoundError(f"Cache source artifact not found: {artifact_id}")
        if target_path.exists():
            target_path.unlink()
        try:
            target_path.hardlink_to(source_path)
        except OSError:
            shutil.copy2(source_path, target_path)
        return ArtifactRef(target_run_id, artifact_id, target_path, content_type, target_path.stat().st_size)
    return target_store.put_bytes(
        target_run_id,
        artifact_id,
        source_store.read_bytes(source_run_id, artifact_id),
        content_type=content_type,
    )


class LocalCacheArtifactStore:
    """Filesystem artifact adapter for cache keys.

    This mirrors ``LocalArtifactStore`` but keeps cache entries in a separate
    root so user run deletion and retention cleanup never remove shared cache
    packages.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def put_bytes(self, run_id: str, artifact_id: str, data: bytes, *, content_type: str) -> ArtifactRef:
        path = self._artifact_path(run_id, artifact_id)
        path.write_bytes(data)
        return ArtifactRef(run_id, artifact_id, path, content_type, path.stat().st_size)

    def put_json(self, run_id: str, artifact_id: str, payload: object) -> ArtifactRef:
        if not artifact_id.endswith(".json"):
            artifact_id = f"{artifact_id}.json"
        data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        return self.put_bytes(run_id, artifact_id, data, content_type="application/json")

    def list(self, run_id: str) -> Iterable[ArtifactRef]:
        from neuralop.flood.serving.storage import _content_type_for_suffix

        path = self._run_dir(run_id)
        refs: list[ArtifactRef] = []
        for item in sorted(path.iterdir()):
            if item.is_file():
                refs.append(ArtifactRef(run_id, item.name, item, _content_type_for_suffix(item), item.stat().st_size))
        return refs

    def read_bytes(self, run_id: str, artifact_id: str) -> bytes:
        path = self._artifact_path(run_id, artifact_id)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Cache artifact not found: {artifact_id}")
        return path.read_bytes()

    def delete_run_artifacts(self, run_id: str) -> None:
        path = self.root / Path(run_id).name
        if path.exists():
            shutil.rmtree(path)

    def _run_dir(self, run_id: str) -> Path:
        safe = Path(run_id).name
        if safe != run_id:
            raise ValueError("cache key may not contain path separators.")
        path = self.root / safe
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _artifact_path(self, run_id: str, artifact_id: str) -> Path:
        safe = Path(artifact_id).name
        if safe != artifact_id:
            raise ValueError("artifact_id may not contain path separators.")
        return self._run_dir(run_id) / safe

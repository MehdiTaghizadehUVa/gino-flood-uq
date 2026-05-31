"""Resolve "what normalizer should training use right now?" — once, for every trainer.

This module owns the decision matrix every flood training entry point should
follow when looking at the normalizer cache on disk. Keeping the logic in one
place is the load-bearing safety property: getting it wrong means silently
miscalibrating a resumed model against a stale normalizer (the bug that burned
the production diffusion ens02 and ens03 runs in May 2026).

Public API:

- :class:`NormalizerResolutionSource` – enum naming each branch of the matrix.
- :class:`NormalizerResolution` – the resolver's return type.
- :func:`resolve_normalizer_artifact` – the resolver itself.

The resolver is dependency-injected so callers can swap in any compatible
``fit_normalizers`` / ``save_normalizers`` / ``load_normalizers`` /
``save_normalizer_metadata`` / ``load_normalizer_metadata`` implementations
(production code wires it to the real ones in ``neuralop.flood.data``; tests
wire it to spies). This keeps the module a pure-Python decision engine —
testable without torch.

Decision matrix (precedence order):

  1. No cache on disk                          -> fit + save                  (REFIT_FRESH)
  2. Metadata MATCHES                          -> load cache                  (CACHED_EXACT)
  3. Metadata MISMATCH + is_resuming           -> load cache, warn            (CACHED_FOR_RESUME)
  4. Metadata MISMATCH + force_load            -> load cache, warn            (CACHED_FORCE_LOADED)
  5. Metadata MISMATCH + neither               -> snapshot, refit, save       (REFIT_REPLACED)

Distributed safety: callers pass ``rank0=True`` for the rank-0 worker and
``rank0=False`` for every other. Non-rank-0 invocations never fit or write —
they return a resolution with ``normalizers=None``, and the caller is
responsible for waiting on rank-0 to finish writing the artifact.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Optional


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class NormalizerResolutionSource(str, Enum):
    """Names each decision-matrix outcome so callers can branch on the source."""

    CACHED_EXACT = "cached_exact"
    CACHED_FOR_RESUME = "cached_for_resume"
    CACHED_FORCE_LOADED = "cached_force_loaded"
    REFIT_FRESH = "refit_fresh"
    REFIT_REPLACED = "refit_replaced"


@dataclass(frozen=True)
class NormalizerResolution:
    """Outcome of resolving a normalizer artifact against a train split."""

    normalizers: Any
    fit_method: str
    source: NormalizerResolutionSource
    mismatch_keys: tuple[str, ...] = ()
    snapshot_path: Optional[Path] = None
    metadata_written: Optional[dict[str, Any]] = field(default=None, compare=False)


# ---------------------------------------------------------------------------
# Internal helpers (no I/O outside the injected callables; snapshot is the
# one exception because it must atomically copy bytes on disk).
# ---------------------------------------------------------------------------


_DEFAULT_PATH_EXISTS = Path.exists  # bound classmethod, called as fn(path)


def _exists(path: Optional[Path], path_exists_fn: Callable[[Path], bool]) -> bool:
    return path is not None and path_exists_fn(path)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _snapshot_artifact(path: Path, *, logger: logging.Logger) -> Optional[Path]:
    """Copy ``path`` to a ``.pre_refit_<UTC>`` sibling. Returns the snapshot path or None on failure."""
    snapshot = path.with_name(f"{path.name}.pre_refit_{_utc_stamp()}")
    try:
        shutil.copy2(path, snapshot)
    except OSError as exc:  # pragma: no cover - filesystem-specific
        logger.error(
            "Failed to snapshot normalizer artifact %s before refit: %s", path, exc
        )
        return None
    return snapshot


def _format_mismatch_summary(mismatch_keys: tuple[str, ...]) -> str:
    return ",".join(mismatch_keys) if mismatch_keys else "<none>"


def _metadata_mismatch_keys(
    expected: Mapping[str, Any], actual: Optional[Mapping[str, Any]]
) -> tuple[str, ...]:
    """Return the keys whose values differ. ``<missing_metadata>`` flags a
    fully-absent on-disk metadata so the caller can distinguish "different"
    from "we have no idea what's there".
    """
    if actual is None:
        return ("<missing_metadata>",)
    diffs = [key for key in expected if actual.get(key) != expected.get(key)]
    return tuple(diffs)


# ---------------------------------------------------------------------------
# The resolver
# ---------------------------------------------------------------------------


def resolve_normalizer_artifact(
    *,
    # --- inputs the trainer already has ---
    train_data_raw: Any,
    normalizer_path: Optional[Path],
    metadata_path: Optional[Path],
    expected_metadata: Mapping[str, Any],
    fit_method: str,
    structural_dry_policy: Any,
    chunk_size: int,
    expect_target: bool,
    # --- contextual flags ---
    is_resuming: bool,
    force_load: bool,
    rank0: bool,
    logger: logging.Logger,
    # --- dependencies (injected for testability) ---
    fit_normalizers_fn: Callable[..., Any],
    save_normalizers_fn: Callable[[Any, Path], None],
    load_normalizers_fn: Callable[..., Any],
    save_normalizer_metadata_fn: Callable[[Path, Mapping[str, Any]], Any],
    load_normalizer_metadata_fn: Callable[[Path], Optional[Mapping[str, Any]]],
    path_exists_fn: Callable[[Path], bool] = _DEFAULT_PATH_EXISTS,
) -> NormalizerResolution:
    """Resolve the normalizer to use for this training/eval session.

    See the module docstring for the decision matrix. The function is
    side-effect-minimal: it only touches disk via the injected callables and,
    in the snapshot branch, a stdlib copy of the existing normalizer file.

    Args:
        train_data_raw: Source dataset passed straight to
            ``fit_normalizers_fn``; the lifecycle never inspects it.
        normalizer_path / metadata_path: Where the cached normalizer artifact
            and its metadata sidecar live, if persistence is desired. Either
            (or both) may be ``None`` — see test
            ``test_no_normalizer_path_means_no_persistence`` for the contract.
        expected_metadata: Metadata describing the current train split.
            Compared against the on-disk metadata to decide compatibility.
        fit_method / structural_dry_policy / chunk_size / expect_target:
            Forwarded to ``fit_normalizers_fn`` unmodified.
        is_resuming: True when the caller plans to load a model checkpoint
            after the normalizer is resolved. The load-bearing flag — when
            True the resolver MUST NOT refit-and-overwrite an existing cache
            even if metadata disagrees, because the model on disk is calibrated
            against the cache that's there.
        force_load: Operator override accepting "use the cache even though
            metadata says it's stale". Implies ``is_resuming=False`` semantics
            for warnings.
        rank0: True for the rank-0 worker. Non-rank-0 workers return
            ``normalizers=None`` and never write to disk.

    Returns:
        :class:`NormalizerResolution` describing what happened.
    """
    cache_exists = _exists(normalizer_path, path_exists_fn)
    metadata_exists = _exists(metadata_path, path_exists_fn)
    on_disk_metadata = (
        load_normalizer_metadata_fn(metadata_path)
        if metadata_path is not None and metadata_exists
        else None
    )

    # ---- Outcome 1: no cache on disk -> fit + save (or refuse on non-rank0)
    if not cache_exists:
        return _refit(
            train_data_raw=train_data_raw,
            normalizer_path=normalizer_path,
            metadata_path=metadata_path,
            expected_metadata=expected_metadata,
            fit_method=fit_method,
            structural_dry_policy=structural_dry_policy,
            chunk_size=chunk_size,
            expect_target=expect_target,
            rank0=rank0,
            logger=logger,
            fit_normalizers_fn=fit_normalizers_fn,
            save_normalizers_fn=save_normalizers_fn,
            save_normalizer_metadata_fn=save_normalizer_metadata_fn,
            source=NormalizerResolutionSource.REFIT_FRESH,
            mismatch_keys=(),
            snapshot_path=None,
        )

    # Cache exists. Decide whether to trust it.
    mismatch_keys = _metadata_mismatch_keys(expected_metadata, on_disk_metadata)
    metadata_matches = not mismatch_keys

    # ---- Outcome 2: metadata matches -> load cache, done.
    if metadata_matches:
        normalizers = load_normalizers_fn(normalizer_path, device=None)
        logger.info(
            "Loaded normalizers from %s (method=%s)", normalizer_path, fit_method
        )
        return NormalizerResolution(
            normalizers=normalizers,
            fit_method=fit_method,
            source=NormalizerResolutionSource.CACHED_EXACT,
            mismatch_keys=(),
        )

    # ---- Outcome 3: mismatch + is_resuming -> KEEP cache. THE FIX.
    if is_resuming:
        normalizers = load_normalizers_fn(normalizer_path, device=None)
        logger.warning(
            "Cached normalizer at %s has metadata fingerprint mismatch "
            "(mismatch_keys=%s) BUT a resume checkpoint is in play. "
            "Keeping the cached normalizer to preserve model–normalizer "
            "consistency; not refitting. If this drift is unexpected, "
            "verify the checkpoint and the on-disk artifact were produced "
            "by the same training run.",
            normalizer_path,
            _format_mismatch_summary(mismatch_keys),
        )
        return NormalizerResolution(
            normalizers=normalizers,
            fit_method=fit_method,
            source=NormalizerResolutionSource.CACHED_FOR_RESUME,
            mismatch_keys=mismatch_keys,
        )

    # ---- Outcome 4: mismatch + force_load -> load cache anyway.
    if force_load:
        normalizers = load_normalizers_fn(normalizer_path, device=None)
        logger.warning(
            "Loading normalizer from %s despite fingerprint mismatch "
            "(mismatch_keys=%s) because force_load=True was set by the "
            "operator. Downstream metrics may be biased.",
            normalizer_path,
            _format_mismatch_summary(mismatch_keys),
        )
        return NormalizerResolution(
            normalizers=normalizers,
            fit_method=fit_method,
            source=NormalizerResolutionSource.CACHED_FORCE_LOADED,
            mismatch_keys=mismatch_keys,
        )

    # ---- Outcome 5: mismatch + neither resume nor force -> snapshot, refit, save.
    snapshot_path: Optional[Path] = None
    if rank0 and normalizer_path is not None and cache_exists:
        snapshot_path = _snapshot_artifact(normalizer_path, logger=logger)
        if snapshot_path is not None:
            logger.warning(
                "Refitting normalizer due to fingerprint mismatch "
                "(mismatch_keys=%s); previous artifact snapshotted to %s.",
                _format_mismatch_summary(mismatch_keys),
                snapshot_path,
            )
    return _refit(
        train_data_raw=train_data_raw,
        normalizer_path=normalizer_path,
        metadata_path=metadata_path,
        expected_metadata=expected_metadata,
        fit_method=fit_method,
        structural_dry_policy=structural_dry_policy,
        chunk_size=chunk_size,
        expect_target=expect_target,
        rank0=rank0,
        logger=logger,
        fit_normalizers_fn=fit_normalizers_fn,
        save_normalizers_fn=save_normalizers_fn,
        save_normalizer_metadata_fn=save_normalizer_metadata_fn,
        source=NormalizerResolutionSource.REFIT_REPLACED,
        mismatch_keys=mismatch_keys,
        snapshot_path=snapshot_path,
    )


def _refit(
    *,
    train_data_raw: Any,
    normalizer_path: Optional[Path],
    metadata_path: Optional[Path],
    expected_metadata: Mapping[str, Any],
    fit_method: str,
    structural_dry_policy: Any,
    chunk_size: int,
    expect_target: bool,
    rank0: bool,
    logger: logging.Logger,
    fit_normalizers_fn: Callable[..., Any],
    save_normalizers_fn: Callable[[Any, Path], None],
    save_normalizer_metadata_fn: Callable[[Path, Mapping[str, Any]], Any],
    source: NormalizerResolutionSource,
    mismatch_keys: tuple[str, ...],
    snapshot_path: Optional[Path],
) -> NormalizerResolution:
    """Shared implementation of outcomes 1 and 5: fit + (rank0) save."""
    if not rank0:
        # Distributed safety: only rank-0 may write the canonical artifact.
        # Caller is responsible for waiting on rank-0 to finish before
        # invoking load_normalizers_fn on the artifact.
        return NormalizerResolution(
            normalizers=None,
            fit_method=fit_method,
            source=source,
            mismatch_keys=mismatch_keys,
            snapshot_path=snapshot_path,
        )

    normalizers, resolved_fit_method = fit_normalizers_fn(
        train_data_raw,
        chunk_size=chunk_size,
        expect_target=expect_target,
        structural_dry_policy=structural_dry_policy,
        method=fit_method,
        return_method=True,
    )
    metadata_written: Optional[dict[str, Any]] = None
    if normalizer_path is not None:
        save_normalizers_fn(normalizers, normalizer_path)
        if metadata_path is not None:
            metadata_written = dict(expected_metadata)
            metadata_written["fit_method"] = resolved_fit_method
            save_normalizer_metadata_fn(metadata_path, metadata_written)
        logger.info(
            "Saved normalizers to %s (method=%s)", normalizer_path, resolved_fit_method
        )
    return NormalizerResolution(
        normalizers=normalizers,
        fit_method=resolved_fit_method,
        source=source,
        mismatch_keys=mismatch_keys,
        snapshot_path=snapshot_path,
        metadata_written=metadata_written,
    )

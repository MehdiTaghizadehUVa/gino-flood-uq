"""Unit tests for neuralop.flood.data.normalizer_lifecycle.

The module owns the decision matrix every training entry point should follow
when resolving "what normalizer should I use right now?". The matrix is the
load-bearing safety contract for resume: if it is wrong, training silently
recalibrates the model against a stale normalizer (the bug that burned the
production diffusion ens02 and ens03 runs).

Decision matrix (precedence order; tested individually below):

  1. No cache on disk                          -> fit + save                  (refit_fresh)
  2. Metadata MATCHES                          -> load cache                  (cached_exact)
  3. Metadata MISMATCH + is_resuming           -> load cache, warn loudly     (cached_for_resume)
  4. Metadata MISMATCH + force_load            -> load cache, warn            (cached_force_loaded)
  5. Metadata MISMATCH + neither               -> snapshot cache, refit + save (refit_replaced)

Every test injects fake fit/save/load functions so the suite does not depend on
torch, numpy, or any dataset machinery.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from neuralop.flood.data.normalizer_lifecycle import (
    NormalizerResolution,
    NormalizerResolutionSource,
    resolve_normalizer_artifact,
)


# ----------------------------------------------------------------------------
# Test doubles
# ----------------------------------------------------------------------------


@dataclass
class _FakeNormalizer:
    """Stand-in for the normalizer object — tests only check identity."""

    tag: str


@dataclass
class _Spies:
    """Records every call the lifecycle code makes against its dependencies."""

    fit_calls: List[Dict[str, Any]]
    save_normalizer_calls: List[Tuple[Path, str]]
    load_normalizer_calls: List[Path]
    save_metadata_calls: List[Tuple[Path, Dict[str, Any]]]
    load_metadata_calls: List[Path]


def _build_fakes(
    *,
    on_disk_metadata: Dict[str, Any] | None,
    cached_normalizer_tag: str = "cached",
    fit_method_out: str = "fast_exact",
) -> Tuple[_Spies, Dict[str, Any]]:
    spies = _Spies(
        fit_calls=[],
        save_normalizer_calls=[],
        load_normalizer_calls=[],
        save_metadata_calls=[],
        load_metadata_calls=[],
    )

    def fake_fit(train_data_raw, **kwargs):
        spies.fit_calls.append({"data": train_data_raw, **kwargs})
        return _FakeNormalizer(tag="fresh"), fit_method_out

    def fake_save_normalizers(normalizer, path):
        spies.save_normalizer_calls.append((Path(path), normalizer.tag))

    def fake_load_normalizers(path, device=None):
        spies.load_normalizer_calls.append(Path(path))
        return _FakeNormalizer(tag=cached_normalizer_tag)

    def fake_save_metadata(path, metadata):
        spies.save_metadata_calls.append((Path(path), dict(metadata)))

    def fake_load_metadata(path):
        spies.load_metadata_calls.append(Path(path))
        return on_disk_metadata

    deps = {
        "fit_normalizers_fn": fake_fit,
        "save_normalizers_fn": fake_save_normalizers,
        "load_normalizers_fn": fake_load_normalizers,
        "save_normalizer_metadata_fn": fake_save_metadata,
        "load_normalizer_metadata_fn": fake_load_metadata,
    }
    return spies, deps


def _resolve(
    *,
    normalizer_path: Path | None,
    metadata_path: Path | None,
    expected_metadata: Dict[str, Any],
    is_resuming: bool = False,
    force_load: bool = False,
    on_disk_metadata: Dict[str, Any] | None = None,
    cache_exists: bool = False,
    metadata_exists: bool = False,
    cached_normalizer_tag: str = "cached",
    fit_method_in: str = "fast_exact",
    logger: logging.Logger | None = None,
) -> Tuple[NormalizerResolution, _Spies]:
    """Tiny harness over resolve_normalizer_artifact for readable test bodies."""
    spies, deps = _build_fakes(
        on_disk_metadata=on_disk_metadata,
        cached_normalizer_tag=cached_normalizer_tag,
    )

    def existence(path: Path | None) -> bool:
        if path is None:
            return False
        if path == normalizer_path:
            return cache_exists
        if path == metadata_path:
            return metadata_exists
        return path.exists()

    resolution = resolve_normalizer_artifact(
        train_data_raw="<sentinel-train-raw>",
        normalizer_path=normalizer_path,
        metadata_path=metadata_path,
        expected_metadata=expected_metadata,
        fit_method=fit_method_in,
        structural_dry_policy="masked_primary",
        chunk_size=10_000,
        expect_target=True,
        is_resuming=is_resuming,
        force_load=force_load,
        rank0=True,
        logger=logger or logging.getLogger("test"),
        path_exists_fn=existence,
        **deps,
    )
    return resolution, spies


def _expected_meta(**overrides: Any) -> Dict[str, Any]:
    base = {
        "fit_method": "fast_exact",
        "structural_dry_policy": "masked_primary",
        "split_fingerprint": "expected-split",
        "boundary_spec_fingerprint": "expected-bs",
        "code_version": "abc",
    }
    base.update(overrides)
    return base


# ----------------------------------------------------------------------------
# Decision matrix tests, one per outcome
# ----------------------------------------------------------------------------


def test_no_cache_on_disk_fits_and_saves(tmp_path: Path):
    """Outcome 1: pure fresh run -> fit + save + write metadata."""
    norm_path = tmp_path / "norm.pt"
    meta_path = tmp_path / "norm.metadata.json"
    expected = _expected_meta()

    resolution, spies = _resolve(
        normalizer_path=norm_path,
        metadata_path=meta_path,
        expected_metadata=expected,
        cache_exists=False,
        metadata_exists=False,
    )

    assert resolution.source is NormalizerResolutionSource.REFIT_FRESH
    assert resolution.normalizers.tag == "fresh"
    assert resolution.fit_method == "fast_exact"
    assert resolution.mismatch_keys == ()
    assert resolution.snapshot_path is None
    assert len(spies.fit_calls) == 1
    assert spies.save_normalizer_calls == [(norm_path, "fresh")]
    assert spies.save_metadata_calls and spies.save_metadata_calls[0][0] == meta_path


def test_metadata_matches_loads_cache(tmp_path: Path):
    """Outcome 2: cache + metadata + match -> load only, no fit, no save."""
    norm_path = tmp_path / "norm.pt"
    meta_path = tmp_path / "norm.metadata.json"
    expected = _expected_meta()

    resolution, spies = _resolve(
        normalizer_path=norm_path,
        metadata_path=meta_path,
        expected_metadata=expected,
        on_disk_metadata=expected.copy(),
        cache_exists=True,
        metadata_exists=True,
    )

    assert resolution.source is NormalizerResolutionSource.CACHED_EXACT
    assert resolution.normalizers.tag == "cached"
    assert resolution.mismatch_keys == ()
    assert spies.fit_calls == []
    assert spies.save_normalizer_calls == []
    assert spies.save_metadata_calls == []
    assert spies.load_normalizer_calls == [norm_path]


def test_mismatch_during_resume_keeps_cache(tmp_path: Path, caplog):
    """Outcome 3 (the bug fix): mismatch + is_resuming -> load cache, do NOT
    refit, do NOT overwrite. This is the load-bearing safety property: every
    other path can be slightly wrong without burning a training run, but this
    one cannot."""
    norm_path = tmp_path / "norm.pt"
    meta_path = tmp_path / "norm.metadata.json"
    expected = _expected_meta(split_fingerprint="EXPECTED")
    drifted = _expected_meta(split_fingerprint="ON_DISK_DRIFTED")

    with caplog.at_level(logging.WARNING):
        resolution, spies = _resolve(
            normalizer_path=norm_path,
            metadata_path=meta_path,
            expected_metadata=expected,
            on_disk_metadata=drifted,
            cache_exists=True,
            metadata_exists=True,
            is_resuming=True,
        )

    assert resolution.source is NormalizerResolutionSource.CACHED_FOR_RESUME
    assert resolution.normalizers.tag == "cached"
    assert resolution.mismatch_keys == ("split_fingerprint",)
    assert resolution.snapshot_path is None
    # Refit must not have run.
    assert spies.fit_calls == []
    # The cache must not have been overwritten.
    assert spies.save_normalizer_calls == []
    assert spies.save_metadata_calls == []
    # A clear warning must reach the operator.
    assert any("split_fingerprint" in m for m in caplog.messages)
    assert any("resume" in m.lower() for m in caplog.messages)


def test_mismatch_with_force_load_keeps_cache(tmp_path: Path, caplog):
    """Outcome 4: mismatch + force_load -> load cache, warn, no fit."""
    norm_path = tmp_path / "norm.pt"
    meta_path = tmp_path / "norm.metadata.json"
    expected = _expected_meta(split_fingerprint="EXPECTED")
    drifted = _expected_meta(split_fingerprint="ON_DISK_DRIFTED")

    with caplog.at_level(logging.WARNING):
        resolution, spies = _resolve(
            normalizer_path=norm_path,
            metadata_path=meta_path,
            expected_metadata=expected,
            on_disk_metadata=drifted,
            cache_exists=True,
            metadata_exists=True,
            force_load=True,
        )

    assert resolution.source is NormalizerResolutionSource.CACHED_FORCE_LOADED
    assert resolution.normalizers.tag == "cached"
    assert resolution.mismatch_keys == ("split_fingerprint",)
    assert spies.fit_calls == []
    assert spies.save_normalizer_calls == []
    assert any("force_load" in m.lower() for m in caplog.messages)


def test_mismatch_without_resume_or_force_snapshots_then_refits(tmp_path: Path):
    """Outcome 5: mismatch + neither -> snapshot existing artifact, refit, save.

    The snapshot rule is the safety net: even when the lifecycle decides to
    overwrite, the previous artifact is recoverable. This prevents the
    diffusion-style 'we overwrote your only normalizer file' loss.
    """
    norm_path = tmp_path / "norm.pt"
    norm_path.write_bytes(b"OLD_NORMALIZER_BYTES")
    meta_path = tmp_path / "norm.metadata.json"
    meta_path.write_text(json.dumps(_expected_meta(split_fingerprint="ON_DISK")))

    expected = _expected_meta(split_fingerprint="EXPECTED")
    on_disk = _expected_meta(split_fingerprint="ON_DISK")

    resolution, spies = _resolve(
        normalizer_path=norm_path,
        metadata_path=meta_path,
        expected_metadata=expected,
        on_disk_metadata=on_disk,
        cache_exists=True,
        metadata_exists=True,
    )

    assert resolution.source is NormalizerResolutionSource.REFIT_REPLACED
    assert resolution.normalizers.tag == "fresh"
    assert "split_fingerprint" in resolution.mismatch_keys
    assert resolution.snapshot_path is not None
    assert resolution.snapshot_path.exists()
    # The snapshot must contain the original bytes — no data loss.
    assert resolution.snapshot_path.read_bytes() == b"OLD_NORMALIZER_BYTES"
    # Refit and save must have happened, in that order.
    assert len(spies.fit_calls) == 1
    assert spies.save_normalizer_calls == [(norm_path, "fresh")]


# ----------------------------------------------------------------------------
# Edge cases that turn into real incidents if left unspecified
# ----------------------------------------------------------------------------


def test_resume_with_no_cache_falls_through_to_refit(tmp_path: Path):
    """If the cache file is missing on a resume (e.g., scratch wipe), the
    lifecycle must NOT silently load nothing. Outcome 1 applies even on
    resume: fit and save."""
    norm_path = tmp_path / "norm.pt"
    meta_path = tmp_path / "norm.metadata.json"

    resolution, spies = _resolve(
        normalizer_path=norm_path,
        metadata_path=meta_path,
        expected_metadata=_expected_meta(),
        cache_exists=False,
        metadata_exists=False,
        is_resuming=True,
    )

    assert resolution.source is NormalizerResolutionSource.REFIT_FRESH
    assert len(spies.fit_calls) == 1


def test_resume_when_metadata_file_missing_but_normalizer_present_treats_as_mismatch(tmp_path: Path):
    """An older normalizer artifact lacking a metadata sidecar should be
    treated as a mismatch (we cannot prove compatibility). Under resume, that
    means we keep the cache and warn — never silently refit."""
    norm_path = tmp_path / "norm.pt"
    norm_path.write_bytes(b"old")
    meta_path = tmp_path / "norm.metadata.json"

    resolution, _ = _resolve(
        normalizer_path=norm_path,
        metadata_path=meta_path,
        expected_metadata=_expected_meta(),
        on_disk_metadata=None,
        cache_exists=True,
        metadata_exists=False,
        is_resuming=True,
    )

    assert resolution.source is NormalizerResolutionSource.CACHED_FOR_RESUME
    assert resolution.mismatch_keys == ("<missing_metadata>",)


def test_non_rank0_does_not_fit_or_save(tmp_path: Path):
    """In a distributed run only rank-0 should ever fit. The lifecycle must
    not silently call fit_normalizers from a non-rank-0 worker even when the
    decision matrix says 'refit', because that would race with rank-0."""
    norm_path = tmp_path / "norm.pt"
    meta_path = tmp_path / "norm.metadata.json"
    expected = _expected_meta()

    spies, deps = _build_fakes(on_disk_metadata=None)

    resolution = resolve_normalizer_artifact(
        train_data_raw="<sentinel-train-raw>",
        normalizer_path=norm_path,
        metadata_path=meta_path,
        expected_metadata=expected,
        fit_method="fast_exact",
        structural_dry_policy="masked_primary",
        chunk_size=10_000,
        expect_target=True,
        is_resuming=False,
        force_load=False,
        rank0=False,
        logger=logging.getLogger("test"),
        path_exists_fn=lambda p: False,
        **deps,
    )

    assert resolution.normalizers is None
    assert resolution.source is NormalizerResolutionSource.REFIT_FRESH
    assert spies.fit_calls == []
    assert spies.save_normalizer_calls == []


def test_no_normalizer_path_means_no_persistence(tmp_path: Path):
    """If the caller passes normalizer_path=None we still must return a
    normalizer (fit it), but we must never try to save."""
    resolution, spies = _resolve(
        normalizer_path=None,
        metadata_path=None,
        expected_metadata=_expected_meta(),
    )

    assert resolution.source is NormalizerResolutionSource.REFIT_FRESH
    assert resolution.normalizers.tag == "fresh"
    assert spies.save_normalizer_calls == []
    assert spies.save_metadata_calls == []

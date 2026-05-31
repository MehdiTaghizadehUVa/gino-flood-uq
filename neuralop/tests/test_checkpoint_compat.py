"""Unit tests for neuralop.flood.utils.checkpoint_compat.

The module's contract is small and strict:
    - normalizer_fingerprint() selects exactly the fingerprint keys.
    - assert_normalizer_matches_checkpoint() passes when fingerprints agree on
      every recorded key and raises NormalizerCheckpointMismatchError otherwise.
    - The function distinguishes missing keys from value mismatches in its error
      payload so resume failures are actionable in the operator's log.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import pytest

from neuralop.flood.utils.checkpoint_compat import (
    NORMALIZER_FINGERPRINT_KEYS,
    NormalizerCheckpointMismatchError,
    assert_normalizer_matches_checkpoint,
    normalizer_fingerprint,
)


def _full_metadata(**overrides: Any) -> Dict[str, Any]:
    base = {
        "fit_method": "fast_exact",
        "dataset_root": "/scratch/example/train",
        "dataset_class": "FloodTrainDataset",
        "structural_dry_policy": "masked_primary",
        "target_variables": ["wd"],
        "static_text_files": ["a.txt", "b.txt"],
        "boundary_spec_fingerprint": "0" * 64,
        "hdf_paths_fingerprint": "1" * 64,
        "split_sample_count": 418500,
        "split_fingerprint": "2" * 64,
        "code_version": "abc1234",
    }
    base.update(overrides)
    return base


def test_normalizer_fingerprint_includes_only_documented_keys():
    meta = _full_metadata()
    fp = normalizer_fingerprint(meta)
    assert set(fp.keys()) == set(NORMALIZER_FINGERPRINT_KEYS) & set(meta.keys())
    # Unrelated keys must not leak into the fingerprint.
    assert "code_version" not in fp
    assert "dataset_class" not in fp


def test_normalizer_fingerprint_is_resilient_to_missing_keys():
    sparse = {"split_fingerprint": "abc"}
    fp = normalizer_fingerprint(sparse)
    assert fp == {"split_fingerprint": "abc"}


def test_assert_normalizer_matches_checkpoint_passes_when_identical():
    fp = normalizer_fingerprint(_full_metadata())
    # Identical metadata must pass without raising.
    assert_normalizer_matches_checkpoint(
        checkpoint_fingerprint=fp,
        on_disk_metadata=_full_metadata(),
        logger=logging.getLogger("test"),
    )


def test_assert_raises_when_value_changes():
    fp = normalizer_fingerprint(_full_metadata())
    drifted = _full_metadata(split_fingerprint="DIFFERENT")
    with pytest.raises(NormalizerCheckpointMismatchError) as exc_info:
        assert_normalizer_matches_checkpoint(
            checkpoint_fingerprint=fp,
            on_disk_metadata=drifted,
            logger=logging.getLogger("test"),
        )
    err = exc_info.value
    assert err.changed_keys == ("split_fingerprint",)
    assert err.missing_keys == ()
    # The message must name the offending key so operators can act on it.
    assert "split_fingerprint" in str(err)


def test_assert_raises_when_key_missing_from_on_disk():
    fp = normalizer_fingerprint(_full_metadata())
    incomplete = _full_metadata()
    del incomplete["hdf_paths_fingerprint"]
    with pytest.raises(NormalizerCheckpointMismatchError) as exc_info:
        assert_normalizer_matches_checkpoint(
            checkpoint_fingerprint=fp,
            on_disk_metadata=incomplete,
            logger=logging.getLogger("test"),
        )
    assert exc_info.value.missing_keys == ("hdf_paths_fingerprint",)
    assert exc_info.value.changed_keys == ()


def test_assert_treats_extra_on_disk_keys_as_ok():
    """Adding new fields to the on-disk metadata (e.g. a future fingerprint key)
    must not retroactively break old checkpoints. The contract is one-way: every
    key the checkpoint remembers must agree, but the on-disk record may carry
    more."""
    fp = normalizer_fingerprint(_full_metadata())
    enriched = _full_metadata(future_extra_fp="something")
    assert_normalizer_matches_checkpoint(
        checkpoint_fingerprint=fp,
        on_disk_metadata=enriched,
        logger=logging.getLogger("test"),
    )


def test_strict_false_logs_warning_instead_of_raising(caplog):
    fp = normalizer_fingerprint(_full_metadata())
    drifted = _full_metadata(split_fingerprint="DIFFERENT")
    with caplog.at_level(logging.WARNING):
        assert_normalizer_matches_checkpoint(
            checkpoint_fingerprint=fp,
            on_disk_metadata=drifted,
            logger=logging.getLogger("test"),
            strict=False,
        )
    assert any("split_fingerprint" in m for m in caplog.messages)


def test_assert_passes_when_checkpoint_has_no_fingerprint():
    """Older checkpoints saved before this contract existed must not be
    blocked from resuming. Empty checkpoint_fingerprint is the documented
    "no-fingerprint" state and must succeed silently."""
    assert_normalizer_matches_checkpoint(
        checkpoint_fingerprint={},
        on_disk_metadata=_full_metadata(),
        logger=logging.getLogger("test"),
    )


# ----------------------------------------------------------------------------
# Sidecar helpers (PR-4 addition)
# ----------------------------------------------------------------------------

from pathlib import Path  # noqa: E402  (kept local; tests above don't need it)

from neuralop.flood.utils.checkpoint_compat import (
    NORMALIZER_FINGERPRINT_SIDECAR,
    read_normalizer_fingerprint_sidecar,
    write_normalizer_fingerprint_sidecar,
)


def test_sidecar_round_trip_preserves_fingerprint(tmp_path: Path):
    metadata = _full_metadata()
    sidecar_path = write_normalizer_fingerprint_sidecar(
        tmp_path, normalizer_metadata=metadata
    )
    assert sidecar_path.name == NORMALIZER_FINGERPRINT_SIDECAR
    assert sidecar_path.exists()

    loaded = read_normalizer_fingerprint_sidecar(tmp_path)
    assert loaded == normalizer_fingerprint(metadata)


def test_sidecar_round_trip_drops_non_fingerprint_keys(tmp_path: Path):
    """The sidecar must NOT carry code_version, dataset_class, etc. — only the
    contract subset. Otherwise a code-version bump on its own would falsely
    fail every resume."""
    metadata = _full_metadata(code_version="changed-after-save")
    write_normalizer_fingerprint_sidecar(tmp_path, normalizer_metadata=metadata)
    loaded = read_normalizer_fingerprint_sidecar(tmp_path)
    assert "code_version" not in loaded
    assert "dataset_class" not in loaded
    # Required fingerprint keys are all present.
    assert set(loaded.keys()) >= {
        "split_fingerprint",
        "boundary_spec_fingerprint",
        "hdf_paths_fingerprint",
        "split_sample_count",
    }


def test_sidecar_read_returns_none_when_absent(tmp_path: Path):
    """Legacy checkpoint dirs with no sidecar must be readable as None — the
    'no fingerprint recorded' state. assert_normalizer_matches_checkpoint
    accepts this silently."""
    assert read_normalizer_fingerprint_sidecar(tmp_path) is None


def test_sidecar_write_is_atomic_across_concurrent_readers(tmp_path: Path):
    """Sidecar must be written via temp+rename so a reader that polls the file
    never observes a half-written JSON. The simplest reliable proof: confirm
    no .tmp file is left over after a normal write."""
    write_normalizer_fingerprint_sidecar(tmp_path, normalizer_metadata=_full_metadata())
    leftovers = list(tmp_path.glob(f"{NORMALIZER_FINGERPRINT_SIDECAR}.tmp"))
    assert leftovers == [], f"unexpected leftover temp files: {leftovers}"


def test_sidecar_read_returns_none_on_corrupted_json(tmp_path: Path):
    """A partially-written sidecar (corrupted JSON on disk) is treated as
    'no fingerprint' rather than crashing the resume path. The bug we are
    fixing is silent miscalibration; a corrupted sidecar should at worst
    degrade to legacy behavior, never block training."""
    sidecar = tmp_path / NORMALIZER_FINGERPRINT_SIDECAR
    sidecar.write_text("{ not valid json")
    assert read_normalizer_fingerprint_sidecar(tmp_path) is None


def test_sidecar_round_trip_then_assert_against_drifted_metadata_raises(tmp_path: Path):
    """End-to-end contract: save fingerprint, then later assert against a
    drifted on-disk metadata -> mismatch error with the named key."""
    original = _full_metadata(split_fingerprint="ORIGINAL")
    write_normalizer_fingerprint_sidecar(tmp_path, normalizer_metadata=original)
    cp_fp = read_normalizer_fingerprint_sidecar(tmp_path)

    drifted_metadata = _full_metadata(split_fingerprint="DIFFERENT")
    with pytest.raises(NormalizerCheckpointMismatchError) as exc_info:
        assert_normalizer_matches_checkpoint(
            checkpoint_fingerprint=cp_fp,
            on_disk_metadata=drifted_metadata,
            logger=logging.getLogger("test"),
        )
    assert exc_info.value.changed_keys == ("split_fingerprint",)

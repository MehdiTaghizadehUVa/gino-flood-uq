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

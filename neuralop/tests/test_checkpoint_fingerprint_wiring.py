"""PR-4 wiring tests: fingerprint sidecar is read+asserted+written by both trainers.

These tests are grep-level — they verify the contract is wired into the
training entry points without needing to spin up a full training run.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest


def _module_source(dotted: str) -> str:
    mod = __import__(dotted, fromlist=["*"])
    src_path = Path(inspect.getsourcefile(mod) or "")
    assert src_path.exists(), f"could not locate source for {dotted!r}: {src_path}"
    return src_path.read_text()


# ----------------------------------------------------------------------------
# Diffusion side
# ----------------------------------------------------------------------------


def test_diffusion_app_writes_fingerprint_sidecar_after_ckpt_dir_mkdir():
    """diffusion_app must call write_normalizer_fingerprint_sidecar after the
    checkpoint dir is created so a *future* resume of this run can verify
    normalizer-checkpoint compatibility."""
    src = _module_source("neuralop.flood.train.diffusion_app")
    assert "write_normalizer_fingerprint_sidecar(" in src, (
        "diffusion_app must call write_normalizer_fingerprint_sidecar() to "
        "pin the active normalizer's fingerprint into the checkpoint dir."
    )


def test_diffusion_app_asserts_fingerprint_on_resume():
    """diffusion_app must call assert_normalizer_matches_checkpoint inside the
    resume_checkpoint branch (before any state loading), so a fingerprint
    drift hard-fails instead of silently miscalibrating the model."""
    src = _module_source("neuralop.flood.train.diffusion_app")
    # The assertion must appear after the resume_checkpoint guard.
    resume_idx = src.find("if resume_checkpoint is not None")
    assert resume_idx != -1, "resume guard not found in diffusion_app.py"
    assert_idx = src.find("assert_normalizer_matches_checkpoint(", resume_idx)
    assert assert_idx != -1, (
        "assert_normalizer_matches_checkpoint must appear inside the resume "
        "branch of diffusion_app."
    )
    # And it must be called BEFORE the resume_bundle is loaded — otherwise
    # the model state would be partially restored against a stale normalizer
    # before we even check.
    bundle_load_idx = src.find("load_checkpoint_bundle(", resume_idx)
    assert bundle_load_idx == -1 or assert_idx < bundle_load_idx, (
        "assert_normalizer_matches_checkpoint must fire BEFORE "
        "load_checkpoint_bundle so the resume hard-fails without touching "
        "model state on mismatch."
    )


# ----------------------------------------------------------------------------
# Operator side
# ----------------------------------------------------------------------------


_OPERATOR_TRAIN_CALL_RE = re.compile(r"^\s+trainer\.train\(\s*$", re.MULTILINE)


def _operator_trainer_train_index(src: str) -> int:
    """Index of the ACTUAL trainer.train(... ) call (not a comment mentioning it).

    The real call is the only one on its own line with an open paren and no
    trailing argument on the same line. Anchoring this way avoids false
    positives from docstring comments containing the substring 'trainer.train('.
    """
    match = _OPERATOR_TRAIN_CALL_RE.search(src)
    assert match, "could not locate the trainer.train(...) call in operator_app.py"
    return match.start()


def test_operator_app_writes_fingerprint_sidecar_before_trainer_train():
    """operator_app must write the fingerprint sidecar before trainer.train()
    so the very first checkpoint Trainer.checkpoint() writes has a sibling
    fingerprint file."""
    src = _module_source("neuralop.flood.train.operator_app")
    write_idx = src.find("write_normalizer_fingerprint_sidecar(")
    train_idx = _operator_trainer_train_index(src)
    assert write_idx != -1, (
        "operator_app must call write_normalizer_fingerprint_sidecar() to "
        "pin the active normalizer's fingerprint into the checkpoint dir."
    )
    assert write_idx < train_idx, (
        "The fingerprint sidecar must be written BEFORE trainer.train() so "
        "the file exists by the time Trainer.checkpoint() runs its first save."
    )


def test_operator_app_asserts_fingerprint_when_resuming():
    """operator_app must call assert_normalizer_matches_checkpoint inside a
    block guarded by checkpoint_resume_dir being non-None — and before
    trainer.train(), so a fingerprint drift hard-fails before any optimizer
    step touches the model."""
    src = _module_source("neuralop.flood.train.operator_app")
    train_idx = _operator_trainer_train_index(src)
    assert_idx = src.find("assert_normalizer_matches_checkpoint(", 0, train_idx)
    assert assert_idx != -1, (
        "operator_app must call assert_normalizer_matches_checkpoint before "
        "trainer.train() so the assertion fires before training resumes."
    )
    # The assertion must be guarded by checkpoint_resume_dir presence so it
    # does not fire on a fresh run.
    guard_idx = src.rfind("if checkpoint_resume_dir is not None", 0, assert_idx)
    assert guard_idx != -1, (
        "The assert must live inside a block guarded by checkpoint_resume_dir "
        "being non-None so fresh runs are unaffected."
    )


# ----------------------------------------------------------------------------
# Sidecar contract end-to-end (no torch needed)
# ----------------------------------------------------------------------------


def test_sidecar_end_to_end_pin_then_verify_then_raise_on_drift(tmp_path: Path):
    """Simulate: a trainer writes the sidecar at save time, a later resume
    reads it, then on-disk normalizer drifts. The contract MUST raise so the
    drift is loud, not silent."""
    from neuralop.flood.utils.checkpoint_compat import (
        NormalizerCheckpointMismatchError,
        assert_normalizer_matches_checkpoint,
        read_normalizer_fingerprint_sidecar,
        write_normalizer_fingerprint_sidecar,
    )
    import logging

    original = {
        "split_fingerprint": "TRAIN_RUN_42",
        "boundary_spec_fingerprint": "bspec",
        "hdf_paths_fingerprint": "hdf",
        "split_sample_count": 418500,
        "code_version": "abc1234",
    }
    write_normalizer_fingerprint_sidecar(tmp_path, normalizer_metadata=original)
    later_resume = read_normalizer_fingerprint_sidecar(tmp_path)
    assert later_resume is not None

    drifted = dict(original)
    drifted["split_fingerprint"] = "DIFFERENT_SPLIT"

    with pytest.raises(NormalizerCheckpointMismatchError) as exc:
        assert_normalizer_matches_checkpoint(
            checkpoint_fingerprint=later_resume,
            on_disk_metadata=drifted,
            logger=logging.getLogger("test"),
        )
    assert exc.value.changed_keys == ("split_fingerprint",)

"""Integration-style tests for the PR-2 wiring of the lifecycle helper into diffusion training.

These tests do not spin up a real torch dataset; they verify:

1. ``_prepare_datasets`` exposes the ``is_resuming`` keyword (so the resume-safety
   contract is reachable from the call site in ``diffusion_app.py``).
2. ``_wait_for_normalizer_artifacts`` defaults to "wait for files to exist"
   without requiring an exact on-disk-metadata match — required so the
   CACHED_FOR_RESUME outcome from the lifecycle does not falsely error out on
   non-rank-0 workers.
3. The strict mode still works when an operator explicitly opts in.
"""

from __future__ import annotations

import inspect
import json
import threading
import time
from pathlib import Path

import pytest

from neuralop.flood.train.diffusion_data import (
    _prepare_datasets,
    _wait_for_normalizer_artifacts,
)


def test_prepare_datasets_exposes_is_resuming_kwarg():
    """The whole point of PR-2 is that the call site can flag resume state.
    If this signature ever loses the kwarg the resume-safety contract is gone."""
    sig = inspect.signature(_prepare_datasets)
    assert "is_resuming" in sig.parameters, sig.parameters
    param = sig.parameters["is_resuming"]
    assert param.kind in {
        inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }
    assert param.default is False, (
        "is_resuming MUST default to False so callers that have not been "
        "migrated yet preserve the historical 'fresh run' behavior."
    )


def test_wait_helper_returns_when_files_appear_even_if_metadata_mismatches(tmp_path: Path):
    """The default (expected_metadata=None) mode must NOT compare on-disk
    metadata to expected. This is what lets CACHED_FOR_RESUME work in
    distributed training: rank-0 deliberately keeps a stale-metadata cache,
    and non-rank-0 must accept it."""
    norm_path = tmp_path / "norm.pt"
    meta_path = tmp_path / "norm.metadata.json"

    # Write a metadata sidecar whose content has nothing to do with any
    # "expected" metadata.
    meta_path.write_text(json.dumps({"fit_method": "fast_exact", "anything": "goes"}))
    norm_path.write_bytes(b"<placeholder normalizer bytes>")

    # Should return quickly without raising; we did not pass expected_metadata.
    _wait_for_normalizer_artifacts(
        norm_path,
        metadata_path=meta_path,
        timeout_seconds=1.0,
        poll_interval_seconds=0.05,
    )


def test_wait_helper_strict_mode_still_errors_on_mismatch(tmp_path: Path):
    """The historical strict-match behavior is opt-in via expected_metadata.

    The wait helper deliberately catches the per-iteration check inside its
    retry loop (the file may be mid-write), so a persistent mismatch surfaces
    as a *timeout* whose ``__cause__`` records the original mismatch error.
    Both layers are part of the contract — the outer timeout signals "I gave
    up", the inner cause names *why* the artifact was unacceptable.
    """
    norm_path = tmp_path / "norm.pt"
    meta_path = tmp_path / "norm.metadata.json"
    meta_path.write_text(json.dumps({"fit_method": "fast_exact", "split_fingerprint": "A"}))
    norm_path.write_bytes(b"<placeholder>")

    with pytest.raises(RuntimeError) as exc_info:
        _wait_for_normalizer_artifacts(
            norm_path,
            metadata_path=meta_path,
            expected_metadata={"fit_method": "fast_exact", "split_fingerprint": "B"},
            timeout_seconds=0.3,
            poll_interval_seconds=0.05,
        )
    cause = exc_info.value.__cause__
    assert cause is not None, "expected the outer timeout to chain the original mismatch error"
    assert "does not match" in str(cause), str(cause)


def test_wait_helper_times_out_when_files_never_appear(tmp_path: Path):
    norm_path = tmp_path / "missing_norm.pt"
    meta_path = tmp_path / "missing_meta.json"
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        _wait_for_normalizer_artifacts(
            norm_path,
            metadata_path=meta_path,
            timeout_seconds=0.3,
            poll_interval_seconds=0.05,
        )
    # Sanity: did not block for orders of magnitude more than the timeout.
    assert time.monotonic() - start < 2.0


def test_wait_helper_retries_until_metadata_becomes_readable(tmp_path: Path):
    """The wait helper must tolerate a brief window during which the metadata
    file exists but is still mid-write (rank-0 just touched it and hasn't
    flushed the JSON body yet). Background writer simulates that race."""
    norm_path = tmp_path / "norm.pt"
    meta_path = tmp_path / "norm.metadata.json"
    norm_path.write_bytes(b"<placeholder>")
    meta_path.write_bytes(b"")  # exists but unreadable as JSON

    def finish_write_after_delay():
        time.sleep(0.15)
        meta_path.write_text(json.dumps({"fit_method": "fast_exact"}))

    threading.Thread(target=finish_write_after_delay, daemon=True).start()

    _wait_for_normalizer_artifacts(
        norm_path,
        metadata_path=meta_path,
        timeout_seconds=2.0,
        poll_interval_seconds=0.05,
    )

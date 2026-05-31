"""PR-5: confirm the legacy primitive ``normalizer_metadata_matches`` is now deprecated.

The deprecation steers new code toward ``resolve_normalizer_artifact``, which
owns the resume-safety contract end-to-end. The legacy primitive is kept for
the small number of legitimate strict-equality use sites (the diffusion wait
helper's opt-in strict mode and two standalone CLI scripts).
"""

from __future__ import annotations

import warnings

import pytest

from neuralop.flood.data.normalization_impl import normalizer_metadata_matches


def _meta(**overrides):
    base = {
        "fit_method": "fast_exact",
        "dataset_root": "/scratch/example",
        "dataset_class": "FloodTrainDataset",
        "structural_dry_policy": "masked_primary",
        "target_variables": ["wd"],
        "static_text_files": [],
        "boundary_spec_fingerprint": "bs",
        "split_sample_count": 100,
        "split_fingerprint": "sf",
        "code_version": "abc",
    }
    base.update(overrides)
    return base


def test_normalizer_metadata_matches_emits_deprecation_warning():
    """A bare call from user code must emit DeprecationWarning. This is the
    'how we steer new code away from the primitive' contract — losing this
    warning means the legacy API silently survives migration."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        normalizer_metadata_matches(_meta(), _meta())
    deprecation_messages = [
        str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert deprecation_messages, "expected DeprecationWarning to be emitted"
    # The deprecation message must direct the reader to the lifecycle helper.
    assert any("resolve_normalizer_artifact" in msg for msg in deprecation_messages), (
        f"deprecation message must reference the replacement; got {deprecation_messages}"
    )


def test_normalizer_metadata_matches_still_returns_correct_decision_under_warning():
    """The primitive still works — the deprecation is a notice, not a removal."""
    a = _meta(split_fingerprint="X")
    b = _meta(split_fingerprint="X")
    c = _meta(split_fingerprint="Y")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        assert normalizer_metadata_matches(a, b) is True
        assert normalizer_metadata_matches(a, c) is False
        assert normalizer_metadata_matches(a, None) is False


def test_diffusion_wait_helper_strict_mode_does_not_propagate_deprecation():
    """The diffusion ``_wait_for_normalizer_artifacts`` strict opt-in
    legitimately calls the deprecated primitive. Its caller must NOT see the
    DeprecationWarning — the wait helper wraps the call with
    ``warnings.catch_warnings()``.
    """
    import json
    from pathlib import Path

    from neuralop.flood.train.diffusion_data import _wait_for_normalizer_artifacts

    # Use a real tmpdir so the helper can read the files it writes.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        norm_path = tmp_path / "n.pt"
        meta_path = tmp_path / "n.metadata.json"
        norm_path.write_bytes(b"placeholder")
        expected = {"fit_method": "fast_exact", "split_fingerprint": "X"}
        meta_path.write_text(json.dumps(expected))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _wait_for_normalizer_artifacts(
                norm_path,
                metadata_path=meta_path,
                expected_metadata=expected,
                timeout_seconds=1.0,
                poll_interval_seconds=0.05,
            )
        deprecations_from_primitive = [
            w
            for w in caught
            if issubclass(w.category, DeprecationWarning)
            and "normalizer_metadata_matches" in str(w.message)
        ]
        assert not deprecations_from_primitive, (
            "diffusion wait helper must suppress the normalizer_metadata_matches "
            f"deprecation noise; got {[str(w.message) for w in deprecations_from_primitive]}"
        )


def test_operator_app_no_longer_imports_normalizer_metadata_matches():
    """Post-PR-3 + PR-5: operator_app.py must NOT import the deprecated symbol
    (it stopped using it). If a future refactor re-adds the import, that's a
    signal someone reintroduced the inline cached-or-refit logic.
    """
    import inspect
    from pathlib import Path

    import neuralop.flood.train.operator_app as mod

    src = Path(inspect.getsourcefile(mod) or "").read_text()
    assert "normalizer_metadata_matches" not in src, (
        "operator_app.py still references normalizer_metadata_matches — either "
        "remove the unused import or migrate the new call site to the lifecycle helper."
    )

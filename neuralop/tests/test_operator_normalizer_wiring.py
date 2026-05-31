"""PR-3 wiring tests for the operator (FGN/MC-dropout/Gaussian) trainer.

These tests verify that ``neuralop.flood.train.operator_app`` integrates with
the shared normalizer lifecycle helper correctly. The operator trainer's
``main()`` is too large to invoke end-to-end here, so the tests focus on the
contract surface that matters for the bug fix:

1. ``operator_app`` imports the lifecycle helper (catches a future refactor
   that accidentally removes the dependency and silently reverts to the old
   inline cached-or-refit branch).
2. The lifecycle helper is the symbol called from inside ``main`` (greppable
   sanity check that the migration didn't get half-applied).
3. The legacy ``normalizer_metadata_matches`` is no longer called from inside
   ``main`` — the lifecycle helper owns the decision now. (The wait helper in
   diffusion_data.py is a different module and is allowed to keep using it
   under PR-5's strict opt-in policy.)
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest


def _operator_app_source() -> str:
    import neuralop.flood.train.operator_app as mod

    src_path = Path(inspect.getsourcefile(mod) or "")
    assert src_path.exists(), f"could not locate operator_app source on disk: {src_path}"
    return src_path.read_text()


def test_operator_app_imports_lifecycle_helper():
    """If this import disappears, every operator training run reverts to the
    pre-PR-3 cached-or-refit behavior and the resume bug returns silently."""
    import neuralop.flood.train.operator_app as mod

    assert hasattr(mod, "resolve_normalizer_artifact"), (
        "operator_app must import resolve_normalizer_artifact from the shared "
        "lifecycle module. The bug-fix path goes through this symbol."
    )


def test_operator_app_main_calls_resolve_normalizer_artifact():
    """Grep-level invariant: the call site exists in main()."""
    src = _operator_app_source()
    assert re.search(r"resolve_normalizer_artifact\s*\(", src), (
        "Expected a call to resolve_normalizer_artifact(...) inside operator_app. "
        "If the migration was reverted, the resume-time bug returns."
    )


def test_operator_app_no_longer_calls_normalizer_metadata_matches_in_main():
    """The legacy decision primitive must not be called from operator_app any
    more — the lifecycle helper is the only place that should make that call.

    We allow it in import lines (for backward compat) and inside the diffusion
    wait helper (operated under PR-5's strict opt-in), but the operator's own
    code path must not branch on it directly."""
    src = _operator_app_source()
    # Drop import lines so we're only matching the operator's actual logic.
    code_only = "\n".join(
        line for line in src.splitlines() if not line.lstrip().startswith(("from ", "import "))
    )
    assert "normalizer_metadata_matches(" not in code_only, (
        "operator_app.main() must not call normalizer_metadata_matches() "
        "directly; the lifecycle helper owns the cached-vs-refit decision."
    )


def test_operator_app_forwards_is_resuming_from_resume_from_dir():
    """The is_resuming flag passed to the lifecycle must derive from the
    config.checkpoint.resume_from_dir setting. Grep-level check on the source
    rather than a full end-to-end run.
    """
    src = _operator_app_source()
    pattern = re.compile(
        r"is_resuming\s*=\s*bool\(\s*_cfg_get\(\s*config\.checkpoint\s*,\s*[\"']resume_from_dir[\"']",
    )
    assert pattern.search(src), (
        "Expected is_resuming = bool(_cfg_get(config.checkpoint, 'resume_from_dir', ...)) "
        "in operator_app.main() so resume intent is forwarded to the lifecycle helper."
    )


def test_operator_app_preserves_force_load_normalizers_flag():
    """force_load_cached_normalizers was a user-facing flag before PR-3 and
    must keep working — operators who set it expect to bypass the metadata
    check. The lifecycle helper supports it via force_load=..."""
    src = _operator_app_source()
    assert re.search(
        r"force_load\s*=\s*force_load_cached_normalizers", src
    ), (
        "Expected force_load=force_load_cached_normalizers in the lifecycle call. "
        "Otherwise the user-facing force_load_normalizers config flag silently "
        "stops working."
    )

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from neuralop.flood.eval.neon_phase5 import write_checksummed_artifact


def _load_runtime():
    path = Path(__file__).resolve().parents[1] / "scripts" / "neon_phase5_runtime.py"
    spec = importlib.util.spec_from_file_location("neon_phase5_runtime_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _root_preflight(runtime, tmp_path: Path, *, head: str = "abc123"):
    inputs = {
        "config": _write(tmp_path / "config.yaml", "config\n"),
        "bundle": _write(tmp_path / "bundle.json", "bundle\n"),
        "checkpoint": _write(tmp_path / "b3" / "neon_stage2_best.pt", "checkpoint\n"),
        "history": _write(tmp_path / "b3" / "history.json", "history\n"),
        "preflight": _write(tmp_path / "b3" / "preflight.json", "preflight\n"),
    }
    cache = tmp_path / "feature_cache_v3"
    canonical = cache / "canonical_0123"
    canonical.mkdir(parents=True)
    root = tmp_path / "phase5" / "PREFLIGHT.json"
    write_checksummed_artifact(
        root,
        {
            "schema_version": "neon_phase5_preflight_v1",
            "analysis_git_head": head,
            "input_paths": {key: str(path) for key, path in inputs.items()},
            "input_sha256": {
                key: runtime.file_sha256(path) for key, path in inputs.items()
            },
            "cache_dir": str(cache),
            "canonical_cache_dir": str(canonical),
        },
    )
    return root, inputs, cache


def test_phase5_frozen_input_verifier_rejects_mutated_b3_checkpoint(tmp_path):
    runtime = _load_runtime()
    root, inputs, cache = _root_preflight(runtime, tmp_path)

    runtime.verify_phase5_frozen_inputs(
        phase5_preflight_path=root,
        config_path=inputs["config"],
        bundle_path=inputs["bundle"],
        checkpoint_path=inputs["checkpoint"],
        history_path=inputs["history"],
        source_preflight_path=inputs["preflight"],
        cache_dir=cache,
        expected_head="abc123",
    )
    inputs["checkpoint"].write_text("mutated\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        runtime.verify_phase5_frozen_inputs(
            phase5_preflight_path=root,
            config_path=inputs["config"],
            bundle_path=inputs["bundle"],
            checkpoint_path=inputs["checkpoint"],
            history_path=inputs["history"],
            source_preflight_path=inputs["preflight"],
            cache_dir=cache,
            expected_head="abc123",
        )


def test_phase5_frozen_input_verifier_requires_signed_completion_for_derived_run(
    tmp_path,
):
    runtime = _load_runtime()
    root, inputs, cache = _root_preflight(runtime, tmp_path)
    derived = {
        "checkpoint": _write(tmp_path / "pilot" / "neon_stage2_best.pt", "pilot checkpoint\n"),
        "history": _write(tmp_path / "pilot" / "history.json", "pilot history\n"),
        "preflight": _write(tmp_path / "pilot" / "preflight.json", "pilot preflight\n"),
    }

    with pytest.raises(FileNotFoundError, match="TRAINING_COMPLETE"):
        runtime.verify_phase5_frozen_inputs(
            phase5_preflight_path=root,
            config_path=inputs["config"],
            bundle_path=inputs["bundle"],
            checkpoint_path=derived["checkpoint"],
            history_path=derived["history"],
            source_preflight_path=derived["preflight"],
            cache_dir=cache,
            expected_head="abc123",
        )

    completion = derived["checkpoint"].parent / "TRAINING_COMPLETE.json"
    write_checksummed_artifact(
        completion,
        {
            "schema_version": "neon_stage2_training_complete_v1",
            "git_head": "abc123",
            "input_paths": {key: str(path) for key, path in derived.items()},
            "input_sha256": {
                key: runtime.file_sha256(path) for key, path in derived.items()
            },
        },
    )
    runtime.verify_phase5_frozen_inputs(
        phase5_preflight_path=root,
        config_path=inputs["config"],
        bundle_path=inputs["bundle"],
        checkpoint_path=derived["checkpoint"],
        history_path=derived["history"],
        source_preflight_path=derived["preflight"],
        cache_dir=cache,
        expected_head="abc123",
    )

    derived["history"].write_text("mutated pilot history\n", encoding="utf-8")
    with pytest.raises(ValueError, match="history SHA-256"):
        runtime.verify_phase5_frozen_inputs(
            phase5_preflight_path=root,
            config_path=inputs["config"],
            bundle_path=inputs["bundle"],
            checkpoint_path=derived["checkpoint"],
            history_path=derived["history"],
            source_preflight_path=derived["preflight"],
            cache_dir=cache,
            expected_head="abc123",
        )

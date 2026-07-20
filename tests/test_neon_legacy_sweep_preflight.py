from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "neon_legacy_sweep_preflight.py"
SPEC = importlib.util.spec_from_file_location("neon_legacy_sweep_preflight", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _sources(tmp_path: Path):
    config = tmp_path / "config.yaml"
    bundle = tmp_path / "bundle.json"
    source = tmp_path / "source"
    config.write_text("x: 1\n")
    bundle.write_text("{}\n")
    for n_train in MODULE.N_VALUES:
        run = source / f"tr_n{n_train}"
        run.mkdir(parents=True)
        (run / "neon_stage2_best.pt").write_bytes(f"checkpoint-{n_train}".encode())
        (run / "history.json").write_text(json.dumps({"n_train": n_train}))
    return config, bundle, source


def test_legacy_export_plan_is_n_major_and_pins_every_input(tmp_path):
    config, bundle, source = _sources(tmp_path)

    plan = MODULE.build_plan(
        config=config,
        bundle=bundle,
        source_root=source,
        output_root=tmp_path / "out",
        expected_head="abc123",
    )

    assert len(plan["tasks"]) == 250
    assert plan["tasks"][0]["n_train"] == 25
    assert plan["tasks"][49]["family_index"] == 49
    assert plan["tasks"][50]["n_train"] == 50
    assert plan["m_eval"] == 16
    assert plan["k_eval"] == 8
    assert all(row["checkpoint_sha256"] for row in plan["checkpoints"])


def test_legacy_export_plan_detects_post_preflight_checkpoint_mutation(tmp_path):
    config, bundle, source = _sources(tmp_path)
    plan_path = tmp_path / "plan.json"
    payload = MODULE.build_plan(
        config=config,
        bundle=bundle,
        source_root=source,
        output_root=tmp_path / "out",
        expected_head="abc123",
    )
    MODULE._atomic_json(plan_path, payload)
    (source / "tr_n25" / "neon_stage2_best.pt").write_bytes(b"changed")

    with pytest.raises(ValueError, match="checkpoint changed"):
        MODULE.load_and_validate_plan(plan_path, expected_head="abc123", task_id=0)

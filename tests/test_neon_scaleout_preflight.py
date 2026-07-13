from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "neon_scaleout_preflight.py"
    spec = importlib.util.spec_from_file_location("neon_scaleout_preflight", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scaleout_plan_has_five_nested_sizes_for_each_of_five_replicates(tmp_path):
    script = _load_script()
    g1 = tmp_path / "g1.json"
    g1.write_text(json.dumps({"schema_version": "neon_g1_gate_v1", "gate_passed": True}))
    plan = script.prepare_scaleout_plan(
        g1_report=g1,
        run_root=tmp_path / "runs",
        cache_dir=tmp_path / "cache",
        expected_head="abc123",
    )
    assert len(plan["tasks"]) == 25
    for replicate in range(5):
        rows = [row for row in plan["tasks"] if row["subset_replicate"] == replicate]
        assert [row["n_train"] for row in rows] == [25, 50, 100, 250, 400]
        assert all(row["config"]["bootstrap_distribution"] == "probit_exponential" for row in rows)
        assert all(row["config"]["member_bootstrap_enabled"] is False for row in rows)


def test_scaleout_plan_rejects_failed_g1(tmp_path):
    script = _load_script()
    g1 = tmp_path / "g1.json"
    g1.write_text(json.dumps({"schema_version": "neon_g1_gate_v1", "gate_passed": False}))
    with pytest.raises(ValueError, match="G1"):
        script.prepare_scaleout_plan(
            g1_report=g1,
            run_root=tmp_path / "runs",
            cache_dir=tmp_path / "cache",
            expected_head="abc123",
        )

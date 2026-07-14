from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "neon_stage2_tr_train.py"
    spec = importlib.util.spec_from_file_location("neon_stage2_tr_train", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolved_ladder_config_separates_b1_mechanisms():
    script = _load_script()
    b1a = script._resolved_ladder_config(
        "B1A", prior_scale="auto_0p10_base_rmse", d_e=16, n_epochs=30
    )
    b1b = script._resolved_ladder_config(
        "B1B", prior_scale="auto_0p10_base_rmse", d_e=16, n_epochs=30
    )
    assert b1a.member_bootstrap_enabled is False
    assert b1a.bootstrap_distribution == "tempered_exponential"
    assert b1b.member_bootstrap_enabled is False
    assert b1b.bootstrap_distribution == "probit_exponential"


def test_preflight_manifest_records_fully_validated_config(tmp_path):
    script = _load_script()
    config = script._resolved_ladder_config(
        "B4", prior_scale="auto_0p10_base_rmse", d_e=16, n_epochs=30,
        de_spread_multiplier=0.5,
    )
    output = tmp_path / "preflight.json"
    script._write_preflight_manifest(
        output,
        config=config,
        rung="B4",
        n_train=450,
        output_dir=tmp_path / "run",
        cache_dir=tmp_path / "cache",
        subset_replicate=0,
    )
    payload = json.loads(output.read_text())
    assert payload["schema_version"] == "neon_repair_preflight_v1"
    assert payload["ladder_rung"] == "B4"
    assert payload["config"]["prior_scale"]["mode"] == "de_spread_target"
    assert payload["config"]["prior_scale"]["target_std_m"] == 0.023


def test_training_state_paths_resume_only_from_an_existing_epoch_state(tmp_path):
    script = _load_script()

    latest, resume = script._training_state_paths(tmp_path)
    assert latest == tmp_path / "neon_stage2_latest_state.pt"
    assert resume is None

    latest.write_bytes(b"completed epoch state")
    latest_again, resume = script._training_state_paths(tmp_path)
    assert latest_again == latest
    assert resume == latest

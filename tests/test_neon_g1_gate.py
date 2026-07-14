from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "neon_g1_gate.py"
    spec = importlib.util.spec_from_file_location("neon_g1_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_run(tmp_path: Path, *, crps=0.018, base_crps=0.020, rmse_delta=0.0005):
    run = tmp_path / "b3"
    run.mkdir()
    row = {
        "epoch": 2,
        "val_base_fair_crps_physical": base_crps,
        "val_deterministic_head_fair_crps_physical": 0.019,
        "val_mixture_fair_crps_physical": crps,
        "val_base_rmse_physical": 0.050,
        "val_deterministic_head_rmse_physical": 0.0502,
        "val_stage2_rmse_physical": 0.050 + rmse_delta,
        "val_stage2_minus_base_rmse_physical": rmse_delta,
        "val_cancellation_fraction": 0.40,
        "val_prior_retention_ratio": 0.60,
        "selection_rmse_margin_m": 0.001,
    }
    (run / "history.json").write_text(json.dumps({"best_epoch": 2, "history": [row]}))
    pairs = [
        {"family_id": f"F{i:03d}", "stage2_minus_base_rmse_physical": rmse_delta}
        for i in range(50)
    ]
    (run / "validation_rmse_pairs_epoch_0003.json").write_text(
        json.dumps({"epoch": 2, "pairs": pairs})
    )
    return run


def test_g1_gate_passes_skill_and_rmse_parity(tmp_path):
    script = _load_script()
    report = script.evaluate_g1(_write_run(tmp_path), bootstrap_replicates=100, seed=7)
    assert report["gate_passed"] is True
    assert report["checks"]["mixture_crps_nonworse"] is True
    assert report["checks"]["rmse_noninferior"] is True
    assert report["n_paired_families"] == 50


def test_g1_gate_fails_worse_crps_even_with_rmse_parity(tmp_path):
    script = _load_script()
    report = script.evaluate_g1(
        _write_run(tmp_path, crps=0.021, base_crps=0.020),
        bootstrap_replicates=100,
        seed=7,
    )
    assert report["gate_passed"] is False
    assert report["checks"]["mixture_crps_nonworse"] is False


def test_g1_report_writer_emits_all_formats(tmp_path):
    script = _load_script()
    report = script.evaluate_g1(
        _write_run(tmp_path), bootstrap_replicates=100, seed=7
    )
    output_dir = tmp_path / "reports"

    script._write_reports(output_dir, report)

    assert (output_dir / "g1_gate.json").is_file()
    assert (output_dir / "g1_gate.csv").is_file()
    markdown = (output_dir / "g1_gate.md").read_text()
    assert "| Frozen model-0 |" in markdown
    assert "**G1 passed:** `true`" in markdown

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "neon_legacy_sweep_analysis.py"
SPEC = importlib.util.spec_from_file_location("neon_legacy_sweep_analysis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_fixture(root: Path, source: Path, n: int) -> None:
    run = root / f"n{n}"
    (run / "remap").mkdir(parents=True)
    (run / "output").mkdir(parents=True)
    (source / f"tr_n{n}").mkdir(parents=True)
    variance = (0.2 * n ** -0.5) ** 2
    (run / "remap" / "legacy_estimator_remap.json").write_text(
        json.dumps(
            {
                "label": "legacy frozen rollout, estimator repaired",
                "n_families": 50,
                "aggregate": {"variance_epistemic_crossed_mean": variance},
            }
        )
    )
    (run / "output" / "neon_eval_metrics.json").write_text(
        json.dumps(
            {
                "aggregate": {
                    "ensemble_mean_rmse": 0.04 + n * 1e-6,
                    "marginal_fair_crps": 0.02,
                },
                "checkpoint_metadata": {"alpha": 0.9},
            }
        )
    )
    ids = [f"TR{i:06d}" for i in range(n)]
    (source / f"tr_n{n}" / "history.json").write_text(
        json.dumps(
            {
                "n_train": n,
                "best_epoch": 4,
                "best_val_fit": 0.02,
                "train_family_ids": ids,
                "history": [{"epoch": 4, "val_fit": 0.02}],
            }
        )
    )


def test_legacy_sweep_analysis_recovers_preliminary_power_law(tmp_path):
    run_root = tmp_path / "runs"
    source_root = tmp_path / "source"
    for n in MODULE.N_VALUES:
        _write_fixture(run_root, source_root, n)

    result = MODULE.analyze_legacy_sweep(run_root, source_root)

    assert result["status"] == "preliminary_descriptive_nonreplicated"
    assert result["gamma_hat"] == pytest.approx(0.5)
    assert len(result["covariates"]) == 5
    assert result["covariates"][0]["subset_sha256"]
    assert "prior_scale_m" in result["covariates"][0]["missing_covariates"]


def test_legacy_sweep_rejects_unrepaired_estimator_label(tmp_path):
    run_root = tmp_path / "runs"
    source_root = tmp_path / "source"
    for n in MODULE.N_VALUES:
        _write_fixture(run_root, source_root, n)
    path = run_root / "n25" / "remap" / "legacy_estimator_remap.json"
    payload = json.loads(path.read_text())
    payload["label"] = "wrong"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="estimator-repair label"):
        MODULE.analyze_legacy_sweep(run_root, source_root)

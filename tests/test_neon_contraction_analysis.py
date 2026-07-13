from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "neon_contraction_analysis.py"
    spec = importlib.util.spec_from_file_location("neon_contraction_analysis", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_run(root: Path, *, replicate: int, n_train: int, gamma: float = 0.5) -> None:
    family_ids = [f"F{i:03d}" for i in range(n_train)]
    run = root / f"rep{replicate}" / f"n{n_train}"
    run.mkdir(parents=True)
    std = (1.0 + 0.1 * replicate) * n_train ** (-gamma)
    (run / "history.json").write_text(
        json.dumps(
            {
                "n_train": n_train,
                "ladder_rung": "B3",
                "subset_replicate": replicate,
                "train_family_ids": family_ids,
                "best_epoch": 2,
                "best_val_fit": 0.01,
                "history": [
                    {"epoch": 0, "val_total_epistemic_std_physical": std * 1.2},
                    {"epoch": 2, "val_total_epistemic_std_physical": std},
                ],
            }
        )
    )


def test_contraction_analysis_recovers_replicated_power_law(tmp_path):
    script = _load_script()
    for replicate in range(5):
        for n_train in script.N_VALUES:
            _write_run(tmp_path, replicate=replicate, n_train=n_train)
    result = script.analyze_scaleout(tmp_path, bootstrap_samples=500, bootstrap_seed=9)
    assert result["schema_version"] == "neon_contraction_analysis_v1"
    assert len(result["replicates"]) == 5
    assert result["gamma_mean"] == pytest.approx(0.5, abs=1e-12)
    assert result["gamma_std"] == pytest.approx(0.0, abs=1e-12)
    assert all(row["nested_prefixes_valid"] for row in result["replicates"])


def test_contraction_analysis_rejects_non_nested_family_prefixes(tmp_path):
    script = _load_script()
    for replicate in range(5):
        for n_train in script.N_VALUES:
            _write_run(tmp_path, replicate=replicate, n_train=n_train)
    path = tmp_path / "rep0" / "n50" / "history.json"
    payload = json.loads(path.read_text())
    payload["train_family_ids"][0] = "NOT_THE_PREFIX"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="nested prefix"):
        script.analyze_scaleout(tmp_path, bootstrap_samples=50)


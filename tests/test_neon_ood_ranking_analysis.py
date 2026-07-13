from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "neon_ood_ranking_analysis.py"
    spec = importlib.util.spec_from_file_location("neon_ood_ranking_analysis", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_metrics(path: Path, rows):
    path.write_text(json.dumps({"per_family": rows}))


def test_ood_ranking_recovers_perfect_event_order_and_matched_ratios(tmp_path):
    script = _load_script()
    ood = [
        {
            "family_id": f"H{i:02d}",
            "ensemble_mean_rmse": float(i + 1),
            "variance_epistemic_anova_corrected_mean": float((i + 1) ** 2),
        }
        for i in range(13)
    ]
    iid = [
        {
            "family_id": f"T{i:02d}",
            "ensemble_mean_rmse": 1.0,
            "variance_epistemic_anova_corrected_mean": 1.0,
        }
        for i in range(5)
    ]
    ood_path, id_path = tmp_path / "ood.json", tmp_path / "id.json"
    _write_metrics(ood_path, ood)
    _write_metrics(id_path, iid)
    result = script.analyze_ood_ranking(
        ood_path,
        id_path,
        expected_ood_events=13,
        bootstrap_samples=500,
        permutation_samples=500,
        seed=4,
    )
    assert result["n_ood_events"] == 13
    assert result["pearson_epistemic_std_rmse"] == pytest.approx(1.0)
    assert result["spearman_epistemic_std_rmse"] == pytest.approx(1.0)
    assert result["top3_error_event_recall"] == pytest.approx(1.0)
    assert result["ood_to_id_epistemic_std_ratio"] == pytest.approx(7.0)
    assert result["ood_to_id_rmse_ratio"] == pytest.approx(7.0)
    assert all(row["pearson"] == pytest.approx(1.0) for row in result["leave_one_out"])
    assert len(result["risk_coverage_curve"]) == 13


def test_ood_ranking_requires_expected_event_count(tmp_path):
    script = _load_script()
    path = tmp_path / "metrics.json"
    _write_metrics(
        path,
        [
            {
                "family_id": "H00",
                "ensemble_mean_rmse": 1.0,
                "variance_epistemic_anova_corrected_mean": 1.0,
            }
        ],
    )
    with pytest.raises(ValueError, match="expected 13"):
        script.analyze_ood_ranking(
            path,
            path,
            expected_ood_events=13,
            bootstrap_samples=50,
            permutation_samples=50,
        )

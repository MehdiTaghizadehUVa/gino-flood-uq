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


def _write_metrics(path: Path, rows, *, checkpoint_sha256: str | None = None):
    payload = {"per_family": rows}
    if checkpoint_sha256 is not None:
        payload["plan"] = {"stage2_checkpoint_sha256": checkpoint_sha256}
    path.write_text(json.dumps(payload))


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


def test_ood_ranking_is_pinned_to_stage2_checkpoint(tmp_path):
    script = _load_script()
    ood = [
        {
            "family_id": f"H{i:02d}",
            "ensemble_mean_rmse": float(i + 1),
            "variance_epistemic_anova_corrected_mean": float((i + 1) ** 2),
        }
        for i in range(3)
    ]
    iid = [
        {
            "family_id": f"T{i:02d}",
            "ensemble_mean_rmse": 1.0,
            "variance_epistemic_anova_corrected_mean": 1.0,
        }
        for i in range(3)
    ]
    ood_path, id_path = tmp_path / "ood.json", tmp_path / "id.json"
    _write_metrics(ood_path, ood, checkpoint_sha256="checkpoint-a")
    _write_metrics(id_path, iid)

    result = script.analyze_ood_ranking(
        ood_path,
        id_path,
        expected_ood_events=3,
        bootstrap_samples=50,
        permutation_samples=50,
        analysis_git_head="head-a",
        protocol_sha256="protocol-a",
        stage2_checkpoint_sha256="checkpoint-a",
    )
    assert result["analysis_git_head"] == "head-a"
    assert result["protocol_sha256"] == "protocol-a"
    assert result["stage2_checkpoint_sha256"] == "checkpoint-a"
    assert len(result["ood_metrics_sha256"]) == 64
    assert len(result["id_metrics_sha256"]) == 64

    with pytest.raises(ValueError, match="OOD metrics/checkpoint mismatch"):
        script.analyze_ood_ranking(
            ood_path,
            id_path,
            expected_ood_events=3,
            bootstrap_samples=50,
            permutation_samples=50,
            stage2_checkpoint_sha256="checkpoint-b",
        )

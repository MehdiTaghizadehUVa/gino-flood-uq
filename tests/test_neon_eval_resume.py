"""Behavioral tests for interruption-safe NEON evaluation shards."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_eval_cli():
    for name in ("neuralop", "neuralop.flood", "neuralop.flood.eval", "neuralop.flood.cli"):
        package = sys.modules.setdefault(name, types.ModuleType(name))
        package_path = str(REPO_ROOT / name.replace(".", "/"))
        existing_paths = list(getattr(package, "__path__", []))
        if package_path not in existing_paths:
            existing_paths.append(package_path)
        package.__path__ = existing_paths
    path = REPO_ROOT / "neuralop/flood/cli/eval_neon_stage2.py"
    spec = importlib.util.spec_from_file_location(
        "neuralop.flood.cli.eval_neon_stage2_resume_test", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cli = _load_eval_cli()


def _plan(output_dir: Path, **updates):
    plan = {
        "config": "/cfg.yaml",
        "stage2_checkpoint": "/stage2.pt",
        "stage1_bundle": "/bundle.json",
        "output_dir": str(output_dir),
        "families": "val",
        "m_eval": 4,
        "k_eval": 8,
        "thresholds": [0.1, 0.3],
        "seed": 7,
        "family_index": None,
        "shard_only": False,
        "merge_only": False,
        "resume": False,
        "expected_families": None,
        "shard_dir": None,
    }
    plan.update(updates)
    return plan


def _write_shard(tmp_path, index, family_id, row, pit, reliability, impact=None):
    return cli.write_evaluation_shard(
        tmp_path / "shards",
        plan=_plan(tmp_path),
        checkpoint_metadata={"epoch": 63, "feature_source": "decoder_pre_projection"},
        family_index=index,
        family_id=family_id,
        row=row,
        pit_rank=pit,
        reliability=reliability,
        impact_metrics=impact,
    )


def test_scientific_signature_ignores_execution_controls_but_not_budget(tmp_path):
    base = _plan(tmp_path)
    moved = _plan(
        tmp_path / "other",
        family_index=17,
        shard_only=True,
        merge_only=True,
        resume=True,
        expected_families=50,
        shard_dir="/different/shards",
    )
    assert cli.scientific_eval_signature(base) == cli.scientific_eval_signature(moved)
    assert cli.scientific_eval_signature(base) != cli.scientific_eval_signature(
        _plan(tmp_path, m_eval=8)
    )


def test_atomic_shards_merge_to_the_same_global_sufficient_statistics(tmp_path):
    _write_shard(
        tmp_path,
        1,
        "TE000002",
        {"family_id": "TE000002", "rmse": 3.0, "spread_error_corr": 0.4},
        {"pit_edges": [0.0, 0.5, 1.0], "pit_counts": [2, 3], "rank_counts": [4, 5, 6]},
        {
            "wd_gt_0.1": [
                {
                    "bin_lo": 0.0,
                    "bin_hi": 0.5,
                    "n": 2.0,
                    "sum_forecast_prob": 0.4,
                    "sum_observed_freq": 1.0,
                }
            ]
        },
        {"family_id": "TE000002", "area_crps": 0.8},
    )
    _write_shard(
        tmp_path,
        0,
        "TE000001",
        {"family_id": "TE000001", "rmse": 1.0, "spread_error_corr": 0.2},
        {"pit_edges": [0.0, 0.5, 1.0], "pit_counts": [5, 7], "rank_counts": [1, 2, 3]},
        {
            "wd_gt_0.1": [
                {
                    "bin_lo": 0.0,
                    "bin_hi": 0.5,
                    "n": 3.0,
                    "sum_forecast_prob": 1.5,
                    "sum_observed_freq": 2.0,
                }
            ]
        },
        {"family_id": "TE000001", "area_crps": 0.4},
    )

    output_path = tmp_path / "neon_eval_metrics.json"
    payload = cli.merge_evaluation_shards(
        tmp_path / "shards", output_path=output_path, expected_families=2
    )
    assert output_path.with_suffix(".json.sha256").is_file()

    assert output_path.exists()
    assert [r["family_id"] for r in payload["per_family"]] == ["TE000001", "TE000002"]
    assert payload["aggregate"]["rmse"] == pytest.approx(2.0)
    assert payload["aggregate"]["spread_error_corr"] == pytest.approx(0.3)
    assert payload["aggregate"]["spread_error_corr_mean"] == pytest.approx(0.3)
    assert payload["plan"]["output_dir"] == str(tmp_path)
    assert payload["plan"]["family_index"] is None
    assert payload["plan"]["shard_only"] is False
    assert payload["pit_rank"]["pit_edges"] == [0.0, 0.5, 1.0]
    assert payload["pit_rank"]["pit_counts"] == [7, 10]
    assert payload["pit_rank"]["rank_counts"] == [5, 7, 9]
    rel = payload["reliability"]["wd_gt_0.1"][0]
    assert rel["n"] == 5
    assert rel["forecast_prob"] == pytest.approx(1.9 / 5.0)
    assert rel["observed_freq"] == pytest.approx(3.0 / 5.0)
    assert [r["family_id"] for r in payload["impact_metrics"]] == [
        "TE000001",
        "TE000002",
    ]
    assert not list((tmp_path / "shards").glob("*.tmp"))
    assert json.loads(output_path.read_text()) == payload


def test_merge_rejects_missing_family_index_instead_of_publishing_partial_summary(tmp_path):
    _write_shard(
        tmp_path,
        1,
        "TE000002",
        {"family_id": "TE000002", "rmse": 1.0, "spread_error_corr": 0.0},
        {"pit_counts": [1], "rank_counts": [1]},
        {},
    )
    with pytest.raises(ValueError, match="contiguous family indices"):
        cli.merge_evaluation_shards(
            tmp_path / "shards",
            output_path=tmp_path / "neon_eval_metrics.json",
            expected_families=2,
        )
    assert not (tmp_path / "neon_eval_metrics.json").exists()


def test_existing_valid_shard_can_be_resumed_without_recomputation(tmp_path):
    path = _write_shard(
        tmp_path,
        0,
        "TE000001",
        {"family_id": "TE000001", "rmse": 1.0, "spread_error_corr": 0.0},
        {"pit_counts": [1], "rank_counts": [1]},
        {},
    )
    assert cli.completed_evaluation_shard(
        path,
        plan=_plan(tmp_path),
        family_index=0,
        family_id="TE000001",
    )
    assert not cli.completed_evaluation_shard(
        path,
        plan=_plan(tmp_path, k_eval=99),
        family_index=0,
        family_id="TE000001",
    )


def test_merge_rejects_mixed_checkpoint_metadata(tmp_path):
    path0 = _write_shard(
        tmp_path,
        0,
        "TE000001",
        {"family_id": "TE000001", "rmse": 1.0, "spread_error_corr": 0.0},
        {"pit_counts": [1], "rank_counts": [1]},
        {},
    )
    path1 = _write_shard(
        tmp_path,
        1,
        "TE000002",
        {"family_id": "TE000002", "rmse": 1.0, "spread_error_corr": 0.0},
        {"pit_counts": [1], "rank_counts": [1]},
        {},
    )
    payload = json.loads(path1.read_text())
    payload["checkpoint_metadata"]["epoch"] = 64
    path1.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="checkpoint metadata"):
        cli.merge_evaluation_shards(
            tmp_path / "shards",
            output_path=tmp_path / "neon_eval_metrics.json",
            expected_families=2,
        )
    assert path0.exists()


def test_merge_only_cli_publishes_without_loading_model_stack(tmp_path, capsys):
    for index, family_id in enumerate(("TE000001", "TE000002")):
        _write_shard(
            tmp_path,
            index,
            family_id,
            {"family_id": family_id, "rmse": float(index), "spread_error_corr": 0.0},
            {"pit_counts": [1], "rank_counts": [1]},
            {},
        )
    rc = cli.main(
        [
            "--config", "/cfg.yaml",
            "--stage2-checkpoint", "/stage2.pt",
            "--stage1-bundle", "/bundle.json",
            "--output-dir", str(tmp_path),
            "--m-eval", "4",
            "--k-eval", "8",
            "--thresholds", "0.1", "0.3",
            "--seed", "7",
            "--merge-only",
            "--expected-families", "2",
        ]
    )
    assert rc == 0
    assert "NEON EVAL MERGE OK" in capsys.readouterr().out
    assert (tmp_path / "neon_eval_metrics.json").exists()

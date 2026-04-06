from __future__ import annotations

import json
from pathlib import Path

import yaml

from neuralop.flood.cli.coastal_fgn_gridsearch import (
    build_run_tag,
    parse_training_log,
    rank_summaries,
    render_config,
    spec_for_index,
)


def test_spec_for_index_grid_order_matches_plan():
    assert spec_for_index(0) == {
        "index": 0,
        "learning_rate": 5.0e-5,
        "weight_decay": 1.0e-4,
        "gno_radius": 0.08,
        "fno_hidden_channels": 64,
        "fgn_noise_dim": 16,
    }
    assert spec_for_index(1)["fgn_noise_dim"] == 32
    assert spec_for_index(48) == {
        "index": 48,
        "learning_rate": 2.0e-4,
        "weight_decay": 1.0e-4,
        "gno_radius": 0.08,
        "fno_hidden_channels": 64,
        "fgn_noise_dim": 16,
    }
    assert spec_for_index(95) == {
        "index": 95,
        "learning_rate": 4.0e-4,
        "weight_decay": 5.0e-4,
        "gno_radius": 0.12,
        "fno_hidden_channels": 96,
        "fgn_noise_dim": 32,
    }


def test_build_run_tag_matches_expected_format():
    spec = spec_for_index(37)
    assert build_run_tag(spec) == "coastal_fgn_gs30_lr1e-4_wd5e-4_r0.08_h64_z32_idx37"


def test_render_config_keeps_multichannel_clean_family_and_sets_overrides(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    base_config = repo_root / "config" / "flood" / "coastal" / "gino_pluvial_flood_config_coastal_depth_only_fgn_grid.yaml"
    output_config = tmp_path / "rendered.yaml"
    checkpoint_dir = tmp_path / "ckpt"
    normalizer_root = checkpoint_dir

    rendered = render_config(
        base_config_path=base_config,
        output_config_path=output_config,
        index=95,
        checkpoint_dir=checkpoint_dir,
        normalizer_root=normalizer_root,
        wandb_group="group",
        wandb_name="name",
        data_root="/tmp/coastal/train",
        clean_boundary_root="/tmp/coastal/clean",
        batch_size=16,
        n_samples_max=4096,
        n_epochs=30,
        seed=123,
        deterministic=False,
        wandb_log=True,
    )

    with output_config.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    flood = payload["flood"]
    channels = flood["data"]["boundary"]["channels"]
    assert [channel["name"] for channel in channels] == ["stage", "precipitation"]
    assert all(channel["mode"] == "clean_family" for channel in channels)
    assert all(channel["clean_boundary_root"] == "/tmp/coastal/clean" for channel in channels)
    assert flood["data"]["static_text_files"] == ["Coastal_Slope.txt", "Coastal_Aspect.txt", "Coastal_FlowDirection.txt", "Coastal_Curvature.txt", "Coastal_FlowAccumulation.txt"]
    assert flood["data"]["root"] == "/tmp/coastal/train"
    assert flood["data"]["batch_size"] == 16
    assert flood["data"]["n_samples_max"] == 4096
    assert flood["data"]["normalizer_root"] == str(normalizer_root)
    assert flood["data"]["normalizer_path"] == "normalizers_depth_only.pt"
    assert flood["checkpoint"]["save_dir"] == str(checkpoint_dir)
    assert flood["checkpoint"]["save_best_metric"] == "test_crps"
    assert flood["checkpoint"]["save_every"] == 30
    assert flood["opt"]["n_epochs"] == 30
    assert flood["opt"]["learning_rate"] == 4.0e-4
    assert flood["opt"]["weight_decay"] == 5.0e-4
    assert flood["opt"]["amp_autocast"] is False
    assert flood["gino"]["gno_radius"] == 0.12
    assert flood["gino"]["fno_hidden_channels"] == 96
    assert flood["gino"]["fgn_noise_dim"] == 32
    assert flood["wandb"]["group"] == "group"
    assert flood["wandb"]["name"] == "name"
    assert flood["wandb"]["log"] is True
    assert flood["verify_training"] is False
    assert flood["rollout"]["run_after_training"] is False
    assert rendered["run_tag"] == build_run_tag(spec_for_index(95))
    assert rendered["batch_size"] == 16
    assert rendered["n_samples_max"] == 4096


def test_parse_training_log_extracts_best_and_final_metrics(tmp_path: Path):
    log_path = tmp_path / "training.log"
    log_path.write_text(
        "Epoch 0 | time=12.34s | avg_loss=0.50000000 | train_err=0.400000 | lr=1.00e-04\n"
        "Eval: test_l2=4.000e-02, test_crps=7.000e-03\n"
        "Epoch 1 | time=12.50s | avg_loss=0.45000000 | train_err=0.350000 | lr=1.00e-04\n"
        "Eval: test_l2=3.800e-02, test_crps=6.500e-03\n",
        encoding="utf-8",
    )

    parsed = parse_training_log(log_path)
    assert parsed["best_epoch"] == 1
    assert parsed["best_test_crps"] == 6.5e-03
    assert parsed["best_epoch_test_l2"] == 3.8e-02
    assert parsed["final_epoch"] == 1
    assert parsed["final_metrics"]["test_l2"] == 3.8e-02
    assert parsed["final_metrics"]["avg_loss"] == 0.45


def test_rank_summaries_uses_crps_then_l2(tmp_path: Path):
    root = tmp_path / "summaries"
    (root / "a").mkdir(parents=True)
    (root / "b").mkdir(parents=True)
    (root / "c").mkdir(parents=True)

    (root / "a" / "run_summary.json").write_text(
        json.dumps({"run_tag": "a", "best_test_crps": 0.01, "best_epoch_test_l2": 0.03}),
        encoding="utf-8",
    )
    (root / "b" / "run_summary.json").write_text(
        json.dumps({"run_tag": "b", "best_test_crps": 0.01, "best_epoch_test_l2": 0.02}),
        encoding="utf-8",
    )
    (root / "c" / "run_summary.json").write_text(
        json.dumps({"run_tag": "c", "best_test_crps": 0.009, "best_epoch_test_l2": 0.05}),
        encoding="utf-8",
    )

    ranked = rank_summaries(root)
    assert [item["run_tag"] for item in ranked] == ["c", "b", "a"]

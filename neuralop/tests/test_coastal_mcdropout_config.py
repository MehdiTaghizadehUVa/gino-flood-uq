from pathlib import Path

import yaml


def test_coastal_mcdropout_config_contract():
    repo = Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load(
        (repo / "config" / "flood" / "coastal" / "gino_coastal_depth_only_mcdropout.yaml").read_text()
    )["flood"]

    assert cfg["data"]["root"].endswith("Coastal_Flood_coastal_v1_5k_train_prod_t2_w64_20260318_233556/train")
    assert cfg["rollout_data"]["root"].endswith("Coastal_Flood_coastal_v1_5k_test_prod_t2_w16_20260414/test")
    assert cfg["data"]["target_variables"] == ["wd"]
    assert cfg["data"]["query_res"] == [48, 48]
    assert cfg["data"]["n_history"] == 3
    assert cfg["data"]["batch_size"] == 128
    assert cfg["data"]["num_workers"] == 4
    assert cfg["data"]["force_load_normalizers"] is True
    assert cfg["data"]["normalizer_path"] == "normalizers_depth_only.pt"
    assert cfg["gino"]["fno_hidden_channels"] == 128
    assert cfg["gino"]["fno_channel_mlp_dropout"] == 0.10
    assert cfg["gino"]["fno_norm"] == "instance_norm"
    assert cfg["gino"]["use_fgn_noise"] is False
    assert cfg["gino"]["output_distribution"] == "deterministic"
    assert cfg["opt"]["training_loss"] == "l2"
    assert cfg["opt"]["learning_rate"] == 0.0005
    assert cfg["opt"]["ar_finetune_start_epoch"] == 150
    assert cfg["opt"]["weight_decay"] == 0.0001
    assert cfg["opt"]["n_epochs"] == 250
    assert cfg["uq"]["method"] == "mc_dropout"
    assert cfg["uq"]["mc_samples"] == 32
    assert cfg["uq"]["mc_dropout"]["dropout_probability"] == 0.10
    assert cfg["uq"]["mc_dropout"]["activate_modules"] == "dropout_only"
    assert cfg["checkpoint"]["save_best_metric"] == "test_l2"
    assert cfg["checkpoint"]["eval_name"] == "best_model"
    assert cfg["opt"]["scheduler_monitor"] == "test_l2"
    assert cfg["opt"]["early_stopping_enabled"] is True
    assert cfg["opt"]["early_stopping_patience"] == 20
    assert float(cfg["opt"]["early_stopping_min_delta"]) == 1e-4
    assert cfg["wandb"]["log"] is True
    assert cfg["structural_dry"]["policy"] == "legacy_full_domain"
    assert cfg["structural_dry"]["mask_path"] is None
    assert cfg["structural_dry"]["canonical_data_root"] is None
    assert cfg["data"]["static_text_files"] == [
        "Coastal_Slope.txt",
        "Coastal_Aspect.txt",
        "Coastal_FlowDirection.txt",
        "Coastal_Curvature.txt",
        "Coastal_FlowAccumulation.txt",
    ]
    assert [c["name"] for c in cfg["data"]["boundary"]["channels"]] == ["stage", "precipitation"]
    assert [c["clean_boundary_file"] for c in cfg["rollout_data"]["boundary"]["channels"]] == [
        "Stage_Hydrographs_Test_Clean.txt",
        "Precipitation_Test_Clean.txt",
    ]
    assert "Portsmouth" in cfg["data"]["hdf_paths"]["wd"]


def test_wv_mcdropout_template_is_deterministic_dropout_not_fgn():
    repo = Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load(
        (repo / "config" / "flood" / "wv" / "gino_pluvial_flood_config_WV_depth_only_mcdropout.yaml").read_text()
    )["flood"]

    assert cfg["uq"]["method"] == "mc_dropout"
    assert cfg["uq"]["mc_samples"] == 32
    assert cfg["gino"]["fno_channel_mlp_dropout"] == cfg["uq"]["mc_dropout"]["dropout_probability"]
    assert cfg["gino"]["use_fgn_noise"] is False
    assert cfg["gino"]["output_distribution"] == "deterministic"
    assert cfg["gino"]["fno_norm"] == "instance_norm"
    assert cfg["opt"]["training_loss"] == "l2"
    assert cfg["checkpoint"]["save_best_metric"] == "test_l2"
    assert cfg["checkpoint"]["eval_name"] == "best_model"
    assert cfg["opt"]["scheduler_monitor"] == "test_l2"
    assert cfg["opt"]["early_stopping_enabled"] is True
    assert cfg["opt"]["early_stopping_patience"] == 20
    assert float(cfg["opt"]["early_stopping_min_delta"]) == 1e-4
    assert cfg["structural_dry"]["policy"] == "masked_primary"


def test_coastal_mcdropout_slurm_wrappers_are_scratch_backed_and_nested_boundary_safe():
    repo = Path(__file__).resolve().parents[2]
    train_text = (repo / "scripts" / "slurm" / "train" / "flood_coastal_train_mcdropout.sh").read_text()
    eval_text = (repo / "scripts" / "slurm" / "eval" / "flood_coastal_eval_mcdropout.sh").read_text()
    chain_text = (repo / "scripts" / "slurm" / "train" / "submit_coastal_mcdropout_chain.sh").read_text()

    assert "gino_coastal_depth_only_mcdropout.yaml" in train_text
    assert "COASTAL_MCD_SMOKE" in train_text
    assert "data.boundary_source" not in train_text
    assert "data.boundary_source" not in eval_text
    assert "max_val_batches" not in train_text
    assert "MAX_VAL_BATCHES" not in chain_text
    assert "Stage_Hydrographs_Train_Clean.txt/Stage_Hydrographs_Test_Clean.txt" in eval_text
    assert "--checkpoint.eval_name" in eval_text
    assert "CHECKPOINT_EVAL_NAME" in eval_text
    assert "SMOKE_N_SAMPLES_MAX=\"${SMOKE_N_SAMPLES_MAX:-32}\"" in chain_text
    assert "SMOKE_BATCH_SIZE=\"${SMOKE_BATCH_SIZE:-2}\"" in chain_text
    assert "SUBMIT_PRODUCTION" in chain_text
    assert "75 150 175 200 215 225 240 250" in chain_text

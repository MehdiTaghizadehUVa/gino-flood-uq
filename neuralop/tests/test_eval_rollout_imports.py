from types import SimpleNamespace

import numpy as np
import torch

from neuralop.flood.eval.metrics import _compute_csi
from neuralop.flood.eval.runtime import (
    build_rollout_initial_histories,
    clone_model_config_for_get_model,
)


def test_compute_csi_matches_expected_value():
    pred = np.array([0.0, 0.1, 0.2, 0.4], dtype=np.float32)
    gt = np.array([0.0, 0.0, 0.3, 0.5], dtype=np.float32)
    val = _compute_csi(0.05, pred, gt)
    # tp=2, fp=1, fn=0 -> 2/3
    assert np.isclose(val, 2.0 / 3.0)


def test_maintained_eval_common_imports():
    from neuralop.flood.eval.common import main  # noqa: F401


def test_maintained_calibrated_eval_imports():
    from neuralop.flood.eval.calibrated import main  # noqa: F401


def test_clone_model_config_handles_keyerror_getattr_dictlikes():
    class WeirdConfig(dict):
        def __getattr__(self, name):
            raise KeyError(name)

    cfg = WeirdConfig(
        {
            "arch": "gino",
            "gino": {"data_channels": 3, "fno_hidden_channels": 64},
            "distributed": WeirdConfig({"seed": 123}),
        }
    )
    cloned = clone_model_config_for_get_model(cfg)
    assert cloned == {
        "arch": "gino",
        "gino": {"data_channels": 3, "fno_hidden_channels": 64},
        "distributed": {"seed": 123},
    }
    cloned["gino"]["data_channels"] = 99
    assert cfg["gino"]["data_channels"] == 3


def test_build_rollout_dataset_groups_hydrographs(monkeypatch):
    from neuralop.flood.eval import datasets as ds_mod

    class DummyRolloutDataset:
        def __init__(self, *args, **kwargs):
            self.valid_run_ids = [
                "M40_TE000001_sim00",
                "M40_TE000001_sim01",
                "M40_TE000002_sim00",
                "M40_TE000002_sim01",
            ]
            self.structural_dry_mask = None

        def __len__(self):
            return len(self.valid_run_ids)

        def set_structural_dry_mask(self, dry_mask):
            self.structural_dry_mask = dry_mask

    def fake_collect_all_fields(dataset, expect_target=False):
        del expect_target
        n = len(dataset.valid_run_ids)
        geom = [torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32) for _ in range(n)]
        static = [torch.zeros((2, 1), dtype=torch.float32) for _ in range(n)]
        boundary = [torch.zeros((6, 2, 1), dtype=torch.float32) for _ in range(n)]
        dynamic = [torch.zeros((6, 2, 3), dtype=torch.float32) for _ in range(n)]
        return geom, static, boundary, dynamic, None

    monkeypatch.setattr(ds_mod, "FloodRolloutTestDatasetHDF", DummyRolloutDataset)
    monkeypatch.setattr(ds_mod, "collect_all_fields", fake_collect_all_fields)

    config = SimpleNamespace(
        data=SimpleNamespace(
            rollout_length=3,
            n_history=2,
            skip_before_timestep=0,
            query_res=[4, 4],
        ),
        rollout_data=SimpleNamespace(
            root="/tmp/unused",
            static_text_files=["M40_CS.txt", "M40_CU.txt", "M40_FA.txt"],
            test_txt="test.txt",
            boundary_source="clean_family",
            clean_boundary_root="/tmp/unused",
            clean_boundary_file="Hydrographs_Test_Clean.txt",
        ),
    )

    rollout_dataset, hydrograph_samples = ds_mod._build_rollout_normalized_dataset(
        config=config,
        normalizers={},
        target_variables=["wd"],
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
    )

    assert len(rollout_dataset) == 4
    assert hydrograph_samples is not None
    assert len(hydrograph_samples) == 2
    assert sorted(sample["hydrograph_id"] for sample in hydrograph_samples) == [
        "M40_TE000001",
        "M40_TE000002",
    ]
    assert hydrograph_samples[0]["reference_run_ids"] == [
        "M40_TE000001_sim00",
        "M40_TE000001_sim01",
    ]
    assert all(sample["dynamic_ref"].shape[0] == 2 for sample in hydrograph_samples)
    assert all(tuple(sample["query_points"].shape) == (4, 4, 2) for sample in hydrograph_samples)


def test_build_rollout_initial_histories_mean_history_repeats_mean():
    dynamic_ref = torch.arange(3 * 2 * 2, dtype=torch.float32).reshape(3, 2, 2, 1)
    histories, member_indices = build_rollout_initial_histories(
        dynamic_ref,
        skip_before_timestep=0,
        start_pred_t=2,
        n_members=4,
        rollout_init_mode="mean_history",
    )
    expected = dynamic_ref.mean(dim=0)
    assert member_indices == [-1, -1, -1, -1]
    assert len(histories) == 4
    assert all(torch.equal(hist, expected) for hist in histories)


def test_build_rollout_initial_histories_member_history_uses_reference_members():
    dynamic_ref = torch.arange(4 * 3 * 1 * 1, dtype=torch.float32).reshape(4, 3, 1, 1)
    histories, member_indices = build_rollout_initial_histories(
        dynamic_ref,
        skip_before_timestep=0,
        start_pred_t=3,
        n_members=3,
        rollout_init_mode="member_history",
    )
    assert member_indices == [0, 2, 3]
    assert len(histories) == 3
    assert torch.equal(histories[0], dynamic_ref[0])
    assert torch.equal(histories[1], dynamic_ref[2])
    assert torch.equal(histories[2], dynamic_ref[3])


def test_operator_eval_passes_fgn_state_update_to_rollout(monkeypatch, tmp_path):
    from neuralop.flood.eval import operator_app as app

    captured = {}

    class DummyLogger:
        def info(self, *args, **kwargs):
            return None

        def warning(self, *args, **kwargs):
            return None

    config = SimpleNamespace(
        deterministic=True,
        distributed=SimpleNamespace(seed=123),
        checkpoint=SimpleNamespace(save_dir=str(tmp_path / "ckpt")),
        data=SimpleNamespace(
            root=str(tmp_path),
            rollout_length=3,
            n_history=2,
            skip_before_timestep=0,
            dt=1200.0,
            batch_size=4,
            query_res=[4, 4],
        ),
        gino=SimpleNamespace(
            output_distribution="deterministic",
            use_fgn_noise=True,
            fgn_noise_dim=32,
            fgn_latent_temporal_mode="persistent",
        ),
        opt=SimpleNamespace(
            training_loss="l2",
            fgn_ar_state_update="member_feedback",
            crps_n_samples=2,
        ),
        rollout=SimpleNamespace(
            run_after_training=True,
            out_dir=str(tmp_path / "out"),
            n_ensemble_samples=2,
            init_mode="member_history",
        ),
    )

    monkeypatch.setattr(
        app,
        "_parse_args",
        lambda: SimpleNamespace(
            run_single_step=False,
            skip_single_step=True,
            run_rollout=True,
            skip_rollout=False,
            gaussian_state_update=None,
            eval_log_file=str(tmp_path / "eval.log"),
        ),
    )
    monkeypatch.setattr(app, "_validate_args", lambda args: None)
    monkeypatch.setattr(app, "_get_cli_arg_value", lambda flag: None)
    monkeypatch.setattr(app, "_resolve_cli_config_path", lambda path: None)
    monkeypatch.setattr(app, "load_config_and_setup", lambda: (config, "cpu", True))
    monkeypatch.setattr(app, "_resolve_device", lambda device: device)
    monkeypatch.setattr(app, "set_seed", lambda seed, deterministic=True: None)
    monkeypatch.setattr(app, "_discover_checkpoint_runs", lambda path: [(tmp_path / "ckpt", "alias", None)])
    monkeypatch.setattr(app, "setup_logging", lambda **kwargs: DummyLogger())
    dummy_ds = SimpleNamespace(dataset=SimpleNamespace(reference_cell_count=1, run_ids=[]))
    monkeypatch.setattr(app, "_build_one_step_datasets", lambda config, seed, logger: (dummy_ds, dummy_ds, ["wd"]))
    monkeypatch.setattr(app, "_load_or_fit_normalizers", lambda config, train_raw, primary_dir, logger: ({"dynamic": None, "target": None}, tmp_path / "norm.pt"))
    monkeypatch.setattr(
        app,
        "_load_structural_dry_artifact_for_eval",
        lambda config, **kwargs: ("legacy_full_domain", None),
    )
    monkeypatch.setattr(app, "NormalizedDatasetOnTheFly", lambda dataset, normalizers, query_res: dataset)
    monkeypatch.setattr(app, "_build_test_loader", lambda test_norm, batch_size: [object()])
    monkeypatch.setattr(app, "_load_models_from_runs", lambda config, device, checkpoint_runs, logger: [object()])
    monkeypatch.setattr(app, "_is_gaussian_mode", lambda config: False)
    monkeypatch.setattr(app, "_build_eval_losses", lambda config, use_fgn: {"l2": object()})
    monkeypatch.setattr(app, "_build_rollout_normalized_dataset", lambda *args, **kwargs: ([object()], [{"hydrograph_id": "H1"}]))
    monkeypatch.setattr(
        app,
        "_rollout_prediction_per_hydrograph",
        lambda **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(app, "_rollout_prediction_generic", lambda **kwargs: captured.update(kwargs))

    class DummyPhaseTimer:
        def __init__(self, logger, message):
            self.logger = logger
            self.message = message

        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(app, "_PhaseTimer", DummyPhaseTimer)

    assert app.main() == 0
    assert captured["fgn_ar_state_update"] == "member_feedback"
    assert captured["rollout_init_mode"] == "member_history"
    assert captured["out_dir"] == str((tmp_path / "out").resolve())

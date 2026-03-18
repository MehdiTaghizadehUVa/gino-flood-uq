import numpy as np
import torch
from types import SimpleNamespace

from neuralop.flood.eval.metrics import _compute_csi


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
    assert all(sample["dynamic_ref"].shape[0] == 2 for sample in hydrograph_samples)
    assert all(tuple(sample["query_points"].shape) == (4, 4, 2) for sample in hydrograph_samples)

"""TDD tests for grouped-hydrograph -> NEONFamilySample conversion (Gap 11)."""

import importlib.util
import sys
import types
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def _ensure_pkg(name: str):
    package = sys.modules.setdefault(name, types.ModuleType(name))
    package.__path__ = [str(REPO_ROOT.joinpath(*name.split(".")))]


def _load_module(name: str, rel_path: str):
    for pkg in ("neuralop", "neuralop.flood", "neuralop.flood.train"):
        _ensure_pkg(pkg)
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


neon = _load_module("neuralop.flood.neon", "neuralop/flood/neon.py")
train_neon = _load_module("neuralop.flood.train.neon", "neuralop/flood/train/neon.py")
fam_mod = _load_module("neuralop.flood.train.neon_families", "neuralop/flood/train/neon_families.py")

grouped_sample_to_family = fam_mod.grouped_sample_to_family
split_families_by_id = fam_mod.split_families_by_id
grouped_samples_to_families = fam_mod.grouped_samples_to_families
build_families_from_config = fam_mod.build_families_from_config
_prepare_family_dataset_config = fam_mod._prepare_family_dataset_config


Nv, Cs, Cb, C, res = 5, 7, 2, 1, 4


def _sample(hid: str, *, R=3, t_total=8, dry=None, area=None):
    sample = {
        "hydrograph_id": hid,
        "geometry": torch.zeros(Nv, 2),
        "static": torch.zeros(Nv, Cs),
        "query_points": torch.zeros(res, res, 2),
        "boundary": torch.arange(t_total * Nv * Cb, dtype=torch.float32).reshape(t_total, Nv, Cb),
        "dynamic_ref": torch.arange(R * t_total * Nv * C, dtype=torch.float32).reshape(R, t_total, Nv, C),
        "n_ref_sims": R,
        "structural_dry_mask": dry,
    }
    if area is not None:
        sample["cell_area"] = area
    return sample


def test_sample_to_family_slices_reference_to_forecast_window():
    skip, n_hist = 2, 3
    t_total, R = 8, 3
    fam = grouped_sample_to_family(_sample("TE1", R=R, t_total=t_total), skip_before_timestep=skip, n_history=n_hist)
    start = skip + n_hist  # 5
    T = t_total - start    # 3
    assert fam.family_id == "TE1"
    assert fam.reference.shape == (R, T, Nv, C)
    assert fam.static.shape == (1, Nv, Cs)
    assert fam.geometry.shape == (1, Nv, 2)
    assert fam.query_points.shape == (1, res, res, 2)
    assert fam.initial_histories.shape == (n_hist, Nv, C)
    expected_history = _sample("TE1", R=R, t_total=t_total)["dynamic_ref"][:, skip : skip + n_hist].mean(dim=0)
    torch.testing.assert_close(fam.initial_histories, expected_history)
    # boundary offset by skip -> length t_total - skip, and covers T + n_history
    assert fam.boundary_sequence.shape[0] == t_total - skip
    assert fam.boundary_sequence.shape[0] >= T + n_hist


def test_sample_to_family_respects_rollout_length_cap():
    fam = grouped_sample_to_family(
        _sample("TE1", t_total=20), skip_before_timestep=2, n_history=3, rollout_length=4
    )
    assert fam.reference.shape[1] == 4


def test_sample_to_family_builds_wettable_weights_from_dry_mask():
    dry = torch.zeros(Nv, dtype=torch.bool)
    dry[:2] = True  # first 2 cells structurally dry
    fam = grouped_sample_to_family(
        _sample("TE1", t_total=8, dry=dry), skip_before_timestep=2, n_history=3
    )
    assert fam.weights is not None
    # dry cells weighted 0, wet cells weighted 1
    assert torch.all(fam.weights[:, :2, :] == 0.0)
    assert torch.all(fam.weights[:, 2:, :] == 1.0)


def test_sample_to_family_uses_uniform_weights_when_mask_and_area_are_absent():
    fam = grouped_sample_to_family(_sample("TE1", dry=None), skip_before_timestep=2, n_history=3)
    assert fam.weights is not None
    assert torch.all(fam.weights == 1.0)


def test_sample_to_family_area_weights_when_cell_area_is_available():
    area = torch.tensor([10.0, 20.0, 0.0, float("nan"), 50.0])
    fam = grouped_sample_to_family(
        _sample("TE1", t_total=8, area=area), skip_before_timestep=2, n_history=3
    )
    assert fam.weights is not None
    torch.testing.assert_close(fam.weights[0, :, 0], torch.tensor([10.0, 20.0, 0.0, 0.0, 50.0]))


def test_sample_to_family_rejects_window_past_horizon():
    with pytest.raises(ValueError, match="start_pred_t"):
        grouped_sample_to_family(_sample("TE1", t_total=4), skip_before_timestep=2, n_history=3)


def test_split_by_explicit_val_ids():
    fams = [
        grouped_sample_to_family(_sample(f"TE{i}"), skip_before_timestep=2, n_history=3)
        for i in range(5)
    ]
    train, val = split_families_by_id(fams, val_family_ids=["TE1", "TE3"])
    assert sorted(f.family_id for f in val) == ["TE1", "TE3"]
    assert sorted(f.family_id for f in train) == ["TE0", "TE2", "TE4"]


def test_split_by_fraction_is_deterministic_and_disjoint():
    fams = [
        grouped_sample_to_family(_sample(f"TE{i:02d}"), skip_before_timestep=2, n_history=3)
        for i in range(10)
    ]
    train, val = split_families_by_id(fams, val_fraction=0.2)
    assert len(val) == 2 and len(train) == 8
    train_ids = {f.family_id for f in train}
    val_ids = {f.family_id for f in val}
    assert train_ids.isdisjoint(val_ids)
    # deterministic: last 2 by sorted id
    assert val_ids == {"TE08", "TE09"}


def test_grouped_samples_to_families_end_to_end_and_max_cap():
    samples = [_sample(f"TE{i}") for i in range(6)]
    train, val = grouped_samples_to_families(
        samples, skip_before_timestep=2, n_history=3, val_fraction=0.34, max_families=5
    )
    assert len(train) + len(val) == 5  # capped
    # produced real families usable by the training loop
    assert all(f.reference.ndim == 4 for f in train + val)


def test_build_families_explicitly_forwards_single_reference_opt_in(monkeypatch):
    captured = {}

    def fake_builder(*args, **kwargs):
        captured.update(kwargs)
        return object(), [_sample("HIST_A", R=1), _sample("HIST_B", R=1)]

    eval_pkg = types.ModuleType("neuralop.flood.eval")
    eval_pkg.__path__ = [str(REPO_ROOT / "neuralop" / "flood" / "eval")]
    datasets_mod = types.ModuleType("neuralop.flood.eval.datasets")
    datasets_mod._build_rollout_normalized_dataset = fake_builder
    monkeypatch.setitem(sys.modules, "neuralop.flood.eval", eval_pkg)
    monkeypatch.setitem(sys.modules, "neuralop.flood.eval.datasets", datasets_mod)

    cfg = SimpleNamespace(
        data=SimpleNamespace(n_history=3, skip_before_timestep=2),
        rollout_data=SimpleNamespace(root="/historical/test", test_txt="historical.txt"),
    )
    train, val = build_families_from_config(
        cfg,
        normalizers={},
        target_variables=["wd"],
        logger=None,
        dataset_split="test",
        allow_single_reference=True,
        val_fraction=0.5,
    )

    assert captured["include_single_reference_groups"] is True
    assert [family.reference.shape[0] for family in train + val] == [1, 1]


def test_training_family_config_uses_train_package_without_mutating_eval_config():
    cfg = SimpleNamespace(
        data=SimpleNamespace(
            root="/scratch/test-package/test",
            train_root="/scratch/train-package/train",
            train_txt="test.txt",  # stale value carried by some eval configs
            n_history=3,
            skip_before_timestep=12,
        ),
        rollout_data=SimpleNamespace(
            root="/scratch/test-package/test",
            test_txt="test.txt",
            boundary=SimpleNamespace(
                channels=[
                    SimpleNamespace(
                        name="stage",
                        mode="clean_family",
                        clean_boundary_file="Stage_Hydrographs_Test_Clean.txt",
                    ),
                    SimpleNamespace(
                        name="precipitation",
                        mode="clean_family",
                        clean_boundary_file="Precipitation_Test_Clean.txt",
                    ),
                ]
            ),
        ),
    )

    prepared, split_name, split_txt = _prepare_family_dataset_config(
        cfg,
        dataset_split="train",
    )

    assert split_name == "train"
    assert split_txt == "train.txt"
    assert prepared.rollout_data.root == "/scratch/train-package/train"
    assert prepared.rollout_data.test_txt == "train.txt"
    clean_files = [
        channel.clean_boundary_file
        for channel in prepared.rollout_data.boundary.channels
    ]
    assert clean_files == [
        "Stage_Hydrographs_Train_Clean.txt",
        "Precipitation_Train_Clean.txt",
    ]

    # The eval config remains a test-package view for ordinary rollout use.
    assert cfg.rollout_data.root == "/scratch/test-package/test"
    assert cfg.rollout_data.test_txt == "test.txt"
    assert cfg.rollout_data.boundary.channels[0].clean_boundary_file.endswith("_Test_Clean.txt")


def test_test_family_config_preserves_rollout_package_view():
    cfg = SimpleNamespace(
        data=SimpleNamespace(train_root="/scratch/train-package/train"),
        rollout_data=SimpleNamespace(root="/scratch/test-package/test", test_txt="test.txt"),
    )

    prepared, split_name, split_txt = _prepare_family_dataset_config(
        cfg,
        dataset_split="test",
    )

    assert prepared is cfg
    assert split_name == "test"
    assert split_txt is None
    assert prepared.rollout_data.root == "/scratch/test-package/test"

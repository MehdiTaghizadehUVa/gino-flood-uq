from pathlib import Path

import numpy as np
import pytest
import torch

try:
    import h5py
except Exception as exc:  # pragma: no cover - environment dependency guard
    pytest.skip(f"h5py unavailable: {exc}", allow_module_level=True)

try:
    from scripts.train_gino_flood_train_rollout_animation_WV import (
        FloodDatasetHDF,
        FloodRolloutTestDatasetHDF,
    )
except Exception as exc:  # pragma: no cover - environment dependency guard
    pytest.skip(f"WV flood dataset loaders unavailable: {exc}", allow_module_level=True)


HDF_GROUPS = {
    "geometry": "Geometry/2D Flow Areas/Cell Points",
    "geometry_cell_centers": "Geometry/2D Flow Areas/Flow Area/Cells Center Coordinate",
    "elevation": "Geometry/2D Flow Areas/Flow Area/Cells Minimum Elevation",
    "area": "Geometry/2D Flow Areas/Flow Area/Cells Surface Area",
    "wd": "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/Flow Area/Cell Hydraulic Depth",
    "vx": "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/Flow Area/Cell Velocity - Velocity X",
    "vy": "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/Flow Area/Cell Velocity - Velocity Y",
    "us_inflow": "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/Flow Area/Boundary Conditions/US Inflow",
}


def _ensure_group(handle: h5py.File, dataset_path: str):
    parts = dataset_path.split("/")
    grp = handle
    for part in parts[:-1]:
        grp = grp.require_group(part)
    return grp, parts[-1]


def _write_dataset(handle: h5py.File, dataset_path: str, value):
    grp, name = _ensure_group(handle, dataset_path)
    grp.create_dataset(name, data=value)


def _write_minimal_hdf(path: Path, *, inflow_offset: float, wd_offset: float, n_time: int = 6, n_cells: int = 3):
    geometry = np.stack(
        [
            np.linspace(0.0, 1.0, n_cells, dtype=np.float32),
            np.linspace(0.0, 1.0, n_cells, dtype=np.float32),
        ],
        axis=1,
    )
    elevation = np.linspace(10.0, 11.0, n_cells, dtype=np.float32)
    area = np.full((n_cells,), 5.0, dtype=np.float32)
    time_axis = np.arange(n_time, dtype=np.float32)
    wd = np.tile(time_axis[:, None], (1, n_cells)).astype(np.float32) + wd_offset
    vx = wd + 0.1
    vy = wd + 0.2
    inflow = np.stack([time_axis, 10.0 + inflow_offset + time_axis], axis=1).astype(np.float32)

    with h5py.File(path, "w") as handle:
        _write_dataset(handle, HDF_GROUPS["geometry"], geometry)
        _write_dataset(handle, HDF_GROUPS["geometry_cell_centers"], geometry)
        _write_dataset(handle, HDF_GROUPS["elevation"], elevation)
        _write_dataset(handle, HDF_GROUPS["area"], area)
        _write_dataset(handle, HDF_GROUPS["wd"], wd)
        _write_dataset(handle, HDF_GROUPS["vx"], vx)
        _write_dataset(handle, HDF_GROUPS["vy"], vy)
        _write_dataset(handle, HDF_GROUPS["us_inflow"], inflow)


def _write_clean_boundary_table(path: Path, family_to_series: dict[str, np.ndarray]):
    families = list(family_to_series.keys())
    matrix = np.column_stack([np.asarray(family_to_series[f], dtype=np.float32) for f in families])
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(families) + "\n")
        np.savetxt(handle, matrix, delimiter="\t", fmt="%.6f")


def _make_fixture(tmp_path: Path):
    data_root = tmp_path / "data"
    meta_root = tmp_path / "metadata"
    data_root.mkdir()
    meta_root.mkdir()

    run_ids = ["TR000001_sim00", "TR000001_sim01"]
    _write_minimal_hdf(data_root / f"{run_ids[0]}.hdf", inflow_offset=0.0, wd_offset=0.0)
    _write_minimal_hdf(data_root / f"{run_ids[1]}.hdf", inflow_offset=50.0, wd_offset=25.0)

    clean_series = np.array([100, 101, 102, 103, 104, 105], dtype=np.float32)
    _write_clean_boundary_table(
        meta_root / "Hydrographs_Train_Clean.txt",
        {"TR000001": clean_series},
    )
    _write_clean_boundary_table(
        meta_root / "Hydrographs_Test_Clean.txt",
        {"TR000001": clean_series},
    )
    return data_root, meta_root, run_ids, clean_series


def _first_index_for_run(dataset: FloodDatasetHDF, run_id: str) -> int:
    for idx, (sample_run_id, _) in enumerate(dataset.sample_index):
        if sample_run_id == run_id:
            return idx
    raise AssertionError(f"run_id {run_id} not found in sample_index")


def test_clean_family_training_boundary_is_shared_within_family(tmp_path: Path):
    data_root, meta_root, run_ids, clean_series = _make_fixture(tmp_path)
    ds = FloodDatasetHDF(
        data_root=str(data_root),
        n_history=2,
        ar_rollout_steps=2,
        run_ids=run_ids,
        boundary_source="clean_family",
        clean_boundary_root=str(meta_root),
        clean_boundary_file="Hydrographs_Train_Clean.txt",
        target_variables=["wd"],
    )

    sample_a = ds[_first_index_for_run(ds, run_ids[0])]
    sample_b = ds[_first_index_for_run(ds, run_ids[1])]

    assert torch.equal(sample_a["boundary"], sample_b["boundary"])
    assert torch.equal(sample_a["boundary_sequence"], sample_b["boundary_sequence"])
    assert not torch.equal(sample_a["dynamic"], sample_b["dynamic"])
    assert not torch.equal(sample_a["target_sequence"], sample_b["target_sequence"])

    expected_hist = torch.tensor(clean_series[:2], dtype=torch.float32).view(2, 1, 1).expand(-1, 3, -1)
    expected_future = torch.tensor(clean_series[2:4], dtype=torch.float32).view(2, 1, 1).expand(-1, 3, -1)
    assert torch.allclose(sample_a["boundary"], expected_hist)
    assert torch.allclose(sample_a["boundary_sequence"], expected_future)


def test_clean_family_rollout_boundary_is_shared_within_family(tmp_path: Path):
    data_root, meta_root, run_ids, clean_series = _make_fixture(tmp_path)
    ds = FloodRolloutTestDatasetHDF(
        rollout_data_root=str(data_root),
        n_history=2,
        rollout_length=3,
        run_ids=run_ids,
        boundary_source="clean_family",
        clean_boundary_root=str(meta_root),
        clean_boundary_file="Hydrographs_Test_Clean.txt",
    )

    sample_a = ds[0]
    sample_b = ds[1]

    assert torch.equal(sample_a["boundary"], sample_b["boundary"])
    assert not torch.equal(sample_a["dynamic"], sample_b["dynamic"])
    expected = torch.tensor(clean_series, dtype=torch.float32).view(-1, 1, 1).expand(-1, 3, -1)
    assert torch.allclose(sample_a["boundary"], expected)


def test_clean_family_missing_family_raises_clear_error(tmp_path: Path):
    data_root, meta_root, run_ids, _ = _make_fixture(tmp_path)
    _write_clean_boundary_table(
        meta_root / "Hydrographs_Train_Clean.txt",
        {"TR999999": np.arange(6, dtype=np.float32)},
    )
    ds = FloodDatasetHDF(
        data_root=str(data_root),
        n_history=2,
        ar_rollout_steps=2,
        run_ids=run_ids[:1],
        boundary_source="clean_family",
        clean_boundary_root=str(meta_root),
        clean_boundary_file="Hydrographs_Train_Clean.txt",
        target_variables=["wd"],
    )
    with pytest.raises(KeyError, match="Family 'TR000001' not found"):
        _ = ds[0]


def test_clean_family_length_mismatch_raises_clear_error(tmp_path: Path):
    data_root, meta_root, run_ids, _ = _make_fixture(tmp_path)
    _write_clean_boundary_table(
        meta_root / "Hydrographs_Train_Clean.txt",
        {"TR000001": np.arange(3, dtype=np.float32)},
    )
    ds = FloodDatasetHDF(
        data_root=str(data_root),
        n_history=2,
        ar_rollout_steps=2,
        run_ids=run_ids[:1],
        boundary_source="clean_family",
        clean_boundary_root=str(meta_root),
        clean_boundary_file="Hydrographs_Train_Clean.txt",
        target_variables=["wd"],
    )
    with pytest.raises(ValueError, match="Clean boundary slice \\[0:4\\] is invalid"):
        _ = ds[0]


def test_member_hdf_mode_preserves_legacy_member_specific_boundary(tmp_path: Path):
    data_root, _, run_ids, _ = _make_fixture(tmp_path)
    ds = FloodDatasetHDF(
        data_root=str(data_root),
        n_history=2,
        ar_rollout_steps=2,
        run_ids=run_ids,
        boundary_source="member_hdf",
        target_variables=["wd"],
    )

    sample_a = ds[_first_index_for_run(ds, run_ids[0])]
    sample_b = ds[_first_index_for_run(ds, run_ids[1])]

    assert not torch.equal(sample_a["boundary"], sample_b["boundary"])

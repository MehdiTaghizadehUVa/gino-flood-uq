from pathlib import Path

import numpy as np
import pytest
import torch

try:
    import h5py
except Exception as exc:  # pragma: no cover - environment dependency guard
    pytest.skip(f"h5py unavailable: {exc}", allow_module_level=True)

try:
    from neuralop.flood.data.wv import (
        FloodDatasetHDF,
        FloodRolloutTestDatasetHDF,
    )
    from neuralop.flood.processing.wv import _build_x_from_dynamic_boundary
    from neuralop.flood.utils.runtime import get_dataset_boundary_kwargs
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
    "stage": "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/Flow Area/Boundary Conditions/Stage Hydrographs",
    "precipitation": "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/Flow Area/Boundary Conditions/Precipitation Hydrographs",
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
    stage = (20.0 + inflow_offset + time_axis).astype(np.float32)
    precipitation = (30.0 + inflow_offset + 2.0 * time_axis).astype(np.float32)

    with h5py.File(path, "w") as handle:
        _write_dataset(handle, HDF_GROUPS["geometry"], geometry)
        _write_dataset(handle, HDF_GROUPS["geometry_cell_centers"], geometry)
        _write_dataset(handle, HDF_GROUPS["elevation"], elevation)
        _write_dataset(handle, HDF_GROUPS["area"], area)
        _write_dataset(handle, HDF_GROUPS["wd"], wd)
        _write_dataset(handle, HDF_GROUPS["vx"], vx)
        _write_dataset(handle, HDF_GROUPS["vy"], vy)
        _write_dataset(handle, HDF_GROUPS["us_inflow"], inflow)
        _write_dataset(handle, HDF_GROUPS["stage"], stage)
        _write_dataset(handle, HDF_GROUPS["precipitation"], precipitation)


def _write_subset_geometry_hdf(path: Path, *, n_time: int = 6):
    centers = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
        ],
        dtype=np.float32,
    )
    point_idx = np.array([0, 2, 4], dtype=np.int64)
    geometry = centers[point_idx]
    elevation = np.array([10, 11, 12, 13, 14], dtype=np.float32)
    area = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    time_axis = np.arange(n_time, dtype=np.float32)
    wd = np.tile(time_axis[:, None], (1, centers.shape[0])).astype(np.float32)
    vx = wd + 0.1
    vy = wd + 0.2
    inflow = np.stack([time_axis, 10.0 + time_axis], axis=1).astype(np.float32)

    with h5py.File(path, "w") as handle:
        _write_dataset(handle, HDF_GROUPS["geometry"], geometry)
        _write_dataset(handle, HDF_GROUPS["geometry_cell_centers"], centers)
        _write_dataset(handle, HDF_GROUPS["elevation"], elevation)
        _write_dataset(handle, HDF_GROUPS["area"], area)
        _write_dataset(handle, HDF_GROUPS["wd"], wd)
        _write_dataset(handle, HDF_GROUPS["vx"], vx)
        _write_dataset(handle, HDF_GROUPS["vy"], vy)
        _write_dataset(handle, HDF_GROUPS["us_inflow"], inflow)
    return point_idx


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
    stage_clean = np.array([100, 101, 102, 103, 104, 105], dtype=np.float32)
    precip_clean = np.array([200, 201, 202, 203, 204, 205], dtype=np.float32)
    _write_clean_boundary_table(
        meta_root / "Hydrographs_Train_Clean.txt",
        {"TR000001": clean_series},
    )
    _write_clean_boundary_table(
        meta_root / "Hydrographs_Test_Clean.txt",
        {"TR000001": clean_series},
    )
    _write_clean_boundary_table(
        meta_root / "Stage_Hydrographs_Train_Clean.txt",
        {"TR000001": stage_clean},
    )
    _write_clean_boundary_table(
        meta_root / "Precipitation_Train_Clean.txt",
        {"TR000001": precip_clean},
    )
    _write_clean_boundary_table(
        meta_root / "Stage_Hydrographs_Test_Clean.txt",
        {"TR000001": stage_clean},
    )
    _write_clean_boundary_table(
        meta_root / "Precipitation_Test_Clean.txt",
        {"TR000001": precip_clean},
    )
    return data_root, meta_root, run_ids, clean_series, stage_clean, precip_clean


def _first_index_for_run(dataset: FloodDatasetHDF, run_id: str) -> int:
    for idx, (sample_run_id, _) in enumerate(dataset.sample_index):
        if sample_run_id == run_id:
            return idx
    raise AssertionError(f"run_id {run_id} not found in sample_index")


def test_clean_family_training_boundary_is_shared_within_family(tmp_path: Path):
    data_root, meta_root, run_ids, clean_series, _, _ = _make_fixture(tmp_path)
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
    data_root, meta_root, run_ids, clean_series, _, _ = _make_fixture(tmp_path)
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


def test_clean_family_diagnostics_load_member_specific_sibling_forcings(tmp_path: Path):
    from neuralop.flood.eval.datasets import _boundary_ensemble_series_from_reference_members

    meta_root = tmp_path / "metadata_member_forcing"
    meta_root.mkdir()
    _write_clean_boundary_table(
        meta_root / "Stage_Hydrographs_Test_Clean.txt",
        {"TE000001": np.array([10.0, 11.0, 12.0], dtype=np.float32)},
    )
    _write_clean_boundary_table(
        meta_root / "Stage_Hydrographs_Test.txt",
        {
            "TE000001_sim00": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "TE000001_sim01": np.array([4.0, 5.0, 6.0], dtype=np.float32),
        },
    )
    fallback = torch.zeros((2, 3, 1), dtype=torch.float32)

    ensemble = _boundary_ensemble_series_from_reference_members(
        [
            {
                "name": "stage",
                "mode": "clean_family",
                "clean_boundary_root": str(meta_root),
                "clean_boundary_file": "Stage_Hydrographs_Test_Clean.txt",
            }
        ],
        ["Flood_coastal_TE000001_sim00", "Flood_coastal_TE000001_sim01"],
        fallback,
    )

    assert ensemble is not None
    assert ensemble.shape == (2, 3, 1)
    expected = torch.tensor([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]])
    assert torch.allclose(ensemble, expected)


def test_clean_family_prefixed_run_id_resolves_to_clean_event_id(tmp_path: Path):
    data_root = tmp_path / "data_prefixed"
    meta_root = tmp_path / "metadata_prefixed"
    data_root.mkdir()
    meta_root.mkdir()

    run_ids = ["M40_TR000001_sim00", "M40_TR000001_sim01"]
    _write_minimal_hdf(data_root / f"{run_ids[0]}.hdf", inflow_offset=0.0, wd_offset=0.0)
    _write_minimal_hdf(data_root / f"{run_ids[1]}.hdf", inflow_offset=50.0, wd_offset=25.0)

    clean_series = np.array([100, 101, 102, 103, 104, 105], dtype=np.float32)
    _write_clean_boundary_table(
        meta_root / "Hydrographs_Train_Clean.txt",
        {"TR000001": clean_series},
    )

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
    expected_hist = torch.tensor(clean_series[:2], dtype=torch.float32).view(2, 1, 1).expand(-1, 3, -1)
    assert torch.allclose(sample_a["boundary"], expected_hist)


def test_clean_family_missing_family_raises_clear_error(tmp_path: Path):
    data_root, meta_root, run_ids, _, _, _ = _make_fixture(tmp_path)
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
    with pytest.raises(KeyError, match="Family 'TR000001' not found in clean boundary file"):
        _ = ds[0]


def test_clean_family_length_mismatch_raises_clear_error(tmp_path: Path):
    data_root, meta_root, run_ids, _, _, _ = _make_fixture(tmp_path)
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
    data_root, _, run_ids, _, _, _ = _make_fixture(tmp_path)
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


def test_multi_channel_clean_family_training_boundary_stacks_channels(tmp_path: Path):
    data_root, meta_root, run_ids, _, stage_clean, precip_clean = _make_fixture(tmp_path)
    ds = FloodDatasetHDF(
        data_root=str(data_root),
        n_history=2,
        ar_rollout_steps=2,
        run_ids=run_ids,
        boundary_spec=[
            {
                "name": "stage",
                "mode": "clean_family",
                "clean_boundary_root": str(meta_root),
                "clean_boundary_file": "Stage_Hydrographs_Train_Clean.txt",
            },
            {
                "name": "precipitation",
                "mode": "clean_family",
                "clean_boundary_root": str(meta_root),
                "clean_boundary_file": "Precipitation_Train_Clean.txt",
            },
        ],
        target_variables=["wd"],
    )

    sample = ds[_first_index_for_run(ds, run_ids[0])]
    expected_hist = torch.tensor(
        np.stack([stage_clean[:2], precip_clean[:2]], axis=1),
        dtype=torch.float32,
    ).view(2, 1, 2).expand(-1, 3, -1)
    expected_future = torch.tensor(
        np.stack([stage_clean[2:4], precip_clean[2:4]], axis=1),
        dtype=torch.float32,
    ).view(2, 1, 2).expand(-1, 3, -1)

    assert sample["boundary"].shape == (2, 3, 2)
    assert sample["boundary_sequence"].shape == (2, 3, 2)
    assert torch.allclose(sample["boundary"], expected_hist)
    assert torch.allclose(sample["boundary_sequence"], expected_future)


def test_multi_channel_member_hdf_training_boundary_preserves_both_channels(tmp_path: Path):
    data_root, _, run_ids, _, _, _ = _make_fixture(tmp_path)
    ds = FloodDatasetHDF(
        data_root=str(data_root),
        n_history=2,
        ar_rollout_steps=2,
        run_ids=run_ids,
        boundary_spec=[
            {
                "name": "stage",
                "mode": "member_hdf",
                "hdf_path": HDF_GROUPS["stage"],
            },
            {
                "name": "precipitation",
                "mode": "member_hdf",
                "hdf_path": HDF_GROUPS["precipitation"],
            },
        ],
        target_variables=["wd"],
    )

    sample_a = ds[_first_index_for_run(ds, run_ids[0])]
    sample_b = ds[_first_index_for_run(ds, run_ids[1])]

    expected_a = torch.tensor(
        [[[20.0, 30.0]], [[21.0, 32.0]]],
        dtype=torch.float32,
    ).expand(-1, 3, -1)
    expected_b = torch.tensor(
        [[[70.0, 80.0]], [[71.0, 82.0]]],
        dtype=torch.float32,
    ).expand(-1, 3, -1)

    assert sample_a["boundary"].shape == (2, 3, 2)
    assert sample_b["boundary"].shape == (2, 3, 2)
    assert torch.allclose(sample_a["boundary"], expected_a)
    assert torch.allclose(sample_b["boundary"], expected_b)


def test_multi_channel_clean_family_rollout_boundary_stacks_channels(tmp_path: Path):
    data_root, meta_root, run_ids, _, stage_clean, precip_clean = _make_fixture(tmp_path)
    ds = FloodRolloutTestDatasetHDF(
        rollout_data_root=str(data_root),
        n_history=2,
        rollout_length=3,
        run_ids=run_ids,
        boundary_spec=[
            {
                "name": "stage",
                "mode": "clean_family",
                "clean_boundary_root": str(meta_root),
                "clean_boundary_file": "Stage_Hydrographs_Test_Clean.txt",
            },
            {
                "name": "precipitation",
                "mode": "clean_family",
                "clean_boundary_root": str(meta_root),
                "clean_boundary_file": "Precipitation_Test_Clean.txt",
            },
        ],
    )

    sample = ds[0]
    expected = torch.tensor(
        np.stack([stage_clean, precip_clean], axis=1),
        dtype=torch.float32,
    ).view(-1, 1, 2).expand(-1, 3, -1)
    assert sample["boundary"].shape == (6, 3, 2)
    assert torch.allclose(sample["boundary"], expected)


def test_get_dataset_boundary_kwargs_normalizes_explicit_multichannel_config(tmp_path: Path):
    section = {
        "root": str(tmp_path),
        "boundary": {
            "channels": [
                {
                    "name": "stage",
                    "mode": "clean_family",
                    "clean_boundary_root": str(tmp_path),
                    "clean_boundary_file": "Stage_Hydrographs_Train_Clean.txt",
                },
                {
                    "name": "precipitation",
                    "mode": "member_hdf",
                    "hdf_path": HDF_GROUPS["precipitation"],
                },
            ]
        },
    }
    kwargs = get_dataset_boundary_kwargs(section, split="train")
    assert kwargs["boundary_source"] == "multi_channel"
    assert [channel["name"] for channel in kwargs["boundary_spec"]] == [
        "stage",
        "precipitation",
    ]
    assert kwargs["boundary_spec"][0]["mode"] == "clean_family"
    assert kwargs["boundary_spec"][1]["mode"] == "member_hdf"


def test_get_dataset_boundary_kwargs_rejects_ambiguous_legacy_and_explicit_config(tmp_path: Path):
    section = {
        "root": str(tmp_path),
        "boundary_source": "member_hdf",
        "boundary": {
            "channels": [
                {
                    "name": "stage",
                    "mode": "member_hdf",
                    "hdf_path": HDF_GROUPS["stage"],
                }
            ]
        },
    }
    with pytest.raises(ValueError, match="When boundary.channels is configured"):
        get_dataset_boundary_kwargs(section, split="train")


def test_processor_flattens_multi_channel_boundary_history():
    static = torch.zeros((1, 3, 2), dtype=torch.float32)
    boundary = torch.arange(12, dtype=torch.float32).reshape(1, 2, 3, 2)
    dynamic = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3, 1)

    x = _build_x_from_dynamic_boundary(static, boundary, dynamic)

    assert x.shape == (1, 3, 2 + 2 * 2 + 2 * 1)
    expected_cell0 = torch.tensor([0.0, 0.0, 0.0, 1.0, 6.0, 7.0, 0.0, 3.0])
    assert torch.allclose(x[0, 0], expected_cell0)


def test_static_text_full_cell_grid_is_aligned_to_cell_points_for_training(tmp_path: Path):
    data_root = tmp_path / "data_static_align"
    data_root.mkdir()
    run_id = "TR000001_sim00"
    point_idx = _write_subset_geometry_hdf(data_root / f"{run_id}.hdf")

    slope_full = np.array([10, 20, 30, 40, 50], dtype=np.float32)
    np.savetxt(data_root / "Slope.txt", slope_full, delimiter="\t", fmt="%.6f")

    ds = FloodDatasetHDF(
        data_root=str(data_root),
        n_history=2,
        ar_rollout_steps=1,
        run_ids=[run_id],
        static_text_files=["Slope.txt"],
        target_variables=["wd"],
    )
    sample = ds[0]

    expected = torch.tensor(slope_full[point_idx], dtype=torch.float32)
    assert sample["static"].shape == (3, 3)
    assert torch.allclose(sample["static"][:, 2], expected)


def test_static_text_full_cell_grid_is_aligned_to_cell_points_for_rollout(tmp_path: Path):
    data_root = tmp_path / "data_static_rollout_align"
    data_root.mkdir()
    run_id = "TR000001_sim00"
    point_idx = _write_subset_geometry_hdf(data_root / f"{run_id}.hdf")

    slope_full = np.array([10, 20, 30, 40, 50], dtype=np.float32)
    np.savetxt(data_root / "Slope.txt", slope_full, delimiter="\t", fmt="%.6f")

    ds = FloodRolloutTestDatasetHDF(
        rollout_data_root=str(data_root),
        n_history=2,
        rollout_length=3,
        run_ids=[run_id],
        static_text_files=["Slope.txt"],
    )
    sample = ds[0]

    expected = torch.tensor(slope_full[point_idx], dtype=torch.float32)
    assert sample["static"].shape == (3, 3)
    assert torch.allclose(sample["static"][:, 2], expected)


def test_static_text_mismatched_length_raises_clear_error(tmp_path: Path):
    data_root = tmp_path / "data_static_bad"
    data_root.mkdir()
    run_id = "TR000001_sim00"
    _write_subset_geometry_hdf(data_root / f"{run_id}.hdf")

    slope_bad = np.array([10, 20, 30, 40], dtype=np.float32)
    np.savetxt(data_root / "Slope.txt", slope_bad, delimiter="\t", fmt="%.6f")

    with pytest.raises(ValueError, match="does not match either reference_cell_count=3 or full_cell_count=5"):
        FloodDatasetHDF(
            data_root=str(data_root),
            n_history=2,
            ar_rollout_steps=1,
            run_ids=[run_id],
            static_text_files=["Slope.txt"],
            target_variables=["wd"],
        )

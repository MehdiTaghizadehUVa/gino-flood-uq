from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Subset

try:
    import h5py
except Exception as exc:  # pragma: no cover - environment dependency guard
    pytest.skip(f"h5py unavailable: {exc}", allow_module_level=True)

from neuralop.flood.data.normalization_impl import (
    build_normalizer_metadata,
    fit_normalizers,
    fit_normalizers_streaming,
    load_normalizer_metadata,
    normalizer_metadata_matches,
    resolve_normalizer_fit_method,
    resolve_normalizer_metadata_path,
    save_normalizer_metadata,
)
from neuralop.flood.data.wv import FloodDatasetHDF


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


def _write_minimal_hdf(path: Path, *, inflow_offset: float, wd_offset: float, n_time: int = 7, n_cells: int = 3):
    geometry = np.stack(
        [
            np.linspace(0.0, 1.0, n_cells, dtype=np.float32),
            np.linspace(0.0, 1.0, n_cells, dtype=np.float32),
        ],
        axis=1,
    )
    elevation = np.linspace(10.0, 11.0, n_cells, dtype=np.float32)
    area = np.linspace(5.0, 7.0, n_cells, dtype=np.float32)
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

    stage_clean = np.array([100, 101, 102, 103, 104, 105, 106], dtype=np.float32)
    precip_clean = np.array([200, 201, 202, 203, 204, 205, 206], dtype=np.float32)
    _write_clean_boundary_table(
        meta_root / "Stage_Hydrographs_Train_Clean.txt",
        {"TR000001": stage_clean},
    )
    _write_clean_boundary_table(
        meta_root / "Precipitation_Train_Clean.txt",
        {"TR000001": precip_clean},
    )
    return data_root, meta_root, run_ids


def _assert_normalizers_close(lhs, rhs):
    for key in ("geometry", "static", "boundary", "target"):
        assert torch.allclose(lhs[key].mean, rhs[key].mean, atol=1e-4, rtol=1e-4), key
        assert torch.allclose(lhs[key].std, rhs[key].std, atol=1e-4, rtol=1e-4), key
    assert lhs["dynamic"] is lhs["target"]
    assert rhs["dynamic"] is rhs["target"]


def test_fast_exact_matches_streaming_for_member_hdf_multichannel_raw_dataset(tmp_path: Path):
    data_root, _, run_ids = _make_fixture(tmp_path)
    dataset = FloodDatasetHDF(
        data_root=str(data_root),
        n_history=2,
        ar_rollout_steps=2,
        run_ids=run_ids,
        boundary_spec=[
            {"name": "stage", "mode": "member_hdf", "hdf_path": HDF_GROUPS["stage"]},
            {"name": "precipitation", "mode": "member_hdf", "hdf_path": HDF_GROUPS["precipitation"]},
        ],
        target_variables=["wd"],
    )

    streaming = fit_normalizers_streaming(
        dataset,
        chunk_size=1,
        expect_target=True,
        structural_dry_policy="legacy_full_domain",
    )
    exact, resolved_method = fit_normalizers(
        dataset,
        chunk_size=1,
        expect_target=True,
        structural_dry_policy="legacy_full_domain",
        method="auto",
        return_method=True,
    )

    assert resolved_method == "fast_exact"
    _assert_normalizers_close(streaming, exact)


def test_fast_exact_matches_streaming_for_clean_family_multichannel_subset(tmp_path: Path):
    data_root, meta_root, run_ids = _make_fixture(tmp_path)
    dataset = FloodDatasetHDF(
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
    subset = Subset(dataset, [0, 1, 3, 4])

    streaming = fit_normalizers_streaming(
        subset,
        chunk_size=1,
        expect_target=True,
        structural_dry_policy="legacy_full_domain",
    )
    exact = fit_normalizers(
        subset,
        chunk_size=1,
        expect_target=True,
        structural_dry_policy="legacy_full_domain",
        method="fast_exact",
    )

    _assert_normalizers_close(streaming, exact)


def test_normalizer_metadata_tracks_subset_fingerprint_and_sidecar_path(tmp_path: Path):
    data_root, meta_root, run_ids = _make_fixture(tmp_path)
    dataset = FloodDatasetHDF(
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
    subset_a = Subset(dataset, [0, 1, 2, 3])
    subset_b = Subset(dataset, [0, 1, 3, 4])

    meta_a = build_normalizer_metadata(
        subset_a,
        structural_dry_policy="legacy_full_domain",
        fit_method="fast_exact",
    )
    meta_b = build_normalizer_metadata(
        subset_b,
        structural_dry_policy="legacy_full_domain",
        fit_method="fast_exact",
    )

    assert meta_a["split_fingerprint"] != meta_b["split_fingerprint"]
    path = tmp_path / "normalizers_depth_only.pt"
    metadata_path = resolve_normalizer_metadata_path(path)
    assert metadata_path.name == "normalizers_depth_only.metadata.json"

    save_normalizer_metadata(metadata_path, meta_a)
    loaded = load_normalizer_metadata(metadata_path)
    assert normalizer_metadata_matches(meta_a, loaded)
    assert not normalizer_metadata_matches(meta_b, loaded)


def test_auto_falls_back_to_streaming_for_masked_primary():
    class _TinyDataset:
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {
                "geometry": torch.zeros((2, 2), dtype=torch.float32),
                "static": torch.zeros((2, 1), dtype=torch.float32),
                "boundary": torch.zeros((2, 2, 1), dtype=torch.float32),
                "dynamic": torch.zeros((2, 2, 1), dtype=torch.float32),
                "target": torch.zeros((2, 1), dtype=torch.float32),
                "structural_dry_mask": torch.tensor([False, True]),
            }

    dataset = _TinyDataset()
    assert (
        resolve_normalizer_fit_method(
            dataset,
            method="auto",
            structural_dry_policy="masked_primary",
        )
        == "streaming"
    )
    with pytest.raises(ValueError):
        resolve_normalizer_fit_method(
            dataset,
            method="fast_exact",
            structural_dry_policy="masked_primary",
        )

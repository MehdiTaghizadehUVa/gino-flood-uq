import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from typing import Dict, Tuple

try:
    import h5py
except Exception as exc:  # pragma: no cover
    pytest.skip(f"h5py unavailable: {exc}", allow_module_level=True)

from neuralop.flood.cli.prepare_flood_training_artifacts import prepare_training_artifacts


HDF_GROUPS = {
    "geometry": "Geometry/2D Flow Areas/Cell Points",
    "geometry_cell_centers": "Geometry/2D Flow Areas/Flow Area/Cells Center Coordinate",
    "elevation": "Geometry/2D Flow Areas/Flow Area/Cells Minimum Elevation",
    "area": "Geometry/2D Flow Areas/Flow Area/Cells Surface Area",
    "wd": "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/Flow Area/Cell Hydraulic Depth",
    "vx": "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/Flow Area/Cell Velocity - Velocity X",
    "vy": "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/Flow Area/Cell Velocity - Velocity Y",
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


def _write_hdf(path: Path, *, wd_offset: float, n_time: int = 6, n_cells: int = 3):
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
    with h5py.File(path, "w") as handle:
        _write_dataset(handle, HDF_GROUPS["geometry"], geometry)
        _write_dataset(handle, HDF_GROUPS["geometry_cell_centers"], geometry)
        _write_dataset(handle, HDF_GROUPS["elevation"], elevation)
        _write_dataset(handle, HDF_GROUPS["area"], area)
        _write_dataset(handle, HDF_GROUPS["wd"], wd)
        _write_dataset(handle, HDF_GROUPS["vx"], vx)
        _write_dataset(handle, HDF_GROUPS["vy"], vy)


def _write_clean_boundary_table(path: Path, family_to_series: Dict[str, np.ndarray]):
    families = list(family_to_series.keys())
    matrix = np.column_stack([np.asarray(family_to_series[f], dtype=np.float32) for f in families])
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(families) + "\n")
        np.savetxt(handle, matrix, delimiter="\t", fmt="%.6f")


def _write_static_file(path: Path, values):
    np.savetxt(path, np.asarray(values, dtype=np.float32), fmt="%.6f")


def _make_fixture(tmp_path: Path) -> Tuple[Path, Path, Path]:
    data_root = tmp_path / "train"
    meta_root = tmp_path / "metadata"
    data_root.mkdir()
    meta_root.mkdir()

    (data_root / "train.txt").write_text("TR000001_sim00\nTR000001_sim01\n", encoding="utf-8")
    _write_hdf(data_root / "TR000001_sim00.hdf", wd_offset=0.0)
    _write_hdf(data_root / "TR000001_sim01.hdf", wd_offset=1.0)
    _write_clean_boundary_table(
        meta_root / "Hydrographs_Train_Clean.txt",
        {"TR000001": np.array([100, 101, 102, 103, 104, 105], dtype=np.float32)},
    )
    for name, base in (("M40_CS.txt", 1.0), ("M40_CU.txt", 2.0), ("M40_FA.txt", 3.0)):
        _write_static_file(data_root / name, [base, base + 1.0, base + 2.0])

    config_path = tmp_path / "masked.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "flood": {
                    "distributed": {"seed": 123},
                    "data": {
                        "root": str(data_root),
                        "boundary_source": "clean_family",
                        "clean_boundary_root": str(meta_root),
                        "clean_boundary_file": "Hydrographs_Train_Clean.txt",
                        "n_history": 2,
                        "query_res": [48, 48],
                        "train_txt": "train.txt",
                        "write_train_txt": False,
                        "static_text_files": ["M40_CS.txt", "M40_CU.txt", "M40_FA.txt"],
                        "noise_type": "none",
                        "noise_std": [0.01],
                        "target_variables": ["wd"],
                        "skip_before_timestep": 0,
                        "normalizer_chunk_size": 2,
                        "normalizer_fit_method": "auto",
                        "normalizer_path": "normalizers_depth_only_masked_primary.pt",
                    },
                    "opt": {"ar_rollout_steps": 2},
                    "structural_dry": {
                        "policy": "masked_primary",
                        "mask_definition": "exact_zero",
                    },
                }
            },
            default_flow_style=False,
        ),
        encoding="utf-8",
    )
    return data_root, meta_root, config_path


def test_prepare_training_artifacts_writes_masked_primary_outputs(tmp_path: Path):
    data_root, _, config_path = _make_fixture(tmp_path)
    artifact_root = tmp_path / "artifacts"
    result = prepare_training_artifacts(
        config_path=config_path,
        artifact_root=artifact_root,
        data_root=str(data_root),
        seed=123,
    )
    assert result["structural_dry_policy"] == "masked_primary"
    assert result["normalizer_fit_method"] == "streaming"
    assert Path(result["normalizer_path"]).exists()
    assert Path(result["normalizer_metadata_path"]).exists()
    assert Path(result["structural_dry_artifact_path"]).exists()
    assert Path(result["structural_dry_summary_path"]).exists()
    summary = json.loads(Path(result["prep_summary_path"]).read_text(encoding="utf-8"))
    assert summary["normalizer_fit_method"] == "streaming"
    assert summary["artifact_root"] == str(artifact_root.resolve())


def test_prepare_training_artifacts_reuses_matching_outputs(tmp_path: Path):
    data_root, _, config_path = _make_fixture(tmp_path)
    artifact_root = tmp_path / "artifacts"
    first = prepare_training_artifacts(
        config_path=config_path,
        artifact_root=artifact_root,
        data_root=str(data_root),
        seed=123,
    )
    second = prepare_training_artifacts(
        config_path=config_path,
        artifact_root=artifact_root,
        data_root=str(data_root),
        seed=123,
    )
    assert first["normalizer_path"] == second["normalizer_path"]
    assert second["artifact_status"] == "reused"
    assert second["normalizer_status"] == "reused"


def test_prepare_training_artifacts_metadata_mismatch_requires_overwrite(tmp_path: Path):
    data_root, _, config_path = _make_fixture(tmp_path)
    artifact_root = tmp_path / "artifacts"
    prepare_training_artifacts(
        config_path=config_path,
        artifact_root=artifact_root,
        data_root=str(data_root),
        seed=123,
    )
    with pytest.raises(RuntimeError, match="metadata mismatch"):
        prepare_training_artifacts(
            config_path=config_path,
            artifact_root=artifact_root,
            data_root=str(data_root),
            seed=999,
        )
    result = prepare_training_artifacts(
        config_path=config_path,
        artifact_root=artifact_root,
        data_root=str(data_root),
        seed=999,
        overwrite=True,
    )
    assert result["normalizer_status"] == "computed"

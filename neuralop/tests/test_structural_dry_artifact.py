import json

import numpy as np
import pytest
import torch

from neuralop.flood.data.structural_dry import (
    HDF_PATHS,
    STRUCTURAL_DRY_MASK_DEFINITION_EXACT_ZERO,
    broadcast_wettable_mask,
    build_structural_dry_artifact,
    load_structural_dry_artifact,
    save_structural_dry_artifact,
    validate_structural_dry_artifact,
)

try:
    import h5py
except Exception as exc:  # pragma: no cover
    h5py = None
    H5PY_IMPORT_ERROR = exc
else:
    H5PY_IMPORT_ERROR = None


pytestmark = pytest.mark.skipif(h5py is None, reason=f"h5py unavailable: {H5PY_IMPORT_ERROR}")


def _write_wd_hdf(path, wd):
    with h5py.File(path, "w") as handle:
        handle.create_dataset(HDF_PATHS["wd"], data=np.asarray(wd, dtype=np.float32))


def test_build_save_load_validate_structural_dry_artifact(tmp_path):
    (tmp_path / "train.txt").write_text("run01\nrun02\n", encoding="utf-8")
    _write_wd_hdf(
        tmp_path / "run01.hdf",
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.5, 0.0],
        ],
    )
    _write_wd_hdf(
        tmp_path / "run02.hdf",
        [
            [0.0, 0.2, 0.0],
            [0.0, 2.0, 0.0],
        ],
    )

    artifact = build_structural_dry_artifact(
        data_root=tmp_path,
        train_txt="train.txt",
        cell_point_index=np.arange(3),
        mask_definition=STRUCTURAL_DRY_MASK_DEFINITION_EXACT_ZERO,
    )

    assert artifact["cell_count"] == 3
    assert artifact["n_dry"] == 2
    assert artifact["n_wettable"] == 1
    assert torch.equal(artifact["dry_mask"], torch.tensor([True, False, True]))
    assert torch.equal(artifact["wettable_mask"], torch.tensor([False, True, False]))

    artifact_path = tmp_path / "structural_dry_mask_exact_zero.pt"
    summary_path = tmp_path / "structural_dry_mask_exact_zero_summary.json"
    save_structural_dry_artifact(
        artifact,
        artifact_path=artifact_path,
        summary_path=summary_path,
    )

    loaded = load_structural_dry_artifact(artifact_path)
    validated = validate_structural_dry_artifact(
        loaded,
        expected_cell_count=3,
        expected_run_ids=["run01", "run02"],
    )
    assert torch.equal(validated["dry_mask"], artifact["dry_mask"])

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["mask_definition"] == "exact_zero"
    assert summary["n_dry"] == 2
    assert summary["n_wettable"] == 1


def test_broadcast_wettable_mask_matches_reference_shape():
    wettable = torch.tensor([True, False, True])
    ref_3d = torch.zeros(2, 3, 1)
    out_3d = broadcast_wettable_mask(wettable, ref_3d)
    assert out_3d.shape == ref_3d.shape
    assert torch.equal(out_3d[0, :, 0].to(dtype=torch.bool), wettable)

    ref_4d = torch.zeros(4, 2, 3, 1)
    out_4d = broadcast_wettable_mask(wettable, ref_4d)
    assert out_4d.shape == ref_4d.shape
    assert torch.equal(out_4d[0, 0, :, 0].to(dtype=torch.bool), wettable)

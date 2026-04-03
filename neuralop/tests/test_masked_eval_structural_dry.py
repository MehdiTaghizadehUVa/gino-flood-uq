from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from neuralop.flood.data.structural_dry import (
    HDF_PATHS,
    STRUCTURAL_DRY_MASK_DEFINITION_EXACT_ZERO,
    build_structural_dry_artifact,
    save_structural_dry_artifact,
)
from neuralop.flood.eval.datasets import _load_structural_dry_artifact_for_eval

try:
    import h5py
except Exception as exc:  # pragma: no cover
    h5py = None
    H5PY_IMPORT_ERROR = exc
else:
    H5PY_IMPORT_ERROR = None


pytestmark = pytest.mark.skipif(h5py is None, reason=f"h5py unavailable: {H5PY_IMPORT_ERROR}")


def _write_wd_hdf(path: Path, wd) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset(HDF_PATHS["wd"], data=np.asarray(wd, dtype=np.float32))


def test_masked_eval_validates_against_canonical_training_package(tmp_path: Path):
    train_root = tmp_path / "train"
    train_root.mkdir()
    (train_root / "train.txt").write_text("run01\nrun02\n", encoding="utf-8")
    _write_wd_hdf(train_root / "run01.hdf", [[0.0, 1.0, 0.0], [0.0, 0.5, 0.0]])
    _write_wd_hdf(train_root / "run02.hdf", [[0.0, 0.2, 0.0], [0.0, 2.0, 0.0]])

    artifact = build_structural_dry_artifact(
        data_root=train_root,
        train_txt="train.txt",
        cell_point_index=np.arange(3),
        mask_definition=STRUCTURAL_DRY_MASK_DEFINITION_EXACT_ZERO,
    )
    artifact_path = tmp_path / "structural_dry_mask_exact_zero.pt"
    save_structural_dry_artifact(artifact, artifact_path=artifact_path)

    config = SimpleNamespace(
        structural_dry=SimpleNamespace(
            policy="masked_primary",
            mask_definition="exact_zero",
            mask_path=str(artifact_path),
        )
    )

    policy_kwargs, validated = _load_structural_dry_artifact_for_eval(
        config,
        normalizer_path=tmp_path / "normalizers_depth_only_masked_primary.pt",
        expected_cell_count=3,
        expected_run_ids=["wrong_run_id"],
        logger=None,
    )

    assert policy_kwargs["policy"] == "masked_primary"
    assert torch.equal(validated["dry_mask"], artifact["dry_mask"])
    assert validated["run_ids"] == ["run01", "run02"]

"""Structural-dry artifact helpers for WV flood datasets."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from neuralop.flood.data.hec_ras import HDF_PATHS, build_cell_point_index, h5py


STRUCTURAL_DRY_MASK_DEFINITION_EXACT_ZERO = "exact_zero"


def _normalize_run_ids(run_ids: Iterable[str]) -> list[str]:
    out = [str(run_id).strip() for run_id in run_ids if str(run_id).strip()]
    if not out:
        raise ValueError("Expected at least one canonical training run ID.")
    return out


def _load_run_ids_from_train_txt(data_root: Path, train_txt: str) -> list[str]:
    txt_path = (Path(data_root) / str(train_txt)).resolve()
    if not txt_path.exists():
        raise FileNotFoundError(f"Training split file not found: {txt_path}")
    with txt_path.open("r", encoding="utf-8-sig") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    if not lines:
        raise ValueError(f"Training split file is empty: {txt_path}")
    if len(lines) == 1 and "," in lines[0]:
        return _normalize_run_ids(token.strip() for token in lines[0].split(","))
    return _normalize_run_ids(lines)


def resolve_canonical_training_run_ids(
    *,
    data_root: str | os.PathLike[str],
    run_ids: Sequence[str] | None = None,
    train_txt: str = "train.txt",
) -> list[str]:
    """Return ordered canonical run IDs for the upstream training package."""
    if run_ids is not None:
        return _normalize_run_ids(run_ids)
    return _load_run_ids_from_train_txt(Path(data_root), train_txt)


def _first_existing_hdf(
    *,
    data_root: Path,
    run_ids: Sequence[str],
    hdf_suffix: str,
) -> Path:
    for run_id in run_ids:
        cand = (data_root / f"{run_id}{hdf_suffix}").resolve()
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"Could not find any HDF files for the canonical run IDs under {data_root}."
    )


def _ensure_cell_point_index(
    *,
    data_root: Path,
    run_ids: Sequence[str],
    hdf_suffix: str,
    hdf_paths: dict[str, str],
    cell_point_index: np.ndarray | None,
) -> np.ndarray:
    if cell_point_index is not None:
        return np.asarray(cell_point_index, dtype=np.intp)
    first_hdf = _first_existing_hdf(
        data_root=data_root,
        run_ids=run_ids,
        hdf_suffix=hdf_suffix,
    )
    return build_cell_point_index(first_hdf, hdf_paths)


def _read_max_wd_for_run(
    hdf_path: Path,
    *,
    wd_path: str,
    cell_point_index: np.ndarray,
) -> np.ndarray:
    if h5py is None:
        raise ImportError("h5py is required for structural-dry artifact building.")
    with h5py.File(hdf_path, "r") as handle:
        wd = np.asarray(handle[wd_path][:, cell_point_index], dtype=np.float32)
    if wd.ndim != 2:
        raise ValueError(
            f"Expected WD tensor to be 2D [T, N] for {hdf_path}, got {tuple(wd.shape)}."
        )
    return np.max(wd, axis=0)


def summarize_structural_dry_mask(dry_mask: torch.Tensor) -> dict[str, Any]:
    dry_mask = dry_mask.to(dtype=torch.bool, device="cpu").reshape(-1)
    n_dry = int(dry_mask.sum().item())
    cell_count = int(dry_mask.numel())
    return {
        "cell_count": cell_count,
        "n_dry": n_dry,
        "n_wettable": cell_count - n_dry,
        "dry_fraction": (float(n_dry) / float(cell_count)) if cell_count > 0 else 0.0,
    }


def build_structural_dry_artifact(
    *,
    data_root: str | os.PathLike[str],
    run_ids: Sequence[str] | None = None,
    train_txt: str = "train.txt",
    hdf_suffix: str = ".hdf",
    hdf_paths: dict[str, str] | None = None,
    cell_point_index: np.ndarray | None = None,
    mask_definition: str = STRUCTURAL_DRY_MASK_DEFINITION_EXACT_ZERO,
) -> dict[str, Any]:
    """Build the canonical structural-dry artifact from the upstream training package."""
    if mask_definition != STRUCTURAL_DRY_MASK_DEFINITION_EXACT_ZERO:
        raise ValueError(
            f"Unsupported structural_dry.mask_definition={mask_definition!r}. "
            f"Only {STRUCTURAL_DRY_MASK_DEFINITION_EXACT_ZERO!r} is supported."
        )
    data_root_path = Path(data_root).resolve()
    resolved_run_ids = resolve_canonical_training_run_ids(
        data_root=data_root_path,
        run_ids=run_ids,
        train_txt=train_txt,
    )
    paths = hdf_paths or HDF_PATHS
    point_index = _ensure_cell_point_index(
        data_root=data_root_path,
        run_ids=resolved_run_ids,
        hdf_suffix=hdf_suffix,
        hdf_paths=paths,
        cell_point_index=cell_point_index,
    )

    max_wd = None
    missing: list[str] = []
    for run_id in resolved_run_ids:
        hdf_path = (data_root_path / f"{run_id}{hdf_suffix}").resolve()
        if not hdf_path.exists():
            missing.append(run_id)
            continue
        run_max = _read_max_wd_for_run(
            hdf_path,
            wd_path=paths["wd"],
            cell_point_index=point_index,
        )
        max_wd = run_max if max_wd is None else np.maximum(max_wd, run_max)

    if max_wd is None:
        raise FileNotFoundError(
            f"Could not build structural-dry artifact: no HDF files found under {data_root_path}."
        )
    if missing:
        raise FileNotFoundError(
            "Canonical training package is missing HDF files for run IDs: "
            + ", ".join(missing[:10])
            + (" ..." if len(missing) > 10 else "")
        )

    dry_mask = torch.from_numpy(max_wd <= 0.0).to(dtype=torch.bool, device="cpu")
    wettable_mask = ~dry_mask
    summary = summarize_structural_dry_mask(dry_mask)
    return {
        "mask_definition": mask_definition,
        "source_root": str(data_root_path),
        "source_train_txt": str((data_root_path / str(train_txt)).resolve()),
        "run_ids": list(resolved_run_ids),
        "cell_count": int(dry_mask.numel()),
        "cell_point_index": torch.as_tensor(point_index, dtype=torch.int64),
        "dry_mask": dry_mask,
        "wettable_mask": wettable_mask,
        "n_dry": int(summary["n_dry"]),
        "n_wettable": int(summary["n_wettable"]),
        "dry_fraction": float(summary["dry_fraction"]),
    }


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _summary_json_payload(artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "mask_definition": str(artifact["mask_definition"]),
        "source_root": str(artifact["source_root"]),
        "source_train_txt": str(artifact["source_train_txt"]),
        "cell_count": int(artifact["cell_count"]),
        "n_dry": int(artifact["n_dry"]),
        "n_wettable": int(artifact["n_wettable"]),
        "dry_fraction": float(artifact["dry_fraction"]),
        "run_ids": list(artifact["run_ids"]),
    }


def save_structural_dry_artifact(
    artifact: dict[str, Any],
    *,
    artifact_path: str | os.PathLike[str],
    summary_path: str | os.PathLike[str] | None = None,
) -> None:
    artifact_path = Path(artifact_path).resolve()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(artifact, artifact_path)
    if summary_path is not None:
        summary_path = Path(summary_path).resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json_dump(_summary_json_payload(artifact), summary_path)


def load_structural_dry_artifact(path: str | os.PathLike[str]) -> dict[str, Any]:
    payload = torch.load(Path(path).resolve(), map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Structural-dry artifact must be a dict, got {type(payload)!r}.")
    if "dry_mask" not in payload or "wettable_mask" not in payload:
        raise KeyError("Structural-dry artifact is missing dry_mask/wettable_mask.")
    payload["dry_mask"] = payload["dry_mask"].to(dtype=torch.bool, device="cpu")
    payload["wettable_mask"] = payload["wettable_mask"].to(dtype=torch.bool, device="cpu")
    return payload


def validate_structural_dry_artifact(
    artifact: dict[str, Any],
    *,
    expected_cell_count: int | None = None,
    expected_run_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    dry_mask = torch.as_tensor(artifact["dry_mask"], dtype=torch.bool, device="cpu").reshape(-1)
    wettable_mask = torch.as_tensor(
        artifact["wettable_mask"], dtype=torch.bool, device="cpu"
    ).reshape(-1)
    if dry_mask.numel() != wettable_mask.numel():
        raise ValueError("dry_mask and wettable_mask must have the same cell count.")
    if torch.any(dry_mask & wettable_mask):
        raise ValueError("Structural-dry artifact has overlapping dry/wettable masks.")
    if torch.any(~(dry_mask | wettable_mask)):
        raise ValueError("Structural-dry artifact masks do not fully cover the mesh.")
    if expected_cell_count is not None and int(dry_mask.numel()) != int(expected_cell_count):
        raise ValueError(
            f"Structural-dry artifact cell_count={dry_mask.numel()} "
            f"does not match expected {expected_cell_count}."
        )
    if expected_run_ids is not None:
        resolved_expected = _normalize_run_ids(expected_run_ids)
        resolved_artifact = _normalize_run_ids(artifact.get("run_ids", []))
        if resolved_artifact != resolved_expected:
            raise ValueError("Structural-dry artifact run_ids do not match the canonical training package.")
    return {
        **artifact,
        "dry_mask": dry_mask,
        "wettable_mask": wettable_mask,
    }


def dry_mask_to_wettable_mask(dry_mask: torch.Tensor) -> torch.Tensor:
    dry_mask = torch.as_tensor(dry_mask, dtype=torch.bool)
    return (~dry_mask).clone()


def broadcast_wettable_mask(
    wettable_mask: torch.Tensor,
    ref: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Broadcast a `[N]` or `[B, N]` wettable mask to the shape of `ref`."""
    mask = torch.as_tensor(wettable_mask, dtype=torch.bool, device=ref.device)
    if ref.ndim == 3:
        if mask.ndim == 1:
            mask = mask.view(1, -1, 1)
        elif mask.ndim == 2:
            mask = mask.unsqueeze(-1)
    elif ref.ndim == 4:
        if mask.ndim == 1:
            mask = mask.view(1, 1, -1, 1)
        elif mask.ndim == 2:
            mask = mask.unsqueeze(1).unsqueeze(-1)
        elif mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask.unsqueeze(1)
    elif mask.ndim != ref.ndim:
        raise ValueError(
            f"Cannot broadcast wettable_mask shape {tuple(mask.shape)} to ref shape {tuple(ref.shape)}."
        )
    mask = mask.expand_as(ref)
    out_dtype = ref.dtype if dtype is None else dtype
    return mask.to(dtype=out_dtype)

"""Dataset construction for flood evaluation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset, random_split

from neuralop.data.transforms.normalizers import load_normalizers
from neuralop.flood.data.wv import (
    FloodDatasetHDF,
    FloodRolloutTestDatasetHDF,
    NormalizedRolloutTestDataset,
    collect_all_fields,
)
from neuralop.flood.eval.runtime import (
    CHANNEL_INDEX,
    DEFAULT_STATIC_FILES,
    DEVICE_REF_KEYS,
    NORMALIZER_KEYS,
    TRAIN_FRAC,
    _PhaseTimer,
    _opt,
)
from neuralop.flood.utils.runtime import (
    get_dataset_boundary_kwargs,
    make_split_generator,
    parse_target_variables,
)

def _load_or_fit_normalizers(
    config: Any,
    train_data: Any,
    save_dir: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Load pre-fit normalizers from training-data location only.

    Evaluation must never fit/refit normalizers and must never resolve a
    relative path against data.root (which is often test data in eval jobs).
    """
    del train_data, save_dir  # kept for backward-compatible call signature
    normalizer_path = _opt(config, "data", "normalizer_path", None)
    if normalizer_path is None:
        raise ValueError(
            "Evaluation requires data.normalizer_path. "
            "Provide training normalizer file path (prefer absolute)."
        )
    normalizer_path = Path(str(normalizer_path))
    if not normalizer_path.is_absolute():
        normalizer_root = _opt(config, "data", "normalizer_root", None)
        if normalizer_root is None:
            normalizer_root = _opt(config, "data", "train_root", None)
        if normalizer_root is None:
            raise ValueError(
                "Relative data.normalizer_path is not allowed without "
                "data.normalizer_root (or data.train_root). "
                "Evaluator refuses to resolve normalizers against data.root."
            )
        normalizer_path = Path(str(normalizer_root)) / normalizer_path
    normalizer_path = normalizer_path.resolve()
    if not normalizer_path.exists():
        raise FileNotFoundError(
            f"Training normalizer file not found: {normalizer_path}"
        )
    with _PhaseTimer(logger, f"Loading normalizers from {normalizer_path}"):
        return load_normalizers(normalizer_path, device=None)


def _build_one_step_datasets(
    config: Any,
    seed: int,
    logger: logging.Logger,
) -> Tuple[Any, Any, List[str]]:
    """Build train/test split and target_variables from config."""
    static_files = _opt(config, "data", "static_text_files", DEFAULT_STATIC_FILES)
    if not isinstance(static_files, list):
        static_files = list(static_files)
    ar_rollout_steps = max(1, int(_opt(config, "opt", "ar_rollout_steps", 1)))
    target_variables = parse_target_variables(
        _opt(config, "data", "target_variables", ["wd", "vx", "vy"])
    )
    n_target = len(target_variables)
    n_static = 2 + len(static_files)
    n_history = config.data.n_history
    data_channels = n_static + n_history * 1 + n_history * n_target
    if hasattr(config, "gino"):
        setattr(config.gino, "data_channels", data_channels)
        setattr(config.gino, "out_channels", n_target)
    data_boundary_kwargs = get_dataset_boundary_kwargs(config.data)
    logger.info(
        "One-step dataset boundary_source=%s%s",
        data_boundary_kwargs["boundary_source"],
        f", clean_boundary_file={data_boundary_kwargs['clean_boundary_file']}"
        if data_boundary_kwargs["boundary_source"] == "clean_family"
        else "",
    )

    with _PhaseTimer(logger, "Building one-step dataset"):
        full = FloodDatasetHDF(
            data_root=config.data.root,
            n_history=config.data.n_history,
            query_res=_opt(config, "data", "query_res", [48, 48]),
            run_ids=None,
            train_txt=_opt(config, "data", "train_txt", "train.txt"),
            static_text_files=static_files,
            hdf_suffix=".hdf",
            raise_on_smaller=True,
            skip_before_timestep=_opt(config, "data", "skip_before_timestep", 0),
            noise_type=_opt(config, "data", "noise_type", "none"),
            noise_std=_opt(config, "data", "noise_std", None),
            ar_rollout_steps=ar_rollout_steps,
            target_variables=target_variables,
            **data_boundary_kwargs,
        )
    n_max = _opt(config, "data", "n_samples_max", None)
    if n_max is not None and int(n_max) > 0:
        full = Subset(full, range(min(int(n_max), len(full))))
    total = len(full)
    train_sz = max(1, int(TRAIN_FRAC * total))
    test_sz = total - train_sz
    train_raw, test_raw = random_split(
        full, [train_sz, test_sz], generator=make_split_generator(seed)
    )
    logger.info(
        "Split: total=%d train=%d test=%d target_variables=%s",
        total, len(train_raw), len(test_raw), target_variables,
    )
    return train_raw, test_raw, target_variables


def _build_test_loader(
    test_norm: Any, batch_size: int
) -> DataLoader:
    """Build test DataLoader (no shuffle, num_workers=0)."""
    return DataLoader(
        test_norm,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


def _build_rollout_normalized_dataset(
    config: Any,
    normalizers: Dict[str, Any],
    target_variables: List[str],
    logger: logging.Logger,
) -> Tuple[Any, Optional[List[Dict[str, Any]]]]:
    """Build rollout dataset and optional grouped hydrograph samples."""
    rollout_length = config.data.rollout_length
    history_steps = config.data.n_history
    skip = _opt(config, "data", "skip_before_timestep", 0)
    target_indices = [CHANNEL_INDEX[v] for v in target_variables]
    rollout_static = _opt(
        config, "rollout_data", "static_text_files", DEFAULT_STATIC_FILES
    )
    if not isinstance(rollout_static, list):
        rollout_static = list(rollout_static)

    with _PhaseTimer(logger, "Building rollout test dataset"):
        rollout_boundary_kwargs = get_dataset_boundary_kwargs(config.rollout_data, split="test")
        logger.info(
            "Rollout dataset boundary_source=%s%s",
            rollout_boundary_kwargs["boundary_source"],
            f", clean_boundary_file={rollout_boundary_kwargs['clean_boundary_file']}"
            if rollout_boundary_kwargs["boundary_source"] == "clean_family"
            else "",
        )
        rds = FloodRolloutTestDatasetHDF(
            rollout_data_root=config.rollout_data.root,
            n_history=history_steps,
            rollout_length=rollout_length,
            run_ids=None,
            test_txt=_opt(config, "rollout_data", "test_txt", "test.txt"),
            static_text_files=rollout_static,
            hdf_suffix=".hdf",
            raise_on_smaller=True,
            skip_before_timestep=skip,
            **rollout_boundary_kwargs,
        )

    groups = group_run_ids_by_hydrograph(rds.valid_run_ids)
    sims_per_hydro = [len(v) for v in groups.values()] if groups else []
    if sims_per_hydro and max(sims_per_hydro) > 1:
        logger.info(
            "Rollout runs: %d total | Hydrographs: %d (sims per hydrograph min=%d max=%d)",
            len(rds.valid_run_ids),
            len(groups),
            min(sims_per_hydro),
            max(sims_per_hydro),
        )
    else:
        logger.info("Rollout runs: %d", len(rds))

    with _PhaseTimer(logger, "Collecting rollout fields"):
        geom_list, static_list, boundary_list, dyn_list, _ = collect_all_fields(
            rds, expect_target=False
        )
    ref_device = _move_normalizers_to_device(normalizers)
    geometry_big = torch.stack(geom_list, dim=0) if geom_list else None
    static_big = torch.stack(static_list, dim=0) if static_list else None
    boundary_big = torch.stack(boundary_list, dim=0) if boundary_list else None
    dynamic_big = torch.stack(dyn_list, dim=0) if dyn_list else None
    if dynamic_big is not None:
        dynamic_big = dynamic_big[..., target_indices]
    if geometry_big is not None and "geometry" in normalizers:
        geometry_big = normalizers["geometry"].transform(geometry_big.to(ref_device))
    if static_big is not None and "static" in normalizers:
        static_big = normalizers["static"].transform(static_big.to(ref_device))
    if boundary_big is not None and "boundary" in normalizers:
        boundary_big = normalizers["boundary"].transform(boundary_big.to(ref_device))
    if dynamic_big is not None and "dynamic" in normalizers:
        dynamic_big = normalizers["dynamic"].transform(dynamic_big.to(ref_device))
    logger.info(
        "Rollout cache tensors normalized on device=%s (per-sample rollout code moves tensors to the requested compute device).",
        ref_device,
    )
    samples = [
        {
            "run_id": rds.valid_run_ids[i],
            "geometry": geometry_big[i],
            "static": static_big[i],
            "boundary": boundary_big[i],
            "dynamic": dynamic_big[i],
        }
        for i in range(len(rds))
    ]
    rollout_dataset = NormalizedRolloutTestDataset(
        normalized_samples=samples,
        query_res=config.data.query_res,
    )
    hydrograph_samples: Optional[List[Dict[str, Any]]] = None
    if sims_per_hydro and max(sims_per_hydro) > 1:
        run_id_to_idx = {rid: i for i, rid in enumerate(rds.valid_run_ids)}
        query_points = _build_query_points_from_geometry(
            geometry_big[0], config.data.query_res
        )
        hydrograph_samples = []
        for hydro_id, run_ids_group in groups.items():
            indices = [run_id_to_idx[rid] for rid in run_ids_group if rid in run_id_to_idx]
            if len(indices) < 2:
                continue
            hydrograph_samples.append(
                {
                    "hydrograph_id": hydro_id,
                    "geometry": geometry_big[indices[0]],
                    "static": static_big[indices[0]],
                    "boundary": boundary_big[indices[0]],
                    "dynamic_ref": torch.stack([dynamic_big[i] for i in indices], dim=0),
                    "query_points": query_points,
                    "n_ref_sims": len(indices),
                }
            )
        logger.info(
            "Built %d grouped hydrograph samples for UQ evaluation.",
            len(hydrograph_samples),
        )

    return rollout_dataset, hydrograph_samples

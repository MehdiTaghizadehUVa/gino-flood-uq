"""Dataset construction for flood evaluation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset, random_split

from neuralop.data.transforms.normalizers import load_normalizers
from neuralop.flood.data.structural_dry import (
    load_structural_dry_artifact,
    resolve_canonical_training_run_ids,
    validate_structural_dry_artifact,
)
from neuralop.flood.data.wv import (
    FloodDatasetHDF,
    FloodRolloutTestDatasetHDF,
    NormalizedRolloutTestDataset,
    collect_all_fields,
)
from neuralop.flood.eval.runtime import (
    CHANNEL_INDEX,
    DEFAULT_STATIC_FILES,
    TRAIN_FRAC,
    _PhaseTimer,
    _opt,
    _move_normalizers_to_device,
    group_run_ids_by_hydrograph,
)
from neuralop.flood.eval.metrics import _build_query_points_from_geometry
from neuralop.flood.utils.runtime import (
    assert_boundary_channel_compatibility,
    describe_boundary_spec,
    get_dataset_boundary_kwargs,
    get_boundary_channel_count,
    get_dataset_hdf_paths,
    get_structural_dry_policy_kwargs,
    make_split_generator,
    parse_target_variables,
)

def _load_or_fit_normalizers(
    config: Any,
    train_data: Any,
    save_dir: Path,
    logger: logging.Logger,
) -> Tuple[Dict[str, Any], Path]:
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
        return load_normalizers(normalizer_path, device=None), normalizer_path


def _set_dataset_structural_dry_mask(dataset: Any, dry_mask: torch.Tensor | None) -> None:
    if dry_mask is None:
        return
    if hasattr(dataset, "set_structural_dry_mask"):
        dataset.set_structural_dry_mask(dry_mask)
        return
    if hasattr(dataset, "dataset"):
        _set_dataset_structural_dry_mask(dataset.dataset, dry_mask)
        return
    raise TypeError(f"Dataset of type {type(dataset)!r} does not support structural_dry_mask injection.")


def _resolve_structural_dry_eval_run_ids(
    config: Any,
    artifact: Dict[str, Any],
    *,
    fallback_run_ids: List[str] | None = None,
    logger: logging.Logger | None = None,
) -> List[str] | None:
    canonical_root = _opt(config, "structural_dry", "canonical_data_root", None)
    canonical_train_txt = _opt(config, "structural_dry", "canonical_train_txt", None)

    if canonical_train_txt is not None:
        configured_train_path = Path(str(canonical_train_txt))
        if configured_train_path.is_absolute():
            canonical_root = str(configured_train_path.parent)
            canonical_train_txt = configured_train_path.name

    source_train_txt = artifact.get("source_train_txt", None)
    if canonical_root is None:
        source_root = artifact.get("source_root", None)
        if source_root is not None:
            canonical_root = str(source_root)
    if canonical_train_txt is None and source_train_txt is not None:
        source_train_path = Path(str(source_train_txt))
        if source_train_path.is_absolute():
            canonical_root = canonical_root or str(source_train_path.parent)
            canonical_train_txt = source_train_path.name
        else:
            canonical_train_txt = str(source_train_txt)

    if canonical_root is None:
        return fallback_run_ids

    resolved_run_ids = list(
        resolve_canonical_training_run_ids(
            data_root=str(canonical_root),
            train_txt=str(canonical_train_txt or "train.txt"),
        )
    )
    if logger is not None:
        logger.info(
            "Structural-dry validation using canonical package root=%s train_txt=%s",
            canonical_root,
            canonical_train_txt or "train.txt",
        )
    return resolved_run_ids


def _load_structural_dry_artifact_for_eval(
    config: Any,
    *,
    normalizer_path: Path,
    expected_cell_count: int | None = None,
    expected_run_ids: List[str] | None = None,
    logger: logging.Logger | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any] | None]:
    policy_kwargs = get_structural_dry_policy_kwargs(
        config,
        normalizer_path=normalizer_path,
        allow_data_root_fallback=False,
    )
    if policy_kwargs["policy"] != "masked_primary":
        return policy_kwargs, None
    artifact = load_structural_dry_artifact(policy_kwargs["artifact_path"])
    validation_run_ids = _resolve_structural_dry_eval_run_ids(
        config,
        artifact,
        fallback_run_ids=expected_run_ids,
        logger=logger,
    )
    artifact = validate_structural_dry_artifact(
        artifact,
        expected_cell_count=expected_cell_count,
        expected_run_ids=validation_run_ids,
    )
    if logger is not None:
        logger.info(
            "Loaded structural-dry artifact policy=%s n_dry=%d n_wettable=%d from %s",
            policy_kwargs["policy"],
            artifact["n_dry"],
            artifact["n_wettable"],
            policy_kwargs["artifact_path"],
        )
    return policy_kwargs, artifact


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
    data_boundary_kwargs = get_dataset_boundary_kwargs(config.data)
    n_boundary_channels = get_boundary_channel_count(data_boundary_kwargs["boundary_spec"])
    data_channels = n_static + n_history * n_boundary_channels + n_history * n_target
    if hasattr(config, "gino"):
        setattr(config.gino, "data_channels", data_channels)
        setattr(config.gino, "out_channels", n_target)
    logger.info(
        "One-step dataset boundary=%s (bc_dim=%d)",
        describe_boundary_spec(data_boundary_kwargs["boundary_spec"]),
        n_boundary_channels,
    )

    with _PhaseTimer(logger, "Building one-step dataset"):
        full = FloodDatasetHDF(
            data_root=config.data.root,
            n_history=config.data.n_history,
            query_res=_opt(config, "data", "query_res", [48, 48]),
            run_ids=None,
            train_txt=_opt(config, "data", "train_txt", "train.txt"),
            static_text_files=static_files,
            hdf_paths=get_dataset_hdf_paths(config.data),
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
    structural_dry_artifact: Dict[str, Any] | None = None,
    split_txt: str | None = None,
    split_name: str = "test",
    config_section: str = "rollout_data",
) -> Tuple[Any, Optional[List[Dict[str, Any]]]]:
    """Build rollout dataset and optional grouped hydrograph samples.

    Grouped hydrograph evaluation can involve thousands of member runs. The
    previous implementation collected every run, stacked the full package into
    large tensors, then regrouped it. That doubled peak host memory and can OOM
    before rollout starts. For grouped runs, normalize and yield one hydrograph
    at a time instead.
    """
    cfg = getattr(config, config_section)
    rollout_length = config.data.rollout_length
    history_steps = config.data.n_history
    skip = _opt(config, "data", "skip_before_timestep", 0)
    target_indices = [CHANNEL_INDEX[v] for v in target_variables]
    rollout_static = _opt(config, config_section, "static_text_files", DEFAULT_STATIC_FILES)
    if not isinstance(rollout_static, list):
        rollout_static = list(rollout_static)

    with _PhaseTimer(logger, "Building rollout test dataset"):
        rollout_boundary_kwargs = get_dataset_boundary_kwargs(cfg, split=split_name)
        data_boundary_kwargs = get_dataset_boundary_kwargs(config.data)
        assert_boundary_channel_compatibility(
            data_boundary_kwargs["boundary_spec"],
            rollout_boundary_kwargs["boundary_spec"],
            label_a="data",
            label_b=config_section,
        )
        logger.info(
            "Rollout dataset boundary=%s (bc_dim=%d)",
            describe_boundary_spec(rollout_boundary_kwargs["boundary_spec"]),
            get_boundary_channel_count(rollout_boundary_kwargs["boundary_spec"]),
        )
        rds = FloodRolloutTestDatasetHDF(
            rollout_data_root=cfg.root,
            n_history=history_steps,
            rollout_length=rollout_length,
            run_ids=None,
            test_txt=split_txt or _opt(config, config_section, "test_txt", "test.txt"),
            static_text_files=rollout_static,
            hdf_paths=get_dataset_hdf_paths(cfg),
            hdf_suffix=".hdf",
            raise_on_smaller=True,
            skip_before_timestep=skip,
            **rollout_boundary_kwargs,
        )
    boundary_channel_names = [
        str(channel.get("name", f"boundary_{idx}"))
        for idx, channel in enumerate(rollout_boundary_kwargs["boundary_spec"])
    ]
    if structural_dry_artifact is not None:
        rds.set_structural_dry_mask(structural_dry_artifact["dry_mask"])

    groups = group_run_ids_by_hydrograph(rds.valid_run_ids)
    sims_per_hydro = [len(v) for v in groups.values()] if groups else []
    grouped_mode = bool(sims_per_hydro and max(sims_per_hydro) > 1)
    if grouped_mode:
        logger.info(
            "Rollout runs: %d total | Hydrographs: %d (sims per hydrograph min=%d max=%d)",
            len(rds.valid_run_ids),
            len(groups),
            min(sims_per_hydro),
            max(sims_per_hydro),
        )
    else:
        logger.info("Rollout runs: %d", len(rds))

    ref_device = _move_normalizers_to_device(normalizers)

    def _normalize_field(key: str, value: torch.Tensor) -> torch.Tensor:
        if key not in normalizers or normalizers[key] is None:
            return value.cpu()
        # Normalizers were fit with a leading sample dimension. Keep that
        # dimension during transform, then drop it to preserve rollout shapes.
        return normalizers[key].transform(value.unsqueeze(0).to(ref_device)).squeeze(0).cpu()

    def _normalize_raw_sample(raw: Dict[str, Any]) -> Dict[str, Any]:
        dynamic = raw["dynamic"][..., target_indices]
        boundary_raw = torch.as_tensor(raw["boundary"])
        out = {
            "run_id": raw["run_id"],
            "geometry": _normalize_field("geometry", raw["geometry"]),
            "static": _normalize_field("static", raw["static"]),
            "boundary": _normalize_field("boundary", boundary_raw),
            "boundary_series_raw": boundary_raw[:, 0, :].detach().cpu().clone(),
            "dynamic": _normalize_field("dynamic", dynamic),
        }
        if structural_dry_artifact is not None:
            out["structural_dry_mask"] = structural_dry_artifact["dry_mask"]
        return out

    if grouped_mode:
        run_id_to_idx = {rid: i for i, rid in enumerate(rds.valid_run_ids)}
        group_items = []
        for hydro_id, run_ids_group in groups.items():
            indices = [run_id_to_idx[rid] for rid in run_ids_group if rid in run_id_to_idx]
            if len(indices) >= 2:
                group_items.append((hydro_id, indices))

        def _build_hydrograph_sample(hydro_id: str, normalized_group: List[Dict[str, Any]]):
            geometry = normalized_group[0]["geometry"]
            query_points = _build_query_points_from_geometry(geometry, config.data.query_res)
            sample = {
                "hydrograph_id": hydro_id,
                "reference_run_ids": [g["run_id"] for g in normalized_group],
                "geometry": geometry,
                "static": normalized_group[0]["static"],
                "boundary": normalized_group[0]["boundary"],
                "boundary_series_raw": normalized_group[0]["boundary_series_raw"],
                "boundary_channel_names": list(boundary_channel_names),
                "dynamic_ref": torch.stack([g["dynamic"] for g in normalized_group], dim=0),
                "query_points": query_points,
                "n_ref_sims": len(normalized_group),
            }
            if structural_dry_artifact is not None:
                sample["structural_dry_mask"] = structural_dry_artifact["dry_mask"]
            return sample

        if not hasattr(rds, "__getitem__"):
            geometry_list, static_list, boundary_list, dynamic_list, _ = collect_all_fields(
                rds, expect_target=False
            )
            normalized_samples = []
            for i, run_id in enumerate(rds.valid_run_ids):
                normalized_samples.append(
                    _normalize_raw_sample(
                        {
                            "run_id": run_id,
                            "geometry": geometry_list[i],
                            "static": static_list[i],
                            "boundary": boundary_list[i],
                            "dynamic": dynamic_list[i],
                        }
                    )
                )
            hydrograph_samples = [
                _build_hydrograph_sample(
                    hydro_id, [normalized_samples[i] for i in indices]
                )
                for hydro_id, indices in group_items
            ]
            rollout_dataset = NormalizedRolloutTestDataset(
                normalized_samples=normalized_samples,
                query_res=config.data.query_res,
            )
            return rollout_dataset, hydrograph_samples

        class _LazyHydrographSamples:
            def __len__(self) -> int:
                return len(group_items)

            def __bool__(self) -> bool:
                return bool(group_items)

            def __iter__(self):
                for idx in range(len(group_items)):
                    yield self[idx]

            def __getitem__(self, idx: int):
                hydro_id, indices = group_items[idx]
                normalized_group = [_normalize_raw_sample(rds[i]) for i in indices]
                return _build_hydrograph_sample(hydro_id, normalized_group)

        class _GroupedRolloutDatasetLengthProxy:
            def __len__(self) -> int:
                return len(rds)

        hydrograph_samples = _LazyHydrographSamples()
        logger.info(
            "Built lazy grouped hydrograph iterator for %d hydrographs; rollout fields will be loaded one hydrograph at a time.",
            len(hydrograph_samples),
        )
        rollout_dataset = _GroupedRolloutDatasetLengthProxy()
        return rollout_dataset, hydrograph_samples

    with _PhaseTimer(logger, "Normalizing rollout fields"):
        samples = [_normalize_raw_sample(rds[i]) for i in range(len(rds))]
    logger.info(
        "Rollout tensors normalized on device=%s and stored on CPU for %d runs.",
        ref_device,
        len(samples),
    )
    rollout_dataset = NormalizedRolloutTestDataset(
        normalized_samples=samples,
        query_res=config.data.query_res,
    )
    return rollout_dataset, None

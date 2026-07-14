"""Dataset construction for flood evaluation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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

def _log_debug(logger: Any, message: str, *args: Any) -> None:
    debug = getattr(logger, "debug", None)
    if callable(debug):
        debug(message, *args)


def _normalizer_tensor_device(normalizer: Any, fallback: torch.device) -> torch.device:
    """Return the current tensor device for a normalizer.

    Lazy rollout hydrograph samples close over the normalizer dict, while rollout
    code may later move target/dynamic normalizers to the sampling device. Device
    choice must therefore be evaluated at transform time, not when the lazy
    iterator is created.
    """
    for attr in ("mean", "std"):
        tensor = getattr(normalizer, attr, None)
        if isinstance(tensor, torch.Tensor):
            return tensor.device
    return fallback


def _transform_with_current_normalizer_device(
    normalizer: Any, value: torch.Tensor, fallback_device: torch.device
) -> torch.Tensor:
    norm_device = _normalizer_tensor_device(normalizer, fallback_device)
    return normalizer.transform(value.unsqueeze(0).to(norm_device)).squeeze(0).cpu()


def _extract_raw_render_context_from_sample(raw: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Preserve raw UTM/elevation context separately from model-normalized tensors."""
    geometry_raw = torch.as_tensor(raw["geometry"]).detach().cpu().clone()
    static_raw = torch.as_tensor(raw["static"]).detach().cpu().clone()
    context: Dict[str, torch.Tensor] = {
        "geometry_raw": geometry_raw,
        "static_raw": static_raw,
    }
    if static_raw.ndim == 2 and static_raw.shape[1] > 0:
        # Static channel 0 is the raw elevation column in FloodRolloutTestDatasetHDF.
        context["elevation_raw"] = static_raw[:, 0].detach().cpu().clone()
    return context


def _clean_family_member_boundary_file_candidates(clean_boundary_file: str) -> List[str]:
    """Return likely member-specific forcing files next to a *_Clean table."""
    file_path = Path(str(clean_boundary_file))
    stem = file_path.stem
    suffix = file_path.suffix
    candidates: List[str] = []
    for token in ("_Clean", "_clean", "Clean", "clean"):
        if token not in stem:
            continue
        candidate = file_path.with_name(stem.replace(token, "") + suffix).as_posix()
        if candidate != file_path.as_posix() and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _reference_member_boundary_column_candidates(run_id: str) -> List[str]:
    """Return exact-first column aliases for member forcing tables.

    Some dataset packages prefix run IDs (for example
    ``Flood_coastal_TE000001_sim00``) while forcing tables may use the bare
    event/member key (``TE000001_sim00``). Keep exact matching first, then try
    suffix aliases that preserve the member id.
    """
    rid = str(run_id).strip()
    candidates: List[str] = []

    def _add(value: str) -> None:
        value = str(value).strip()
        if value and value not in candidates:
            candidates.append(value)

    _add(rid)
    if "_sim" not in rid:
        return candidates
    family_id, _, sim_suffix = rid.rpartition("_sim")
    if not family_id or not sim_suffix:
        return candidates
    parts = [part for part in family_id.split("_") if part]
    family_candidates = [family_id]
    if len(parts) > 1:
        for start in range(1, len(parts)):
            alias = "_".join(parts[start:])
            if alias and alias not in family_candidates:
                family_candidates.append(alias)
    for family_candidate in family_candidates:
        _add(f"{family_candidate}_sim{sim_suffix}")
    return candidates


def _load_reference_member_boundary_table(
    channel: Dict[str, Any],
    *,
    logger: logging.Logger | None = None,
) -> Optional[Dict[str, Any]]:
    candidates = _clean_family_member_boundary_file_candidates(
        str(channel.get("clean_boundary_file", ""))
    )
    if not candidates:
        return None

    from neuralop.flood.utils.runtime_core import _load_clean_boundary_table

    for candidate in candidates:
        try:
            return _load_clean_boundary_table(channel["clean_boundary_root"], candidate)
        except FileNotFoundError:
            continue
    _log_debug(
        logger,
        "No member-specific boundary table found for clean table %s; tried %s",
        channel.get("clean_boundary_file"),
        candidates,
    )
    return None


def _reference_member_boundary_series_from_table(
    channel: Dict[str, Any],
    run_ids: List[str],
    *,
    logger: logging.Logger | None = None,
) -> Optional[np.ndarray]:
    bundle = _load_reference_member_boundary_table(channel, logger=logger)
    if bundle is None:
        return None
    boundary_by_member = bundle["boundary_by_family"]
    member_series: List[np.ndarray] = []
    missing: List[str] = []
    for run_id in run_ids:
        matched_key = None
        for candidate in _reference_member_boundary_column_candidates(run_id):
            if candidate in boundary_by_member:
                matched_key = candidate
                break
        if matched_key is None:
            missing.append(str(run_id))
            continue
        member_series.append(
            np.asarray(boundary_by_member[matched_key], dtype=np.float32).copy()
        )
    if missing:
        if logger is not None:
            logger.warning(
                "Member-specific boundary table %s is missing %d/%d reference members "
                "for channel %s; falling back to configured boundary series for this channel.",
                bundle["path"],
                len(missing),
                len(run_ids),
                channel.get("name"),
            )
        return None
    if not member_series:
        return None
    lengths = {int(series.shape[0]) for series in member_series}
    if len(lengths) != 1:
        if logger is not None:
            logger.warning(
                "Member-specific boundary table %s has inconsistent time lengths %s "
                "for channel %s; falling back to configured boundary series for this channel.",
                bundle["path"],
                sorted(lengths),
                channel.get("name"),
            )
        return None
    return np.stack(member_series, axis=0)


def _boundary_ensemble_series_from_reference_members(
    boundary_spec: List[Dict[str, Any]],
    run_ids: List[str],
    fallback_boundary_series_raw: torch.Tensor,
    *,
    logger: logging.Logger | None = None,
) -> Optional[torch.Tensor]:
    """Build raw forcing ensemble for grouped rollout diagnostics.

    The model may be configured to consume clean-family forcings, but the
    grouped reference simulations are generated by member-specific perturbations
    in sibling tables. The renderer should show those reference-member forcings
    when available and fall back to the configured series otherwise.
    """
    fallback = torch.as_tensor(fallback_boundary_series_raw).detach().cpu()
    if fallback.ndim == 2:
        fallback = fallback.unsqueeze(0)
    if fallback.ndim != 3 or fallback.shape[0] == 0 or fallback.shape[1] == 0:
        return None

    channels = list(boundary_spec or [])
    if not channels:
        return fallback.clone()

    per_channel: List[np.ndarray] = []
    for channel_idx, channel in enumerate(channels):
        channel_series = None
        if str(channel.get("mode", "")).strip().lower() == "clean_family":
            channel_series = _reference_member_boundary_series_from_table(
                channel, run_ids, logger=logger
            )
        if channel_series is None:
            if channel_idx >= int(fallback.shape[2]):
                if logger is not None:
                    logger.warning(
                        "Boundary spec has channel %d but fallback boundary series has shape %s; "
                        "skipping ensemble forcing diagnostics for this hydrograph.",
                        channel_idx,
                        tuple(fallback.shape),
                    )
                return None
            channel_series = fallback[:, :, channel_idx].numpy().astype(np.float32, copy=True)
        if int(channel_series.shape[0]) != int(fallback.shape[0]):
            if logger is not None:
                logger.warning(
                    "Boundary ensemble member count mismatch for channel %s: got %d, expected %d; "
                    "skipping ensemble forcing diagnostics for this hydrograph.",
                    channel.get("name"),
                    int(channel_series.shape[0]),
                    int(fallback.shape[0]),
                )
            return None
        per_channel.append(np.asarray(channel_series, dtype=np.float32))

    min_time = min(int(arr.shape[1]) for arr in per_channel)
    if min_time <= 0:
        return None
    stacked = np.stack([arr[:, :min_time] for arr in per_channel], axis=-1)
    return torch.as_tensor(stacked, dtype=torch.float32)


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
    include_single_reference_groups: bool = False,
) -> Tuple[Any, Optional[List[Dict[str, Any]]]]:
    """Build rollout dataset and optional grouped hydrograph samples.

    Grouped hydrograph evaluation can involve thousands of member runs. The
    previous implementation collected every run, stacked the full package into
    large tensors, then regrouped it. That doubled peak host memory and can OOM
    before rollout starts. For grouped runs, normalize and yield one hydrograph
    at a time instead. Single-run groups remain disabled by default and are
    exposed only for explicit single-reference retrospective evaluation.
    """
    cfg = getattr(config, config_section)
    rollout_length = config.data.rollout_length
    history_steps = config.data.n_history
    skip = _opt(config, "data", "skip_before_timestep", 0)
    rollout_full_length = int(rollout_length) == -1
    validation_rollout_length = 1 if rollout_full_length else rollout_length
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
            rollout_length=validation_rollout_length,
            run_ids=None,
            test_txt=split_txt or _opt(config, config_section, "test_txt", "test.txt"),
            static_text_files=rollout_static,
            hdf_paths=get_dataset_hdf_paths(cfg),
            hdf_suffix=".hdf",
            raise_on_smaller=True,
            skip_before_timestep=skip,
            **rollout_boundary_kwargs,
        )
    available_lengths = getattr(rds, "available_rollout_lengths", {})
    available_min = int(getattr(rds, "min_available_rollout_length", 0) or 0)
    available_max = int(getattr(rds, "max_available_rollout_length", 0) or 0)
    if rollout_full_length:
        if available_min < 1:
            raise ValueError(
                "data.rollout_length=-1 requested full-length rollout, but found no forecast steps after "
                f"skip_before_timestep={skip} and n_history={history_steps}."
            )
        if available_min != available_max:
            logger.warning(
                "Full-length rollout found variable HDF time-series lengths; using shortest "
                "common forecast horizon=%d steps (max available=%d) so metrics remain stackable.",
                available_min,
                available_max,
            )
        else:
            logger.info(
                "data.rollout_length=-1 resolved to full available forecast horizon=%d steps.",
                available_min,
            )
    else:
        available_min = int(rollout_length)
    boundary_channel_names = [
        str(channel.get("name", f"boundary_{idx}"))
        for idx, channel in enumerate(rollout_boundary_kwargs["boundary_spec"])
    ]
    if structural_dry_artifact is not None:
        rds.set_structural_dry_mask(structural_dry_artifact["dry_mask"])

    groups = group_run_ids_by_hydrograph(rds.valid_run_ids)
    sims_per_hydro = [len(v) for v in groups.values()] if groups else []
    grouped_mode = bool(
        sims_per_hydro
        and (
            bool(include_single_reference_groups)
            or max(sims_per_hydro) > 1
        )
    )
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
        return _transform_with_current_normalizer_device(normalizers[key], value, ref_device)

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
            "available_rollout_length": int(
                raw.get(
                    "available_rollout_length",
                    max(0, int(dynamic.shape[0]) - int(skip) - int(history_steps)),
                )
            ),
            **_extract_raw_render_context_from_sample(raw),
        }
        if structural_dry_artifact is not None:
            out["structural_dry_mask"] = structural_dry_artifact["dry_mask"]
        return out

    if grouped_mode:
        run_id_to_idx = {rid: i for i, rid in enumerate(rds.valid_run_ids)}
        group_items = []
        minimum_group_size = 1 if include_single_reference_groups else 2
        for hydro_id, run_ids_group in groups.items():
            indices = [run_id_to_idx[rid] for rid in run_ids_group if rid in run_id_to_idx]
            if len(indices) >= minimum_group_size:
                group_items.append((hydro_id, indices))

        def _build_hydrograph_sample(hydro_id: str, normalized_group: List[Dict[str, Any]]):
            geometry = normalized_group[0]["geometry"]
            query_points = _build_query_points_from_geometry(geometry, config.data.query_res)
            reference_run_ids = [g["run_id"] for g in normalized_group]
            fallback_boundary_ensemble_raw = torch.stack(
                [g["boundary_series_raw"] for g in normalized_group], dim=0
            )
            boundary_member_series = torch.stack(
                [torch.as_tensor(g["boundary"])[:, 0, :].detach().cpu().clone() for g in normalized_group],
                dim=0,
            )
            sample = {
                "hydrograph_id": hydro_id,
                "reference_run_ids": reference_run_ids,
                "geometry": geometry,
                "geometry_raw": normalized_group[0].get("geometry_raw", geometry),
                "static": normalized_group[0]["static"],
                "static_raw": normalized_group[0].get("static_raw"),
                "elevation_raw": normalized_group[0].get("elevation_raw"),
                "boundary": normalized_group[0]["boundary"],
                "boundary_series_raw": normalized_group[0]["boundary_series_raw"],
                "boundary_member_series": boundary_member_series,
                "boundary_ensemble_series_raw": _boundary_ensemble_series_from_reference_members(
                    rollout_boundary_kwargs["boundary_spec"],
                    reference_run_ids,
                    fallback_boundary_ensemble_raw,
                    logger=logger,
                ),
                "boundary_channel_names": list(boundary_channel_names),
                "dynamic_ref": torch.stack([g["dynamic"] for g in normalized_group], dim=0),
                "query_points": query_points,
                "n_ref_sims": len(normalized_group),
                "available_rollout_length": min(
                    int(g.get("available_rollout_length", 0)) for g in normalized_group
                ),
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
        rollout_dataset.available_rollout_length = available_min
        rollout_dataset.available_rollout_length_max = available_max
        rollout_dataset.rollout_full_length = rollout_full_length
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
    rollout_dataset.available_rollout_length = available_min
    rollout_dataset.available_rollout_length_max = available_max
    rollout_dataset.rollout_full_length = rollout_full_length
    return rollout_dataset, None

"""Convert grouped-hydrograph rollout samples into NEON Stage-2 families.

The coastal FGN rollout evaluator yields per-hydrograph sample dicts (built by
``neuralop.flood.eval.datasets._build_rollout_normalized_dataset``) with keys:

    hydrograph_id : str
    geometry      : [Nv, 2]            (normalized mesh coords)
    static        : [Nv, Cs]           (normalized static channels)
    query_points  : [res, res, 2]      (normalized latent-grid queries)
    boundary      : [T_total, Nv, Cb]  (normalized boundary forcing)
    dynamic_ref   : [R, T_total, Nv, C] (HEC-RAS reference ensemble)
    n_ref_sims    : int (R)
    structural_dry_mask : optional [Nv] bool

This module turns those into :class:`NEONFamilySample` objects (reference sliced
to the forecast window; shared pre-issue history from the reference-ensemble
mean; wettable/area weights from available mesh metadata) and performs a
deterministic family-level split. It is torch-only (no dataset/IO deps) so it
is unit-testable with fake samples.
"""

from __future__ import annotations

import copy
from collections.abc import MutableMapping
from typing import Any, Mapping, Optional, Sequence, Tuple

import torch

from neuralop.flood.train.neon import NEONFamilySample


def _get(sample: Any, key: str, default=None):
    if isinstance(sample, Mapping):
        return sample.get(key, default)
    return getattr(sample, key, default)


def _set(sample: Any, key: str, value: Any) -> None:
    if isinstance(sample, MutableMapping):
        sample[key] = value
    else:
        setattr(sample, key, value)


def _as_lower_name(value: Any) -> str:
    return str(value or "").strip().lower()


def _copy_config_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        copied = {key: _copy_config_value(val) for key, val in value.items()}
        try:
            return type(value)(copied)
        except Exception:
            return copied
    if isinstance(value, list):
        return [_copy_config_value(val) for val in value]
    if isinstance(value, tuple):
        return tuple(_copy_config_value(val) for val in value)
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _clean_boundary_file_for_split(channel: Any, split_name: str) -> str | None:
    """Return the clean forcing file for a train/test family split.

    Historical NEON scripts load an eval config whose ``rollout_data`` section
    points at the held-out test package. For Stage-2 training, we intentionally
    reuse that config shape but switch the grouped-family source to the training
    package and the matching clean forcing files.
    """

    current = _get(channel, "clean_boundary_file")
    if current is None:
        return None

    split_token = "Train" if split_name == "train" else "Test"
    other_token = "Test" if split_name == "train" else "Train"
    updated = str(current)
    replacements = (
        (f"_{other_token}_Clean", f"_{split_token}_Clean"),
        (f"_{other_token.lower()}_clean", f"_{split_token.lower()}_clean"),
        (f"_{other_token}", f"_{split_token}"),
        (f"_{other_token.lower()}", f"_{split_token.lower()}"),
    )
    for old, new in replacements:
        if old in updated:
            return updated.replace(old, new)

    channel_name = _as_lower_name(_get(channel, "name"))
    file_name = _as_lower_name(current)
    if "precip" in channel_name or "precip" in file_name:
        return f"Precipitation_{split_token}_Clean.txt"
    if "stage" in channel_name or "stage" in file_name:
        return f"Stage_Hydrographs_{split_token}_Clean.txt"
    return updated


def _rewrite_clean_boundary_files(rollout_section: Any, split_name: str) -> None:
    boundary = _get(rollout_section, "boundary")
    if boundary is None:
        return
    channels = _get(boundary, "channels")
    if channels is None:
        replacement = _clean_boundary_file_for_split(boundary, split_name)
        if replacement is not None:
            _set(boundary, "clean_boundary_file", replacement)
        return
    for channel in channels:
        replacement = _clean_boundary_file_for_split(channel, split_name)
        if replacement is not None:
            _set(channel, "clean_boundary_file", replacement)


def _train_split_txt(config: Any, explicit_split_txt: str | None) -> str:
    if explicit_split_txt:
        return str(explicit_split_txt)
    data = _get(config, "data")
    candidate = _get(data, "train_txt") if data is not None else None
    # Several eval configs carry ``data.train_txt: test.txt`` because they were
    # authored for rollout evaluation. Do not let that stale value leak into
    # Stage-2 training.
    if candidate and _as_lower_name(candidate) != "test.txt":
        return str(candidate)
    return "train.txt"


def _prepare_family_dataset_config(
    config: Any,
    *,
    dataset_split: str = "train",
    split_txt: str | None = None,
) -> tuple[Any, str, str | None]:
    """Return a dataset-builder config for NEON family construction.

    ``dataset_split='train'`` is the default because this module is used by
    Stage-2 training. It converts an eval-style config into a training-package
    view while leaving the caller's config untouched.
    """

    split = _as_lower_name(dataset_split)
    if split in {"test", "rollout"}:
        return config, "test", split_txt
    if split != "train":
        raise ValueError(f"dataset_split must be 'train' or 'test'; got {dataset_split!r}.")

    try:
        prepared = copy.deepcopy(config)
    except Exception:
        prepared = _copy_config_value(config)
    data = _get(prepared, "data")
    rollout = _get(prepared, "rollout_data")
    if data is None or rollout is None:
        raise ValueError(
            "NEON Stage-2 training requires config.data and config.rollout_data sections."
        )

    train_root = _get(data, "train_root") or _get(data, "root")
    if not train_root:
        raise ValueError(
            "NEON Stage-2 training requires data.train_root (preferred) or data.root "
            "to locate the grouped training package."
        )
    resolved_split_txt = _train_split_txt(prepared, split_txt)
    _set(rollout, "root", str(train_root))
    _set(rollout, "test_txt", resolved_split_txt)
    _rewrite_clean_boundary_files(rollout, "train")
    return prepared, "train", resolved_split_txt


def _cell_area_vector(sample: Any, *, nv: int, dtype: torch.dtype) -> torch.Tensor | None:
    """Return positive cell areas when the grouped sample exposes them."""

    for key in ("cell_area", "cell_area_m2", "area", "areas"):
        value = _get(sample, key)
        if value is not None:
            area = torch.as_tensor(value, dtype=dtype).reshape(-1)
            if area.numel() == nv:
                return torch.where(torch.isfinite(area) & (area > 0), area, torch.zeros_like(area))
    static_raw = _get(sample, "static_raw")
    if static_raw is not None:
        raw = torch.as_tensor(static_raw, dtype=dtype)
        if raw.ndim == 2 and raw.shape[0] == nv and raw.shape[1] > 1:
            area = raw[:, 1].reshape(-1)
            return torch.where(torch.isfinite(area) & (area > 0), area, torch.zeros_like(area))
    return None


def grouped_sample_to_family(
    sample: Any,
    *,
    skip_before_timestep: int,
    n_history: int,
    rollout_length: Optional[int] = None,
    wettable_area_weights: bool = True,
) -> NEONFamilySample:
    """Convert one grouped-hydrograph sample dict into a NEONFamilySample.

    The reference ensemble is ``dynamic_ref`` sliced to the forecast window
    ``[start_pred_t : start_pred_t + T]`` where ``start_pred_t =
    skip_before_timestep + n_history`` and ``T`` is ``rollout_length`` (or the
    remaining horizon). The boundary sequence is offset to start at
    ``skip_before_timestep`` so the AR step ``t`` window ``[t : t + n_history]``
    aligns with the coastal rollout convention. Initial histories are the
    shared pre-issue reference-ensemble mean over the history window.
    """
    family_id = str(_get(sample, "hydrograph_id", "unknown"))
    geometry = _get(sample, "geometry")
    static = _get(sample, "static")
    query_points = _get(sample, "query_points")
    boundary = _get(sample, "boundary")
    dynamic_ref = _get(sample, "dynamic_ref")
    if any(v is None for v in (geometry, static, query_points, boundary, dynamic_ref)):
        raise ValueError(f"grouped sample {family_id!r} missing a required field.")

    if dynamic_ref.ndim != 4:
        raise ValueError(
            f"dynamic_ref must be [R, T_total, Nv, C]; got {tuple(dynamic_ref.shape)}."
        )
    R, t_total, nv, c = (int(v) for v in dynamic_ref.shape)
    start_pred_t = int(skip_before_timestep) + int(n_history)
    if start_pred_t >= t_total:
        raise ValueError(
            f"start_pred_t={start_pred_t} >= T_total={t_total} for family {family_id!r}."
        )
    available = t_total - start_pred_t
    T = available if rollout_length in (None, -1) else min(int(rollout_length), available)

    reference = dynamic_ref[:, start_pred_t : start_pred_t + T].contiguous()  # [R, T, Nv, C]
    # Boundary offset so step t's window [t : t+n_history] matches the coastal
    # rollout (window covers [skip+t, skip+t+n_history)).
    boundary_sequence = boundary[int(skip_before_timestep):].contiguous()      # [>=T+n_history, Nv, Cb]
    history_start = int(skip_before_timestep)
    history_stop = history_start + int(n_history)
    history_ref = dynamic_ref[:, history_start:history_stop]
    if history_ref.shape[1] != int(n_history):
        raise ValueError(
            f"not enough pre-issue history for family {family_id!r}: need {n_history} "
            f"frames from t={history_start}, got {history_ref.shape[1]}."
        )
    initial_histories = history_ref.mean(dim=0).contiguous()  # [n_history, Nv, C]

    weights = None
    if wettable_area_weights:
        dry = _get(sample, "structural_dry_mask")
        wettable = torch.ones(nv, dtype=reference.dtype)
        if dry is not None:
            wettable = (~torch.as_tensor(dry, dtype=torch.bool).reshape(nv)).to(reference.dtype)
        area = _cell_area_vector(sample, nv=nv, dtype=reference.dtype)
        if area is not None:
            wettable = wettable * area
        # [T, Nv, C] weights, zero on structural-dry cells, area-weighted when
        # physical cell areas are available and uniform otherwise.
        weights = wettable.view(1, nv, 1).expand(T, nv, c).contiguous()

    return NEONFamilySample(
        family_id=family_id,
        reference=reference,
        weights=weights,
        static=static.unsqueeze(0).contiguous(),
        geometry=geometry.unsqueeze(0).contiguous(),
        query_points=query_points.unsqueeze(0).contiguous(),
        boundary_sequence=boundary_sequence,
        initial_histories=initial_histories,
    )


def split_families_by_id(
    families: Sequence[NEONFamilySample],
    *,
    val_family_ids: Optional[Sequence[str]] = None,
    val_fraction: float = 0.1,
) -> Tuple[list[NEONFamilySample], list[NEONFamilySample]]:
    """Deterministic family-level train/val split (no member leakage).

    If ``val_family_ids`` is given, those families form the validation set.
    Otherwise the last ``ceil(val_fraction * N)`` families by sorted family_id
    are held out, so the split is stable across runs.
    """
    families = list(families)
    if not families:
        raise ValueError("no families to split.")
    if val_family_ids is not None:
        val_ids = {str(x) for x in val_family_ids}
        train = [f for f in families if f.family_id not in val_ids]
        val = [f for f in families if f.family_id in val_ids]
        if not val:
            raise ValueError("val_family_ids matched no families.")
        return train, val

    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1); got {val_fraction}.")
    ordered = sorted(families, key=lambda f: f.family_id)
    import math

    n_val = max(1, math.ceil(val_fraction * len(ordered)))
    val = ordered[-n_val:]
    train = ordered[:-n_val]
    if not train:
        raise ValueError("val_fraction left no training families.")
    return train, val


def grouped_samples_to_families(
    hydrograph_samples: Sequence[Any],
    *,
    skip_before_timestep: int,
    n_history: int,
    rollout_length: Optional[int] = None,
    val_family_ids: Optional[Sequence[str]] = None,
    val_fraction: float = 0.1,
    wettable_area_weights: bool = True,
    max_families: Optional[int] = None,
) -> Tuple[list[NEONFamilySample], list[NEONFamilySample]]:
    """Convert grouped-hydrograph samples to (train, val) NEONFamilySample splits."""
    families: list[NEONFamilySample] = []
    for idx, sample in enumerate(hydrograph_samples):
        if max_families is not None and idx >= int(max_families):
            break
        families.append(
            grouped_sample_to_family(
                sample,
                skip_before_timestep=skip_before_timestep,
                n_history=n_history,
                rollout_length=rollout_length,
                wettable_area_weights=wettable_area_weights,
            )
        )
    return split_families_by_id(
        families, val_family_ids=val_family_ids, val_fraction=val_fraction
    )


def build_families_from_config(
    config: Any,
    normalizers: Mapping[str, Any],
    target_variables: Sequence[str],
    logger: Any,
    *,
    structural_dry_artifact: Optional[Mapping[str, Any]] = None,
    rollout_length: Optional[int] = None,
    max_families: Optional[int] = None,
    val_fraction: float = 0.1,
    val_family_ids: Optional[Sequence[str]] = None,
    wettable_area_weights: bool = True,
    dataset_split: str = "train",
    split_txt: str | None = None,
) -> Tuple[list[NEONFamilySample], list[NEONFamilySample]]:
    """Build (train, val) NEONFamilySample splits from a flood eval config.

    Calls the shared grouped-hydrograph rollout dataset builder (the same one
    the coastal FGN evaluator uses) and converts its per-hydrograph samples
    into NEON families. ``config`` must expose ``data`` (n_history,
    skip_before_timestep, rollout_length, query_res) and ``rollout_data``
    sections. By default, an eval-style ``rollout_data`` test view is converted
    to the grouped training package described by ``data.train_root``. Requires
    grouped hydrographs (R>1 references).
    """
    # Lazy import so this module's converter unit tests stay free of the heavy
    # dataset/IO dependency chain.
    from neuralop.flood.eval.datasets import _build_rollout_normalized_dataset

    family_config, split_name, resolved_split_txt = _prepare_family_dataset_config(
        config,
        dataset_split=dataset_split,
        split_txt=split_txt,
    )
    rollout_section = _get(family_config, "rollout_data")
    if logger is not None:
        logger.info(
            "building NEON families from %s package: root=%s split_txt=%s",
            split_name,
            _get(rollout_section, "root"),
            resolved_split_txt or _get(rollout_section, "test_txt", "test.txt"),
        )
    _rollout_ds, hydrograph_samples = _build_rollout_normalized_dataset(
        family_config,
        dict(normalizers),
        list(target_variables),
        logger,
        structural_dry_artifact=structural_dry_artifact,
        split_txt=resolved_split_txt,
        split_name=split_name,
        config_section="rollout_data",
    )
    if not hydrograph_samples:
        raise ValueError(
            "dataset builder produced no grouped hydrograph samples; NEON Stage-2 "
            "requires grouped families with R>1 HEC-RAS references per hydrograph."
        )
    data_section = _get(family_config, "data")
    skip = int(_get(data_section, "skip_before_timestep", 0)) if data_section is not None else 0
    n_hist = int(_get(data_section, "n_history", 3)) if data_section is not None else 3
    return grouped_samples_to_families(
        hydrograph_samples,
        skip_before_timestep=skip,
        n_history=n_hist,
        rollout_length=rollout_length,
        val_family_ids=val_family_ids,
        val_fraction=val_fraction,
        wettable_area_weights=wettable_area_weights,
        max_families=max_families,
    )

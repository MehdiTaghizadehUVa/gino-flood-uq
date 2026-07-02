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
to the forecast window; dry/baseline initial histories; wettable-area weights
from the structural-dry mask) and performs a deterministic family-level split.
It is torch-only (no dataset/IO deps) so it is unit-testable with fake samples.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple

import torch

from neuralop.flood.train.neon import NEONFamilySample


def _get(sample: Any, key: str, default=None):
    if isinstance(sample, Mapping):
        return sample.get(key, default)
    return getattr(sample, key, default)


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
    aligns with the coastal rollout convention. Initial histories are dry/
    baseline zeros (the plan's default warm start).
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
    initial_histories = torch.zeros(int(n_history), nv, c, dtype=reference.dtype)

    weights = None
    if wettable_area_weights:
        dry = _get(sample, "structural_dry_mask")
        if dry is not None:
            wettable = (~torch.as_tensor(dry, dtype=torch.bool).reshape(nv)).to(reference.dtype)
            # [T, Nv, C] weights, zero on structural-dry cells, uniform elsewhere.
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
) -> Tuple[list[NEONFamilySample], list[NEONFamilySample]]:
    """Build (train, val) NEONFamilySample splits from a flood eval config.

    Calls the shared grouped-hydrograph rollout dataset builder (the same one
    the coastal FGN evaluator uses) and converts its per-hydrograph samples
    into NEON families. ``config`` must expose ``data`` (n_history,
    skip_before_timestep, rollout_length, query_res) and ``rollout_data``
    (grouped test set) sections. Requires grouped hydrographs (R>1 references).
    """
    # Lazy import so this module's converter unit tests stay free of the heavy
    # dataset/IO dependency chain.
    from neuralop.flood.eval.datasets import _build_rollout_normalized_dataset

    _rollout_ds, hydrograph_samples = _build_rollout_normalized_dataset(
        config,
        dict(normalizers),
        list(target_variables),
        logger,
        structural_dry_artifact=structural_dry_artifact,
    )
    if not hydrograph_samples:
        raise ValueError(
            "dataset builder produced no grouped hydrograph samples; NEON Stage-2 "
            "requires grouped families with R>1 HEC-RAS references per hydrograph."
        )
    skip = int(_get(config.data, "skip_before_timestep", 0)) if hasattr(config, "data") else 0
    n_hist = int(_get(config.data, "n_history", 3)) if hasattr(config, "data") else 3
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

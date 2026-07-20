"""Flood-impact uncertainty metrics for rollout evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


DEFAULT_IMPACT_THRESHOLD_M = 0.10
DEFAULT_POOLED_RADII_M = (100.0, 250.0, 500.0, 1000.0)


@dataclass(frozen=True)
class ImpactMetricsConfig:
    enabled: bool = True
    inundation_threshold_m: float = DEFAULT_IMPACT_THRESHOLD_M
    pooled_radii_m: Tuple[float, ...] = DEFAULT_POOLED_RADII_M


def _cfg_get(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    try:
        return getattr(config, key)
    except (AttributeError, KeyError, TypeError):
        return default


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(default)


def normalize_impact_metrics_config(config: Any = None) -> ImpactMetricsConfig:
    """Normalize optional ``rollout.impact_metrics`` config with safe defaults."""
    enabled = _as_bool(_cfg_get(config, "enabled", True), True)
    threshold = float(_cfg_get(config, "inundation_threshold_m", DEFAULT_IMPACT_THRESHOLD_M))
    raw_radii = _cfg_get(config, "pooled_radii_m", DEFAULT_POOLED_RADII_M)
    if isinstance(raw_radii, (int, float, str)):
        raw_radii = [raw_radii]
    radii = tuple(float(radius) for radius in raw_radii)
    if threshold < 0.0:
        raise ValueError("rollout.impact_metrics.inundation_threshold_m must be non-negative.")
    if any(radius <= 0.0 for radius in radii):
        raise ValueError("rollout.impact_metrics.pooled_radii_m values must be positive.")
    return ImpactMetricsConfig(
        enabled=enabled,
        inundation_threshold_m=threshold,
        pooled_radii_m=radii,
    )


def _to_numpy(value: Any, *, dtype=np.float64) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _crps_ensemble_vs_reference(
    forecast_ens: np.ndarray,
    reference_ens: np.ndarray,
) -> np.ndarray:
    """Fair CRPS for each column of forecast/reference ensemble arrays.

    The first term averages over forecast and reference members. The forecast
    self-distance term uses the finite-ensemble fair denominator K*(K-1),
    excluding diagonal pairs. With one forecast member, the score reduces to MAE
    against the reference ensemble because no fair spread correction is defined.
    """
    forecast = np.asarray(forecast_ens, dtype=np.float64)
    reference = np.asarray(reference_ens, dtype=np.float64)
    if forecast.ndim != 2 or reference.ndim != 2:
        raise ValueError("forecast_ens and reference_ens must be [members, locations].")
    if forecast.shape[1] != reference.shape[1]:
        raise ValueError("forecast/reference location dimensions differ.")
    k = forecast.shape[0]
    if k < 1 or reference.shape[0] < 1:
        raise ValueError("fair CRPS requires at least one forecast and one reference member.")
    term_1 = np.mean(np.abs(forecast[:, None, :] - reference[None, :, :]), axis=(0, 1))
    if k < 2:
        return term_1
    pair_sum = np.sum(np.abs(forecast[:, None, :] - forecast[None, :, :]), axis=(0, 1))
    term_2 = pair_sum / float(2 * k * (k - 1))
    return term_1 - term_2


def ensemble_crps_scalar(forecast_ens: np.ndarray, reference_ens: np.ndarray) -> float:
    """CRPS for scalar ensemble-valued quantities."""
    forecast = np.asarray(forecast_ens, dtype=np.float64).reshape(-1, 1)
    reference = np.asarray(reference_ens, dtype=np.float64).reshape(-1, 1)
    return float(_crps_ensemble_vs_reference(forecast, reference)[0])


def _radius_key(radius_m: float) -> str:
    rounded = int(round(radius_m))
    if np.isclose(radius_m, rounded):
        return f"r{rounded}"
    return f"r{str(radius_m).replace('.', 'p')}"


def _active_geometry_and_area(
    geometry: Any,
    static_raw: Any,
    wettable_mask: Optional[np.ndarray],
    n_cells: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = _to_numpy(geometry)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("Impact metrics require geometry with shape (n_cells, >=2).")
    if coords.shape[0] < n_cells:
        raise ValueError(
            f"Impact metric geometry has {coords.shape[0]} cells, expected at least {n_cells}."
        )
    coords = coords[:n_cells, :2]

    if wettable_mask is None:
        active = np.ones(n_cells, dtype=bool)
    else:
        active = np.asarray(wettable_mask, dtype=bool).reshape(-1)
        if active.size != n_cells:
            raise ValueError(
                f"Impact metric wettable mask has {active.size} cells, expected {n_cells}."
            )
    if not np.any(active):
        return coords[:0], np.ones(0, dtype=np.float64), active

    area = None
    if static_raw is not None:
        static = _to_numpy(static_raw)
        if static.ndim == 2 and static.shape[0] >= n_cells and static.shape[1] >= 2:
            area = static[:n_cells, 1]
    if area is None or not np.any(np.isfinite(area) & (area > 0.0)):
        area = np.ones(n_cells, dtype=np.float64)
    else:
        area = np.where(np.isfinite(area) & (area > 0.0), area, 0.0)
        if float(np.sum(area)) <= 0.0:
            area = np.ones(n_cells, dtype=np.float64)
    return coords[active], area[active].astype(np.float64, copy=False), active


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    mask = np.isfinite(vals) & np.isfinite(w) & (w > 0.0)
    if not np.any(mask):
        return 0.0
    return float(np.average(vals[mask], weights=w[mask]))


def radius_neighborhoods(coords: np.ndarray, radius_m: float) -> List[np.ndarray]:
    """Return local-index neighbor lists for each coordinate within ``radius_m``."""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.size == 0:
        return []
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(coords)
        return [np.asarray(nbrs, dtype=np.int64) for nbrs in tree.query_ball_point(coords, r=radius_m)]
    except Exception:
        neighborhoods: List[np.ndarray] = []
        for start in range(0, coords.shape[0], 1024):
            chunk = coords[start:start + 1024]
            dist2 = np.sum((chunk[:, None, :] - coords[None, :, :]) ** 2, axis=2)
            for row in dist2:
                neighborhoods.append(np.flatnonzero(row <= radius_m * radius_m).astype(np.int64))
        return neighborhoods


def _average_pool_matrix(neighborhoods: List[np.ndarray], area: np.ndarray):
    from scipy.sparse import csr_matrix

    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    for row_idx, nbrs in enumerate(neighborhoods):
        if nbrs.size == 0:
            nbrs = np.asarray([row_idx], dtype=np.int64)
        weights = np.asarray(area[nbrs], dtype=np.float64)
        total = float(np.sum(weights))
        if total <= 0.0:
            weights = np.full(nbrs.size, 1.0 / max(nbrs.size, 1), dtype=np.float64)
        else:
            weights = weights / total
        rows.extend([row_idx] * int(nbrs.size))
        cols.extend(int(col) for col in nbrs)
        vals.extend(float(val) for val in weights)
    n = len(neighborhoods)
    return csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)


def _apply_average_pool(values: np.ndarray, weight_matrix: Any) -> np.ndarray:
    return np.asarray(weight_matrix.dot(np.asarray(values, dtype=np.float64).T)).T


def _apply_max_pool(values: np.ndarray, neighborhoods: List[np.ndarray]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    pooled = np.empty((arr.shape[0], len(neighborhoods)), dtype=np.float64)
    max_flat_neighbors = 250_000
    start = 0
    while start < len(neighborhoods):
        flat_parts: List[np.ndarray] = []
        lengths: List[int] = []
        end = start
        flat_count = 0
        while end < len(neighborhoods):
            nbrs = neighborhoods[end]
            if nbrs.size == 0:
                nbrs = np.asarray([end], dtype=np.int64)
            if flat_parts and flat_count + int(nbrs.size) > max_flat_neighbors:
                break
            flat_parts.append(nbrs)
            lengths.append(int(nbrs.size))
            flat_count += int(nbrs.size)
            end += 1
        flat = np.concatenate(flat_parts)
        offsets = np.concatenate([[0], np.cumsum(lengths[:-1])]).astype(np.int64)
        gathered = arr[:, flat]
        pooled[:, start:end] = np.maximum.reduceat(gathered, offsets, axis=1)
        start = end
    return pooled


def inundated_area_series(
    wd_rollout: np.ndarray,
    area: np.ndarray,
    threshold_m: float,
) -> np.ndarray:
    """Return inundated area per lead time and ensemble member."""
    wd = np.asarray(wd_rollout, dtype=np.float64)
    cell_area = np.asarray(area, dtype=np.float64)
    return np.sum((wd >= float(threshold_m)) * cell_area[None, None, :], axis=2)


def peak_inundated_area_series(area_series: np.ndarray) -> np.ndarray:
    """Return cumulative peak inundated area per lead time and ensemble member."""
    return np.maximum.accumulate(np.asarray(area_series, dtype=np.float64), axis=0)


def arrival_times(
    wd_rollout: np.ndarray,
    threshold_m: float,
    *,
    never_time: float,
) -> np.ndarray:
    """Return first 1-based lead-time step reaching threshold, or ``never_time``."""
    wet = np.asarray(wd_rollout, dtype=np.float64) >= float(threshold_m)
    any_wet = np.any(wet, axis=0)
    first = np.argmax(wet, axis=0).astype(np.float64) + 1.0
    first[~any_wet] = float(never_time)
    return first


def _nested_score_function(sampling_design: Any):
    """Resolve the finite-ensemble score implied by a nested sampling design."""

    from neuralop.flood.neon import (
        crossed_fair_crps_members,
        fixed_support_fair_crps_members,
        independent_nested_fair_crps_members,
    )

    kind = getattr(sampling_design, "kind", sampling_design)
    scorers = {
        "crossed_common_random_numbers": crossed_fair_crps_members,
        "crossed": crossed_fair_crps_members,
        "fixed_epistemic_support_common_random_numbers": fixed_support_fair_crps_members,
        "fixed_support": fixed_support_fair_crps_members,
        "independent_nested": independent_nested_fair_crps_members,
    }
    try:
        return scorers[str(kind)]
    except KeyError as exc:
        raise ValueError(
            "Impact metrics require a recognized nested sampling design; got "
            f"{kind!r}."
        ) from exc


def _nested_crps_scalar(
    forecast: np.ndarray,
    reference: np.ndarray,
    *,
    scorer: Any,
    weights: np.ndarray | None = None,
) -> float:
    """Score ``[M,K,L]`` against ``[R,L]`` without flattening the design."""

    import torch

    pred = torch.as_tensor(np.asarray(forecast), dtype=torch.float64)
    ref = torch.as_tensor(np.asarray(reference), dtype=torch.float64)
    if pred.ndim != 3 or ref.ndim != 2 or pred.shape[2] != ref.shape[1]:
        raise ValueError(
            "Nested impact score expects forecast [M,K,L] and reference [R,L]."
        )
    pred = pred.unsqueeze(0).unsqueeze(3).unsqueeze(-1)  # [1,M,K,1,L,1]
    ref = ref.unsqueeze(0).unsqueeze(2).unsqueeze(-1)  # [1,R,1,L,1]
    score_weights = None
    if weights is not None:
        score_weights = torch.as_tensor(weights, dtype=torch.float64).reshape(1, -1, 1)
    return float(
        scorer(
            pred,
            ref,
            weights=score_weights,
            reduction="mean",
        ).item()
    )


def compute_nested_flood_impact_crps_metrics(
    pred_wd_nested: np.ndarray,
    ref_wd_rollout: np.ndarray,
    geometry: Any,
    static_raw: Any = None,
    wettable_mask: Optional[np.ndarray] = None,
    config: Any = None,
    *,
    sampling_design: Any,
) -> Dict[str, np.ndarray | float]:
    """Compute design-aware impact CRPS for one nested NEON forecast.

    The forecast must retain its ``[M,K,T,Nv]`` epistemic-by-aleatory axes.
    Inundated areas are reported in km2 and use the configured depth threshold
    (0.1 m by default). Arrival time is right-censored at ``T+1`` for members
    and reference realizations that never exceed the threshold.
    """

    cfg = normalize_impact_metrics_config(config)
    if not cfg.enabled:
        return {}
    pred = np.asarray(pred_wd_nested, dtype=np.float64)
    ref = np.asarray(ref_wd_rollout, dtype=np.float64)
    if pred.ndim != 4 or ref.ndim != 3:
        raise ValueError(
            "Nested impact metrics expect prediction [M,K,T,Nv] and reference [R,T,Nv]."
        )
    M, K, n_steps, n_cells = pred.shape
    if M < 2 or K < 2:
        raise ValueError("Nested impact metrics require M >= 2 and K >= 2.")
    if ref.shape[1:] != (n_steps, n_cells):
        raise ValueError(
            "Forecast/reference nested impact arrays must share time and cell dimensions."
        )
    scorer = _nested_score_function(sampling_design)
    coords, area_m2, active = _active_geometry_and_area(
        geometry, static_raw, wettable_mask, n_cells
    )
    if coords.shape[0] == 0:
        return {}
    pred = pred[..., active]
    ref = ref[..., active]
    domain_area_m2 = float(np.sum(area_m2))
    if not np.isfinite(domain_area_m2) or domain_area_m2 <= 0.0:
        raise ValueError("Impact metric normalization requires positive active-domain area.")

    area_km2 = area_m2 / 1_000_000.0
    threshold = float(cfg.inundation_threshold_m)
    pred_area = np.sum((pred >= threshold) * area_km2[None, None, None, :], axis=-1)
    ref_area = np.sum((ref >= threshold) * area_km2[None, None, :], axis=-1)
    pred_peak = np.maximum.accumulate(pred_area, axis=2)
    ref_peak = np.maximum.accumulate(ref_area, axis=1)

    area_scores = []
    peak_scores = []
    for t in range(n_steps):
        area_scores.append(
            _nested_crps_scalar(
                pred_area[:, :, t, None],
                ref_area[:, t, None],
                scorer=scorer,
            )
        )
        peak_scores.append(
            _nested_crps_scalar(
                pred_peak[:, :, t, None],
                ref_peak[:, t, None],
                scorer=scorer,
            )
        )

    pred_wet = pred >= threshold
    ref_wet = ref >= threshold
    censor_step = float(n_steps + 1)
    pred_any = np.any(pred_wet, axis=2)
    ref_any = np.any(ref_wet, axis=1)
    pred_arrival = np.argmax(pred_wet, axis=2).astype(np.float64) + 1.0
    ref_arrival = np.argmax(ref_wet, axis=1).astype(np.float64) + 1.0
    pred_arrival[~pred_any] = censor_step
    ref_arrival[~ref_any] = censor_step
    arrival_score = _nested_crps_scalar(
        pred_arrival,
        ref_arrival,
        scorer=scorer,
        weights=area_m2,
    )

    return {
        "crps_total_inundated_area_km2": np.asarray(area_scores, dtype=np.float64),
        "crps_peak_inundated_area_km2": np.asarray(peak_scores, dtype=np.float64),
        "crps_arrival_time_step": float(arrival_score),
        "inundation_threshold_m": threshold,
        "area_unit_scale_m2_per_output_unit": 1_000_000.0,
        "arrival_censor_step": censor_step,
    }


def compute_flood_impact_crps_metrics(
    pred_wd_rollout: np.ndarray,
    ref_wd_rollout: np.ndarray,
    geometry: Any,
    static_raw: Any = None,
    wettable_mask: Optional[np.ndarray] = None,
    config: Any = None,
) -> Dict[str, np.ndarray | float]:
    """Compute flood-impact CRPS metrics for one hydrograph rollout."""
    cfg = normalize_impact_metrics_config(config)
    if not cfg.enabled:
        return {}
    pred = np.asarray(pred_wd_rollout, dtype=np.float64)
    ref = np.asarray(ref_wd_rollout, dtype=np.float64)
    if pred.ndim != 3 or ref.ndim != 3:
        raise ValueError("Impact metrics expect rollout arrays shaped (time, ensemble, cells).")
    if pred.shape[0] != ref.shape[0] or pred.shape[2] != ref.shape[2]:
        raise ValueError(
            "Forecast/reference rollout arrays must share time and cell dimensions for impact metrics."
        )
    n_steps, _, n_cells = pred.shape
    coords, area, active = _active_geometry_and_area(geometry, static_raw, wettable_mask, n_cells)
    if coords.shape[0] == 0:
        return {}
    pred = pred[:, :, active]
    ref = ref[:, :, active]

    metrics: Dict[str, np.ndarray | float] = {}
    domain_area = float(np.sum(area))
    if not np.isfinite(domain_area) or domain_area <= 0.0:
        raise ValueError("Impact metric normalization requires positive active-domain area.")

    pred_area = inundated_area_series(pred, area, cfg.inundation_threshold_m)
    ref_area = inundated_area_series(ref, area, cfg.inundation_threshold_m)
    metrics["crps_total_inundated_area_wd"] = np.asarray(
        [ensemble_crps_scalar(pred_area[t], ref_area[t]) for t in range(n_steps)],
        dtype=np.float64,
    )
    pred_area_fraction = pred_area / domain_area
    ref_area_fraction = ref_area / domain_area
    metrics["crps_total_inundated_area_fraction_wd"] = np.asarray(
        [
            ensemble_crps_scalar(pred_area_fraction[t], ref_area_fraction[t])
            for t in range(n_steps)
        ],
        dtype=np.float64,
    )

    pred_peak = peak_inundated_area_series(pred_area)
    ref_peak = peak_inundated_area_series(ref_area)
    metrics["crps_peak_inundated_area_wd"] = np.asarray(
        [ensemble_crps_scalar(pred_peak[t], ref_peak[t]) for t in range(n_steps)],
        dtype=np.float64,
    )
    pred_peak_fraction = peak_inundated_area_series(pred_area_fraction)
    ref_peak_fraction = peak_inundated_area_series(ref_area_fraction)
    metrics["crps_peak_inundated_area_fraction_wd"] = np.asarray(
        [
            ensemble_crps_scalar(pred_peak_fraction[t], ref_peak_fraction[t])
            for t in range(n_steps)
        ],
        dtype=np.float64,
    )

    never_time = float(n_steps + 1)
    pred_arrival = arrival_times(pred, cfg.inundation_threshold_m, never_time=never_time)
    ref_arrival = arrival_times(ref, cfg.inundation_threshold_m, never_time=never_time)
    arrival_crps = _crps_ensemble_vs_reference(pred_arrival, ref_arrival)
    metrics["crps_arrival_time_wd"] = _weighted_mean(arrival_crps, area)
    arrival_fraction_crps = _crps_ensemble_vs_reference(
        pred_arrival / never_time,
        ref_arrival / never_time,
    )
    metrics["crps_arrival_time_fraction_wd"] = _weighted_mean(arrival_fraction_crps, area)

    for radius in cfg.pooled_radii_m:
        radius_name = _radius_key(radius)
        neighborhoods = radius_neighborhoods(coords, radius)
        avg_pool = _average_pool_matrix(neighborhoods, area)
        avg_vals = []
        max_vals = []
        for t in range(n_steps):
            pred_avg = _apply_average_pool(pred[t], avg_pool)
            ref_avg = _apply_average_pool(ref[t], avg_pool)
            avg_vals.append(_weighted_mean(_crps_ensemble_vs_reference(pred_avg, ref_avg), area))

            pred_max = _apply_max_pool(pred[t], neighborhoods)
            ref_max = _apply_max_pool(ref[t], neighborhoods)
            max_vals.append(_weighted_mean(_crps_ensemble_vs_reference(pred_max, ref_max), area))

        metrics[f"pooled_avg_crps_wd_{radius_name}"] = np.asarray(avg_vals, dtype=np.float64)
        metrics[f"pooled_max_crps_wd_{radius_name}"] = np.asarray(max_vals, dtype=np.float64)

    return metrics

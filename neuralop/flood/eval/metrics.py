"""Metrics and ensemble-statistics helpers for flood evaluation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from neuralop.flood.losses import (
    FloodDryBackgroundFalseWetRate,
    FloodDryBackgroundMAE,
    FloodDryBackgroundRMSE,
    FloodEnsembleDryPredStdMean,
    FloodGaussianDryPredStdMean,
    FloodMaskedCRPSLoss,
    FloodMaskedGaussianNLLLoss,
    FloodMaskedRelLpLoss,
)
from neuralop.flood.eval.runtime import MIN_EPS, _opt, _opt_float
from neuralop.flood.utils.runtime import parse_target_variables
from neuralop.losses.data_losses import LpLoss
from neuralop.losses.probabilistic_losses import CRPSLoss, GaussianNLLLoss, split_gaussian_packed

def _is_gaussian_mode(config: Any) -> bool:
    out_dist = str(_opt(config, "gino", "output_distribution", "deterministic")).strip().lower()
    train_loss = str(_opt(config, "opt", "training_loss", "l2")).strip().lower()
    return out_dist == "gaussian" and train_loss == "gaussian_nll"


def _sample_from_packed_gaussian(
    out: torch.Tensor,
    n_channels: int,
    min_logvar: float,
    max_logvar: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu, logvar = split_gaussian_packed(out, n_channels=n_channels)
    logvar = torch.clamp(logvar, min=float(min_logvar), max=float(max_logvar))
    sample = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
    return sample, mu, logvar


def _resolve_target_variables(config: Any, target_variables: Optional[List[str]]) -> List[str]:
    if target_variables is not None:
        return parse_target_variables(target_variables)
    return parse_target_variables(_opt(config, "data", "target_variables", ["wd", "vx", "vy"]))


def _add_dry_background_wd_metrics(
    out: Dict[str, Any],
    *,
    wd_idx: int | None,
    gaussian_n_channels: int | None = None,
    gaussian_min_logvar: float = -9.0,
    gaussian_max_logvar: float = 4.0,
    ensemble_std: bool = False,
) -> None:
    if wd_idx is None:
        return
    out["rmse_dry_background_wd"] = FloodDryBackgroundRMSE(channel_idx=wd_idx)
    out["mae_dry_background_wd"] = FloodDryBackgroundMAE(channel_idx=wd_idx)
    out["falsewet_rate_001_dry_background_wd"] = FloodDryBackgroundFalseWetRate(
        0.01,
        channel_idx=wd_idx,
    )
    out["falsewet_rate_005_dry_background_wd"] = FloodDryBackgroundFalseWetRate(
        0.05,
        channel_idx=wd_idx,
    )
    if gaussian_n_channels is not None:
        out["pred_std_mean_dry_background_wd"] = FloodGaussianDryPredStdMean(
            n_channels=gaussian_n_channels,
            min_logvar=gaussian_min_logvar,
            max_logvar=gaussian_max_logvar,
            channel_idx=wd_idx,
        )
    elif ensemble_std:
        out["pred_std_mean_dry_background_wd"] = FloodEnsembleDryPredStdMean(
            channel_idx=wd_idx,
        )


def _build_eval_losses(
    config: Any,
    use_fgn: bool,
    target_variables: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build loss dict for one-step evaluation (L2 and optionally CRPS)."""
    l2_loss = LpLoss(d=2, p=2)
    resolved_targets = _resolve_target_variables(config, target_variables)
    wd_idx = resolved_targets.index("wd") if "wd" in resolved_targets else None
    structural_policy = str(
        _opt(config, "structural_dry", "policy", "legacy_full_domain")
    ).strip().lower()
    primary_l2 = FloodMaskedRelLpLoss(
        policy=structural_policy,
        base_loss=l2_loss,
        reduction="sum",
    )
    if _is_gaussian_mode(config):
        out = {
            "l2": primary_l2,
            "gaussian_nll": FloodMaskedGaussianNLLLoss(
                policy=structural_policy,
                base_loss=GaussianNLLLoss(
                    channel_weights=_opt(config, "opt", "crps_channel_weights", None),
                    reduction="mean",
                    min_logvar=_opt_float(config, "opt", "gaussian_min_logvar", -9.0),
                    max_logvar=_opt_float(config, "opt", "gaussian_max_logvar", 4.0),
                    logvar_reg_weight=0.0,
                ),
            ),
        }
        if structural_policy == "masked_primary":
            out["l2_full_domain"] = l2_loss
            out["gaussian_nll_full_domain"] = GaussianNLLLoss(
                channel_weights=_opt(config, "opt", "crps_channel_weights", None),
                reduction="mean",
                min_logvar=_opt_float(config, "opt", "gaussian_min_logvar", -9.0),
                max_logvar=_opt_float(config, "opt", "gaussian_max_logvar", 4.0),
                logvar_reg_weight=0.0,
            )
            _add_dry_background_wd_metrics(
                out,
                wd_idx=wd_idx,
                gaussian_n_channels=len(resolved_targets),
                gaussian_min_logvar=_opt_float(config, "opt", "gaussian_min_logvar", -9.0),
                gaussian_max_logvar=_opt_float(config, "opt", "gaussian_max_logvar", 4.0),
            )
        return out
    if use_fgn and _opt(config, "opt", "training_loss", "l2") == "crps":
        n_samples = max(2, int(_opt(config, "opt", "crps_n_samples", 2)))
        ch_weights = _opt(config, "opt", "crps_channel_weights", None)
        out = {
            "l2": primary_l2,
            "crps": FloodMaskedCRPSLoss(
                policy=structural_policy,
                base_loss=CRPSLoss(
                    n_samples=n_samples, channel_weights=ch_weights, reduction="mean"
                ),
            ),
        }
        if structural_policy == "masked_primary":
            out["l2_full_domain"] = l2_loss
            out["crps_full_domain"] = FloodMaskedCRPSLoss(
                policy="legacy_full_domain",
                base_loss=CRPSLoss(
                    n_samples=n_samples,
                    channel_weights=ch_weights,
                    reduction="mean",
                ),
            )
            _add_dry_background_wd_metrics(out, wd_idx=wd_idx, ensemble_std=True)
        return out
    test_loss_name = _opt(config, "opt", "testing_loss", "l2")
    out = {test_loss_name: primary_l2}
    if structural_policy == "masked_primary":
        out["l2_full_domain"] = l2_loss
        _add_dry_background_wd_metrics(out, wd_idx=wd_idx)
    return out


def _compute_csi(threshold: float, pred: np.ndarray, gt: np.ndarray) -> float:
    """Critical Success Index at a given threshold."""
    event_pred = np.asarray(pred) >= float(threshold)
    event_gt = np.asarray(gt) >= float(threshold)
    tp = np.sum(event_pred & event_gt)
    fp = np.sum(event_pred & (~event_gt))
    fn = np.sum((~event_pred) & event_gt)
    denom = tp + fp + fn
    return float(tp / denom) if denom > 0 else 1.0


def _build_query_points_from_geometry(
    geometry: torch.Tensor, query_res: List[int]
) -> torch.Tensor:
    """Build query grid points from geometry bounds."""
    geom = geometry.detach().cpu().numpy()
    x_vals = geom[:, 0]
    y_vals = geom[:, 1]
    tx = np.linspace(float(x_vals.min()), float(x_vals.max()), query_res[0], dtype=np.float32)
    ty = np.linspace(float(y_vals.min()), float(y_vals.max()), query_res[1], dtype=np.float32)
    grid_x, grid_y = np.meshgrid(tx, ty, indexing="ij")
    q_pts = np.stack([grid_x, grid_y], axis=-1)
    return torch.tensor(q_pts, device=geometry.device, dtype=geometry.dtype)


def _crps_ensemble_vs_reference(
    forecast_ens: np.ndarray, reference_ens: np.ndarray
) -> np.ndarray:
    """
    Fair CRPS per location for forecast ensemble against reference ensemble.

    Parameters
    ----------
    forecast_ens: [n_forecast, n_locations]
    reference_ens: [n_reference, n_locations]
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


def _build_member_model_indices(n_models: int, n_members: int) -> List[int]:
    """Allocate ensemble members to models as evenly as possible."""
    if n_models <= 0:
        raise ValueError("n_models must be >= 1")
    if n_members <= 0:
        raise ValueError("n_members must be >= 1")
    base = n_members // n_models
    rem = n_members % n_models
    out: List[int] = []
    for model_idx in range(n_models):
        count = base + (1 if model_idx < rem else 0)
        out.extend([model_idx] * count)
    return out


def _variance_decomposition_by_model(
    pred_ens: np.ndarray, member_model_indices: List[int], n_models: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose predictive variance into within-model and between-model components.

    Parameters
    ----------
    pred_ens: [n_ens, n_locations]
    member_model_indices: model index for each ensemble member
    n_models: number of trained models
    """
    if pred_ens.ndim != 2:
        raise ValueError("pred_ens must be 2D [n_ens, n_locations]")
    n_ens = pred_ens.shape[0]
    if n_ens != len(member_model_indices):
        raise ValueError("member_model_indices length must match n_ens")
    n_loc = pred_ens.shape[1]
    if n_models <= 0:
        z = np.zeros(n_loc, dtype=np.float64)
        return z, z, z

    member_idx = np.asarray(member_model_indices, dtype=np.int64)
    model_means: List[np.ndarray] = []
    within_vars: List[np.ndarray] = []
    for m in range(n_models):
        sel = np.where(member_idx == m)[0]
        if sel.size == 0:
            continue
        vals = pred_ens[sel, :]
        model_means.append(np.mean(vals, axis=0))
        if sel.size > 1:
            within_vars.append(np.var(vals, axis=0, ddof=0))
        else:
            within_vars.append(np.zeros(n_loc, dtype=np.float64))
    if not model_means:
        z = np.zeros(n_loc, dtype=np.float64)
        return z, z, z
    within = np.mean(np.stack(within_vars, axis=0), axis=0)
    between = (
        np.var(np.stack(model_means, axis=0), axis=0, ddof=0)
        if len(model_means) > 1
        else np.zeros(n_loc, dtype=np.float64)
    )
    total = within + between
    return within, between, total


def _pit_rank_counts_from_reference(
    pred_ens: np.ndarray,
    ref_ens: np.ndarray,
    pit_edges: np.ndarray,
    n_ens: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    PIT and rank histogram counts using all reference members as pseudo-observations.

    pred_ens shape: [n_pred, n_locations]
    ref_ens shape:  [n_ref, n_locations]
    """
    if pred_ens.ndim != 2 or ref_ens.ndim != 2:
        raise ValueError("pred_ens/ref_ens must be 2D arrays")
    if pred_ens.shape[1] != ref_ens.shape[1]:
        raise ValueError("pred_ens and ref_ens must share n_locations")

    # Compare each reference member against predictive ensemble.
    less = np.sum(pred_ens[:, None, :] < ref_ens[None, :, :], axis=0).astype(np.int64)
    equal = np.sum(pred_ens[:, None, :] == ref_ens[None, :, :], axis=0).astype(np.int64)
    u = rng.random(size=less.shape, dtype=np.float64)
    pit = (less + u * equal) / max(float(n_ens), 1.0)
    valid = np.isfinite(pit)
    pit_counts = np.zeros(len(pit_edges) - 1, dtype=np.float64)
    rank_counts = np.zeros(n_ens + 1, dtype=np.float64)
    if not np.any(valid):
        return pit_counts, rank_counts

    pit_counts += np.histogram(pit[valid], bins=pit_edges)[0].astype(np.float64)
    # Randomized rank with tie handling: integer rank in [0, n_ens].
    rank = less + np.floor(u * (equal + 1)).astype(np.int64)
    rank = np.clip(rank, 0, n_ens)
    rank_counts += np.bincount(rank[valid], minlength=n_ens + 1).astype(np.float64)
    return pit_counts, rank_counts


def _nanmax_floor(arr: np.ndarray, floor: float = MIN_EPS) -> float:
    """Safe nanmax with lower floor."""
    if arr.size == 0:
        return floor
    try:
        val = float(np.nanmax(arr))
    except ValueError:
        return floor
    if not np.isfinite(val):
        return floor
    return max(val, floor)


def _median_positive_step(vals: np.ndarray) -> Optional[float]:
    """Median positive spacing between unique coordinates."""
    if vals.size < 2:
        return None
    uniq = np.unique(np.asarray(vals, dtype=np.float64))
    if uniq.size < 2:
        return None
    diffs = np.diff(np.sort(uniq))
    diffs = diffs[diffs > 0.0]
    if diffs.size == 0:
        return None
    return float(np.median(diffs))

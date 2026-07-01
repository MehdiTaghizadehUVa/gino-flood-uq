"""Evaluation helpers for NEON Stage-2 FGNO predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from neuralop.flood.neon import (
    _weighted_mean,
    anova_corrected_epistemic_variance,
    base_rmse_from_reference,
    flatten_nested_predictions,
    nested_member_metadata,
    nested_variance_components,
    per_epistemic_fair_crps,
)


@dataclass
class NEONFlattenedArtifacts:
    """Legacy-compatible flattened ensemble plus nested member metadata."""

    prediction: torch.Tensor
    member_id: torch.Tensor
    member_epistemic_id: torch.Tensor
    member_aleatory_id: torch.Tensor

    def metadata_numpy(self) -> dict[str, np.ndarray]:
        return {
            "member_id": self.member_id.detach().cpu().numpy(),
            "member_epistemic_id": self.member_epistemic_id.detach().cpu().numpy(),
            "member_aleatory_id": self.member_aleatory_id.detach().cpu().numpy(),
        }


def flatten_for_legacy_metrics(prediction: torch.Tensor) -> NEONFlattenedArtifacts:
    """Flatten ``[B, M, K, T, Nv, C]`` predictions for existing metric code."""

    if prediction.ndim != 6:
        raise ValueError(
            "prediction must have shape [B, M, K, T, Nv, C], "
            f"got {tuple(prediction.shape)}."
        )
    _, M, K, _, _, _ = prediction.shape
    metadata = nested_member_metadata(int(M), int(K))
    return NEONFlattenedArtifacts(
        prediction=flatten_nested_predictions(prediction),
        member_id=metadata["member_id"].to(prediction.device),
        member_epistemic_id=metadata["member_epistemic_id"].to(prediction.device),
        member_aleatory_id=metadata["member_aleatory_id"].to(prediction.device),
    )


def neon_variance_summary(prediction: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute nested variance fields for NEON-FGNO diagnostics."""

    components = nested_variance_components(prediction)
    corrected = anova_corrected_epistemic_variance(prediction)
    ratio = corrected / (components.aleatory + corrected).clamp_min(1.0e-12)
    return {
        "variance_aleatory": components.aleatory,
        "variance_epistemic": components.epistemic,
        "variance_epistemic_anova_corrected": corrected,
        "variance_total": components.total,
        "variance_epistemic_fraction_anova_corrected": ratio,
    }


def domain_average_variance_summary(
    prediction: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Average NEON variance fields over mesh/time/channel dimensions."""

    summary = neon_variance_summary(prediction)
    if weights is None:
        return {key: value.mean(dim=(1, 2, 3)) for key, value in summary.items()}
    weights = weights.to(device=prediction.device, dtype=prediction.dtype)
    if weights.ndim == 3:
        weights = weights.unsqueeze(0)
    if weights.ndim != 4:
        raise ValueError(
            "weights must have shape [T, Nv, C] or [B, T, Nv, C], "
            f"got {tuple(weights.shape)}."
        )
    denom = weights.sum(dim=(1, 2, 3)).clamp_min(1.0e-12)
    out = {}
    for key, value in summary.items():
        if weights.shape[0] == 1 and value.shape[0] > 1:
            w = weights.expand(value.shape[0], -1, -1, -1)
            d = denom.expand(value.shape[0])
        else:
            w = weights
            d = denom
        out[key] = (value * w).sum(dim=(1, 2, 3)) / d
    return out


def _as_numpy_array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def save_nested_forecast_artifact(
    path,
    *,
    hydrograph_id: str,
    prediction: torch.Tensor,
    ref_members_wd,
    **artifact_kwargs,
):
    """Write a NEON nested prediction artifact through the common HDF5 schema.

    ``prediction`` must have shape ``[B, M, K, T, Nv, C]``.  The common artifact
    schema is per hydrograph, so this adapter requires ``B=1`` and depth-only
    ``C=1`` and flattens ``M*K`` forecast members while preserving the nested
    epistemic and aleatory member IDs.
    """

    if prediction.ndim != 6:
        raise ValueError(
            "prediction must have shape [B, M, K, T, Nv, C], "
            f"got {tuple(prediction.shape)}."
        )
    if prediction.shape[0] != 1:
        raise ValueError("save_nested_forecast_artifact expects one hydrograph at a time (B=1).")
    if prediction.shape[-1] != 1:
        raise ValueError("save_nested_forecast_artifact currently supports depth-only C=1 artifacts.")

    flattened = flatten_for_legacy_metrics(prediction)
    pred_members_wd = flattened.prediction[0, :, :, :, 0].detach().cpu().numpy()
    ref = _as_numpy_array(ref_members_wd)
    if ref.ndim == 4 and ref.shape[-1] == 1:
        ref = ref[..., 0]
    if ref.ndim != 3:
        raise ValueError(
            "ref_members_wd must have shape [R, T, Nv] or [R, T, Nv, 1], "
            f"got {tuple(ref.shape)}."
        )

    from neuralop.flood.eval.scientific_calibration import save_forecast_artifact

    kwargs = dict(artifact_kwargs)
    kwargs.setdefault("member_epistemic_id", flattened.member_epistemic_id.detach().cpu().numpy().tolist())
    kwargs.setdefault("member_aleatory_id", flattened.member_aleatory_id.detach().cpu().numpy().tolist())
    return save_forecast_artifact(
        path,
        hydrograph_id=hydrograph_id,
        pred_members_wd=pred_members_wd,
        ref_members_wd=ref,
        **kwargs,
    )



# ---------------------------------------------------------------------------
# Integrated nested evaluation metrics (Gap 5)
# ---------------------------------------------------------------------------


def _rowwise_pearson(x: torch.Tensor, y: torch.Tensor, *, eps: float = 1.0e-8) -> torch.Tensor:
    """Pearson correlation per row of ``[B, N]`` tensors, returning ``[B]``."""
    xm = x - x.mean(dim=1, keepdim=True)
    ym = y - y.mean(dim=1, keepdim=True)
    num = (xm * ym).sum(dim=1)
    den = torch.sqrt(xm.pow(2).sum(dim=1) * ym.pow(2).sum(dim=1)).clamp_min(eps)
    return num / den


def neon_predictive_metrics(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    *,
    thresholds=(0.1, 0.3, 0.5),
    weights: torch.Tensor | None = None,
    csi_thresholds=None,
) -> dict[str, float]:
    """Predictive metrics for a nested NEON forecast vs a reference ensemble.

    ``prediction`` is ``[B, M, K, T, Nv, C]`` and ``reference`` is
    ``[B, R, T, Nv, C]``. Returns ensemble-mean RMSE, marginal (flattened
    ``M*K``) fair CRPS, per-threshold exceedance Brier scores, and CSI at
    matching inundation thresholds.
    """
    if prediction.ndim != 6:
        raise ValueError(
            f"prediction must be [B,M,K,T,Nv,C]; got {tuple(prediction.shape)}."
        )
    if reference.ndim != 5:
        raise ValueError(f"reference must be [B,R,T,Nv,C]; got {tuple(reference.shape)}.")

    flat = flatten_nested_predictions(prediction)  # [B, M*K, T, Nv, C]
    out: dict[str, float] = {}
    out["ensemble_mean_rmse"] = base_rmse_from_reference(flat, reference, weights=weights)
    # Marginal (flattened M*K) fair CRPS needs >= 2 members; a degenerate
    # single-member ensemble is reported as NaN rather than crashing.
    if int(flat.shape[1]) >= 2:
        out["marginal_fair_crps"] = float(
            per_epistemic_fair_crps(
                flat.unsqueeze(1), reference, weights=weights, reduction="mean"
            ).item()
        )
    else:
        out["marginal_fair_crps"] = float("nan")

    for thr in thresholds:
        p_pred = (flat > float(thr)).to(flat.dtype).mean(dim=1)         # [B,T,Nv,C]
        p_ref = (reference > float(thr)).to(reference.dtype).mean(dim=1)
        out[f"brier_wd_exceed_{float(thr):g}m"] = float(
            _weighted_mean((p_pred - p_ref).pow(2), weights).item()
        )

    csi_thresholds = thresholds if csi_thresholds is None else csi_thresholds
    pred_mean = flat.mean(dim=1)       # [B,T,Nv,C]
    ref_mean = reference.mean(dim=1)
    cell_mask = None
    if weights is not None:
        cell_mask = weights.to(device=pred_mean.device) > 0
    for thr in csi_thresholds:
        pred_wet = pred_mean > float(thr)
        ref_wet = ref_mean > float(thr)
        if cell_mask is not None:
            m = torch.broadcast_to(cell_mask, pred_wet.shape)
            tp = (pred_wet & ref_wet & m).sum()
            fp = (pred_wet & (~ref_wet) & m).sum()
            fn = ((~pred_wet) & ref_wet & m).sum()
        else:
            tp = (pred_wet & ref_wet).sum()
            fp = (pred_wet & (~ref_wet)).sum()
            fn = ((~pred_wet) & ref_wet).sum()
        denom = (tp + fp + fn).clamp_min(1)
        out[f"csi_{float(thr):g}m"] = float((tp.to(torch.float64) / denom.to(torch.float64)).item())
    return out


def neon_epistemic_error_correlation(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
) -> dict[str, float]:
    """Spatial Pearson correlation between ANOVA epistemic variance and |mean error|.

    A well-behaved epistemic map should be elevated where the ensemble mean is
    wrong. Returned value is the per-batch spatial correlation averaged over the
    batch.
    """
    epi = anova_corrected_epistemic_variance(prediction)               # [B,T,Nv,C]
    flat = flatten_nested_predictions(prediction)
    abs_err = (flat.mean(dim=1) - reference.mean(dim=1)).abs()         # [B,T,Nv,C]
    x = epi.reshape(epi.shape[0], -1)
    y = abs_err.reshape(abs_err.shape[0], -1)
    corr = _rowwise_pearson(x, y)
    return {"epistemic_abs_error_spatial_corr": float(corr.mean().item())}


def evaluate_neon_nested(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    *,
    thresholds=(0.1, 0.3, 0.5),
    weights: torch.Tensor | None = None,
) -> dict[str, float]:
    """Full NEON nested-evaluation bundle: predictive + epistemic diagnostics."""
    out: dict[str, float] = {}
    out.update(neon_predictive_metrics(prediction, reference, thresholds=thresholds, weights=weights))
    variance = domain_average_variance_summary(prediction)
    for key, value in variance.items():
        out[f"{key}_mean"] = float(value.mean().item())
    out.update(neon_epistemic_error_correlation(prediction, reference, weights=weights))
    return out

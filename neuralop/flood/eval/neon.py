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
    if weights is not None:
        weights = weights.to(device=prediction.device, dtype=prediction.dtype)
        if weights.ndim == 3:
            weights = weights.unsqueeze(0)
        if weights.ndim != 4:
            raise ValueError(
                "weights must have shape [T, Nv, C] or [B, T, Nv, C], "
                f"got {tuple(weights.shape)}."
            )
        if weights.shape[0] == 1 and x.shape[0] > 1:
            weights = weights.expand(x.shape[0], -1, -1, -1)
        mask = weights.reshape(x.shape[0], -1) > 0
        corrs = []
        for b in range(x.shape[0]):
            if int(mask[b].sum().item()) < 2:
                corrs.append(torch.zeros((), device=x.device, dtype=x.dtype))
            else:
                corrs.append(_rowwise_pearson(x[b : b + 1, mask[b]], y[b : b + 1, mask[b]])[0])
        corr = torch.stack(corrs)
        return {"epistemic_abs_error_spatial_corr": float(corr.mean().item())}
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
    variance = domain_average_variance_summary(prediction, weights=weights)
    for key, value in variance.items():
        out[f"{key}_mean"] = float(value.mean().item())
    out.update(neon_epistemic_error_correlation(prediction, reference, weights=weights))
    return out


# ---------------------------------------------------------------------------
# Deep-ensemble comparison + variance-map plotting (Gap 7)
# ---------------------------------------------------------------------------


def deep_ensemble_epistemic_variance(model_means: torch.Tensor) -> torch.Tensor:
    """Deep-ensemble epistemic variance: Var over models of per-model ensemble means.

    ``model_means`` is ``[B, J, T, Nv, C]`` where ``J`` indexes deep-ensemble
    members (each already averaged over its ``K`` aleatory samples). Returns
    ``[B, T, Nv, C]``. Used as a reference baseline against the NEON epistemic
    map, not as the formal definition of epistemic uncertainty.
    """
    if model_means.ndim != 5:
        raise ValueError(
            f"model_means must be [B, J, T, Nv, C]; got {tuple(model_means.shape)}."
        )
    if int(model_means.shape[1]) < 2:
        return torch.zeros_like(model_means[:, 0])
    return model_means.var(dim=1, unbiased=True)


def compare_epistemic_maps(
    neon_epistemic: torch.Tensor,
    other_epistemic: torch.Tensor,
    *,
    top_q: float = 0.1,
) -> dict[str, float]:
    """Compare two epistemic-variance maps (e.g. NEON vs deep ensemble).

    Returns spatial Pearson correlation, top-``q`` high-variance region overlap
    fraction, and the domain-average variance ratio.
    """
    x = neon_epistemic.reshape(-1).to(torch.float64)
    y = other_epistemic.reshape(-1).to(torch.float64)
    if x.numel() != y.numel():
        raise ValueError(
            f"maps must have the same number of cells; got {x.numel()} and {y.numel()}."
        )
    xm = x - x.mean()
    ym = y - y.mean()
    denom = torch.sqrt((xm.pow(2).sum() * ym.pow(2).sum())).clamp_min(1.0e-12)
    corr = float((xm * ym).sum() / denom)

    n = int(x.numel())
    k = max(1, int(round(float(top_q) * n)))
    top_x = set(torch.topk(x, k).indices.tolist())
    top_y = set(torch.topk(y, k).indices.tolist())
    overlap = len(top_x & top_y) / float(k)

    ratio = float(x.mean() / y.mean().clamp_min(1.0e-12))
    return {"spatial_corr": corr, "topq_overlap": overlap, "variance_ratio": ratio}


def write_variance_maps(
    prediction: torch.Tensor,
    *,
    geometry_xy,
    output_dir,
    label: str = "neon",
    time_index: int = 0,
):
    """Render aleatory / epistemic / ANOVA-corrected / total variance maps to PNG.

    ``prediction`` is ``[B, M, K, T, Nv, C]``. Renders the ``B=0``, ``C=0`` slice
    at ``time_index`` as UTM scatter maps. Returns the list of written paths.
    """
    from pathlib import Path as _Path

    summary = neon_variance_summary(prediction)
    xy = _as_numpy_array(geometry_xy)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"geometry_xy must be [Nv, 2]; got {tuple(xy.shape)}.")

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    out_dir = _Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        ("aleatory", summary["variance_aleatory"], "viridis"),
        ("epistemic", summary["variance_epistemic"], "magma"),
        ("epistemic_anova_corrected", summary["variance_epistemic_anova_corrected"], "magma"),
        ("total", summary["variance_total"], "cividis"),
    ]
    paths = []
    for name, field, cmap in fields:
        values = _as_numpy_array(field[0, int(time_index), :, 0])
        fig, ax = plt.subplots(figsize=(7.0, 6.0), dpi=140)
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=values, s=4.0, cmap=cmap, linewidths=0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{label} {name} variance | t{int(time_index) + 1}")
        ax.set_xlabel("UTM Easting (m)")
        ax.set_ylabel("UTM Northing (m)")
        fig.colorbar(sc, ax=ax, shrink=0.82)
        fig.tight_layout()
        path = out_dir / f"{label}_variance_{name}_t{int(time_index) + 1:03d}.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Legacy-evaluator integration: PIT/rank, reliability, spread-skill
# ---------------------------------------------------------------------------


def nested_pit_rank_histograms(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    *,
    pit_bins: int = 20,
    seed: int = 0,
    min_ref_depth: float | None = None,
):
    """PIT and rank histograms of the flattened ``M*K`` ensemble vs references.

    Reuses the rollout evaluator's randomized tie-handling counts
    (``_pit_rank_counts_from_reference``), looping over time steps so the
    member-by-reference comparison never materializes the full horizon at once.
    ``min_ref_depth`` restricts to cells where any reference member exceeds the
    depth (PIT over the wettable signal instead of the dry background).
    Depth-only (C=1) predictions are assumed.
    """
    import numpy as np

    def _pit_rank_counts(pred_ens, ref_ens, pit_edges, n_ens, rng):
        # Mirrors eval.metrics._pit_rank_counts_from_reference (randomized
        # tie handling) without importing the heavy eval dependency chain.
        less = np.sum(pred_ens[:, None, :] < ref_ens[None, :, :], axis=0).astype(np.int64)
        equal = np.sum(pred_ens[:, None, :] == ref_ens[None, :, :], axis=0).astype(np.int64)
        u = rng.random(size=less.shape, dtype=np.float64)
        pit = (less + u * equal) / max(float(n_ens), 1.0)
        valid = np.isfinite(pit)
        pc = np.histogram(pit[valid], bins=pit_edges)[0].astype(np.float64)
        rank = np.clip(less + np.floor(u * (equal + 1)).astype(np.int64), 0, n_ens)
        rc = np.bincount(rank[valid], minlength=n_ens + 1).astype(np.float64)
        return pc, rc

    flat = flatten_nested_predictions(prediction)
    B, MK, T, _, _ = flat.shape
    pit_edges = np.linspace(0.0, 1.0, int(pit_bins) + 1)
    rng = np.random.default_rng(int(seed))
    pit_counts = np.zeros(int(pit_bins), dtype=np.float64)
    rank_counts = np.zeros(MK + 1, dtype=np.float64)
    for b in range(B):
        for t in range(T):
            pred = flat[b, :, t, :, 0].detach().cpu().numpy()
            ref = reference[b, :, t, :, 0].detach().cpu().numpy()
            if min_ref_depth is not None:
                wet = ref.max(axis=0) > float(min_ref_depth)
                if not wet.any():
                    continue
                pred = pred[:, wet]
                ref = ref[:, wet]
            pc, rc = _pit_rank_counts(pred, ref, pit_edges, MK, rng)
            pit_counts += pc
            rank_counts += rc
    return {
        "pit_edges": pit_edges.tolist(),
        "pit_counts": pit_counts.tolist(),
        "rank_counts": rank_counts.tolist(),
    }


def exceedance_reliability(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    *,
    thresholds=(0.1, 0.3, 0.5),
    n_bins: int = 10,
):
    """Reliability of exceedance probabilities vs reference frequencies.

    For each threshold, bins the flattened-ensemble forecast exceedance
    probability into ``n_bins`` and reports per-bin counts plus raw sums so
    multiple families can be aggregated exactly before computing the curve.
    """
    flat = flatten_nested_predictions(prediction)
    edges = torch.linspace(0.0, 1.0, int(n_bins) + 1)
    out = {}
    for thr in thresholds:
        p_pred = (flat > float(thr)).to(torch.float64).mean(dim=1).reshape(-1)
        p_ref = (reference > float(thr)).to(torch.float64).mean(dim=1).reshape(-1)
        idx = torch.clamp(torch.bucketize(p_pred, edges[1:-1].to(p_pred.dtype)), 0, n_bins - 1)
        bins = []
        for i in range(int(n_bins)):
            mask = idx == i
            n = int(mask.sum())
            bins.append(
                {
                    "bin_lo": float(edges[i]),
                    "bin_hi": float(edges[i + 1]),
                    "n": n,
                    "sum_forecast_prob": float(p_pred[mask].sum()) if n else 0.0,
                    "sum_observed_freq": float(p_ref[mask].sum()) if n else 0.0,
                    "forecast_prob": float(p_pred[mask].mean()) if n else None,
                    "observed_freq": float(p_ref[mask].mean()) if n else None,
                }
            )
        out[f"{float(thr):g}m"] = bins
    return out


def spread_error_diagnostics(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    *,
    n_bins: int = 10,
):
    """Spread-skill diagnostics of the flattened ``M*K`` ensemble.

    Correlates per-(time, cell) ensemble spread (std over members) with the
    absolute error of the ensemble mean vs the reference-ensemble mean, and
    returns quantile-binned mean spread vs error RMSE (a calibrated ensemble
    tracks the 1:1 line).
    """
    flat = flatten_nested_predictions(prediction)
    spread = flat.std(dim=1, unbiased=True).reshape(-1).to(torch.float64)
    err = (flat.mean(dim=1) - reference.mean(dim=1)).abs().reshape(-1).to(torch.float64)
    xm = spread - spread.mean()
    ym = err - err.mean()
    denom = torch.sqrt(xm.pow(2).sum() * ym.pow(2).sum()).clamp_min(1.0e-12)
    corr = float((xm * ym).sum() / denom)

    qs = torch.quantile(spread, torch.linspace(0.0, 1.0, int(n_bins) + 1, dtype=torch.float64))
    bins = []
    for i in range(int(n_bins)):
        lo, hi = qs[i], qs[i + 1]
        mask = (spread >= lo) & (spread <= hi if i == n_bins - 1 else spread < hi)
        n = int(mask.sum())
        bins.append(
            {
                "mean_spread": float(spread[mask].mean()) if n else None,
                "error_rmse": float(err[mask].pow(2).mean().sqrt()) if n else None,
                "n": n,
            }
        )
    return {"spread_error_corr": corr, "bins": bins}

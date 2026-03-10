"""Spatial and non-spatial rendering helpers for flood evaluation outputs."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.animation as animation
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch

from neuralop.flood.eval.metrics import (
    _build_member_model_indices,
    _median_positive_step,
    _nanmax_floor,
    _pit_rank_counts_from_reference,
    _variance_decomposition_by_model,
)
from neuralop.flood.eval.runtime import (
    ANIMATION_FPS,
    ANIMATION_INTERVAL_MS,
    CBAR_FRAC,
    CBAR_PAD,
    CSI_THRESHOLDS,
    MIN_EPS,
    PUBLICATION_TIMESTEPS,
    UQ_BOXPLOT_PNG,
    UQ_EXCEEDANCE_THRESHOLD,
    UQ_INTERVAL_COVERAGE_PNG,
    UQ_PIT_RANK_PNG,
    UQ_RELIABILITY_PNG,
    UQ_SPREAD_SKILL_PNG,
    UQ_VAR_DECOMP_PNG,
)

def _geometry_xy(geometry):
    """Extract x, y coordinates from geometry tensor or array."""
    arr = geometry.detach().cpu().numpy() if hasattr(geometry, "detach") else np.asarray(geometry)
    return arr[:, 0], arr[:, 1]


def _channel_vmin_vmax_cmap(
    ch: str, gt: np.ndarray, pred: np.ndarray
) -> Tuple[float, float, str]:
    """Return (vmin, vmax, cmap) for a channel (wd vs velocity-style)."""
    if ch == "wd":
        vmax = max(float(np.nanmax(gt)), float(np.nanmax(pred)), MIN_EPS)
        return 0.0, vmax, "viridis"
    vmax = max(
        float(np.nanmax(np.abs(gt))), float(np.nanmax(np.abs(pred))), MIN_EPS
    )
    return -vmax, vmax, "coolwarm"

def _save_generic_rollout_visuals(
    geometry: torch.Tensor,
    pred_by_channel: Dict[str, np.ndarray],
    gt_by_channel: Dict[str, np.ndarray],
    target_variables: List[str],
    out_dir: str,
    run_id: str,
    dt_seconds: float,
) -> None:
    """Write publication maps and a comparison GIF for generic target variables."""
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    x, y = _geometry_xy(geometry)
    rid = run_id or "unknown"
    n_steps = next(iter(gt_by_channel.values())).shape[0]
    steps = [s for s in PUBLICATION_TIMESTEPS if 0 <= s < n_steps] or [
        min(n_steps - 1, 0)
    ]
    n_rows = len(target_variables)

    for t in steps:
        fig, axs = plt.subplots(
            n_rows, 3, figsize=(18, 5 * n_rows), dpi=250, constrained_layout=True
        )
        axs = np.atleast_2d(axs)
        for r, ch in enumerate(target_variables):
            gt_t = gt_by_channel[ch][t]
            pred_t = pred_by_channel[ch][t]
            err_t = np.abs(pred_t - gt_t)
            vmin, vmax, cmap = _channel_vmin_vmax_cmap(ch, gt_t, pred_t)
            emax = max(float(np.nanmax(err_t)), MIN_EPS)
            panels = [
                (f"{ch.upper()} Ground Truth", gt_t, cmap, vmin, vmax, ch),
                (f"{ch.upper()} Prediction", pred_t, cmap, vmin, vmax, ch),
                (f"{ch.upper()} Abs Error", err_t, "magma", 0.0, emax, "error"),
            ]
            for c, (title, arr, pcmap, pvmin, pvmax, cblabel) in enumerate(panels):
                ax = axs[r, c]
                sc = ax.scatter(
                    x, y, c=arr, s=6, marker="s", linewidths=0,
                    cmap=pcmap, vmin=pvmin, vmax=pvmax, rasterized=True
                )
                ax.set_title(title)
                ax.set_aspect("equal")
                ax.axis("off")
                cb = fig.colorbar(sc, ax=ax, fraction=CBAR_FRAC, pad=CBAR_PAD)
                cb.set_label(cblabel)
        fig.savefig(
            os.path.join(fig_dir, f"rollout_{rid}_t{t}.png"),
            bbox_inches="tight", pad_inches=0.1
        )
        plt.close(fig)

    fig, axs = plt.subplots(
        n_rows, 2, figsize=(12, 4 * n_rows), constrained_layout=True
    )
    axs = np.atleast_2d(axs)
    scatters: List[Tuple[str, Any, Any]] = []
    for r, ch in enumerate(target_variables):
        gt0 = gt_by_channel[ch][0]
        pred0 = pred_by_channel[ch][0]
        vmin, vmax, cmap = _channel_vmin_vmax_cmap(
            ch, gt_by_channel[ch], pred_by_channel[ch]
        )
        s_gt = axs[r, 0].scatter(x, y, c=gt0, s=12, cmap=cmap, vmin=vmin, vmax=vmax)
        s_pr = axs[r, 1].scatter(x, y, c=pred0, s=12, cmap=cmap, vmin=vmin, vmax=vmax)
        axs[r, 0].set_title(f"{ch.upper()} Ground Truth")
        axs[r, 1].set_title(f"{ch.upper()} Prediction")
        axs[r, 0].axis("off")
        axs[r, 1].axis("off")
        fig.colorbar(s_gt, ax=axs[r, 0], fraction=CBAR_FRAC, pad=0.03)
        fig.colorbar(s_pr, ax=axs[r, 1], fraction=CBAR_FRAC, pad=0.03)
        scatters.append((ch, s_gt, s_pr))

    def _animate(frame_idx: int) -> List[Any]:
        time_hours = (frame_idx + 1) * dt_seconds / 3600.0
        fig.suptitle(
            f"Rollout Comparison (Run: {rid}) - Time: {time_hours:.2f} hrs",
            fontsize=16,
        )
        artists: List[Any] = []
        for ch, s_gt, s_pr in scatters:
            s_gt.set_array(gt_by_channel[ch][frame_idx])
            s_pr.set_array(pred_by_channel[ch][frame_idx])
            artists.extend([s_gt, s_pr])
        return artists

    ani = animation.FuncAnimation(
        fig, _animate, frames=n_steps, interval=ANIMATION_INTERVAL_MS, blit=False
    )
    ani.save(os.path.join(out_dir, f"rollout_{rid}.gif"), writer="pillow", fps=ANIMATION_FPS)
    plt.close(fig)


def _compute_csi(threshold: float, pred: np.ndarray, gt: np.ndarray) -> float:
    """Critical Success Index at given threshold."""
    event_pred = pred >= threshold
    event_gt = gt >= threshold
    tp = np.sum(event_pred & event_gt)

def _adaptive_marker_size(
    x: np.ndarray,
    y: np.ndarray,
    figsize: Tuple[float, float],
    dpi: int,
    n_rows: int = 1,
    n_cols: int = 1,
    fill_factor: float = 1.20,
) -> float:
    """
    Choose square scatter marker area to visually fill gaps for point maps.

    Uses coordinate spacing + panel geometry, not only number of points.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n_points = int(x.size)
    if n_points <= 1:
        return 12.0

    xr = max(float(np.nanmax(x) - np.nanmin(x)), MIN_EPS)
    yr = max(float(np.nanmax(y) - np.nanmin(y)), MIN_EPS)
    dx = _median_positive_step(x)
    dy = _median_positive_step(y)
    if dx is None or not np.isfinite(dx):
        dx = np.sqrt((xr * yr) / max(n_points, 1))
    if dy is None or not np.isfinite(dy):
        dy = np.sqrt((xr * yr) / max(n_points, 1))

    panel_w_in = 0.82 * float(figsize[0]) / max(int(n_cols), 1)
    panel_h_in = 0.78 * float(figsize[1]) / max(int(n_rows), 1)
    px_per_x = panel_w_in * float(dpi) / xr
    px_per_y = panel_h_in * float(dpi) / yr
    step_x_px = max(dx * px_per_x, 0.0)
    step_y_px = max(dy * px_per_y, 0.0)

    # Use the larger axis spacing to prevent visible striping.
    side_px = fill_factor * max(step_x_px, step_y_px, 1.0)
    side_pt = side_px * 72.0 / float(dpi)
    marker_area_pt2 = side_pt ** 2
    return float(np.clip(marker_area_pt2, 4.0, 180.0))


def _scatter_style(marker_size: float) -> Dict[str, Any]:
    """Consistent style for continuous-looking rasterized point maps."""
    return {
        "s": marker_size,
        "marker": "s",
        "linewidths": 0,
        "edgecolors": "none",
        "antialiaseds": False,
        "rasterized": True,
    }


def _compute_cell_edges(centers: np.ndarray) -> np.ndarray:
    """Compute cell edges from sorted cell centers."""
    c = np.asarray(centers, dtype=np.float64)
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5], dtype=np.float64)
    mids = 0.5 * (c[:-1] + c[1:])
    left = c[0] - (mids[0] - c[0])
    right = c[-1] + (c[-1] - mids[-1])
    return np.concatenate(([left], mids, [right])).astype(np.float64)


def _build_structured_renderer(
    x: np.ndarray, y: np.ndarray
) -> Optional[Dict[str, Any]]:
    """
    Recover a structured 2D grid from point coordinates for seam-free rendering.

    Returns None when geometry cannot be mapped cleanly to a unique rectilinear grid.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n_points = int(x.size)
    if n_points <= 1:
        return None

    ux = np.unique(x)
    uy = np.unique(y)
    nx = int(ux.size)
    ny = int(uy.size)
    if nx < 2 or ny < 2:
        return None
    # Structured renderer is only appropriate when grid occupancy is high.
    coverage = float(n_points) / float(nx * ny)
    if coverage < 0.95:
        return None

    xr = max(float(np.nanmax(x) - np.nanmin(x)), 1.0)
    yr = max(float(np.nanmax(y) - np.nanmin(y)), 1.0)
    atol_x = 1e-10 * xr
    atol_y = 1e-10 * yr

    ix = np.searchsorted(ux, x)
    iy = np.searchsorted(uy, y)
    in_bounds = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    if not np.all(in_bounds):
        return None
    if not np.all(np.isclose(x, ux[ix], rtol=0.0, atol=atol_x)):
        return None
    if not np.all(np.isclose(y, uy[iy], rtol=0.0, atol=atol_y)):
        return None

    linear = iy * nx + ix
    if np.unique(linear).size != n_points:
        # Duplicate points in same cell -> ambiguous grid assignment.
        return None

    flat_to_point = np.full(nx * ny, -1, dtype=np.int64)
    flat_to_point[linear] = np.arange(n_points, dtype=np.int64)
    return {
        "mode": "structured",
        "nx": nx,
        "ny": ny,
        "flat_to_point": flat_to_point,
        "x_edges": _compute_cell_edges(ux),
        "y_edges": _compute_cell_edges(uy),
    }


def _build_triangulation_renderer(x: np.ndarray, y: np.ndarray) -> Optional[Dict[str, Any]]:
    """Build point-only triangulation renderer with long-edge masking."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3:
        return None
    tri = mtri.Triangulation(x, y)
    if tri.triangles.size == 0:
        return None

    tris = tri.triangles
    dx = _median_positive_step(x)
    dy = _median_positive_step(y)
    sx = dx if dx is not None and np.isfinite(dx) and dx > 0.0 else 1.0
    sy = dy if dy is not None and np.isfinite(dy) and dy > 0.0 else 1.0
    xn = x / sx
    yn = y / sy

    x0, y0 = xn[tris[:, 0]], yn[tris[:, 0]]
    x1, y1 = xn[tris[:, 1]], yn[tris[:, 1]]
    x2, y2 = xn[tris[:, 2]], yn[tris[:, 2]]
    l01 = np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
    l12 = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    l20 = np.sqrt((x2 - x0) ** 2 + (y2 - y0) ** 2)
    lmax = np.maximum(l01, np.maximum(l12, l20))
    finite = np.isfinite(lmax) & (lmax > 0.0)
    if np.any(finite):
        med = float(np.median(lmax[finite]))
        cutoff = 2.5 * med
        tri.set_mask(lmax > cutoff)
    return {"mode": "tri", "triangulation": tri}


def _build_spatial_renderer(
    x: np.ndarray,
    y: np.ndarray,
    figsize: Tuple[float, float],
    dpi: int,
    n_rows: int,
    n_cols: int,
) -> Dict[str, Any]:
    """Build renderer config: structured grid first, scatter fallback."""
    structured = _build_structured_renderer(x, y)
    if structured is not None:
        return structured
    tri = _build_triangulation_renderer(x, y)
    if tri is not None:
        return tri
    marker_size = _adaptive_marker_size(
        x, y, figsize=figsize, dpi=dpi, n_rows=n_rows, n_cols=n_cols, fill_factor=1.20
    )
    return {
        "mode": "scatter",
        "marker_size": marker_size,
    }


def _field_to_structured_grid(arr: np.ndarray, renderer: Dict[str, Any]) -> np.ndarray:
    """Map 1D point values to structured 2D grid with NaN for absent cells."""
    flat_to_point = renderer["flat_to_point"]
    flat = np.full(flat_to_point.shape[0], np.nan, dtype=np.float64)
    valid = flat_to_point >= 0
    flat[valid] = np.asarray(arr, dtype=np.float64)[flat_to_point[valid]]
    return flat.reshape(int(renderer["ny"]), int(renderer["nx"]))


def _plot_spatial_field(
    ax: Any,
    x: np.ndarray,
    y: np.ndarray,
    arr: np.ndarray,
    renderer: Dict[str, Any],
    cmap: str,
    vmin: float,
    vmax: float,
    norm: Optional[Any] = None,
) -> Any:
    """Plot one spatial field with best available renderer."""
    kwargs: Dict[str, Any] = {"cmap": cmap}
    if norm is not None:
        kwargs["norm"] = norm
    else:
        kwargs["vmin"] = vmin
        kwargs["vmax"] = vmax
    if renderer["mode"] == "structured":
        grid = np.ma.masked_invalid(_field_to_structured_grid(arr, renderer))
        return ax.pcolormesh(
            renderer["x_edges"],
            renderer["y_edges"],
            grid,
            shading="flat",
            antialiased=False,
            rasterized=True,
            **kwargs,
        )
    if renderer["mode"] == "tri":
        return ax.tripcolor(
            renderer["triangulation"],
            np.asarray(arr, dtype=np.float64),
            shading="gouraud",
            rasterized=True,
            **kwargs,
        )
    return ax.scatter(
        x,
        y,
        c=arr,
        **_scatter_style(float(renderer["marker_size"])),
        **kwargs,
    )


def _update_spatial_artist(artist: Any, arr: np.ndarray, renderer: Dict[str, Any]) -> None:
    """Update an existing spatial artist for animation frame."""
    if renderer["mode"] == "structured":
        grid = np.ma.masked_invalid(_field_to_structured_grid(arr, renderer))
        artist.set_array(grid.ravel())
        return
    if renderer["mode"] == "tri":
        artist.set_array(np.asarray(arr, dtype=np.float64))
        return
    artist.set_array(arr)


def _save_nonspatial_uq_diagnostics(
    out_dir: str,
    time_hours: np.ndarray,
    metrics: Dict[str, np.ndarray],
    reliability_bins: Dict[str, np.ndarray],
    pit_hist_counts: np.ndarray,
    pit_edges: np.ndarray,
    rank_hist_counts: np.ndarray,
    spread_skill_samples: np.ndarray,
    interval_coverage: Dict[float, np.ndarray],
    interval_width: Dict[float, np.ndarray],
    wasserstein_wd: Optional[np.ndarray],
    logger: logging.Logger,
) -> None:
    """Write non-spatial UQ figures and overall metric summary for publication use."""
    os.makedirs(out_dir, exist_ok=True)

    overall: Dict[str, Any] = {}
    for key, arr in metrics.items():
        overall[f"{key}_overall_mean"] = float(np.mean(arr))
        overall[f"{key}_overall_std"] = float(np.std(arr))
        overall[f"{key}_leadtime_mean_last"] = float(np.mean(arr[:, -1]))

    if wasserstein_wd is not None:
        overall["wasserstein_wd_overall_mean"] = float(np.mean(wasserstein_wd))
        overall["wasserstein_wd_overall_std"] = float(np.std(wasserstein_wd))

    if reliability_bins.get("count", np.array([])).sum() > 0:
        count = reliability_bins["count"]
        mean_pred = reliability_bins["mean_pred"]
        mean_obs = reliability_bins["mean_obs"]
        rel_mask = count > 0
        ece = float(
            np.sum(count[rel_mask] * np.abs(mean_pred[rel_mask] - mean_obs[rel_mask]))
            / np.sum(count[rel_mask])
        )
        overall["wd_exceed_reliability_ece"] = ece
        if "brier_overall" in reliability_bins:
            overall["wd_exceed_brier_overall"] = float(reliability_bins["brier_overall"])
        elif "all_pred" in reliability_bins and "all_obs" in reliability_bins:
            overall["wd_exceed_brier_overall"] = float(
                np.mean((reliability_bins["all_pred"] - reliability_bins["all_obs"]) ** 2)
            )

    if pit_hist_counts.sum() > 0:
        pit_pdf = pit_hist_counts / pit_hist_counts.sum()
        uniform = np.full_like(pit_pdf, 1.0 / len(pit_pdf))
        overall["pit_l1_distance"] = float(np.sum(np.abs(pit_pdf - uniform)))

    if rank_hist_counts.sum() > 0:
        rank_pdf = rank_hist_counts / rank_hist_counts.sum()
        uniform_rank = np.full_like(rank_pdf, 1.0 / len(rank_pdf))
        overall["rank_hist_l1_distance"] = float(np.sum(np.abs(rank_pdf - uniform_rank)))

    if spread_skill_samples.size > 0:
        x = spread_skill_samples[:, 0]
        y = spread_skill_samples[:, 1]
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if x.size > 2:
            corr = float(np.corrcoef(x, y)[0, 1])
            slope, intercept = np.polyfit(x, y, deg=1)
            overall["spread_skill_corr"] = corr
            overall["spread_skill_slope"] = float(slope)
            overall["spread_skill_intercept"] = float(intercept)

    json_path = os.path.join(out_dir, UQ_OVERALL_JSON)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, sort_keys=True)
    logger.info("Saved overall UQ metrics to %s", json_path)
    for key in sorted(overall.keys()):
        if key.endswith("_overall_mean") or key in {
            "wd_exceed_reliability_ece",
            "pit_l1_distance",
            "rank_hist_l1_distance",
            "spread_skill_corr",
            "spread_skill_slope",
        }:
            logger.info("Overall UQ metric %s=%.6e", key, overall[key])

    # Reliability diagram + bin counts
    if reliability_bins.get("count", np.array([])).sum() > 0:
        centers = reliability_bins["centers"]
        count = reliability_bins["count"]
        mean_pred = reliability_bins["mean_pred"]
        mean_obs = reliability_bins["mean_obs"]
        mask = count > 0
        fig, axs = plt.subplots(2, 1, figsize=(8.2, 7.0), dpi=280, constrained_layout=True)
        ax = axs[0]
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.0, label="Perfect calibration")
        ax.plot(mean_pred[mask], mean_obs[mask], "o-", color="#1f77b4", linewidth=1.6, markersize=4, label="Model")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Forecast probability")
        ax.set_ylabel("Empirical probability")
        ax.set_title(f"Reliability: P(wd > {UQ_EXCEEDANCE_THRESHOLD:.2f})")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
        axs[1].bar(centers, count, width=(centers[1] - centers[0]) * 0.9, color="#4c78a8")
        axs[1].set_xlabel("Forecast probability bin")
        axs[1].set_ylabel("Sample count")
        axs[1].set_title("Forecast probability histogram")
        axs[1].grid(True, axis="y", alpha=0.3)
        fig.savefig(os.path.join(out_dir, UQ_RELIABILITY_PNG), bbox_inches="tight")
        plt.close(fig)

    # PIT + rank histogram
    if pit_hist_counts.sum() > 0 and rank_hist_counts.sum() > 0:
        pit_centers = 0.5 * (pit_edges[:-1] + pit_edges[1:])
        pit_pdf = pit_hist_counts / max(pit_hist_counts.sum(), 1.0)
        rank_idx = np.arange(rank_hist_counts.size)
        rank_pdf = rank_hist_counts / max(rank_hist_counts.sum(), 1.0)
        fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.6), dpi=280, constrained_layout=True)
        axs[0].bar(pit_centers, pit_pdf, width=(pit_edges[1] - pit_edges[0]) * 0.9, color="#59a14f", alpha=0.9)
        axs[0].axhline(1.0 / len(pit_pdf), linestyle="--", color="gray", linewidth=1.1)
        axs[0].set_title("PIT histogram")
        axs[0].set_xlabel("PIT value")
        axs[0].set_ylabel("Density")
        axs[0].grid(True, axis="y", alpha=0.3)
        axs[1].bar(rank_idx, rank_pdf, color="#f28e2b", alpha=0.9)
        axs[1].axhline(1.0 / len(rank_pdf), linestyle="--", color="gray", linewidth=1.1)
        axs[1].set_title("Rank histogram")
        axs[1].set_xlabel("Rank bin")
        axs[1].set_ylabel("Density")
        axs[1].grid(True, axis="y", alpha=0.3)
        fig.savefig(os.path.join(out_dir, UQ_PIT_RANK_PNG), bbox_inches="tight")
        plt.close(fig)

    # Spread-skill scatter
    if spread_skill_samples.size > 0:
        x = spread_skill_samples[:, 0]
        y = spread_skill_samples[:, 1]
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if x.size > 0:
            vmax = max(_nanmax_floor(np.quantile(x, 0.995)), _nanmax_floor(np.quantile(y, 0.995)))
            fig, ax = plt.subplots(1, 1, figsize=(6.8, 5.6), dpi=280, constrained_layout=True)
            hb = ax.hexbin(x, y, gridsize=48, mincnt=1, bins="log", cmap="viridis")
            cb = fig.colorbar(hb, ax=ax, fraction=0.05, pad=0.03)
            cb.set_label("log10(count)")
            ax.plot([0, vmax], [0, vmax], "--", color="white", linewidth=1.2, label="Ideal y=x")
            if x.size > 2:
                slope, intercept = np.polyfit(x, y, deg=1)
                xx = np.linspace(0.0, vmax, 100)
                ax.plot(xx, slope * xx + intercept, color="#d62728", linewidth=1.4, label="Fit")
                corr = np.corrcoef(x, y)[0, 1]
                ax.text(
                    0.02,
                    0.95,
                    f"corr={corr:.3f}\nslope={slope:.3f}",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
                )
            ax.set_xlim(0.0, vmax)
            ax.set_ylim(0.0, vmax)
            ax.set_xlabel("Forecast spread (std)")
            ax.set_ylabel("Absolute mean error")
            ax.set_title("Spread-skill relationship (WD)")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper left", fontsize=8)
            fig.savefig(os.path.join(out_dir, UQ_SPREAD_SKILL_PNG), bbox_inches="tight")
            plt.close(fig)

    # Interval coverage + sharpness
    if interval_coverage:
        alphas = sorted(interval_coverage.keys())
        fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.4), dpi=280, constrained_layout=True)
        for a in alphas:
            cov = interval_coverage[a]
            width = interval_width[a]
            cov_mean = np.mean(cov, axis=0)
            cov_std = np.std(cov, axis=0)
            width_mean = np.mean(width, axis=0)
            axs[0].plot(time_hours, cov_mean, linewidth=1.3, label=f"{int(a*100)}% interval")
            axs[0].fill_between(time_hours, cov_mean - cov_std, cov_mean + cov_std, alpha=0.12)
            axs[0].axhline(a, linestyle="--", linewidth=0.9, alpha=0.6)
            axs[1].plot(time_hours, width_mean, linewidth=1.3, label=f"{int(a*100)}% interval")
        axs[0].set_title("Empirical coverage vs lead time")
        axs[0].set_xlabel("Lead time (hour)")
        axs[0].set_ylabel("Coverage")
        axs[0].set_ylim(0.0, 1.0)
        axs[0].grid(True, alpha=0.3)
        axs[0].legend(fontsize=8, ncol=2)
        axs[1].set_title("Prediction interval width (sharpness)")
        axs[1].set_xlabel("Lead time (hour)")
        axs[1].set_ylabel("Width")
        axs[1].grid(True, alpha=0.3)
        axs[1].legend(fontsize=8, ncol=2)
        fig.savefig(os.path.join(out_dir, UQ_INTERVAL_COVERAGE_PNG), bbox_inches="tight")
        plt.close(fig)

    # Predictive variance decomposition (epistemic vs stochastic).
    if (
        "within_var_wd" in metrics
        and "between_var_wd" in metrics
        and "between_frac_wd" in metrics
    ):
        within = metrics["within_var_wd"]
        between = metrics["between_var_wd"]
        total = metrics.get("total_var_wd", within + between)
        frac = metrics["between_frac_wd"]
        ratio = metrics.get("between_to_within_wd", None)
        within_m = np.mean(within, axis=0)
        between_m = np.mean(between, axis=0)
        total_m = np.mean(total, axis=0)
        frac_m = np.mean(frac, axis=0)

        fig, axs = plt.subplots(1, 2, figsize=(12.4, 4.8), dpi=280, constrained_layout=True)
        ax0, ax1 = axs
        ax0.plot(time_hours, total_m, linewidth=1.6, color="#111111", label="Total variance")
        ax0.plot(time_hours, within_m, linewidth=1.4, color="#1f77b4", label="Within-model (noise)")
        ax0.plot(time_hours, between_m, linewidth=1.4, color="#d62728", label="Between-model")
        ax0.set_yscale("log")
        ax0.set_xlabel("Lead time (hour)")
        ax0.set_ylabel("Variance (log scale)")
        ax0.set_title("Variance decomposition (WD)")
        ax0.grid(True, alpha=0.3)
        ax0.legend(fontsize=8, loc="upper left")

        ax1.plot(time_hours, frac_m, linewidth=1.6, color="#2ca02c", label="Between / total")
        ax1.axhline(0.5, linestyle="--", color="gray", linewidth=0.9, alpha=0.7)
        if ratio is not None:
            ratio_m = np.mean(ratio, axis=0)
            ax1_t = ax1.twinx()
            ax1_t.plot(time_hours, ratio_m, linewidth=1.2, color="#9467bd", label="Between / within")
            ax1_t.set_ylabel("Variance ratio")
            ax1_t.grid(False)
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax1_t.get_legend_handles_labels()
            ax1_t.legend(lines + lines2, labels + labels2, fontsize=8, loc="upper right")
        else:
            ax1.legend(fontsize=8, loc="upper right")
        ax1.set_ylim(0.0, 1.0)
        ax1.set_xlabel("Lead time (hour)")
        ax1.set_ylabel("Fraction")
        ax1.set_title("Epistemic share and dominance")
        ax1.grid(True, alpha=0.3)
        fig.savefig(os.path.join(out_dir, UQ_VAR_DECOMP_PNG), bbox_inches="tight")
        plt.close(fig)

    # Per-hydrograph metric boxplots (time-averaged)
    small_box_data: List[np.ndarray] = []
    small_box_labels: List[str] = []
    for key in ["rmse_wd", "crps_wd", "gaussian_nll_wd", "brier_wd_exceed", "wasserstein_wd"]:
        if key in metrics:
            small_box_data.append(np.mean(metrics[key], axis=1))
            small_box_labels.append(key)
    ratio_data = (
        np.mean(metrics["spread_ratio_wd"], axis=1)
        if "spread_ratio_wd" in metrics
        else None
    )
    if small_box_data or ratio_data is not None:
        fig, axs = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=280, constrained_layout=True)
        ax_small, ax_ratio = axs
        if small_box_data:
            ax_small.boxplot(
                small_box_data,
                labels=small_box_labels,
                showfliers=False,
                whis=(5, 95),
            )
            ax_small.set_yscale("log")
            ax_small.set_title("Error/score metrics (log scale)")
            ax_small.set_ylabel("Metric value")
            ax_small.grid(True, axis="y", alpha=0.3)
            ax_small.tick_params(axis="x", rotation=18)
        else:
            ax_small.set_visible(False)

        if ratio_data is not None:
            ax_ratio.boxplot(
                [ratio_data],
                labels=["spread_ratio_wd"],
                showfliers=False,
                whis=(5, 95),
            )
            ax_ratio.axhline(1.0, linestyle="--", color="gray", linewidth=1.0, alpha=0.8)
            ax_ratio.set_title("Dispersion ratio")
            ax_ratio.set_ylabel("Predicted spread / GT spread")
            lo = np.quantile(ratio_data, 0.01)
            hi = np.quantile(ratio_data, 0.99)
            margin = 0.10 * max(hi - lo, 0.1)
            ax_ratio.set_ylim(max(0.0, lo - margin), hi + margin)
            ax_ratio.grid(True, axis="y", alpha=0.3)
        else:
            ax_ratio.set_visible(False)

        fig.suptitle("Per-hydrograph time-mean UQ metrics", fontsize=12.5)
        fig.savefig(os.path.join(out_dir, UQ_BOXPLOT_PNG), bbox_inches="tight")
        plt.close(fig)

def _save_hydrograph_uq_figures_and_animation(
    geometry: Any,
    pred_mean_by_channel: Dict[str, np.ndarray],
    pred_std_by_channel: Dict[str, np.ndarray],
    gt_mean_by_channel: Dict[str, np.ndarray],
    gt_std_by_channel: Dict[str, np.ndarray],
    target_variables: List[str],
    out_dir: str,
    hydrograph_id: str,
    dt_seconds: float,
    n_ref_sims: int,
    n_ens: int,
    pred_prob_wd: Optional[np.ndarray] = None,
    gt_prob_wd: Optional[np.ndarray] = None,
    crps_map_wd: Optional[np.ndarray] = None,
) -> None:
    """Generate publication-ready UQ figures and animations per hydrograph."""
    import matplotlib as mpl

    mpl.rc("font", family="serif", size=11)
    x, y = _geometry_xy(geometry)
    renderer_3x2 = _build_spatial_renderer(x, y, figsize=(12.4, 14.5), dpi=320, n_rows=3, n_cols=2)
    renderer_1x3 = _build_spatial_renderer(x, y, figsize=(16.5, 5.2), dpi=320, n_rows=1, n_cols=3)
    renderer_1x1 = _build_spatial_renderer(x, y, figsize=(6.8, 5.8), dpi=320, n_rows=1, n_cols=1)
    renderer_2x3 = _build_spatial_renderer(x, y, figsize=(14.8, 9.6), dpi=260, n_rows=2, n_cols=3)
    hid = hydrograph_id or "unknown"
    uq_dir = os.path.join(out_dir, "uq_figures_per_hydrograph")
    os.makedirs(uq_dir, exist_ok=True)
    n_steps = next(iter(pred_mean_by_channel.values())).shape[0]
    steps = [s for s in PUBLICATION_TIMESTEPS if 0 <= s < n_steps] or [0]

    for t in steps:
        for ch in target_variables:
            pred_mean = pred_mean_by_channel[ch][t]
            pred_std = pred_std_by_channel[ch][t]
            gt_mean = gt_mean_by_channel[ch][t]
            gt_std = gt_std_by_channel[ch][t]
            vmin_m, vmax_m, cmap_mean = _channel_vmin_vmax_cmap(ch, gt_mean, pred_mean)
            spread_max = max(_nanmax_floor(gt_std), _nanmax_floor(pred_std), MIN_EPS)
            bias = pred_mean - gt_mean
            abs_err = np.abs(bias)
            bmax = max(_nanmax_floor(np.abs(bias)), MIN_EPS)
            emax = max(_nanmax_floor(abs_err), MIN_EPS)

            fig, axs = plt.subplots(
                3, 2, figsize=(12.4, 14.5), dpi=320, constrained_layout=True
            )
            fig.suptitle(
                f"Hydrograph {hid} | {ch.upper()} | t={t} | GT ({n_ref_sims} sims) vs Forecast ({n_ens} ens)",
                fontsize=12.5,
            )

            panels = [
                ("GT mean", gt_mean, cmap_mean, vmin_m, vmax_m, ch),
                ("Forecast mean", pred_mean, cmap_mean, vmin_m, vmax_m, ch),
                ("Mean bias (pred - gt)", bias, "coolwarm", -bmax, bmax, "bias"),
                ("Absolute error", abs_err, "magma", 0.0, emax, "abs err"),
                ("GT spread (std)", gt_std, "plasma", 0.0, spread_max, "std"),
                ("Forecast spread (std)", pred_std, "plasma", 0.0, spread_max, "std"),
            ]
            for ax, (title, arr, cmap, vmin, vmax, cblabel) in zip(axs.flatten(), panels):
                sc = _plot_spatial_field(
                    ax=ax,
                    x=x,
                    y=y,
                    arr=arr,
                    renderer=renderer_3x2,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
                ax.set_title(title)
                ax.set_aspect("equal")
                ax.axis("off")
                cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
                cbar.set_label(cblabel)

            fig.savefig(
                os.path.join(uq_dir, f"uq_{ch}_{hid}_t{t}.png"),
                bbox_inches="tight",
                pad_inches=0.1,
            )
            plt.close(fig)

    if pred_prob_wd is not None and gt_prob_wd is not None:
        pred_prob_mean = np.mean(pred_prob_wd, axis=0)
        gt_prob_mean = np.mean(gt_prob_wd, axis=0)
        diff_abs = np.abs(pred_prob_mean - gt_prob_mean)
        fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.2), dpi=320, constrained_layout=True)
        items = [
            (f"GT mean P(wd>{UQ_EXCEEDANCE_THRESHOLD:.2f})", gt_prob_mean, "viridis", 0.0, 1.0),
            (f"Forecast mean P(wd>{UQ_EXCEEDANCE_THRESHOLD:.2f})", pred_prob_mean, "viridis", 0.0, 1.0),
            ("|Probability error|", diff_abs, "magma", 0.0, _nanmax_floor(diff_abs)),
        ]
        for ax, (title, arr, cmap, vmin, vmax) in zip(axs, items):
            sc = _plot_spatial_field(
                ax=ax,
                x=x,
                y=y,
                arr=arr,
                renderer=renderer_1x3,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(title)
            ax.set_aspect("equal")
            ax.axis("off")
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        fig.savefig(
            os.path.join(uq_dir, f"uq_wd_timeavg_prob_{hid}.png"),
            bbox_inches="tight",
            pad_inches=0.1,
        )
        plt.close(fig)

    if crps_map_wd is not None:
        crps_mean = np.mean(crps_map_wd, axis=0)
        vmax = _nanmax_floor(crps_mean)
        fig, ax = plt.subplots(1, 1, figsize=(6.8, 5.8), dpi=320, constrained_layout=True)
        sc = _plot_spatial_field(
            ax=ax,
            x=x,
            y=y,
            arr=crps_mean,
            renderer=renderer_1x1,
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_title("WD CRPS map (time-mean)")
        ax.set_aspect("equal")
        ax.axis("off")
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("CRPS")
        fig.savefig(
            os.path.join(uq_dir, f"uq_wd_timeavg_crps_{hid}.png"),
            bbox_inches="tight",
            pad_inches=0.1,
        )
        plt.close(fig)

    if "wd" in pred_mean_by_channel:
        wd_pred_mean = pred_mean_by_channel["wd"]
        wd_pred_std = pred_std_by_channel["wd"]
        wd_gt_mean = gt_mean_by_channel["wd"]
        wd_gt_std = gt_std_by_channel["wd"]
        wd_abs_err = np.abs(wd_pred_mean - wd_gt_mean)
        wd_ratio = np.clip(wd_pred_std / np.maximum(wd_gt_std, MIN_EPS), 0.0, 5.0)
        vmax = max(_nanmax_floor(wd_pred_mean), _nanmax_floor(wd_gt_mean), MIN_EPS)
        spread_max = max(_nanmax_floor(wd_pred_std), _nanmax_floor(wd_gt_std), MIN_EPS)
        err_max = _nanmax_floor(wd_abs_err)
        ratio_vals = wd_ratio[np.isfinite(wd_ratio)]
        if ratio_vals.size > 0:
            q_low = float(np.quantile(ratio_vals, 0.02))
            q_high = float(np.quantile(ratio_vals, 0.98))
            margin = max(1.0 - q_low, q_high - 1.0, 0.15)
            ratio_vmin = max(0.4, 1.0 - margin)
            ratio_vmax = min(2.5, 1.0 + margin)
        else:
            ratio_vmin, ratio_vmax = 0.5, 1.5
        ratio_norm = mcolors.TwoSlopeNorm(vmin=ratio_vmin, vcenter=1.0, vmax=ratio_vmax)

        fig, axs = plt.subplots(2, 3, figsize=(14.8, 9.6), dpi=260, constrained_layout=True)
        ax_gt_m, ax_pr_m, ax_err, ax_gt_s, ax_pr_s, ax_ratio = axs.flatten()
        s_gt_m = _plot_spatial_field(
            ax=ax_gt_m,
            x=x,
            y=y,
            arr=wd_gt_mean[0],
            renderer=renderer_2x3,
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
        )
        s_pr_m = _plot_spatial_field(
            ax=ax_pr_m,
            x=x,
            y=y,
            arr=wd_pred_mean[0],
            renderer=renderer_2x3,
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
        )
        s_err = _plot_spatial_field(
            ax=ax_err,
            x=x,
            y=y,
            arr=wd_abs_err[0],
            renderer=renderer_2x3,
            cmap="magma",
            vmin=0.0,
            vmax=err_max,
        )
        s_gt_s = _plot_spatial_field(
            ax=ax_gt_s,
            x=x,
            y=y,
            arr=wd_gt_std[0],
            renderer=renderer_2x3,
            cmap="plasma",
            vmin=0.0,
            vmax=spread_max,
        )
        s_pr_s = _plot_spatial_field(
            ax=ax_pr_s,
            x=x,
            y=y,
            arr=wd_pred_std[0],
            renderer=renderer_2x3,
            cmap="plasma",
            vmin=0.0,
            vmax=spread_max,
        )
        s_ratio = _plot_spatial_field(
            ax=ax_ratio,
            x=x,
            y=y,
            arr=wd_ratio[0],
            renderer=renderer_2x3,
            cmap="RdBu_r",
            vmin=ratio_vmin,
            vmax=ratio_vmax,
            norm=ratio_norm,
        )
        for ax, title in [
            (ax_gt_m, f"GT mean ({n_ref_sims} sims)"),
            (ax_pr_m, f"Forecast mean ({n_ens} ens)"),
            (ax_err, "Absolute error |mean|"),
            (ax_gt_s, "GT spread (std)"),
            (ax_pr_s, "Forecast spread (std)"),
            (ax_ratio, "Spread ratio (pred/gt)"),
        ]:
            ax.set_title(title)
            ax.set_aspect("equal")
            ax.axis("off")
        fig.colorbar(s_gt_m, ax=ax_gt_m, fraction=0.046, pad=0.02)
        fig.colorbar(s_pr_m, ax=ax_pr_m, fraction=0.046, pad=0.02)
        fig.colorbar(s_err, ax=ax_err, fraction=0.046, pad=0.02)
        fig.colorbar(s_gt_s, ax=ax_gt_s, fraction=0.046, pad=0.02)
        fig.colorbar(s_pr_s, ax=ax_pr_s, fraction=0.046, pad=0.02)
        cb_ratio = fig.colorbar(s_ratio, ax=ax_ratio, fraction=0.046, pad=0.02)
        cb_ratio.set_label("ratio (center=1)")

        def _animate(frame_idx: int) -> List[Any]:
            time_hours = (frame_idx + 1) * dt_seconds / 3600.0
            fig.suptitle(
                f"Hydrograph {hid} | t={frame_idx} ({time_hours:.2f} h)",
                fontsize=13,
            )
            _update_spatial_artist(s_gt_m, wd_gt_mean[frame_idx], renderer_2x3)
            _update_spatial_artist(s_pr_m, wd_pred_mean[frame_idx], renderer_2x3)
            _update_spatial_artist(s_err, wd_abs_err[frame_idx], renderer_2x3)
            _update_spatial_artist(s_gt_s, wd_gt_std[frame_idx], renderer_2x3)
            _update_spatial_artist(s_pr_s, wd_pred_std[frame_idx], renderer_2x3)
            _update_spatial_artist(s_ratio, wd_ratio[frame_idx], renderer_2x3)
            return [s_gt_m, s_pr_m, s_err, s_gt_s, s_pr_s, s_ratio]

        ani = animation.FuncAnimation(
            fig, _animate, frames=n_steps, interval=ANIMATION_INTERVAL_MS, blit=False
        )
        ani.save(
            os.path.join(uq_dir, f"uq_rollout_{hid}.gif"),
            writer="pillow",
            fps=ANIMATION_FPS,
        )
        plt.close(fig)

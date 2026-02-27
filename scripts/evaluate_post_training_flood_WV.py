#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone post-training evaluation for flood GINO WV pipeline.

Reproduces evaluation logic from train_gino_flood_train_rollout_animation_WV.py
without retraining:
  1) One-step test evaluation on a saved checkpoint
  2) Optional rollout evaluation and plots

Example:
  python scripts/evaluate_post_training_flood_WV.py \\
    --config_path config/gino_pluvial_flood_config_WV_depth_only.yaml \\
    --checkpoint.save_dir /path/to/checkpoints
"""

import argparse
import copy
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train_gino_flood_train_rollout_animation_WV import (  # noqa: E402
    load_config_and_setup,
    setup_logging,
    set_seed,
    write_train_txt_from_data_root,
    make_split_generator,
    FloodDatasetHDF,
    fit_normalizers_streaming,
    NormalizedDatasetOnTheFly,
    FloodGINODataProcessor,
    FGNTrainer,
    Trainer,
    FloodRolloutTestDatasetHDF,
    collect_all_fields,
    NormalizedRolloutTestDataset,
    create_rollout_animation,
    generate_publication_maps,
    parse_target_variables,
)
from neuralop import get_model  # noqa: E402
from neuralop.models.base_model import BaseModel  # noqa: E402
from neuralop.losses.data_losses import LpLoss  # noqa: E402
from neuralop.losses.probabilistic_losses import CRPSLoss  # noqa: E402
from neuralop.data.transforms.normalizers import load_normalizers, save_normalizers  # noqa: E402
from neuralop.training.training_state import load_training_state  # noqa: E402

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
CHECKPOINT_BEST = "best_model"
CHECKPOINT_LAST = "model"
CHECKPOINT_FILES = ("best_model_state_dict.pt", "model_state_dict.pt")
DEFAULT_STATIC_FILES = ["M40_CS.txt", "M40_CU.txt", "M40_FA.txt"]
PUBLICATION_TIMESTEPS = [12, 24, 36, 48, 60, 72]
TRAIN_FRAC = 0.9
MIN_EPS = 1e-9
CBAR_FRAC = 0.046
CBAR_PAD = 0.02
ANIMATION_FPS = 5
ANIMATION_INTERVAL_MS = 200
LEGACY_3CH = ("wd", "vx", "vy")
CSI_THRESHOLDS = (0.05, 0.3)
NORMALIZER_KEYS = ("geometry", "static", "boundary", "dynamic", "target")
DEVICE_REF_KEYS = ("dynamic", "target", "geometry")
CHANNEL_INDEX = {"wd": 0, "vx": 1, "vy": 2}
ROLLOUT_METRICS_NPZ = "rollout_metrics_data.npz"
ROLLOUT_SUMMARY_PNG = "rollout_metrics_summary.png"
ROLLOUT_METRICS_HYDRO_NPZ = "rollout_metrics_per_hydrograph.npz"
ROLLOUT_SUMMARY_HYDRO_PNG = "rollout_metrics_per_hydrograph.png"
UQ_OVERALL_JSON = "uq_overall_metrics.json"
UQ_RELIABILITY_PNG = "uq_reliability_wd_exceedance.png"
UQ_PIT_RANK_PNG = "uq_pit_rank_histograms.png"
UQ_SPREAD_SKILL_PNG = "uq_spread_skill_scatter.png"
UQ_INTERVAL_COVERAGE_PNG = "uq_interval_coverage.png"
UQ_BOXPLOT_PNG = "uq_metric_boxplots.png"
DEFAULT_EVAL_LOG = "eval_post_training.log"
HYDROGRAPH_SIM_PATTERN = re.compile(r"^(.+)_sim(\d+)$")
UQ_EXCEEDANCE_THRESHOLD = 0.05


def _opt(config: Any, section: Optional[str], key: str, default: Any) -> Any:
    """Get config.section.key with default. Use section=None for top-level config keys."""
    def _safe_get(obj: Any, name: str, dflt: Any) -> Any:
        try:
            return getattr(obj, name)
        except (AttributeError, KeyError, TypeError):
            pass
        if isinstance(obj, dict):
            return obj.get(name, dflt)
        try:
            return obj[name]
        except Exception:
            return dflt

    if section is None:
        return _safe_get(config, key, default)
    obj = _safe_get(config, section, None)
    if obj is None:
        return default
    return _safe_get(obj, key, default)


def parse_hydrograph_run_id(run_id: str) -> Tuple[str, Optional[int]]:
    """Parse run_id convention: {hydrograph_id}_sim{sim_id}."""
    rid = str(run_id).strip()
    m = HYDROGRAPH_SIM_PATTERN.match(rid)
    if m:
        return m.group(1), int(m.group(2))
    return rid, None


def group_run_ids_by_hydrograph(run_ids: List[str]) -> Dict[str, List[str]]:
    """Group run IDs by hydrograph ID inferred from *_simN naming."""
    groups: Dict[str, List[str]] = {}
    for rid in run_ids:
        hydro_id, _ = parse_hydrograph_run_id(rid)
        groups.setdefault(hydro_id, []).append(rid)
    for hydro_id in groups:
        groups[hydro_id] = sorted(groups[hydro_id])
    return groups


class _PhaseTimer:
    """Context manager that logs phase name and duration."""

    def __init__(self, logger: logging.Logger, phase_name: str) -> None:
        self.logger = logger
        self.phase_name = phase_name
        self._t0 = 0.0

    def __enter__(self) -> "_PhaseTimer":
        self._t0 = time.perf_counter()
        self.logger.info(">>> %s", self.phase_name)
        return self

    def __exit__(self, exc_type: type, exc: BaseException, _tb: Any) -> bool:
        dt = time.perf_counter() - self._t0
        if exc is None:
            self.logger.info("<<< %s completed in %.2fs", self.phase_name, dt)
        else:
            self.logger.exception("<<< %s failed after %.2fs: %s", self.phase_name, dt, exc)
        return False


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


def _build_eval_losses(config: Any, use_fgn: bool) -> Dict[str, Any]:
    """Build loss dict for one-step evaluation (L2 and optionally CRPS)."""
    l2_loss = LpLoss(d=2, p=2)
    if use_fgn and _opt(config, "opt", "training_loss", "l2") == "crps":
        n_samples = max(2, int(_opt(config, "opt", "crps_n_samples", 2)))
        ch_weights = _opt(config, "opt", "crps_channel_weights", None)
        return {
            "l2": l2_loss,
            "crps": CRPSLoss(
                n_samples=n_samples, channel_weights=ch_weights, reduction="mean"
            ),
        }
    test_loss_name = _opt(config, "opt", "testing_loss", "l2")
    return {test_loss_name: l2_loss}


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
    CRPS per location for forecast ensemble against reference ensemble.

    Parameters
    ----------
    forecast_ens: [n_forecast, n_locations]
    reference_ens: [n_reference, n_locations]
    """
    term_1 = np.mean(
        np.abs(forecast_ens[:, None, :] - reference_ens[None, :, :]), axis=(0, 1)
    )
    term_2 = 0.5 * np.mean(
        np.abs(forecast_ens[:, None, :] - forecast_ens[None, :, :]), axis=(0, 1)
    )
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
            vmax = max(_nanmax_floor(x), _nanmax_floor(y))
            fig, ax = plt.subplots(1, 1, figsize=(6.8, 5.6), dpi=280, constrained_layout=True)
            hb = ax.hexbin(x, y, gridsize=45, mincnt=1, cmap="viridis")
            cb = fig.colorbar(hb, ax=ax, fraction=0.05, pad=0.03)
            cb.set_label("Count")
            ax.plot([0, vmax], [0, vmax], "--", color="white", linewidth=1.2, label="Ideal y=x")
            if x.size > 2:
                slope, intercept = np.polyfit(x, y, deg=1)
                xx = np.linspace(0.0, vmax, 100)
                ax.plot(xx, slope * xx + intercept, color="#d62728", linewidth=1.4, label="Fit")
                corr = np.corrcoef(x, y)[0, 1]
                ax.text(
                    0.02,
                    0.98,
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

    # Per-hydrograph metric boxplots (time-averaged)
    box_data: List[np.ndarray] = []
    box_labels: List[str] = []
    for key in ["rmse_wd", "crps_wd", "brier_wd_exceed", "spread_ratio_wd"]:
        if key in metrics:
            box_data.append(np.mean(metrics[key], axis=1))
            box_labels.append(key)
    if wasserstein_wd is not None:
        box_data.append(np.mean(wasserstein_wd, axis=1))
        box_labels.append("wasserstein_wd")
    if box_data:
        fig, ax = plt.subplots(1, 1, figsize=(8.2, 4.8), dpi=280, constrained_layout=True)
        ax.boxplot(box_data, labels=box_labels, showfliers=False)
        ax.set_title("Per-hydrograph time-mean UQ metrics")
        ax.set_ylabel("Metric value")
        ax.grid(True, axis="y", alpha=0.3)
        fig.autofmt_xdate(rotation=20, ha="right")
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
            marker_size = _adaptive_marker_size(
                x, y, figsize=(12.4, 14.5), dpi=320, n_rows=3, n_cols=2, fill_factor=1.20
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
                sc = ax.scatter(
                    x,
                    y,
                    c=arr,
                    **_scatter_style(marker_size),
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
        marker_size = _adaptive_marker_size(
            x, y, figsize=(16.5, 5.2), dpi=320, n_rows=1, n_cols=3, fill_factor=1.20
        )
        items = [
            (f"GT mean P(wd>{UQ_EXCEEDANCE_THRESHOLD:.2f})", gt_prob_mean, "viridis", 0.0, 1.0),
            (f"Forecast mean P(wd>{UQ_EXCEEDANCE_THRESHOLD:.2f})", pred_prob_mean, "viridis", 0.0, 1.0),
            ("|Probability error|", diff_abs, "magma", 0.0, _nanmax_floor(diff_abs)),
        ]
        for ax, (title, arr, cmap, vmin, vmax) in zip(axs, items):
            sc = ax.scatter(
                x, y, c=arr, **_scatter_style(marker_size), cmap=cmap, vmin=vmin, vmax=vmax
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
        marker_size = _adaptive_marker_size(
            x, y, figsize=(6.8, 5.8), dpi=320, n_rows=1, n_cols=1, fill_factor=1.20
        )
        sc = ax.scatter(
            x, y, c=crps_mean, **_scatter_style(marker_size), cmap="magma", vmin=0.0, vmax=vmax
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
        wd_ratio = np.clip(wd_pred_std / np.maximum(wd_gt_std, MIN_EPS), 0.0, 3.0)
        vmax = max(_nanmax_floor(wd_pred_mean), _nanmax_floor(wd_gt_mean), MIN_EPS)
        spread_max = max(_nanmax_floor(wd_pred_std), _nanmax_floor(wd_gt_std), MIN_EPS)
        err_max = _nanmax_floor(wd_abs_err)
        ratio_max = max(_nanmax_floor(wd_ratio), 1.5)

        fig, axs = plt.subplots(2, 3, figsize=(14.8, 9.6), dpi=260, constrained_layout=True)
        marker_size = _adaptive_marker_size(
            x, y, figsize=(14.8, 9.6), dpi=260, n_rows=2, n_cols=3, fill_factor=1.20
        )
        ax_gt_m, ax_pr_m, ax_err, ax_gt_s, ax_pr_s, ax_ratio = axs.flatten()
        s_gt_m = ax_gt_m.scatter(
            x, y, c=wd_gt_mean[0], **_scatter_style(marker_size), cmap="viridis", vmin=0.0, vmax=vmax
        )
        s_pr_m = ax_pr_m.scatter(
            x, y, c=wd_pred_mean[0], **_scatter_style(marker_size), cmap="viridis", vmin=0.0, vmax=vmax
        )
        s_err = ax_err.scatter(
            x, y, c=wd_abs_err[0], **_scatter_style(marker_size), cmap="magma", vmin=0.0, vmax=err_max
        )
        s_gt_s = ax_gt_s.scatter(
            x, y, c=wd_gt_std[0], **_scatter_style(marker_size), cmap="plasma", vmin=0.0, vmax=spread_max
        )
        s_pr_s = ax_pr_s.scatter(
            x, y, c=wd_pred_std[0], **_scatter_style(marker_size), cmap="plasma", vmin=0.0, vmax=spread_max
        )
        s_ratio = ax_ratio.scatter(
            x, y, c=wd_ratio[0], **_scatter_style(marker_size), cmap="RdYlGn_r", vmin=0.5, vmax=ratio_max
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
        fig.colorbar(s_ratio, ax=ax_ratio, fraction=0.046, pad=0.02)

        def _animate(frame_idx: int) -> List[Any]:
            time_hours = (frame_idx + 1) * dt_seconds / 3600.0
            fig.suptitle(
                f"Hydrograph {hid} | t={frame_idx} ({time_hours:.2f} h)",
                fontsize=13,
            )
            s_gt_m.set_array(wd_gt_mean[frame_idx])
            s_pr_m.set_array(wd_pred_mean[frame_idx])
            s_err.set_array(wd_abs_err[frame_idx])
            s_gt_s.set_array(wd_gt_std[frame_idx])
            s_pr_s.set_array(wd_pred_std[frame_idx])
            s_ratio.set_array(wd_ratio[frame_idx])
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


def _rollout_prediction_per_hydrograph(
    models: List[Any],
    hydrograph_samples: List[Dict[str, Any]],
    rollout_length: int,
    history_steps: int,
    dynamic_norm: Any,
    target_norm: Any,
    device: torch.device,
    skip_before_timestep: int,
    dt: float,
    out_dir: str,
    target_variables: List[str],
    logger: logging.Logger,
    fgn_noise_dim: Optional[int],
    n_ensemble_samples: int,
) -> None:
    """
    Evaluate per hydrograph using all reference simulations as ground-truth uncertainty.

    Ground-truth uncertainty: variability across reference simulations (Manning's n).
    Prediction uncertainty: variability across stochastic ensemble rollouts.
    """
    if not models:
        raise ValueError("No models provided for rollout evaluation.")
    for model in models:
        model.eval()
    dynamic_norm.to(device)
    target_norm.to(device)
    os.makedirs(out_dir, exist_ok=True)

    n_models = len(models)
    n_ens = max(1, int(n_ensemble_samples))
    if n_models > 1 and n_ens < n_models:
        logger.warning(
            "n_ensemble_samples=%d is smaller than number of models=%d. "
            "Raising n_ensemble_samples to %d for paper-style equal model usage.",
            n_ens, n_models, n_models,
        )
        n_ens = n_models
    use_ensemble = n_ens > 1 or n_models > 1
    member_model_indices = _build_member_model_indices(n_models, n_ens)
    model_counts = [member_model_indices.count(i) for i in range(n_models)]
    logger.info(
        "Hydrograph rollout ensemble members=%d across models=%d with per-model counts=%s",
        n_ens, n_models, model_counts,
    )
    if not use_ensemble:
        logger.warning(
            "Per-hydrograph UQ run without ensemble spread (single model, single member)."
        )

    start_pred_t = skip_before_timestep + history_steps
    end_pred_t = start_pred_t + rollout_length

    per_channel_rmse: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
    per_channel_crps: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
    per_channel_spread_pred: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
    per_channel_spread_gt: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
    per_channel_spread_ratio: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
    wd_prob_brier: List[np.ndarray] = []
    wd_prob_mae: List[np.ndarray] = []
    wd_wasserstein: List[np.ndarray] = []

    interval_levels = (0.50, 0.80, 0.90, 0.95)
    wd_interval_coverage: Dict[float, List[np.ndarray]] = {a: [] for a in interval_levels}
    wd_interval_width: Dict[float, List[np.ndarray]] = {a: [] for a in interval_levels}

    rel_n_bins = 12
    rel_edges = np.linspace(0.0, 1.0, rel_n_bins + 1, dtype=np.float64)
    rel_count = np.zeros(rel_n_bins, dtype=np.float64)
    rel_sum_pred = np.zeros(rel_n_bins, dtype=np.float64)
    rel_sum_obs = np.zeros(rel_n_bins, dtype=np.float64)
    rel_brier_sum = 0.0
    rel_brier_count = 0

    pit_edges = np.linspace(0.0, 1.0, 21, dtype=np.float64)
    pit_hist_counts = np.zeros(len(pit_edges) - 1, dtype=np.float64)
    rank_hist_counts = np.zeros(n_ens + 1, dtype=np.float64)

    spread_skill_samples: List[np.ndarray] = []
    max_scatter_points_per_step = 250
    w_quantiles = np.linspace(0.0, 1.0, 21, dtype=np.float64)

    for sample in tqdm(hydrograph_samples, desc="Hydrograph rollout evaluation"):
        hydro_id = sample["hydrograph_id"]
        geometry = sample["geometry"]
        static_0 = sample["static"].to(device).unsqueeze(0)
        geom_0 = geometry.to(device).unsqueeze(0)
        query_0 = sample["query_points"].to(device).unsqueeze(0)
        full_boundary = sample["boundary"].to(device)
        dynamic_ref = sample["dynamic_ref"].to(device)
        n_ref = int(sample["n_ref_sims"])

        gt_rollout_ref = dynamic_ref[:, start_pred_t:end_pred_t]
        gt_boundary_rollout = full_boundary[start_pred_t:end_pred_t]

        run_rmse: Dict[str, List[float]] = {name: [] for name in target_variables}
        run_crps: Dict[str, List[float]] = {name: [] for name in target_variables}
        run_spread_pred: Dict[str, List[float]] = {name: [] for name in target_variables}
        run_spread_gt: Dict[str, List[float]] = {name: [] for name in target_variables}
        run_spread_ratio: Dict[str, List[float]] = {name: [] for name in target_variables}
        run_pred_mean_by_channel: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
        run_pred_std_by_channel: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
        run_gt_mean_by_channel: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
        run_gt_std_by_channel: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
        run_wd_pred_prob: List[np.ndarray] = []
        run_wd_gt_prob: List[np.ndarray] = []
        run_wd_crps_map: List[np.ndarray] = []
        run_wd_brier: List[float] = []
        run_wd_mae: List[float] = []
        run_wd_wasserstein: List[float] = []
        run_interval_coverage: Dict[float, List[float]] = {a: [] for a in interval_levels}
        run_interval_width: Dict[float, List[float]] = {a: [] for a in interval_levels}

        # Do not condition on a specific hidden Manning's-n realization:
        # initialize from the mean history across reference simulations.
        init_history = dynamic_ref[:, skip_before_timestep:start_pred_t].mean(dim=0)
        if use_ensemble:
            current_dynamics = [init_history.clone() for _ in range(n_ens)]
        else:
            current_dynamic = init_history.clone()
        current_boundary = full_boundary[skip_before_timestep:start_pred_t].clone()

        for t in range(rollout_length):
            with torch.no_grad():
                if use_ensemble:
                    pred_members: List[torch.Tensor] = []
                    for ens_idx in range(n_ens):
                        dyn_hist = current_dynamics[ens_idx]
                        model_idx = member_model_indices[ens_idx]
                        model = models[model_idx]
                        dyn_flat = dyn_hist.permute(1, 0, 2).reshape(1, dyn_hist.shape[1], -1)
                        bc_flat = current_boundary.permute(1, 0, 2).reshape(1, current_boundary.shape[1], -1)
                        x = torch.cat([static_0, bc_flat, dyn_flat], dim=2)
                        if fgn_noise_dim is not None:
                            z = torch.randn(fgn_noise_dim, device=device, dtype=x.dtype)
                            pred_members.append(
                                model(
                                    input_geom=geom_0,
                                    latent_queries=query_0,
                                    output_queries=geom_0,
                                    x=x,
                                    ada_in=z,
                                )
                            )
                        else:
                            pred_members.append(
                                model(
                                    input_geom=geom_0,
                                    latent_queries=query_0,
                                    output_queries=geom_0,
                                    x=x,
                                )
                            )
                    pred_stack = torch.stack(pred_members, dim=0)  # [n_ens, 1, n_cells, n_target]
                else:
                    model = models[0]
                    dyn_flat = current_dynamic.permute(1, 0, 2).reshape(
                        1, current_dynamic.shape[1], -1
                    )
                    bc_flat = current_boundary.permute(1, 0, 2).reshape(
                        1, current_boundary.shape[1], -1
                    )
                    x = torch.cat([static_0, bc_flat, dyn_flat], dim=2)
                    if fgn_noise_dim is not None:
                        z = torch.randn(fgn_noise_dim, device=device, dtype=x.dtype)
                        pred = model(
                            input_geom=geom_0,
                            latent_queries=query_0,
                            output_queries=geom_0,
                            x=x,
                            ada_in=z,
                        )
                    else:
                        pred = model(
                            input_geom=geom_0,
                            latent_queries=query_0,
                            output_queries=geom_0,
                            x=x,
                        )
                    pred_stack = pred.unsqueeze(0)

            inv_pred_ens = target_norm.inverse_transform(pred_stack.squeeze(1))
            inv_gt_ref = dynamic_norm.inverse_transform(gt_rollout_ref[:, t])

            pred_mean_field = inv_pred_ens.mean(dim=0)
            pred_std_field = inv_pred_ens.std(dim=0, unbiased=False)
            gt_mean_field = inv_gt_ref.mean(dim=0)
            gt_std_field = inv_gt_ref.std(dim=0, unbiased=False)

            for ch_idx, ch_name in enumerate(target_variables):
                pred_ens_ch = inv_pred_ens[:, :, ch_idx].detach().cpu().numpy()
                gt_ref_ch = inv_gt_ref[:, :, ch_idx].detach().cpu().numpy()
                pred_mean_ch = pred_mean_field[:, ch_idx].detach().cpu().numpy()
                pred_std_ch = pred_std_field[:, ch_idx].detach().cpu().numpy()
                gt_mean_ch = gt_mean_field[:, ch_idx].detach().cpu().numpy()
                gt_std_ch = gt_std_field[:, ch_idx].detach().cpu().numpy()

                run_pred_mean_by_channel[ch_name].append(pred_mean_ch)
                run_pred_std_by_channel[ch_name].append(pred_std_ch)
                run_gt_mean_by_channel[ch_name].append(gt_mean_ch)
                run_gt_std_by_channel[ch_name].append(gt_std_ch)

                rmse_t = float(np.sqrt(np.mean((pred_mean_ch - gt_mean_ch) ** 2)))
                crps_map = _crps_ensemble_vs_reference(pred_ens_ch, gt_ref_ch)
                crps_t = float(np.mean(crps_map))
                spread_pred_t = float(np.mean(np.std(pred_ens_ch, axis=0)))
                spread_gt_t = float(np.mean(np.std(gt_ref_ch, axis=0)))
                spread_ratio_t = spread_pred_t / max(spread_gt_t, MIN_EPS)

                run_rmse[ch_name].append(rmse_t)
                run_crps[ch_name].append(crps_t)
                run_spread_pred[ch_name].append(spread_pred_t)
                run_spread_gt[ch_name].append(spread_gt_t)
                run_spread_ratio[ch_name].append(spread_ratio_t)

                if ch_name == "wd":
                    pred_prob = np.mean(pred_ens_ch >= UQ_EXCEEDANCE_THRESHOLD, axis=0)
                    gt_prob = np.mean(gt_ref_ch >= UQ_EXCEEDANCE_THRESHOLD, axis=0)
                    run_wd_pred_prob.append(pred_prob)
                    run_wd_gt_prob.append(gt_prob)
                    run_wd_crps_map.append(crps_map)
                    run_wd_brier.append(float(np.mean((pred_prob - gt_prob) ** 2)))
                    run_wd_mae.append(float(np.mean(np.abs(pred_prob - gt_prob))))

                    # Reliability bins for event probability calibration.
                    bins = np.clip(
                        np.digitize(pred_prob, rel_edges, right=False) - 1, 0, rel_n_bins - 1
                    )
                    for b in range(rel_n_bins):
                        mask_b = bins == b
                        if not np.any(mask_b):
                            continue
                        rel_count[b] += float(np.sum(mask_b))
                        rel_sum_pred[b] += float(np.sum(pred_prob[mask_b]))
                        rel_sum_obs[b] += float(np.sum(gt_prob[mask_b]))
                    rel_brier_sum += float(np.sum((pred_prob - gt_prob) ** 2))
                    rel_brier_count += int(pred_prob.size)

                    # Interval coverage + sharpness.
                    for alpha in interval_levels:
                        q_lo = 0.5 * (1.0 - alpha)
                        q_hi = 1.0 - q_lo
                        lo = np.quantile(pred_ens_ch, q_lo, axis=0)
                        hi = np.quantile(pred_ens_ch, q_hi, axis=0)
                        cover = np.mean((gt_ref_ch >= lo[None, :]) & (gt_ref_ch <= hi[None, :]))
                        width = np.mean(hi - lo)
                        run_interval_coverage[alpha].append(float(cover))
                        run_interval_width[alpha].append(float(width))

                    # Distribution distance (approximate Wasserstein-1 via quantiles).
                    pred_q = np.quantile(pred_ens_ch, w_quantiles, axis=0)
                    gt_q = np.quantile(gt_ref_ch, w_quantiles, axis=0)
                    run_wd_wasserstein.append(float(np.mean(np.abs(pred_q - gt_q))))

                    # PIT and rank histogram against reference ensemble mean (single pseudo-observation).
                    ref_vals = gt_mean_ch
                    less = np.sum(pred_ens_ch < ref_vals[None, :], axis=0)
                    equal = np.sum(pred_ens_ch == ref_vals[None, :], axis=0)
                    pit = (less + 0.5 * equal) / float(n_ens)
                    valid = np.isfinite(pit)
                    if np.any(valid):
                        pit_hist_counts += np.histogram(pit[valid], bins=pit_edges)[0].astype(np.float64)
                        rank = np.clip(less[valid], 0, n_ens).astype(np.int64)
                        rank_hist_counts += np.bincount(rank, minlength=n_ens + 1).astype(np.float64)

                    # Spread-skill diagnostic samples (subsampled for plotting efficiency).
                    spread_loc = np.std(pred_ens_ch, axis=0)
                    abs_err_loc = np.abs(pred_mean_ch - gt_mean_ch)
                    n_loc = spread_loc.size
                    n_take = min(max_scatter_points_per_step, n_loc)
                    if n_take > 0:
                        idx = np.linspace(0, n_loc - 1, n_take, dtype=np.int64)
                        spread_skill_samples.append(
                            np.stack([spread_loc[idx], abs_err_loc[idx]], axis=1).astype(np.float64)
                        )

            if use_ensemble:
                for ens_idx in range(n_ens):
                    current_dynamics[ens_idx] = torch.cat(
                        [current_dynamics[ens_idx][1:], pred_stack[ens_idx, 0].unsqueeze(0)],
                        dim=0,
                    )
            else:
                current_dynamic = torch.cat(
                    [current_dynamic[1:], pred_stack[0, 0].unsqueeze(0)],
                    dim=0,
                )
            current_boundary = torch.cat(
                [current_boundary[1:], gt_boundary_rollout[t].unsqueeze(0)], dim=0
            )

        pred_mean_by_channel = {
            k: np.stack(v, axis=0) for k, v in run_pred_mean_by_channel.items()
        }
        pred_std_by_channel = {
            k: np.stack(v, axis=0) for k, v in run_pred_std_by_channel.items()
        }
        gt_mean_by_channel = {
            k: np.stack(v, axis=0) for k, v in run_gt_mean_by_channel.items()
        }
        gt_std_by_channel = {
            k: np.stack(v, axis=0) for k, v in run_gt_std_by_channel.items()
        }
        pred_prob_wd = np.stack(run_wd_pred_prob, axis=0) if run_wd_pred_prob else None
        gt_prob_wd = np.stack(run_wd_gt_prob, axis=0) if run_wd_gt_prob else None
        crps_map_wd = np.stack(run_wd_crps_map, axis=0) if run_wd_crps_map else None

        _save_hydrograph_uq_figures_and_animation(
            geometry=geometry,
            pred_mean_by_channel=pred_mean_by_channel,
            pred_std_by_channel=pred_std_by_channel,
            gt_mean_by_channel=gt_mean_by_channel,
            gt_std_by_channel=gt_std_by_channel,
            target_variables=target_variables,
            out_dir=out_dir,
            hydrograph_id=hydro_id,
            dt_seconds=dt,
            n_ref_sims=n_ref,
            n_ens=n_ens,
            pred_prob_wd=pred_prob_wd,
            gt_prob_wd=gt_prob_wd,
            crps_map_wd=crps_map_wd,
        )

        for ch_name in target_variables:
            per_channel_rmse[ch_name].append(np.asarray(run_rmse[ch_name], dtype=np.float64))
            per_channel_crps[ch_name].append(np.asarray(run_crps[ch_name], dtype=np.float64))
            per_channel_spread_pred[ch_name].append(np.asarray(run_spread_pred[ch_name], dtype=np.float64))
            per_channel_spread_gt[ch_name].append(np.asarray(run_spread_gt[ch_name], dtype=np.float64))
            per_channel_spread_ratio[ch_name].append(np.asarray(run_spread_ratio[ch_name], dtype=np.float64))
        if run_wd_brier:
            wd_prob_brier.append(np.asarray(run_wd_brier, dtype=np.float64))
            wd_prob_mae.append(np.asarray(run_wd_mae, dtype=np.float64))
            wd_wasserstein.append(np.asarray(run_wd_wasserstein, dtype=np.float64))
            for alpha in interval_levels:
                wd_interval_coverage[alpha].append(
                    np.asarray(run_interval_coverage[alpha], dtype=np.float64)
                )
                wd_interval_width[alpha].append(
                    np.asarray(run_interval_width[alpha], dtype=np.float64)
                )

        logger.info("Completed hydrograph %s (n_ref=%d, n_ens=%d)", hydro_id, n_ref, n_ens)

    if not any(len(v) > 0 for v in per_channel_rmse.values()):
        logger.warning("No per-hydrograph metrics were produced.")
        return

    metrics: Dict[str, np.ndarray] = {}
    for ch_name in target_variables:
        metrics[f"rmse_{ch_name}"] = np.stack(per_channel_rmse[ch_name], axis=0)
        metrics[f"crps_{ch_name}"] = np.stack(per_channel_crps[ch_name], axis=0)
        metrics[f"spread_pred_{ch_name}"] = np.stack(per_channel_spread_pred[ch_name], axis=0)
        metrics[f"spread_gt_{ch_name}"] = np.stack(per_channel_spread_gt[ch_name], axis=0)
        metrics[f"spread_ratio_{ch_name}"] = np.stack(per_channel_spread_ratio[ch_name], axis=0)
    if wd_prob_brier:
        metrics["brier_wd_exceed"] = np.stack(wd_prob_brier, axis=0)
    if wd_prob_mae:
        metrics["prob_mae_wd_exceed"] = np.stack(wd_prob_mae, axis=0)
    if wd_wasserstein:
        metrics["wasserstein_wd"] = np.stack(wd_wasserstein, axis=0)
    for alpha in interval_levels:
        if wd_interval_coverage[alpha]:
            pct = int(round(alpha * 100))
            metrics[f"coverage_wd_{pct}"] = np.stack(wd_interval_coverage[alpha], axis=0)
            metrics[f"width_wd_{pct}"] = np.stack(wd_interval_width[alpha], axis=0)

    stats = {k: {"mean": v.mean(axis=0), "std": v.std(axis=0)} for k, v in metrics.items()}
    time_hours = (np.arange(1, rollout_length + 1) * dt) / 3600.0
    npz_data: Dict[str, Any] = {"time_hours": time_hours}
    for key, stat_dict in stats.items():
        npz_data[f"{key}_mean"] = stat_dict["mean"]
        npz_data[f"{key}_std"] = stat_dict["std"]
        npz_data[f"{key}_all"] = metrics[key]
    data_path = os.path.join(out_dir, ROLLOUT_METRICS_HYDRO_NPZ)
    np.savez(data_path, **npz_data)
    logger.info("Saved per-hydrograph metrics to %s", data_path)

    plot_keys = list(stats.keys())
    n_plots = len(plot_keys)
    n_cols = 2
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(8 * n_cols, 4.8 * n_rows), tight_layout=True
    )
    axs_flat = np.array(axs).reshape(-1)
    for i, key in enumerate(plot_keys):
        mean = stats[key]["mean"]
        std = stats[key]["std"]
        ax = axs_flat[i]
        ax.plot(time_hours, mean, marker="o", markersize=3, linewidth=1.2, label=f"{key} mean")
        ax.fill_between(time_hours, mean - std, mean + std, alpha=0.25, label="±1 std")
        ax.set_title(f"{key} over time (across hydrographs)")
        ax.set_xlabel("Time (hour)")
        if key.startswith("spread_ratio"):
            ax.set_ylabel("Spread ratio (pred/gt)")
            ax.axhline(1.0, color="gray", linestyle="--", alpha=0.8)
        elif key.startswith("rmse_"):
            ax.set_ylabel("RMSE")
        elif key.startswith("crps_"):
            ax.set_ylabel("CRPS")
        elif key.startswith("coverage_"):
            ax.set_ylabel("Coverage")
            if key.endswith("_50"):
                ax.axhline(0.5, color="gray", linestyle="--", alpha=0.6)
            elif key.endswith("_80"):
                ax.axhline(0.8, color="gray", linestyle="--", alpha=0.6)
            elif key.endswith("_90"):
                ax.axhline(0.9, color="gray", linestyle="--", alpha=0.6)
            elif key.endswith("_95"):
                ax.axhline(0.95, color="gray", linestyle="--", alpha=0.6)
            ax.set_ylim(0.0, 1.0)
        elif key.startswith("width_"):
            ax.set_ylabel("Interval width")
        elif key.startswith("brier"):
            ax.set_ylabel("Brier score")
        elif key.startswith("wasserstein"):
            ax.set_ylabel("Wasserstein-1")
        else:
            ax.set_ylabel("Metric")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    for j in range(n_plots, len(axs_flat)):
        axs_flat[j].set_visible(False)
    summary_path = os.path.join(out_dir, ROLLOUT_SUMMARY_HYDRO_PNG)
    fig.savefig(summary_path, dpi=220)
    plt.close(fig)
    logger.info("Saved per-hydrograph summary figure to %s", summary_path)

    rel_mean_pred = np.divide(
        rel_sum_pred, rel_count, out=np.zeros_like(rel_sum_pred), where=rel_count > 0
    )
    rel_mean_obs = np.divide(
        rel_sum_obs, rel_count, out=np.zeros_like(rel_sum_obs), where=rel_count > 0
    )
    reliability_bins = {
        "edges": rel_edges,
        "centers": 0.5 * (rel_edges[:-1] + rel_edges[1:]),
        "count": rel_count,
        "mean_pred": rel_mean_pred,
        "mean_obs": rel_mean_obs,
        "brier_overall": rel_brier_sum / max(float(rel_brier_count), 1.0),
    }
    interval_cov_stack = {
        a: np.stack(wd_interval_coverage[a], axis=0)
        for a in interval_levels
        if wd_interval_coverage[a]
    }
    interval_width_stack = {
        a: np.stack(wd_interval_width[a], axis=0)
        for a in interval_levels
        if wd_interval_width[a]
    }
    spread_skill_arr = (
        np.concatenate(spread_skill_samples, axis=0)
        if spread_skill_samples
        else np.empty((0, 2), dtype=np.float64)
    )
    wasserstein_arr = metrics.get("wasserstein_wd", None)
    _save_nonspatial_uq_diagnostics(
        out_dir=out_dir,
        time_hours=time_hours,
        metrics=metrics,
        reliability_bins=reliability_bins,
        pit_hist_counts=pit_hist_counts,
        pit_edges=pit_edges,
        rank_hist_counts=rank_hist_counts,
        spread_skill_samples=spread_skill_arr,
        interval_coverage=interval_cov_stack,
        interval_width=interval_width_stack,
        wasserstein_wd=wasserstein_arr,
        logger=logger,
    )


def _rollout_prediction_generic(
    models: List[Any],
    rollout_dataset: Any,
    rollout_length: int,
    history_steps: int,
    dynamic_norm: Any,
    target_norm: Any,
    device: torch.device,
    skip_before_timestep: int,
    dt: float,
    out_dir: str,
    target_variables: List[str],
    logger: logging.Logger,
    fgn_noise_dim: Optional[int] = None,
    n_ensemble_samples: int = 1,
) -> None:
    """
    Generic rollout mode (single reference trajectory per run).

    Ensemble mode follows paper-style AR updates: each member keeps its own state.
    """
    if not models:
        raise ValueError("No models provided for rollout evaluation.")
    for model in models:
        model.eval()
    dynamic_norm.to(device)
    target_norm.to(device)
    os.makedirs(out_dir, exist_ok=True)

    n_models = len(models)
    n_ens = max(1, int(n_ensemble_samples))
    if n_models > 1 and n_ens < n_models:
        logger.warning(
            "n_ensemble_samples=%d is smaller than number of models=%d. "
            "Raising n_ensemble_samples to %d for paper-style equal model usage.",
            n_ens, n_models, n_models,
        )
        n_ens = n_models
    use_ensemble = n_ens > 1 or n_models > 1
    member_model_indices = _build_member_model_indices(n_models, n_ens)
    model_counts = [member_model_indices.count(i) for i in range(n_models)]
    logger.info(
        "Generic rollout ensemble members=%d across models=%d with per-model counts=%s",
        n_ens, n_models, model_counts,
    )

    per_channel_rmse: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
    per_channel_spread: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
    per_channel_spread_skill: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
    wd_csi_005: List[np.ndarray] = []
    wd_csi_03: List[np.ndarray] = []

    for idx, sample in enumerate(tqdm(rollout_dataset, desc="Rollout evaluation")):
        run_id = sample.get("run_id", f"sample_{idx}")
        full_dynamic = sample["dynamic"].to(device)
        full_boundary = sample["boundary"].to(device)
        geometry = sample["geometry"]
        start_pred_t = skip_before_timestep + history_steps
        end_pred_t = start_pred_t + rollout_length
        gt_rollout = full_dynamic[start_pred_t:end_pred_t]
        gt_boundary_rollout = full_boundary[start_pred_t:end_pred_t]

        run_rmse: Dict[str, List[float]] = {name: [] for name in target_variables}
        run_spread: Dict[str, List[float]] = {name: [] for name in target_variables}
        run_spread_skill: Dict[str, List[float]] = {name: [] for name in target_variables}
        run_csi_005: List[float] = []
        run_csi_03: List[float] = []
        run_pred_by_channel: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}
        run_gt_by_channel: Dict[str, List[np.ndarray]] = {name: [] for name in target_variables}

        if use_ensemble:
            current_dynamics = [
                full_dynamic[skip_before_timestep:start_pred_t].clone() for _ in range(n_ens)
            ]
        else:
            current_dynamic = full_dynamic[skip_before_timestep:start_pred_t].clone()
        current_boundary = full_boundary[skip_before_timestep:start_pred_t].clone()

        static_0 = sample["static"].to(device).unsqueeze(0)
        geom_0 = geometry.to(device).unsqueeze(0)
        query_0 = sample["query_points"].to(device).unsqueeze(0)

        for t in range(rollout_length):
            with torch.no_grad():
                if use_ensemble:
                    pred_members: List[torch.Tensor] = []
                    for ens_idx in range(n_ens):
                        dyn_hist = current_dynamics[ens_idx]
                        model_idx = member_model_indices[ens_idx]
                        model = models[model_idx]
                        dyn_flat = dyn_hist.permute(1, 0, 2).reshape(1, dyn_hist.shape[1], -1)
                        bc_flat = current_boundary.permute(1, 0, 2).reshape(1, current_boundary.shape[1], -1)
                        x = torch.cat([static_0, bc_flat, dyn_flat], dim=2)
                        if fgn_noise_dim is not None:
                            z = torch.randn(fgn_noise_dim, device=device, dtype=x.dtype)
                            pred_members.append(
                                model(
                                    input_geom=geom_0,
                                    latent_queries=query_0,
                                    output_queries=geom_0,
                                    x=x,
                                    ada_in=z,
                                )
                            )
                        else:
                            pred_members.append(
                                model(
                                    input_geom=geom_0,
                                    latent_queries=query_0,
                                    output_queries=geom_0,
                                    x=x,
                                )
                            )
                    pred_stack = torch.stack(pred_members, dim=0)
                    pred = pred_stack.mean(dim=0)
                else:
                    model = models[0]
                    dyn_flat = current_dynamic.permute(1, 0, 2).reshape(
                        1, current_dynamic.shape[1], -1
                    )
                    bc_flat = current_boundary.permute(1, 0, 2).reshape(
                        1, current_boundary.shape[1], -1
                    )
                    x = torch.cat([static_0, bc_flat, dyn_flat], dim=2)
                    if fgn_noise_dim is not None:
                        z = torch.randn(fgn_noise_dim, device=device, dtype=x.dtype)
                        pred = model(
                            input_geom=geom_0,
                            latent_queries=query_0,
                            output_queries=geom_0,
                            x=x,
                            ada_in=z,
                        )
                    else:
                        pred = model(
                            input_geom=geom_0,
                            latent_queries=query_0,
                            output_queries=geom_0,
                            x=x,
                        )
                    pred_stack = pred.unsqueeze(0)

            inv_pred = target_norm.inverse_transform(pred)
            inv_gt = dynamic_norm.inverse_transform(gt_rollout[t].unsqueeze(0))
            inv_pred_ens = target_norm.inverse_transform(pred_stack.squeeze(1))

            for ch_idx, ch_name in enumerate(target_variables):
                ch_pred = inv_pred[0, :, ch_idx].detach().cpu().numpy()
                ch_gt = inv_gt[0, :, ch_idx].detach().cpu().numpy()
                run_rmse[ch_name].append(float(np.sqrt(np.mean((ch_pred - ch_gt) ** 2))))
                run_pred_by_channel[ch_name].append(ch_pred)
                run_gt_by_channel[ch_name].append(ch_gt)
                if use_ensemble:
                    ens_ch = inv_pred_ens[:, :, ch_idx].detach().cpu().numpy()
                    spread_t = float(np.mean(np.std(ens_ch, axis=0)))
                    skill_t = float(np.sqrt(np.mean((ch_pred - ch_gt) ** 2)))
                    run_spread[ch_name].append(spread_t)
                    run_spread_skill[ch_name].append(
                        spread_t / skill_t if skill_t > MIN_EPS else 0.0
                    )
                if ch_name == "wd":
                    run_csi_005.append(_compute_csi(0.05, ch_pred, ch_gt))
                    run_csi_03.append(_compute_csi(0.3, ch_pred, ch_gt))

            if use_ensemble:
                for ens_idx in range(n_ens):
                    current_dynamics[ens_idx] = torch.cat(
                        [current_dynamics[ens_idx][1:], pred_stack[ens_idx, 0].unsqueeze(0)],
                        dim=0,
                    )
            else:
                current_dynamic = torch.cat(
                    [current_dynamic[1:], pred.squeeze(0).unsqueeze(0)], dim=0
                )
            current_boundary = torch.cat(
                [current_boundary[1:], gt_boundary_rollout[t].unsqueeze(0)], dim=0
            )

        for ch_name in target_variables:
            per_channel_rmse[ch_name].append(np.array(run_rmse[ch_name], dtype=np.float64))
            if use_ensemble:
                per_channel_spread[ch_name].append(np.array(run_spread[ch_name], dtype=np.float64))
                per_channel_spread_skill[ch_name].append(np.array(run_spread_skill[ch_name], dtype=np.float64))
        if "wd" in target_variables:
            wd_csi_005.append(np.array(run_csi_005, dtype=np.float64))
            wd_csi_03.append(np.array(run_csi_03, dtype=np.float64))

        pred_arr = {k: np.stack(v, axis=0) for k, v in run_pred_by_channel.items()}
        gt_arr = {k: np.stack(v, axis=0) for k, v in run_gt_by_channel.items()}
        if all(k in pred_arr for k in LEGACY_3CH):
            generate_publication_maps(
                geometry=geometry,
                wd_gt_array=gt_arr["wd"],
                wd_pred_array=pred_arr["wd"],
                vx_gt_array=gt_arr["vx"],
                vy_gt_array=gt_arr["vy"],
                vx_pred_array=pred_arr["vx"],
                vy_pred_array=pred_arr["vy"],
                steps=list(PUBLICATION_TIMESTEPS),
                out_dir=os.path.join(out_dir, "figures"),
                run_id=run_id,
                filename_prefix="flood",
            )
            create_rollout_animation(
                geometry=geometry,
                wd_gt=gt_arr["wd"],
                wd_pred=pred_arr["wd"],
                vx_gt=gt_arr["vx"],
                vy_gt=gt_arr["vy"],
                vx_pred=pred_arr["vx"],
                vy_pred=pred_arr["vy"],
                run_id=run_id,
                out_dir=out_dir,
                dt_seconds=dt,
            )
        else:
            _save_generic_rollout_visuals(
                geometry=geometry,
                pred_by_channel=pred_arr,
                gt_by_channel=gt_arr,
                target_variables=target_variables,
                out_dir=out_dir,
                run_id=run_id,
                dt_seconds=dt,
            )
        logger.info("Completed rollout run_id=%s", run_id)

    if not any(len(v) > 0 for v in per_channel_rmse.values()):
        logger.warning("No rollout metrics were produced.")
        return

    metrics: Dict[str, np.ndarray] = {}
    for ch_name in target_variables:
        metrics[f"rmse_{ch_name}"] = np.stack(per_channel_rmse[ch_name], axis=0)
        if use_ensemble:
            metrics[f"spread_{ch_name}"] = np.stack(per_channel_spread[ch_name], axis=0)
            metrics[f"spread_skill_{ch_name}"] = np.stack(per_channel_spread_skill[ch_name], axis=0)
    if "wd" in target_variables and wd_csi_005 and wd_csi_03:
        metrics["csi_005"] = np.stack(wd_csi_005, axis=0)
        metrics["csi_03"] = np.stack(wd_csi_03, axis=0)

    stats = {k: {"mean": v.mean(axis=0), "std": v.std(axis=0)} for k, v in metrics.items()}
    time_hours = (np.arange(1, rollout_length + 1) * dt) / 3600.0
    npz_data: Dict[str, Any] = {"time_hours": time_hours}
    for key, stat_dict in stats.items():
        npz_data[f"{key}_mean"] = stat_dict["mean"]
        npz_data[f"{key}_std"] = stat_dict["std"]
        npz_data[f"{key}_all"] = metrics[key]
    data_path = os.path.join(out_dir, ROLLOUT_METRICS_NPZ)
    np.savez(data_path, **npz_data)
    logger.info("Saved rollout metrics to %s", data_path)

    plot_keys = list(stats.keys())
    n_plots = len(plot_keys)
    n_cols = 2
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(8 * n_cols, 5 * n_rows), tight_layout=True
    )
    axs_flat = np.array(axs).reshape(-1)
    for i, key in enumerate(plot_keys):
        mean, std = stats[key]["mean"], stats[key]["std"]
        ax = axs_flat[i]
        ax.plot(time_hours, mean, marker="o", label=f"{key} mean")
        ax.fill_between(time_hours, mean - std, mean + std, alpha=0.3, label="±1 std")
        ax.set_title(f"{key} over time")
        ax.set_xlabel("Time (hour)")
        if key.startswith("spread_skill"):
            ax.set_ylabel("Spread-skill ratio")
            ax.axhline(1.0, color="gray", linestyle="--", alpha=0.7)
        else:
            ax.set_ylabel("RMSE" if key.startswith("rmse_") else "CSI")
        ax.grid(True)
        ax.legend()
    for j in range(n_plots, len(axs_flat)):
        axs_flat[j].set_visible(False)
    summary_path = os.path.join(out_dir, ROLLOUT_SUMMARY_PNG)
    plt.savefig(summary_path)
    plt.close(fig)
    logger.info("Saved rollout summary plot to %s", summary_path)


def _make_trainer(
    config: Any, model: Any, data_processor: Any, device: torch.device, logger: Optional[logging.Logger]
) -> Any:
    """Build FGNTrainer or Trainer from config (no training, eval only)."""
    use_fgn = _opt(config, "gino", "use_fgn_noise", False)
    use_progress_bar = _opt(config, None, "use_progress_bar", True)
    scheduler_monitor = _opt(config, "opt", "scheduler_monitor", "train_err")
    eval_interval = _opt(config, "wandb", "eval_interval", 1)
    is_logger = logger is not None
    n_epochs = max(1, int(_opt(config, "opt", "n_epochs", 1)))

    common = dict(
        model=model,
        n_epochs=n_epochs,
        data_processor=data_processor,
        device=device,
        wandb_log=False,
        verbose=is_logger,
        logger=logger,
        use_progress_bar=use_progress_bar,
        scheduler_monitor=scheduler_monitor,
        eval_interval=eval_interval,
    )
    if use_fgn and _opt(config, "opt", "training_loss", "l2") == "crps":
        return FGNTrainer(
            **common,
            fgn_noise_dim=_opt(config, "gino", "fgn_noise_dim", 32),
            crps_n_samples=max(2, int(_opt(config, "opt", "crps_n_samples", 2))),
            rel_l2_loss_fn=LpLoss(d=2, p=2),
            crps_l2_weight=float(_opt(config, "opt", "crps_l2_weight", 0.0)),
            ar_finetune_start_epoch=max(
                0, int(_opt(config, "opt", "ar_finetune_start_epoch", 0))
            ),
            ar_rollout_steps=max(1, int(_opt(config, "opt", "ar_rollout_steps", 1))),
            ar_curriculum_epochs_per_step=max(
                0, int(_opt(config, "opt", "ar_curriculum_epochs_per_step", 0))
            ),
            use_flood_crps_spatial_weights=bool(
                _opt(config, "opt", "flood_crps_spatial_weights", False)
            ),
            flood_crps_wet_threshold=float(
                _opt(config, "opt", "wet_threshold", 0.01)
            ),
            flood_crps_wet_smooth_scale=float(
                _opt(config, "opt", "wet_smooth_scale", 0.02)
            ),
            flood_crps_dry_weight_alpha=float(
                _opt(config, "opt", "dry_weight_alpha", 0.1)
            ),
            static_normalizer=None,
            use_hazard_proxy_crps=bool(_opt(config, "opt", "hazard_proxy_crps", False)),
            hazard_proxy_crps_weight=float(
                _opt(config, "opt", "hazard_proxy_crps_weight", 0.15)
            ),
            ar_pooled_crps_gamma=float(
                _opt(config, "opt", "ar_pooled_crps_gamma", 1.0)
            ),
        )
    return Trainer(**common)


def _resolve_device(device: Union[str, torch.device]) -> torch.device:
    """Resolve device string to torch.device; fallback to CPU if CUDA unavailable."""
    if isinstance(device, torch.device):
        return device
    if "cuda" in device and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _resolve_checkpoint_in_dir(save_dir: Path) -> Tuple[Path, str]:
    """Return (dir, alias) for a single checkpoint directory."""
    if (save_dir / "best_model_state_dict.pt").exists():
        return save_dir, CHECKPOINT_BEST
    if (save_dir / "model_state_dict.pt").exists():
        return save_dir, CHECKPOINT_LAST
    found = [p.name for p in save_dir.iterdir()] if save_dir.exists() else []
    raise FileNotFoundError(
        f"No checkpoint in {save_dir}. Expected one of {CHECKPOINT_FILES}. Found: {found}"
    )


def _discover_checkpoint_runs(checkpoint_path: Path) -> List[Tuple[Path, str, str]]:
    """
    Discover checkpoint runs.

    Returns
    -------
    list of (checkpoint_dir, alias, label)
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")

    runs: List[Tuple[Path, str, str]] = []
    try:
        run_dir, alias = _resolve_checkpoint_in_dir(checkpoint_path)
        runs.append((run_dir, alias, checkpoint_path.name or "model0"))
        return runs
    except FileNotFoundError:
        pass

    child_dirs = sorted([p for p in checkpoint_path.iterdir() if p.is_dir()])
    for child in child_dirs:
        try:
            run_dir, alias = _resolve_checkpoint_in_dir(child)
            runs.append((run_dir, alias, child.name))
        except FileNotFoundError:
            continue

    if not runs:
        found = [p.name for p in checkpoint_path.iterdir()]
        raise FileNotFoundError(
            "No valid checkpoints found in "
            f"{checkpoint_path}. Expected checkpoint files in this directory or immediate subdirectories. "
            f"Found entries: {found}"
        )
    return runs


def _checkpoint_metadata_candidates(run_dir: Path, alias: str) -> List[Path]:
    """Candidate metadata files storing the exact model init kwargs used at training."""
    candidates = [
        run_dir / f"{alias}_metadata.pkl",
        run_dir / "model_metadata.pkl",
        run_dir / "best_model_metadata.pkl",
    ]
    seen: set = set()
    unique: List[Path] = []
    for p in candidates:
        key = str(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def _instantiate_model_from_metadata(metadata: Dict[str, Any]) -> Any:
    """
    Build model directly from checkpoint metadata.

    This reproduces the exact init kwargs used when the checkpoint was trained.
    """
    arch_name = str(metadata.get("_name", "")).strip().lower()
    if not arch_name:
        raise KeyError("Missing '_name' in checkpoint metadata.")
    model_cls = BaseModel._models.get(arch_name)
    if model_cls is None:
        raise KeyError(
            f"Unknown model name '{arch_name}' in checkpoint metadata. "
            f"Known models: {list(BaseModel._models.keys())}"
        )
    init_kwargs = dict(metadata)
    init_args = init_kwargs.pop("args", ())
    init_kwargs.pop("_version", None)
    init_kwargs.pop("_name", None)
    if not isinstance(init_args, (list, tuple)):
        init_args = (init_args,)
    return model_cls(*init_args, **init_kwargs)


def _build_model_for_run(
    config: Any,
    run_dir: Path,
    alias: str,
    label: str,
    logger: logging.Logger,
) -> Any:
    """
    Build model for a checkpoint run.

    Priority:
      1) checkpoint metadata (exact training init kwargs)
      2) deep-copied config fallback (avoids in-place config mutation in get_model)
    """
    for meta_path in _checkpoint_metadata_candidates(run_dir, alias):
        if not meta_path.exists():
            continue
        try:
            metadata = torch.load(meta_path, map_location="cpu")
            if isinstance(metadata, dict):
                model = _instantiate_model_from_metadata(metadata)
                logger.info(
                    "Model '%s': initialized from checkpoint metadata %s",
                    label,
                    meta_path,
                )
                return model
            logger.warning(
                "Model '%s': metadata file %s is not a dict (type=%s); falling back to config.",
                label,
                meta_path,
                type(metadata).__name__,
            )
        except Exception as exc:
            logger.warning(
                "Model '%s': failed loading metadata %s (%s); falling back to config.",
                label,
                meta_path,
                exc,
            )

    model_cfg = copy.deepcopy(config)
    logger.info(
        "Model '%s': metadata not used; initializing from evaluation config snapshot.",
        label,
    )
    return get_model(model_cfg)


def _load_models_from_runs(
    config: Any,
    device: torch.device,
    checkpoint_runs: List[Tuple[Path, str, str]],
    logger: logging.Logger,
) -> List[Any]:
    """Load all models from discovered checkpoints."""
    models: List[Any] = []
    for run_dir, alias, label in checkpoint_runs:
        model = _build_model_for_run(config, run_dir, alias, label, logger)
        load_training_state(save_dir=run_dir, save_name=alias, model=model)
        model = model.to(device).eval()
        models.append(model)
        logger.info("Loaded model '%s' from %s (%s)", label, run_dir, alias)
    return models


def _get_cli_arg_value(flag: str) -> Optional[str]:
    """Return CLI value for '--flag value' or '--flag=value', without consuming argv."""
    argv = sys.argv[1:]
    for i, token in enumerate(argv):
        if token == flag and i + 1 < len(argv):
            return argv[i + 1]
        prefix = f"{flag}="
        if token.startswith(prefix):
            return token[len(prefix):]
    return None


def _resolve_cli_config_path(cli_value: Optional[str]) -> Optional[Path]:
    """Resolve --config_path value using the same relative-path logic as training script."""
    if not cli_value:
        return None
    cfg = Path(cli_value)
    if not cfg.is_absolute():
        cfg = _REPO_ROOT / cfg
    return cfg.resolve()


def _move_normalizers_to_device(normalizers: Dict[str, Any]) -> torch.device:
    """Move all normalizers to one device (in-place). Return that device."""
    ref: torch.device = torch.device("cpu")
    for key in DEVICE_REF_KEYS:
        if key in normalizers and normalizers[key] is not None:
            if hasattr(normalizers[key], "mean"):
                ref = normalizers[key].mean.device
                break
    for key in NORMALIZER_KEYS:
        if key in normalizers and normalizers[key] is not None:
            normalizers[key].to(ref)
    return ref


def _parse_args() -> argparse.Namespace:
    """Parse CLI and strip known args so config pipeline only sees its own."""
    parser = argparse.ArgumentParser(
        description="Post-training evaluation for flood GINO WV."
    )
    parser.add_argument(
        "--eval_log_file",
        type=str,
        default=DEFAULT_EVAL_LOG,
        help="Log file path (relative to checkpoint dir or absolute).",
    )
    parser.add_argument(
        "--run_single_step",
        action="store_true",
        help="Run one-step test evaluation.",
    )
    parser.add_argument(
        "--skip_single_step",
        action="store_true",
        help="Skip one-step evaluation.",
    )
    parser.add_argument(
        "--run_rollout",
        action="store_true",
        help="Force rollout evaluation even if config says no.",
    )
    parser.add_argument(
        "--skip_rollout",
        action="store_true",
        help="Skip rollout evaluation.",
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


def _validate_args(args: argparse.Namespace) -> None:
    """Raise if mutually exclusive flags are both set."""
    if args.run_single_step and args.skip_single_step:
        raise ValueError("Cannot set both --run_single_step and --skip_single_step")
    if args.run_rollout and args.skip_rollout:
        raise ValueError("Cannot set both --run_rollout and --skip_rollout")


def _load_or_fit_normalizers(
    config: Any,
    train_data: Any,
    save_dir: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Load normalizers from config path or fit on train split and optionally save."""
    normalizer_path = _opt(config, "data", "normalizer_path", None)
    if normalizer_path is not None:
        normalizer_path = Path(normalizer_path)
        if not normalizer_path.is_absolute():
            normalizer_path = Path(config.data.root) / normalizer_path
    if normalizer_path is not None and normalizer_path.exists():
        with _PhaseTimer(logger, f"Loading normalizers from {normalizer_path}"):
            return load_normalizers(normalizer_path, device=None)
    with _PhaseTimer(logger, "Fitting normalizers on training split"):
        normalizers = fit_normalizers_streaming(
            train_data,
            chunk_size=int(_opt(config, "data", "normalizer_chunk_size", 10000)),
            expect_target=True,
        )
    if normalizer_path is not None:
        save_normalizers(normalizers, normalizer_path)
        logger.info("Saved normalizers to %s", normalizer_path)
    return normalizers


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
    logger.info("Rollout tensors on device=%s", ref_device)
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


def main() -> int:
    """Run post-training one-step and/or rollout evaluation."""
    args = _parse_args()
    _validate_args(args)
    cli_config_path = _resolve_cli_config_path(_get_cli_arg_value("--config_path"))
    config, device, is_logger = load_config_and_setup()
    device = _resolve_device(device)
    seed = _opt(config, "distributed", "seed", 123)
    deterministic = _opt(config, None, "deterministic", True)
    set_seed(seed, deterministic=deterministic)

    checkpoint_path = Path(_opt(config, "checkpoint", "save_dir", "."))
    if not checkpoint_path.is_absolute():
        checkpoint_path = checkpoint_path.resolve()
    checkpoint_runs = _discover_checkpoint_runs(checkpoint_path)
    primary_dir, primary_alias, _ = checkpoint_runs[0]
    eval_log = Path(args.eval_log_file)
    if not eval_log.is_absolute():
        eval_log = primary_dir / eval_log
    logger = setup_logging(
        log_level=_opt(config, None, "log_level", "INFO"),
        log_file=str(eval_log),
        logger_name="flood_eval",
    )
    if is_logger:
        logger.info("Post-training evaluation started")
        logger.info("Device=%s | Seed=%s | Deterministic=%s", device, seed, deterministic)
        if cli_config_path is not None:
            logger.info("Config source (--config_path): %s", cli_config_path)
        else:
            logger.warning(
                "No --config_path provided. Evaluator is using the training script default config path."
            )
        logger.info(
            "Checkpoint path=%s | discovered models=%d",
            checkpoint_path,
            len(checkpoint_runs),
        )
        logger.info("data.root=%s", _opt(config, "data", "root", "N/A"))

    if _opt(config, "data", "write_train_txt", False):
        with _PhaseTimer(logger, "Refreshing train.txt"):
            run_ids = write_train_txt_from_data_root(
                config.data.root,
                train_txt=_opt(config, "data", "train_txt", "train.txt"),
                hdf_suffix=".hdf",
            )
        logger.info("train.txt refreshed: %d run IDs", len(run_ids))

    train_raw, test_raw, target_variables = _build_one_step_datasets(
        config, seed, logger
    )
    normalizers = _load_or_fit_normalizers(config, train_raw, primary_dir, logger)
    with _PhaseTimer(logger, "Wrapping normalized datasets"):
        train_norm = NormalizedDatasetOnTheFly(
            train_raw, normalizers, query_res=config.data.query_res
        )
        test_norm = NormalizedDatasetOnTheFly(
            test_raw, normalizers, query_res=config.data.query_res
        )
    test_loader = _build_test_loader(test_norm, config.data.batch_size)
    logger.info(
        "Test loader: batch_size=%s batches=%d",
        config.data.batch_size, len(test_loader),
    )

    with _PhaseTimer(logger, "Loading model checkpoint(s)"):
        models = _load_models_from_runs(config, device, checkpoint_runs, logger)
    inverse_test = _opt(config, None, "inverse_test", True)
    if normalizers.get("target") is not None:
        normalizers["target"] = normalizers["target"].to(device)
    use_fgn = _opt(config, "gino", "use_fgn_noise", False)
    eval_losses = _build_eval_losses(config, use_fgn)
    logger.info(
        "Eval losses=%s inverse_test=%s n_models=%d",
        list(eval_losses.keys()),
        inverse_test,
        len(models),
    )

    run_single = args.run_single_step and not args.skip_single_step
    if run_single:
        model_metrics: List[Dict[str, float]] = []
        with _PhaseTimer(logger, "One-step evaluation (all models)"):
            for model_idx, model in enumerate(models):
                data_processor = FloodGINODataProcessor(
                    device=device,
                    target_norm=normalizers.get("target"),
                    inverse_test=inverse_test,
                )
                data_processor.wrap(model)
                trainer = _make_trainer(config, model, data_processor, device, logger)
                metrics = trainer.evaluate(
                    eval_losses, test_loader, log_prefix=f"test_m{model_idx}"
                )
                clean = {
                    k: float(v.item() if hasattr(v, "item") else v)
                    for k, v in metrics.items()
                    if not k.endswith("_outputs")
                }
                model_metrics.append(clean)
                logger.info("One-step metrics model[%d]: %s", model_idx, clean)
        logger.info("One-step TEST metrics summary across %d models:", len(model_metrics))
        common_keys = sorted(set().union(*[m.keys() for m in model_metrics]))
        for key in common_keys:
            vals = [m[key] for m in model_metrics if key in m]
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals))
            logger.info("  %s: mean=%.6e std=%.6e", key, mean_v, std_v)
            print(f"  {key}: mean={mean_v:.6e} std={std_v:.6e}")

    run_rollout_cfg = bool(_opt(config, "rollout", "run_after_training", False))
    run_rollout = (run_rollout_cfg or args.run_rollout) and not args.skip_rollout
    if not run_rollout:
        logger.info(
            "Skipping rollout (run_after_training=false and --run_rollout not set)."
        )
        return 0

    rollout_n_ensemble = int(_opt(config, "rollout", "n_ensemble_samples", 1))
    ens_per_model = _opt(config, "rollout", "n_ensemble_samples_per_model", None)
    if ens_per_model is not None:
        ens_per_model_int = int(ens_per_model)
        if ens_per_model_int < 1:
            raise ValueError(
                "rollout.n_ensemble_samples_per_model must be >= 1 "
                f"(got {ens_per_model_int})."
            )
        rollout_n_ensemble = ens_per_model_int * max(1, len(models))
        logger.info(
            "Using rollout.n_ensemble_samples_per_model=%d with n_models=%d => total ensemble members=%d",
            ens_per_model_int,
            len(models),
            rollout_n_ensemble,
        )
    else:
        logger.info(
            "Using rollout.n_ensemble_samples=%d total members across %d model(s).",
            rollout_n_ensemble,
            len(models),
        )

    rollout_norm_ds, hydrograph_samples = _build_rollout_normalized_dataset(
        config, normalizers, target_variables, logger
    )
    logger.info("Rollout normalized dataset: %d runs", len(rollout_norm_ds))
    if hydrograph_samples:
        logger.info(
            "Hydrograph-grouped mode enabled: %d hydrographs with reference ensembles.",
            len(hydrograph_samples),
        )
    with _PhaseTimer(logger, "Rollout evaluation + plotting"):
        if hydrograph_samples:
            _rollout_prediction_per_hydrograph(
                models=models,
                hydrograph_samples=hydrograph_samples,
                rollout_length=config.data.rollout_length,
                history_steps=config.data.n_history,
                dynamic_norm=normalizers["dynamic"],
                target_norm=normalizers["target"],
                device=device,
                skip_before_timestep=_opt(config, "data", "skip_before_timestep", 0),
                dt=config.data.dt,
                out_dir=config.rollout.out_dir,
                target_variables=target_variables,
                logger=logger,
                fgn_noise_dim=_opt(config, "gino", "fgn_noise_dim", 32) if use_fgn else None,
                n_ensemble_samples=rollout_n_ensemble,
            )
        else:
            _rollout_prediction_generic(
                models=models,
                rollout_dataset=rollout_norm_ds,
                rollout_length=config.data.rollout_length,
                history_steps=config.data.n_history,
                dynamic_norm=normalizers["dynamic"],
                target_norm=normalizers["target"],
                device=device,
                skip_before_timestep=_opt(config, "data", "skip_before_timestep", 0),
                dt=config.data.dt,
                out_dir=config.rollout.out_dir,
                target_variables=target_variables,
                logger=logger,
                fgn_noise_dim=_opt(config, "gino", "fgn_noise_dim", 32) if use_fgn else None,
                n_ensemble_samples=rollout_n_ensemble,
            )
    logger.info("Evaluation finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

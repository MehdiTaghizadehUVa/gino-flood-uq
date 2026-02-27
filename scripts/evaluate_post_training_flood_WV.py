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
import logging
import os
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
DEFAULT_EVAL_LOG = "eval_post_training.log"


def _opt(config: Any, section: Optional[str], key: str, default: Any) -> Any:
    """Get config.section.key with default. Use section=None for top-level config keys."""
    if section is None:
        return getattr(config, key, default)
    obj = getattr(config, section, None)
    if obj is None:
        return default
    return getattr(obj, key, default)


class _PhaseTimer:
    """Context manager that logs phase name and duration."""

    def __init__(self, logger: logging.Logger, phase_name: str) -> None:
        self.logger = logger
        self.phase_name = phase_name
        self._t0 = 0.0

    def __enter__(self) -> _PhaseTimer:
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


def _rollout_prediction_generic(
    trainer: Any,
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
    """Run autoregressive rollout, compute RMSE/CSI, save visuals and aggregate metrics."""
    model = trainer.model
    model.eval()
    dynamic_norm.to(device)
    target_norm.to(device)
    os.makedirs(out_dir, exist_ok=True)

    per_channel_rmse: Dict[str, List[np.ndarray]] = {
        name: [] for name in target_variables
    }
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
        run_rmse = {name: [] for name in target_variables}
        run_csi_005: List[float] = []
        run_csi_03: List[float] = []
        run_pred_by_channel = {name: [] for name in target_variables}
        run_gt_by_channel = {name: [] for name in target_variables}
        current_dynamic = full_dynamic[skip_before_timestep:start_pred_t].clone()
        current_boundary = full_boundary[skip_before_timestep:start_pred_t].clone()

        static_0 = sample["static"].to(device).unsqueeze(0)
        geom_0 = geometry.to(device).unsqueeze(0)
        query_0 = sample["query_points"].to(device).unsqueeze(0)

        for t in range(rollout_length):
            dyn_flat = current_dynamic.permute(1, 0, 2).reshape(
                1, current_dynamic.shape[1], -1
            )
            bc_flat = current_boundary.permute(1, 0, 2).reshape(
                1, current_boundary.shape[1], -1
            )
            x = torch.cat([static_0, bc_flat, dyn_flat], dim=2)
            with torch.no_grad():
                if fgn_noise_dim is not None and n_ensemble_samples > 1:
                    preds = []
                    for _ in range(n_ensemble_samples):
                        z = torch.randn(fgn_noise_dim, device=device, dtype=x.dtype)
                        p = model(
                            input_geom=geom_0,
                            latent_queries=query_0,
                            output_queries=geom_0,
                            x=x,
                            ada_in=z,
                        )
                        preds.append(p)
                    pred = torch.stack(preds, dim=0).mean(dim=0)
                else:
                    pred = model(
                        input_geom=geom_0,
                        latent_queries=query_0,
                        output_queries=geom_0,
                        x=x,
                    )
            inv_pred = target_norm.inverse_transform(pred)
            inv_gt = dynamic_norm.inverse_transform(gt_rollout[t].unsqueeze(0))
            for ch_idx, ch_name in enumerate(target_variables):
                ch_pred = inv_pred[0, :, ch_idx].detach().cpu().numpy()
                ch_gt = inv_gt[0, :, ch_idx].detach().cpu().numpy()
                run_rmse[ch_name].append(
                    float(np.sqrt(np.mean((ch_pred - ch_gt) ** 2)))
                )
                run_pred_by_channel[ch_name].append(ch_pred)
                run_gt_by_channel[ch_name].append(ch_gt)
                if ch_name == "wd":
                    run_csi_005.append(_compute_csi(0.05, ch_pred, ch_gt))
                    run_csi_03.append(_compute_csi(0.3, ch_pred, ch_gt))
            current_dynamic = torch.cat(
                [current_dynamic[1:], pred.squeeze(0).unsqueeze(0)], dim=0
            )
            current_boundary = torch.cat(
                [current_boundary[1:], gt_boundary_rollout[t].unsqueeze(0)], dim=0
            )

        for ch_name in target_variables:
            per_channel_rmse[ch_name].append(
                np.array(run_rmse[ch_name], dtype=np.float64)
            )
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
    if "wd" in target_variables and wd_csi_005 and wd_csi_03:
        metrics["csi_005"] = np.stack(wd_csi_005, axis=0)
        metrics["csi_03"] = np.stack(wd_csi_03, axis=0)
    stats = {
        k: {"mean": v.mean(axis=0), "std": v.std(axis=0)}
        for k, v in metrics.items()
    }
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


def _resolve_checkpoint(save_dir: Path) -> Tuple[Path, str]:
    """Return (save_dir, checkpoint_name). Raises FileNotFoundError if no checkpoint."""
    if not save_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {save_dir}")
    if (save_dir / "best_model_state_dict.pt").exists():
        return save_dir, CHECKPOINT_BEST
    if (save_dir / "model_state_dict.pt").exists():
        return save_dir, CHECKPOINT_LAST
    found = [p.name for p in save_dir.iterdir()]
    raise FileNotFoundError(
        f"No checkpoint in {save_dir}. Expected one of {CHECKPOINT_FILES}. Found: {found}"
    )


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
) -> Any:
    """Build rollout test HDF dataset, collect fields, normalize, return NormalizedRolloutTestDataset."""
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
    return NormalizedRolloutTestDataset(
        normalized_samples=samples,
        query_res=config.data.query_res,
    )


def main() -> int:
    """Run post-training one-step and/or rollout evaluation."""
    args = _parse_args()
    _validate_args(args)
    config, device, is_logger = load_config_and_setup()
    device = _resolve_device(device)
    seed = _opt(config, "distributed", "seed", 123)
    deterministic = _opt(config, None, "deterministic", True)
    set_seed(seed, deterministic=deterministic)

    save_dir = Path(_opt(config, "checkpoint", "save_dir", "."))
    if not save_dir.is_absolute():
        save_dir = save_dir.resolve()
    save_dir, save_name = _resolve_checkpoint(save_dir)
    eval_log = Path(args.eval_log_file)
    if not eval_log.is_absolute():
        eval_log = save_dir / eval_log
    logger = setup_logging(
        log_level=_opt(config, None, "log_level", "INFO"),
        log_file=str(eval_log),
        logger_name="flood_eval",
    )
    if is_logger:
        logger.info("Post-training evaluation started")
        logger.info("Device=%s | Seed=%s | Deterministic=%s", device, seed, deterministic)
        logger.info("Checkpoint dir=%s | alias=%s", save_dir, save_name)
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
    normalizers = _load_or_fit_normalizers(config, train_raw, save_dir, logger)
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

    with _PhaseTimer(logger, "Loading model and checkpoint"):
        model = get_model(config)
        load_training_state(save_dir=save_dir, save_name=save_name, model=model)
        model = model.to(device).eval()
    inverse_test = _opt(config, None, "inverse_test", True)
    if normalizers.get("target") is not None:
        normalizers["target"] = normalizers["target"].to(device)
    data_processor = FloodGINODataProcessor(
        device=device,
        target_norm=normalizers.get("target"),
        inverse_test=inverse_test,
    )
    data_processor.wrap(model)
    trainer = _make_trainer(config, model, data_processor, device, logger)
    use_fgn = _opt(config, "gino", "use_fgn_noise", False)
    eval_losses = _build_eval_losses(config, use_fgn)
    logger.info("Eval losses=%s inverse_test=%s", list(eval_losses.keys()), inverse_test)

    run_single = args.run_single_step and not args.skip_single_step
    if run_single:
        with _PhaseTimer(logger, "One-step evaluation"):
            metrics = trainer.evaluate(eval_losses, test_loader, log_prefix="test")
        clean = {
            k: (v.item() if hasattr(v, "item") else v)
            for k, v in metrics.items()
            if not k.endswith("_outputs")
        }
        logger.info("One-step TEST metrics:")
        for k, v in clean.items():
            logger.info("  %s: %.6e", k, v)
            print(f"  {k}: {v:.6e}")

    run_rollout_cfg = bool(_opt(config, "rollout", "run_after_training", False))
    run_rollout = (run_rollout_cfg or args.run_rollout) and not args.skip_rollout
    if not run_rollout:
        logger.info(
            "Skipping rollout (run_after_training=false and --run_rollout not set)."
        )
        return 0

    rollout_norm_ds = _build_rollout_normalized_dataset(
        config, normalizers, target_variables, logger
    )
    logger.info("Rollout normalized dataset: %d runs", len(rollout_norm_ds))
    with _PhaseTimer(logger, "Rollout evaluation + plotting"):
        _rollout_prediction_generic(
            trainer=trainer,
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
            n_ensemble_samples=_opt(config, "rollout", "n_ensemble_samples", 1),
        )
    logger.info("Evaluation finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
